import asyncio
import copy
import hashlib
import inspect
import logging
import os
import time
from argparse import Namespace
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

import numpy as np
import pybase64
import sglang_router
from packaging.version import parse
from tqdm import tqdm

from slime.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from slime.rollout.filter_hub.base_types import MetricGatherer, call_dynamic_filter
from slime.utils.async_utils import run
from slime.utils.data import Dataset
from slime.utils.eval_config import EvalDatasetConfig
from slime.utils.http_utils import get, post
from slime.utils.metric_utils import compute_rollout_step
from slime.utils.misc import SingletonMeta, load_function
from slime.utils.processing_utils import encode_image_for_rollout_engine, load_processor, load_tokenizer
from slime.utils.types import Sample

from .rm_hub import async_rm, batched_async_rm

__all__ = ["generate_rollout"]

logger = logging.getLogger(__name__)


def _iter_leaf_samples(value: Any):
    if isinstance(value, Sample):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _iter_leaf_samples(item)


def _group_sample_stats(group: list[Sample] | list[list[Sample]]) -> dict[str, int]:
    samples = list(_iter_leaf_samples(group))
    removed = sum(1 for sample in samples if getattr(sample, "remove_sample", False))
    failed = sum(1 for sample in samples if sample.status == Sample.Status.FAILED)
    aborted = sum(1 for sample in samples if sample.status == Sample.Status.ABORTED)
    return {
        "samples": len(samples),
        "removed": removed,
        "failed": failed,
        "aborted": aborted,
        "all_removed": int(bool(samples) and removed == len(samples)),
        "all_failed": int(bool(samples) and failed == len(samples)),
        "any_removed": int(removed > 0),
        "any_failed": int(failed > 0),
    }


def _dynamic_sampling_max_groups(args: Namespace, target_data_size: int) -> int | None:
    raw = getattr(args, "dynamic_sampling_max_groups", None)
    if raw is not None:
        try:
            value = int(raw)
        except (TypeError, ValueError):
            value = 0
        return value if value > 0 else None

    if getattr(args, "dynamic_sampling_filter_path", None) is None:
        return None

    # Keep the default permissive for ordinary DAPO rejection sampling, while
    # still preventing unbounded replenishment when the environment repeatedly
    # returns non-trainable samples.
    over_sampling_batch_size = max(1, int(getattr(args, "over_sampling_batch_size", target_data_size) or 1))
    return max(target_data_size * 64, over_sampling_batch_size * 64)


def _dynamic_sampling_max_seconds(args: Namespace) -> float | None:
    raw = getattr(args, "dynamic_sampling_max_seconds", None)
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _rollout_abort_wait_timeout(args: Namespace) -> float:
    raw = getattr(args, "rollout_abort_wait_timeout", None)
    if raw is None:
        raw = getattr(args, "dynamic_sampling_max_seconds", None)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 300.0
    return value if value > 0 else 300.0


def _dynamic_sampling_failed_group_abort_min_groups(args: Namespace, target_data_size: int) -> int | None:
    raw = getattr(args, "dynamic_sampling_failed_group_abort_min_groups", None)
    if raw is None:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    return max(1, min(value, target_data_size))


def _dynamic_sampling_failed_group_abort_ratio(args: Namespace) -> float:
    raw = getattr(args, "dynamic_sampling_failed_group_abort_ratio", 1.0)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 1.0
    return min(max(value, 0.0), 1.0)


def _should_abort_for_failed_rollout_groups(
    *,
    completed_groups: int,
    kept_groups: int,
    failed_groups: int,
    min_groups: int | None,
    ratio: float,
) -> bool:
    if min_groups is None:
        return False
    if kept_groups > 0 or completed_groups < min_groups or completed_groups <= 0:
        return False
    if failed_groups <= 0:
        return False
    return failed_groups / completed_groups >= ratio


def _format_dynamic_sampling_state(
    *,
    rollout_id: int,
    target_data_size: int,
    submitted_groups: int,
    completed_groups: int,
    kept_groups: int,
    dropped_groups: int,
    removed_groups: int,
    failed_groups: int,
    failed_samples: int,
    removed_samples: int,
    pending_groups: int,
    max_groups: int | None,
    max_seconds: float | None,
    failed_group_abort_min_groups: int | None = None,
    failed_group_abort_ratio: float | None = None,
) -> str:
    return (
        f"rollout_id={rollout_id} target_groups={target_data_size} "
        f"kept={kept_groups} dropped={dropped_groups} "
        f"submitted={submitted_groups} completed={completed_groups} pending={pending_groups} "
        f"removed_groups={removed_groups} failed_groups={failed_groups} "
        f"removed_samples={removed_samples} failed_samples={failed_samples} "
        f"max_groups={max_groups if max_groups is not None else 'disabled'} "
        f"max_seconds={max_seconds if max_seconds is not None else 'disabled'} "
        f"failed_group_abort_min_groups="
        f"{failed_group_abort_min_groups if failed_group_abort_min_groups is not None else 'disabled'} "
        f"failed_group_abort_ratio="
        f"{failed_group_abort_ratio if failed_group_abort_ratio is not None else 'disabled'}"
    )


async def _cancel_pending_generation_tasks(args: Namespace, reason: str) -> None:
    state = GenerateState(args)
    pending = set(state.pendings)
    if not pending:
        state.reset()
        return

    logger.warning("Canceling %d pending rollout generation tasks: %s", len(pending), reason)
    state.aborted = True
    for task in pending:
        task.cancel()
    try:
        await asyncio.wait_for(
            asyncio.gather(*pending, return_exceptions=True),
            timeout=_rollout_abort_wait_timeout(args),
        )
    except asyncio.TimeoutError:
        logger.warning(
            "Timed out waiting for %d canceled rollout generation tasks: %s",
            len(pending),
            reason,
        )
    state.reset()


def _train_step_start(args: Namespace, rollout_id: int) -> int:
    try:
        steps_per_rollout = int(getattr(args, "num_steps_per_rollout", 1) or 1)
    except (TypeError, ValueError):
        steps_per_rollout = 1
    return int(rollout_id) * max(1, steps_per_rollout)


def _agent57_lite_enabled() -> bool:
    for name in (
        "EXPLORE_AGENT57_LITE_ENABLED",
        "EXPLORE_AGENT57_LITE",
        "EXPLORE_AGENT57_LIFELONG_ENABLED",
        "EXPLORE_AGENT57_LIFELONG",
    ):
        if os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}:
            return True
    return os.getenv("EXPLORE_AGENT57_CONTROLLER", "fixed").strip().lower() == "ucb"


def _fallback_agent57_arms(group_size: int) -> list[int]:
    try:
        k = int(os.getenv("EXPLORE_AGENT57_K", "8") or "8")
    except ValueError:
        k = 8
    k = max(1, k)
    return [idx % k for idx in range(max(0, group_size))]


def _agent57_float_list(name: str) -> list[float]:
    values: list[float] = []
    for part in os.getenv(name, "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            values.append(float(part))
        except ValueError:
            continue
    return values


def _agent57_int_list(name: str) -> list[int]:
    values: list[int] = []
    for part in os.getenv(name, "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            values.append(int(part))
        except ValueError:
            continue
    return values


def _agent57_int_env(name: str, default: int = 0) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _agent57_value_for_arm(values: list[Any], arm_id: int) -> Any | None:
    if not values:
        return None
    return values[int(arm_id) % len(values)]


def _agent57_sampling_warmup_active(metadata: dict[str, Any]) -> bool:
    warmup_rollouts = _agent57_int_env("EXPLORE_AGENT57_ARM_TEMPERATURE_WARMUP_ROLLOUTS", 0)
    if warmup_rollouts <= 0:
        return False
    try:
        rollout_id = int(metadata.get("rollout_id", warmup_rollouts))
    except (TypeError, ValueError):
        rollout_id = warmup_rollouts
    return rollout_id < warmup_rollouts


def _apply_agent57_sampling_params(sample: Sample, sampling_params: dict[str, Any]) -> None:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    if not metadata.get("agent57_lite_enabled"):
        return
    if _agent57_sampling_warmup_active(metadata):
        metadata["agent57_sampling_warmup_active"] = 1
        return
    metadata["agent57_sampling_warmup_active"] = 0
    try:
        arm_id = int(metadata.get("agent57_arm_id", 0))
    except (TypeError, ValueError):
        arm_id = 0

    temp = _agent57_value_for_arm(_agent57_float_list("EXPLORE_AGENT57_ARM_TEMPERATURES"), arm_id)
    top_p = _agent57_value_for_arm(_agent57_float_list("EXPLORE_AGENT57_ARM_TOP_PS"), arm_id)
    top_k = _agent57_value_for_arm(_agent57_int_list("EXPLORE_AGENT57_ARM_TOP_KS"), arm_id)
    if temp is not None:
        sampling_params["temperature"] = float(temp)
    if top_p is not None:
        sampling_params["top_p"] = float(top_p)
    if top_k is not None:
        sampling_params["top_k"] = int(top_k)


def _agent57_dataset_context(group: list[Sample]) -> str | None:
    if not group:
        return None
    sample = group[0]
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    prompt = sample.prompt if isinstance(sample.prompt, dict) else {}
    task_meta = metadata.get("task_meta") if isinstance(metadata.get("task_meta"), dict) else {}
    raw = (
        metadata.get("data_source")
        or task_meta.get("data_source")
        or prompt.get("data_source")
    )
    if raw:
        return str(raw)
    task_path = str(metadata.get("task_path") or task_meta.get("task_path") or "")
    if task_path.startswith("agent_safetybench/"):
        return "agent_safetybench"
    if task_path.startswith("seta_env/") or "seta" in task_path:
        return "seta"
    if task_path.startswith("agentharm/") or "agentharm" in task_path:
        return "agentharm"
    return None


def _assign_agent57_arms(
    group_size: int,
    *,
    evaluation: bool,
    dataset: str | None = None,
) -> list[int]:
    if evaluation or not _agent57_lite_enabled():
        return []
    try:
        from harborrl.algorithms.dive_po.exploration.agent57.controller import assign_group_arms

        return assign_group_arms(group_size, evaluation=evaluation, dataset=dataset)
    except Exception as exc:
        logger.warning("Agent57-lite arm assignment fallback: %s", exc)
        return _fallback_agent57_arms(group_size)


def _annotate_rollout_sample(args: Namespace, sample: Sample, rollout_id: int, *, evaluation: bool) -> None:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    sample.metadata = dict(metadata)
    sample.metadata["rollout_id"] = int(rollout_id)
    sample.metadata["rollout_step"] = int(compute_rollout_step(args, rollout_id))
    sample.metadata["train_step"] = _train_step_start(args, rollout_id)
    sample.metadata["evaluation"] = bool(evaluation)


def _annotate_rollout_groups(
    args: Namespace, samples: list[list[Sample]], rollout_id: int, *, evaluation: bool
) -> None:
    for group in samples:
        agent57_dataset = _agent57_dataset_context(group)
        agent57_arms = _assign_agent57_arms(
            len(group),
            evaluation=evaluation,
            dataset=agent57_dataset,
        )
        for idx, sample in enumerate(group):
            _annotate_rollout_sample(args, sample, rollout_id, evaluation=evaluation)
            if agent57_arms:
                sample.metadata["agent57_lite_enabled"] = True
                sample.metadata["agent57_arm_id"] = int(agent57_arms[idx])
                sample.metadata["agent57_group_position"] = int(idx)
                if agent57_dataset:
                    sample.metadata["agent57_dataset"] = agent57_dataset


class GenerateState(metaclass=SingletonMeta):
    """
    The global state for the generation process.
    """

    def __init__(self, args: Namespace) -> None:
        # persistent state for the generation process
        self.args = args
        self.tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        self.processor = load_processor(args.hf_checkpoint, trust_remote_code=True)

        self.semaphore = asyncio.Semaphore(
            args.sglang_server_concurrency * args.rollout_num_gpus // args.rollout_num_gpus_per_engine
        )
        self.sampling_params: dict[str, Any] = dict(
            temperature=args.rollout_temperature,
            top_p=args.rollout_top_p,
            top_k=args.rollout_top_k,
            max_new_tokens=args.rollout_max_response_len,
            stop=args.rollout_stop,
            stop_token_ids=args.rollout_stop_token_ids,
            skip_special_tokens=args.rollout_skip_special_tokens,
            no_stop_trim=True,
            spaces_between_special_tokens=False,
        )

        if getattr(args, "sglang_enable_deterministic_inference", False):
            sampling_seed_base = args.rollout_seed
            self.group_sampling_seeds = [sampling_seed_base + i for i in range(args.n_samples_per_prompt)]

        # dp rank balancing
        self.dp_counts = [0] * (args.sglang_dp_size or 1)
        self.dp_rank = 0

        self.reset()

    @contextmanager
    def dp_rank_context(self):
        candidates = [i for i, count in enumerate(self.dp_counts) if count == min(self.dp_counts)]
        dp_rank = int(np.random.choice(candidates))
        self.dp_counts[dp_rank] += 1
        self.dp_rank = dp_rank
        try:
            yield dp_rank
        finally:
            self.dp_counts[dp_rank] -= 1
            assert self.dp_counts[dp_rank] >= 0

    def reset(self) -> None:
        self.remaining_batch_size = 0
        self.pendings = set()
        self.aborted = False

    def submit_generate_tasks(self, samples: list[list[Sample]]) -> None:
        for group in samples:
            generate_group_func = generate_and_rm_group
            if os.getenv("HARBORRL_NATIVE_ROLLOUT") == "1":
                from harborrl.rollout.native_generate import generate_group as native_generate_group

                generate_group_func = native_generate_group
            self.pendings.add(
                asyncio.create_task(
                    # submit a group of samples as a single task.
                    generate_group_func(
                        self.args,
                        group,
                        sampling_params=self.sampling_params.copy(),
                        evaluation=False,
                    )
                )
            )
        self.remaining_batch_size += len(samples)


async def generate(args: Namespace, sample: Sample, sampling_params: dict[str, Any]) -> Sample:
    """Generate using traditional SGLang router with token-based workflow"""
    if args.ci_test:
        assert isinstance(sample.prompt, str)

    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    assert (
        sample.status == Sample.Status.PENDING or sample.status == Sample.Status.ABORTED
    ), f"Sample status is {sample.status}"

    if state.processor:
        processor_output = state.processor(text=sample.prompt, **sample.multimodal_inputs)
        prompt_ids = processor_output["input_ids"][0]
        sample.multimodal_train_inputs = {
            k: v for k, v in processor_output.items() if k not in ["input_ids", "attention_mask"]
        } or None
    else:
        prompt_ids = state.tokenizer.encode(sample.prompt, add_special_tokens=False)

    if len(sample.response) > 0:
        sampling_params["max_new_tokens"] -= len(sample.tokens) - len(prompt_ids)

    assert (
        sampling_params["max_new_tokens"] >= 0
    ), f"max_new_tokens: {sampling_params['max_new_tokens']} should not be less than 0"
    if sampling_params["max_new_tokens"] == 0:
        sample.status = Sample.Status.TRUNCATED
        return sample

    # Prepare payload for sglang server
    payload = {
        "sampling_params": sampling_params,
        "return_logprob": True,
    }

    if args.use_rollout_routing_replay:
        payload["return_routed_experts"] = True

    if sample.multimodal_inputs and sample.multimodal_inputs["images"]:
        image_data = sample.multimodal_inputs["images"]
        payload["image_data"] = [encode_image_for_rollout_engine(image) for image in image_data]

    # Use existing tokens for multi-turn or tokenize the new prompt
    if len(sample.response) > 0:
        payload["input_ids"] = sample.tokens
    else:
        payload["input_ids"] = prompt_ids
        if not sample.tokens:  # Initialize sample.tokens for the first turn
            sample.tokens = prompt_ids

    output = await post(url, payload)

    if args.use_slime_router and "RadixTreeMiddleware" in args.slime_router_middleware_paths:
        from slime.router.middleware_hub.radix_tree_middleware import postprocess_sample_with_radix_tree

        sample = await postprocess_sample_with_radix_tree(args, sample, output)
    else:
        if "output_token_logprobs" in output["meta_info"]:
            new_response_tokens = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
            new_response_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
        else:
            new_response_tokens, new_response_log_probs = [], []

        # Update sample with tokens directly - avoiding re-tokenization
        sample.tokens = sample.tokens + new_response_tokens
        sample.response_length += len(new_response_tokens)
        sample.response += output["text"]

        # When partial rollout and masking off policy is enabled, update the loss mask
        if sample.loss_mask is not None:
            assert args.partial_rollout and args.mask_offpolicy_in_partial_rollout
            sample.loss_mask += [1] * len(new_response_tokens)

        if sample.rollout_log_probs is None:
            sample.rollout_log_probs = []
        sample.rollout_log_probs += new_response_log_probs

    if "routed_experts" in output["meta_info"]:
        sample.rollout_routed_experts = np.frombuffer(
            pybase64.b64decode(output["meta_info"]["routed_experts"].encode("ascii")),
            dtype=np.int32,
        ).reshape(
            len(sample.tokens) - 1,
            args.num_layers,
            args.moe_router_topk,
        )

    sample.update_from_meta_info(args, output["meta_info"])

    return sample


async def generate_and_rm(
    args: Namespace,
    sample: Sample | list[Sample],
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> Sample | list[Sample]:
    # mask previous off-policy generation for partial rollout
    if args.partial_rollout and args.mask_offpolicy_in_partial_rollout and sample.response_length > 0:
        sample.loss_mask = [0] * sample.response_length

    # For samples with existing response, check if they're complete
    if sample.status == Sample.Status.COMPLETED or sample.status == Sample.Status.TRUNCATED:
        assert sample.response is not None
        if not args.group_rm:
            assert sample.reward is not None
        return sample

    state = GenerateState(args)

    # generate
    async with state.semaphore:
        if state.aborted:
            sample.status = Sample.Status.ABORTED
            return sample

        with state.dp_rank_context() as _:
            # Check sample.generate_function_path for per-sample custom_generate_function_path (e.g., from eval dataset config)
            custom_func_path = getattr(sample, "generate_function_path", None) or args.custom_generate_function_path

            if custom_func_path is not None:
                custom_generate_func = load_function(custom_func_path)
                # if signature has evaluation, pass evaluation
                if "evaluation" in inspect.signature(custom_generate_func).parameters:
                    sample = await custom_generate_func(args, sample, sampling_params, evaluation=evaluation)
                else:
                    sample = await custom_generate_func(args, sample, sampling_params)
            else:
                sample = await generate(args, sample, sampling_params)

    # for the rm that need the whole group, we will not do the rm here
    if args.group_rm:
        return sample

    # multi samples
    if isinstance(sample, list):
        samples = sample
        if any([sample.status == Sample.Status.ABORTED for sample in samples]):
            return samples

        # for multi agent system, the reward of some sample is calculated during generation.
        samples_need_reward = [sample for sample in samples if sample.reward is None]
        rewards = await batched_async_rm(args, samples_need_reward)
        for sample, reward in zip(samples_need_reward, rewards, strict=False):
            sample.reward = reward
        return samples
    else:
        if sample.status == Sample.Status.ABORTED:
            return sample
        # for multi-turn environment, a reward could be assigned to the agent.
        if sample.reward is None:
            sample.reward = await async_rm(args, sample)

    return sample


async def generate_and_rm_group(
    args: Namespace, group: list[Sample], sampling_params: dict[str, Any], evaluation: bool = False
) -> list[Sample]:
    state = GenerateState(args)

    if state.aborted:
        return group

    tasks = []
    for idx, sample in enumerate(group):
        current_sampling_params = sampling_params.copy()
        _apply_agent57_sampling_params(sample, current_sampling_params)
        if getattr(args, "sglang_enable_deterministic_inference", False):
            seed = state.group_sampling_seeds[idx]
            current_sampling_params["sampling_seed"] = seed
        tasks.append(
            asyncio.create_task(generate_and_rm(args, sample, current_sampling_params, evaluation=evaluation))
        )

    group = await asyncio.gather(*tasks)

    # for the rm that need the whole group, we will do the rm here
    if not state.aborted and args.group_rm:
        rewards = await batched_async_rm(args, group)
        for sample, reward in zip(group, rewards, strict=False):
            sample.reward = reward

    return group


async def abort(args: Namespace, rollout_id: int) -> list[list[Sample]]:
    aborted_samples = []

    state = GenerateState(args)
    assert not state.aborted
    state.aborted = True

    # Aborting is cleanup: a worker can disappear while the router is
    # recovering it.  Do not turn one dead endpoint into a fatal driver error.
    try:
        if parse(sglang_router.__version__) <= parse("0.2.1") or args.use_slime_router:
            response = await get(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/list_workers")
            urls = response["urls"]
        else:
            response = await get(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/workers")
            urls = [worker["url"] for worker in response["workers"]]
    except Exception as exc:
        logger.warning("Unable to list rollout workers during abort; continuing cleanup: %r", exc)
        urls = []

    async def _abort_worker(url: str) -> None:
        try:
            await post(f"{url}/abort_request", {"abort_all": True})
        except Exception as exc:
            logger.warning("Unable to abort rollout worker %s; it may already be dead: %r", url, exc)

    logger.info("Abort request for %s", urls)
    await asyncio.gather(*[_abort_worker(url) for url in urls])

    # make sure all the pending tasks are finished
    count = 0
    deadline = time.monotonic() + _rollout_abort_wait_timeout(args)
    while state.pendings:
        timeout = max(0.0, deadline - time.monotonic())
        if timeout <= 0:
            logger.warning(
                "Timed out waiting for %d pending rollout generation tasks during abort; canceling leftovers",
                len(state.pendings),
            )
            await _cancel_pending_generation_tasks(args, "rollout abort wait timeout")
            break

        done, state.pendings = await asyncio.wait(
            state.pendings,
            timeout=timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if not done:
            logger.warning(
                "Timed out waiting for %d pending rollout generation tasks during abort; canceling leftovers",
                len(state.pendings),
            )
            await _cancel_pending_generation_tasks(args, "rollout abort wait timeout")
            break

        if args.partial_rollout:
            # for partial rollout, collect the partial samples into the data buffer
            for task in done:
                group = task.result()
                for sample in group:
                    if sample.response and "start_rollout_id" not in sample.metadata:
                        sample.metadata["start_rollout_id"] = rollout_id
                aborted_samples.append(group)
                count += len(group)

    if args.partial_rollout:
        logger.info(f"Collected {count} partial samples into the data buffer")

    state.reset()
    return aborted_samples


async def generate_rollout_async(
    args: Namespace, rollout_id: int, data_source: Callable[[int], list[list[Sample]]]
) -> tuple[RolloutFnTrainOutput, list[list[Sample]]]:
    """An example to implement the generate_rollout function for an rule based rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        data_source: the data source to fetch

    Returns:
        tuple[RolloutFnTrainOutput, list[list[Sample]]]:
            - data: a list of groups of samples generated by the rollout, length equals `rollout_batch_size`
            - aborted_samples: any partial groups collected during abort when partial_rollout is enabled
    """
    assert args.rollout_global_dataset

    state = GenerateState(args)

    # instantiate data filters
    dynamic_filter = (
        load_function(args.dynamic_sampling_filter_path) if args.dynamic_sampling_filter_path is not None else None
    )

    metric_gatherer = MetricGatherer()

    # target_data_size is the total number of valid samples to get
    target_data_size = args.rollout_batch_size
    max_groups = _dynamic_sampling_max_groups(args, target_data_size)
    max_seconds = _dynamic_sampling_max_seconds(args)
    failed_group_abort_min_groups = _dynamic_sampling_failed_group_abort_min_groups(args, target_data_size)
    failed_group_abort_ratio = _dynamic_sampling_failed_group_abort_ratio(args)
    deadline = time.monotonic() + max_seconds if max_seconds is not None else None

    data = []
    all_data = []
    do_print = True
    submitted_groups = 0
    completed_groups = 0
    kept_groups = 0
    dropped_groups = 0
    removed_groups = 0
    failed_groups = 0
    removed_samples = 0
    failed_samples = 0
    exhausted = False
    pbar = tqdm(total=target_data_size * args.n_samples_per_prompt, desc="Rollout generation")
    try:
        while len(data) < target_data_size:
            while state.remaining_batch_size < target_data_size:
                if max_groups is not None and submitted_groups >= max_groups:
                    exhausted = True
                    break

                # get samples from the buffer and submit the generation requests.
                fetch_size = args.over_sampling_batch_size
                if max_groups is not None:
                    fetch_size = min(fetch_size, max_groups - submitted_groups)
                if fetch_size <= 0:
                    exhausted = True
                    break

                samples = data_source(fetch_size)
                if not samples:
                    reason = "dynamic sampling data source returned no groups"
                    details = _format_dynamic_sampling_state(
                        rollout_id=rollout_id,
                        target_data_size=target_data_size,
                        submitted_groups=submitted_groups,
                        completed_groups=completed_groups,
                        kept_groups=kept_groups,
                        dropped_groups=dropped_groups,
                        removed_groups=removed_groups,
                        failed_groups=failed_groups,
                        failed_samples=failed_samples,
                        removed_samples=removed_samples,
                        pending_groups=len(state.pendings),
                        max_groups=max_groups,
                        max_seconds=max_seconds,
                        failed_group_abort_min_groups=failed_group_abort_min_groups,
                        failed_group_abort_ratio=failed_group_abort_ratio,
                    )
                    raise RuntimeError(f"{reason}: {details}")
                _annotate_rollout_groups(args, samples, rollout_id, evaluation=False)
                state.submit_generate_tasks(samples)
                submitted_groups += len(samples)

            if len(data) >= target_data_size:
                break

            if not state.pendings:
                reason = "dynamic sampling exhausted before collecting enough kept groups"
                details = _format_dynamic_sampling_state(
                    rollout_id=rollout_id,
                    target_data_size=target_data_size,
                    submitted_groups=submitted_groups,
                    completed_groups=completed_groups,
                    kept_groups=kept_groups,
                    dropped_groups=dropped_groups,
                    removed_groups=removed_groups,
                    failed_groups=failed_groups,
                    failed_samples=failed_samples,
                    removed_samples=removed_samples,
                    pending_groups=0,
                    max_groups=max_groups,
                    max_seconds=max_seconds,
                    failed_group_abort_min_groups=failed_group_abort_min_groups,
                    failed_group_abort_ratio=failed_group_abort_ratio,
                )
                raise RuntimeError(f"{reason}: {details}")

            timeout = None
            if deadline is not None:
                timeout = max(0.0, deadline - time.monotonic())
            done, state.pendings = await asyncio.wait(
                state.pendings,
                timeout=timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                reason = "dynamic sampling timed out before collecting enough kept groups"
                details = _format_dynamic_sampling_state(
                    rollout_id=rollout_id,
                    target_data_size=target_data_size,
                    submitted_groups=submitted_groups,
                    completed_groups=completed_groups,
                    kept_groups=kept_groups,
                    dropped_groups=dropped_groups,
                    removed_groups=removed_groups,
                    failed_groups=failed_groups,
                    failed_samples=failed_samples,
                    removed_samples=removed_samples,
                    pending_groups=len(state.pendings),
                    max_groups=max_groups,
                    max_seconds=max_seconds,
                    failed_group_abort_min_groups=failed_group_abort_min_groups,
                    failed_group_abort_ratio=failed_group_abort_ratio,
                )
                raise RuntimeError(f"{reason}: {details}")

            for task in done:
                group: list[Sample] = task.result()
                completed_groups += 1

                if do_print:
                    sample = group[0][0] if isinstance(group[0], list) else group[0]
                    logger.info(
                        f"First rollout sample: {[str(sample.prompt) + sample.response]}, label: {str(sample.label)[:100]}, reward: {sample.reward}",
                    )
                    do_print = False

                stats = _group_sample_stats(group)
                removed_groups += stats["all_removed"]
                failed_groups += stats["all_failed"]
                removed_samples += stats["removed"]
                failed_samples += stats["failed"]

                assert len(group) == args.n_samples_per_prompt
                all_data.append(group)
                dynamic_filter_output = call_dynamic_filter(dynamic_filter, args, group)
                if not dynamic_filter_output.keep:
                    dropped_groups += 1
                    metric_gatherer.on_dynamic_filter_drop(reason=dynamic_filter_output.reason)
                    state.remaining_batch_size -= 1
                    continue

                # add the samples to the data
                # NOTE: here we have not stored all the unused samples back to the data buffer.
                if len(data) < target_data_size:
                    data.append(group)
                    kept_groups += 1
                    pbar.update(args.n_samples_per_prompt)

            if _should_abort_for_failed_rollout_groups(
                completed_groups=completed_groups,
                kept_groups=kept_groups,
                failed_groups=failed_groups,
                min_groups=failed_group_abort_min_groups,
                ratio=failed_group_abort_ratio,
            ):
                reason = "dynamic sampling aborted after repeated all-failed rollout groups"
                details = _format_dynamic_sampling_state(
                    rollout_id=rollout_id,
                    target_data_size=target_data_size,
                    submitted_groups=submitted_groups,
                    completed_groups=completed_groups,
                    kept_groups=kept_groups,
                    dropped_groups=dropped_groups,
                    removed_groups=removed_groups,
                    failed_groups=failed_groups,
                    failed_samples=failed_samples,
                    removed_samples=removed_samples,
                    pending_groups=len(state.pendings),
                    max_groups=max_groups,
                    max_seconds=max_seconds,
                    failed_group_abort_min_groups=failed_group_abort_min_groups,
                    failed_group_abort_ratio=failed_group_abort_ratio,
                )
                raise RuntimeError(f"{reason}: {details}")

            if exhausted and len(data) < target_data_size and not state.pendings:
                reason = "dynamic sampling reached max groups before collecting enough kept groups"
                details = _format_dynamic_sampling_state(
                    rollout_id=rollout_id,
                    target_data_size=target_data_size,
                    submitted_groups=submitted_groups,
                    completed_groups=completed_groups,
                    kept_groups=kept_groups,
                    dropped_groups=dropped_groups,
                    removed_groups=removed_groups,
                    failed_groups=failed_groups,
                    failed_samples=failed_samples,
                    removed_samples=removed_samples,
                    pending_groups=0,
                    max_groups=max_groups,
                    max_seconds=max_seconds,
                    failed_group_abort_min_groups=failed_group_abort_min_groups,
                    failed_group_abort_ratio=failed_group_abort_ratio,
                )
                raise RuntimeError(f"{reason}: {details}")
    except Exception:
        pbar.close()
        await _cancel_pending_generation_tasks(args, "dynamic sampling did not complete")
        raise

    pbar.close()
    sample = data[-1][0][0] if isinstance(data[-1][0], list) else data[-1][0]
    logger.info(
        f"Finish rollout: {[str(sample.prompt) + sample.response]}, label: {str(sample.label)[:100]}, reward: {sample.reward}",
    )

    # there are still some unfinished requests, abort them
    aborted_samples = await abort(args, rollout_id)

    assert len(data) == args.rollout_batch_size, f"Got {len(data)} samples, expected {args.rollout_batch_size}"
    data = sorted(data, key=lambda group: group[0][0].index if isinstance(group[0], list) else group[0].index)
    all_samples = sorted(
        all_data, key=lambda group: group[0][0].index if isinstance(group[0], list) else group[0].index
    )

    # reset the global state to prevent effects on the next rollout or eval.
    state.reset()
    if args.rollout_sample_filter_path is not None:
        filter_func = load_function(args.rollout_sample_filter_path)
        filter_func(args, data)

    # There can be circumstances where users want to process all samples including filtered ones.
    if args.rollout_all_samples_process_path is not None:
        process_func = load_function(args.rollout_all_samples_process_path)
        process_func(args, all_samples, data_source)

    metrics = metric_gatherer.collect()
    metrics.update(
        {
            "rollout/dynamic_sampling/submitted_groups": submitted_groups,
            "rollout/dynamic_sampling/completed_groups": completed_groups,
            "rollout/dynamic_sampling/kept_groups": kept_groups,
            "rollout/dynamic_sampling/dropped_groups": dropped_groups,
            "rollout/dynamic_sampling/removed_groups": removed_groups,
            "rollout/dynamic_sampling/failed_groups": failed_groups,
            "rollout/dynamic_sampling/removed_samples": removed_samples,
            "rollout/dynamic_sampling/failed_samples": failed_samples,
            "rollout/dynamic_sampling/max_groups": max_groups or 0,
            "rollout/dynamic_sampling/max_seconds": max_seconds or 0,
            "rollout/dynamic_sampling/failed_group_abort_min_groups": failed_group_abort_min_groups or 0,
            "rollout/dynamic_sampling/failed_group_abort_ratio": failed_group_abort_ratio,
        }
    )

    return RolloutFnTrainOutput(samples=data, metrics=metrics), aborted_samples


EVAL_PROMPT_DATASET = {}


async def eval_rollout(args: Namespace, rollout_id: int) -> tuple[dict[str, dict[str, list[Any]]], list[list[Sample]]]:
    assert not args.group_rm, "Group RM is not supported for eval rollout"

    coros = []
    for dataset_cfg in getattr(args, "eval_datasets", []) or []:
        coros.append(eval_rollout_single_dataset(args, rollout_id, dataset_cfg))
    results_list = await asyncio.gather(*coros)
    results = {}
    for r in results_list:
        results.update(r)
    return RolloutFnEvalOutput(data=results), []


async def eval_rollout_single_dataset(
    args: Namespace, rollout_id: int, dataset_cfg: EvalDatasetConfig
) -> dict[str, dict[str, list[Any]]]:
    """An example to implement the eval_rollout function for an rule based rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        dataset_cfg: configuration of the dataset
    """
    assert not args.group_rm, "Group RM is not supported for eval rollout"

    global EVAL_PROMPT_DATASET

    cache_key = dataset_cfg.cache_key + (args.hf_checkpoint, args.apply_chat_template)
    if cache_key not in EVAL_PROMPT_DATASET:
        tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        processor = load_processor(args.hf_checkpoint, trust_remote_code=True)
        EVAL_PROMPT_DATASET[cache_key] = Dataset(
            path=dataset_cfg.path,
            tokenizer=tokenizer,
            processor=processor,
            max_length=args.eval_max_prompt_len,
            prompt_key=dataset_cfg.input_key,
            label_key=dataset_cfg.label_key,
            multimodal_keys=args.multimodal_keys,
            metadata_key=dataset_cfg.metadata_key,
            tool_key=dataset_cfg.tool_key,
            apply_chat_template=args.apply_chat_template,
            apply_chat_template_kwargs=args.apply_chat_template_kwargs,
        )
    dataset = EVAL_PROMPT_DATASET[cache_key]

    base_sampling_params = dict(
        temperature=dataset_cfg.temperature,
        top_p=dataset_cfg.top_p,
        top_k=dataset_cfg.top_k,
        max_new_tokens=dataset_cfg.max_response_len,
        stop=args.rollout_stop,
        stop_token_ids=args.rollout_stop_token_ids,
        skip_special_tokens=args.rollout_skip_special_tokens,
        no_stop_trim=True,
        spaces_between_special_tokens=False,
    )

    eval_max_concurrency = int(
        os.getenv("EVAL_ROLLOUT_MAX_CONCURRENCY", "0") or 0
    )
    eval_semaphore = (
        asyncio.Semaphore(eval_max_concurrency) if eval_max_concurrency > 0 else None
    )

    async def _generate_eval_sample(
        sample: Sample, sampling_params: dict[str, Any]
    ) -> list[Sample]:
        if eval_semaphore is None:
            return await generate_and_rm(
                args,
                sample,
                sampling_params=sampling_params,
                evaluation=True,
            )
        async with eval_semaphore:
            return await generate_and_rm(
                args,
                sample,
                sampling_params=sampling_params,
                evaluation=True,
            )

    tasks = []
    # do multiple samples for eval prompts
    sample_index = 0
    for _i, prompt_sample in enumerate(dataset.samples):
        for j in range(dataset_cfg.n_samples_per_eval_prompt):
            # use the same prompt for multiple samples
            sample = copy.deepcopy(prompt_sample)
            sample.index = sample_index
            sample_index += 1
            sample.metadata = dataset_cfg.inject_metadata(getattr(sample, "metadata", None))
            _annotate_rollout_sample(args, sample, rollout_id, evaluation=True)
            sample.metadata["eval_dataset"] = dataset_cfg.name
            sample.metadata["eval_prompt_index"] = _i
            sample.metadata["eval_sample_index"] = j
            sample.generate_function_path = getattr(dataset_cfg, "custom_generate_function_path", None)
            sampling_params = base_sampling_params
            eval_seed = getattr(args, "eval_seed", None)
            if eval_seed is not None:
                sampling_params = base_sampling_params.copy()
                seed_payload = f"{eval_seed}:{dataset_cfg.name}:{_i}:{j}".encode()
                sampling_seed = int.from_bytes(hashlib.sha256(seed_payload).digest()[:4], "big")
                sampling_params["sampling_seed"] = sampling_seed
                sample.metadata["eval_sampling_seed"] = sampling_seed
            elif getattr(args, "sglang_enable_deterministic_inference", False):
                sampling_params = base_sampling_params.copy()
                sampling_params["sampling_seed"] = args.rollout_seed + j
                sample.metadata["eval_sampling_seed"] = args.rollout_seed + j
            tasks.append(
                asyncio.create_task(
                    _generate_eval_sample(sample, sampling_params=sampling_params)
                )
            )

    data = []
    do_print = True
    pbar = tqdm(total=len(tasks), desc=f"Eval {dataset_cfg.name}", disable=not do_print)
    for coro in asyncio.as_completed(tasks):
        sample = await coro
        if do_print:
            example_sample = sample[0] if isinstance(sample, list) and sample else sample
            example_prompt = getattr(example_sample, "prompt", "<empty sample>")
            example_response = getattr(example_sample, "response", "") or ""
            example_reward = getattr(example_sample, "reward", None)
            logger.info(
                "eval_rollout_single_dataset example data: "
                f"{[str(example_prompt) + example_response]} "
                f"reward={example_reward}"
            )
            do_print = False
        if isinstance(sample, list):
            data.extend(sample)
        else:
            data.append(sample)
        pbar.update(1)
    pbar.close()

    data.sort(key=lambda sample: sample.index)

    reward_key = args.eval_reward_key or args.reward_key
    return {
        dataset_cfg.name: {
            "rewards": [sample.reward if not reward_key else sample.reward[reward_key] for sample in data],
            "truncated": [sample.status == Sample.Status.TRUNCATED for sample in data],
            "samples": data,
        }
    }


def generate_rollout(
    args: Namespace, rollout_id: int, data_source: Any, evaluation: bool = False
) -> RolloutFnTrainOutput | RolloutFnEvalOutput:
    """An example to implement the generate_rollout function for an rule based rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        data_buffer: the data buffer to store the generated samples
        evaluation: bool, whether the rollout is for evaluation or not

    Returns:
        list[list[Sample]]: a list of list of samples generated by the rollout
    """
    assert args.rollout_global_dataset
    if evaluation:
        output, _ = run(eval_rollout(args, rollout_id))
        return output

    output, aborted_samples = run(generate_rollout_async(args, rollout_id, data_source.get_samples))
    data_source.add_samples(aborted_samples)
    return output
