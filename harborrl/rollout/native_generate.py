"""Slime group hook for native Harbor trials.

The hook keeps one trajectory per Slime slot and stores its immutable IR in
metadata. Turn expansion happens only at the training converter boundary.
"""
from __future__ import annotations

import asyncio
import json
import os
import secrets
from pathlib import Path

from harborrl.rollout.harbor_job.coordinator import GroupSlots, run_attempt
from harborrl.trajectories.native import Identity, RewardProfile, digest, read_json


class NativeRolloutRuntime:
    def __init__(self, args):
        from slime.utils.processing_utils import load_tokenizer

        from harborrl.gateway.server import Gateway, make_server
        from harborrl.gateway.sglang_backend import SGLangBackend
        from harborrl.gateway.trace import TraceRegistry

        self.args = args
        self.run_root = Path(os.environ["RUN_DIR"]).resolve()
        self.catalog = read_json(os.environ["HARBORRL_NATIVE_CATALOG"])
        self.profile = read_json(os.environ["HARBORRL_NATIVE_PROFILE"])
        self.rollout_root = self.run_root / "native-rollouts"
        self.rollout_root.mkdir(parents=True, exist_ok=True)
        self.registry = TraceRegistry(self.run_root / "native-traces")
        engine_count = int(args.rollout_num_gpus) // int(args.rollout_num_gpus_per_engine)
        limit = max(1, int(args.sglang_server_concurrency) * engine_count)
        self.semaphore = asyncio.Semaphore(limit)
        tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        tokenizer_digest, template_digest = tokenizer_fingerprint(tokenizer)
        expected_tokenizer = os.environ["HARBORRL_NATIVE_TOKENIZER_DIGEST"]
        expected_template = os.environ["HARBORRL_NATIVE_TEMPLATE_DIGEST"]
        if (tokenizer_digest != expected_tokenizer or template_digest != expected_template):
            raise ValueError("live tokenizer/template digest differs from the native lock")
        endpoint = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
        backend = SGLangBackend(
            endpoint,
            tokenizer,
            tokenizer_digest=tokenizer_digest,
            template_digest=template_digest,
            max_context=int(args.rollout_max_context_len),
                audited_raw_logprobs=os.environ.get("HARBORRL_NATIVE_AUDITED_LOGPROBS") == "1",
        )
        self.backend = backend
        gateway = Gateway(
            self.registry,
            backend,
            model=self.profile["model"],
            max_output_tokens=int(args.rollout_max_response_len),
        )
        self.server = make_server(
            gateway,
            host=os.environ.get("HARBORRL_NATIVE_GATEWAY_HOST", "0.0.0.0"),
            port=int(os.environ["HARBORRL_NATIVE_GATEWAY_PORT"]),
        )
        self.thread = asyncio.get_running_loop().run_in_executor(None, self.server.serve_forever)

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def tokenizer_fingerprint(tokenizer):
    messages = [{"role": "user", "content": "HarborRL native tokenizer/template fingerprint"}]
    kwargs = {"tokenize": True, "add_generation_prompt": True, "enable_thinking": False}
    rendered = tokenizer.apply_chat_template(messages, **kwargs)
    template = digest({"messages": messages, "kwargs": kwargs, "token_ids": rendered})
    vocabulary = tokenizer.get_vocab()
    tokenizer_digest = digest({
        "class": type(tokenizer).__name__,
        "vocab": sorted((str(key), int(value)) for key, value in vocabulary.items()),
    })
    return tokenizer_digest, template


_RUNTIME = None
_RUNTIME_LOCK = asyncio.Lock()


async def runtime(args):
    global _RUNTIME
    if _RUNTIME is None:
        async with _RUNTIME_LOCK:
            if _RUNTIME is None:
                _RUNTIME = NativeRolloutRuntime(args)
    return _RUNTIME


def publish_native_version(states, expected):
    """Open the next admission epoch only after the complete pool is healthy."""
    if _RUNTIME is None:
        return
    _RUNTIME.registry.drain(float(os.environ.get("HARBORRL_NATIVE_DRAIN_TIMEOUT", "30")))
    _RUNTIME.registry.publish_version(states, str(expected))


def current_policy_version(runtime_):
    """Publish and lock the version currently served by the rollout pool."""
    states = runtime_.backend.pool_states()
    from harborrl.platform.policy_pool import publish_pool
    version = publish_pool(states)
    runtime_.registry.drain(float(os.environ.get("HARBORRL_NATIVE_DRAIN_TIMEOUT", "30")))
    runtime_.registry.publish_version(states, version)
    return version


def close_runtime():
    global _RUNTIME
    if _RUNTIME is not None:
        _RUNTIME.close()
        _RUNTIME = None


def _task_entry(sample):
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    entry = metadata.get("native_task")
    required = {"id", "revision", "path", "task_digest", "reward_profile"}
    if not isinstance(entry, dict) or set(entry) != required:
        raise ValueError("native rollout requires a locked task catalog entry")
    return entry


def _identity(
    args,
    sample,
    slot,
    rollout_id,
    task_digest,
    reward_digest,
    sampling_digest,
    policy_version,
):
    group_index = getattr(sample, "group_index", None)
    if type(group_index) is not int:
        raise ValueError("Slime must provide group identity before native generation")
    return Identity(
        run_id=os.environ["RUN_ID"],
        batch_id=f"rollout-{rollout_id}",
        group_id=f"group-{group_index}",
        slot_id=str(slot),
        attempt_id=secrets.token_hex(16),
        trajectory_id=secrets.token_hex(16),
        task_digest=task_digest,
        policy_version=str(policy_version),
        harness_digest=digest(read_json(os.environ["HARBORRL_NATIVE_PROFILE"])),
        reward_digest=reward_digest,
        sampling_digest=sampling_digest,
    )


async def _run_slot(
    args,
    runtime_,
    sample,
    slot,
    rollout_id,
    entry,
    sampling_digest,
    policy_version,
):
    reward_profile = RewardProfile(**entry["reward_profile"])
    identities = [
        _identity(
            args,
            sample,
            index,
            rollout_id=rollout_id,
            task_digest=entry["task_digest"],
            reward_digest=reward_profile.digest,
            sampling_digest=sampling_digest,
            policy_version=policy_version,
        )
        for index in range(args.n_samples_per_prompt)
    ]
    slots = GroupSlots(identities, max_attempts=int(os.environ["HARBORRL_NATIVE_MAX_ATTEMPTS"]))
    workers = json.loads(os.environ["HARBORRL_NATIVE_RUNNER_WORKERS"])
    for attempt_index in range(slots.max_attempts):
        identity = slots.start(str(slot))
        root = runtime_.rollout_root / identity.group_id / identity.attempt_id
        try:
            async with runtime_.semaphore:
                ir = await run_attempt(
                    identity,
                    task_path=entry["path"],
                    profile=runtime_.profile,
                    gateway_url=os.environ["HARBORRL_NATIVE_GATEWAY_URL"],
                    runner_python=os.environ["HARBORRL_NATIVE_RUNNER_PYTHON"],
                    # First attempts spread across workers by slot; a retry shifts to the next worker.
                    runner_host=workers[(slot + attempt_index) % len(workers)],
                    root=root,
                    registry=runtime_.registry,
                    reward_profile=reward_profile,
                    max_output_tokens=int(args.rollout_max_response_len),
                )
        except (OSError, ValueError, RuntimeError, asyncio.TimeoutError) as exc:
            slots.finish(str(slot), error=f"{type(exc).__name__}: {exc}")
            continue
        slots.finish(str(slot), ir)
        return ir
    raise RuntimeError(f"native slot {slot} exhausted its retry budget")


async def generate_group(args, group, sampling_params, rollout_id, evaluation=False):
    """Run one complete native group; evaluation is intentionally unsupported."""
    from harborrl.data.harbor.native_inspector import inspect_native

    if evaluation:
        raise ValueError("native training rollout cannot be reused as evaluation")
    if len(group) != args.n_samples_per_prompt:
        raise ValueError("native rollout group is incomplete before launch")
    entries = [_task_entry(sample) for sample in group]
    if any(entry != entries[0] for entry in entries[1:]):
        raise ValueError("native rollout group mixes task locks")
    entry = entries[0]
    report = inspect_native(entry["path"], expected_digest=entry["task_digest"])
    if report["status"] != "NEEDS_PROBE":
        raise ValueError(f"native task preflight failed: {report['reasons']}")
    stop = list(sampling_params.get("stop") or [])
    sampling_digest = digest({
        "temperature": sampling_params.get("temperature"),
        "top_p": sampling_params.get("top_p"),
        "top_k": sampling_params.get("top_k"),
        "max_new_tokens": sampling_params.get("max_new_tokens"),
        "stop": stop,
    })
    runtime_ = await runtime(args)
    policy_version = current_policy_version(runtime_)
    trajectories = await asyncio.gather(*[
        _run_slot(
            args, runtime_, sample, slot, rollout_id, entry, sampling_digest, policy_version
        )
        for slot, sample in enumerate(group)
    ])
    output = []
    for sample, ir in zip(group, trajectories):
        sample.metadata["native_ir"] = ir
        sample.response = "native Harbor trial"
        sample.reward = {"score": ir["evaluation"]["training_reward"]}
        sample.status = sample.Status.COMPLETED
        try:
            sample.policy_version = int(ir["identity"]["policy_version"])
        except ValueError as exc:
            raise ValueError("native policy version must be an integer in Slime") from exc
        output.append(sample)
    return output
