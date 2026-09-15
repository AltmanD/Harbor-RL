import logging
from argparse import Namespace
from collections.abc import Callable, Iterator
from typing import Any

import torch
import torch.distributed as dist
from megatron.core import mpu
from torch.utils.checkpoint import checkpoint

from slime.utils.distributed_utils import distributed_masked_whiten
from slime.utils.misc import load_function
from .initialize import is_megatron_main_rank

logger = logging.getLogger(__name__)
from slime.utils.ppo_utils import (
    calculate_log_probs_and_entropy,
    compute_approx_kl,
    compute_gspo_kl,
    compute_opsm_mask,
    compute_policy_loss,
    get_advantages_and_returns_batch,
    get_grpo_returns,
    get_reinforce_plus_plus_baseline_advantages,
    get_reinforce_plus_plus_returns,
)
from slime.utils.types import RolloutBatch
from slime.world_model.loss_hook import apply_world_model_loss

from .cp_utils import (
    all_gather_with_cp,
    get_logits_and_tokens_offset_with_cp,
    get_sum_of_sample_mean,
    slice_log_prob_with_cp,
)


def _local_response_loss_masks(
    loss_masks: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    qkv_format: str,
    max_seq_lens: list[int] | None,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> list[torch.Tensor]:
    """Return response-space masks aligned to this rank's local log-prob layout."""
    local_masks = []
    for i, (loss_mask, total_length, response_length) in enumerate(
        zip(loss_masks, total_lengths, response_lengths, strict=False)
    ):
        mask = loss_mask.to(device=device, dtype=dtype)
        max_seq_len = max_seq_lens[i] if max_seq_lens is not None else None
        if mpu.get_context_parallel_world_size() > 1:
            mask = slice_log_prob_with_cp(mask, total_length, response_length, qkv_format, max_seq_len)
        local_masks.append(mask)
    return local_masks


def _local_sum_of_sample_mean(
    local_loss_masks: list[torch.Tensor],
    calculate_per_token_loss: bool,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Reducer for vectors already sliced to this CP rank's response-token layout."""
    lengths = [int(mask.numel()) for mask in local_loss_masks]
    if not local_loss_masks:
        return lambda x: x.sum()

    denoms = torch.stack([mask.sum() for mask in local_loss_masks])
    if mpu.get_context_parallel_world_size() > 1 and dist.is_initialized():
        denoms = denoms.clone()
        dist.all_reduce(denoms, group=mpu.get_context_parallel_group())

    def sum_of_sample_mean(x: torch.Tensor) -> torch.Tensor:
        total = torch.zeros((), dtype=x.dtype, device=x.device)
        for x_i, mask_i, denom_i in zip(x.split(lengths, dim=0), local_loss_masks, denoms, strict=False):
            total = total + (x_i * mask_i.to(device=x.device, dtype=x.dtype)).sum() / torch.clamp_min(
                denom_i.to(device=x.device, dtype=x.dtype), 1
            )
        return total

    def sum_of_token(x: torch.Tensor) -> torch.Tensor:
        total = torch.zeros((), dtype=x.dtype, device=x.device)
        for x_i, mask_i in zip(x.split(lengths, dim=0), local_loss_masks, strict=False):
            total = total + (x_i * mask_i.to(device=x.device, dtype=x.dtype)).sum()
        return total

    return sum_of_token if calculate_per_token_loss else sum_of_sample_mean


def get_responses(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    max_seq_lens: list[int] | None = None,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield response-aligned `(logits_chunk, tokens_chunk)` pairs per sample.

    After squeezing batch dimension and applying temperature scaling, this
    function extracts the logits and tokens corresponding to response segments
    for each sample.  When context parallelism is disabled, it slices directly
    from the concatenated sequence.  With context parallelism enabled, it
    handles split sequences across ranks.

    Logits may arrive in any floating-point dtype (fp32, bf16, fp16).  The
    yielded chunks preserve the original dtype — no fp32 up-cast is performed
    here, so GPU memory stays bounded by the micro-batch logits tensor.
    Downstream consumers (``fused_vocab_parallel_cross_entropy``, etc.) handle
    mixed-precision arithmetic internally.

    Args:
        logits: Model outputs with shape ``[1, T, V]`` (policy) or
            ``[1, T, 1]`` (value).
        args: Configuration containing ``rollout_temperature`` for scaling.
        unconcat_tokens: List of token tensors (prompt+response) per sample.
        total_lengths: Total sequence lengths (prompt+response) per sample.
        response_lengths: Response segment lengths per sample.

    Yields:
        Tuple of ``(logits_chunk, tokens_chunk)`` where ``logits_chunk`` is
        shape ``[R, V]`` (policy) or ``[R, 1]`` (value) and ``tokens_chunk``
        is shape ``[R]`` (1-D int64), both aligned to response tokens for one
        sample.
    """
    qkv_format = args.qkv_format

    assert len(logits.shape) == 3, f"{logits.shape}"

    logits_gib = logits.nelement() * logits.element_size() / (1 << 30)
    if logits_gib > 2.0:
        logger.warning(
            "get_responses: large logits tensor %s dtype=%s (%.2f GiB), "
            "num_samples=%d, sum(total_lengths)=%d, sum(response_lengths)=%d",
            list(logits.shape),
            logits.dtype,
            logits_gib,
            len(unconcat_tokens),
            sum(total_lengths),
            sum(response_lengths),
        )

    if qkv_format == "thd":
        assert logits.size(0) == 1, f"{logits.shape}"
        logits = logits.squeeze(0)
    else:
        assert max_seq_lens is not None
        logits = logits.view(-1, logits.size(-1))

    apply_temp = args.rollout_temperature != 1.0

    cp_size = mpu.get_context_parallel_world_size()
    end = 0
    for i, (tokens, total_length, response_length) in enumerate(
        zip(unconcat_tokens, total_lengths, response_lengths, strict=False)
    ):
        max_seq_len = max_seq_lens[i] if max_seq_lens is not None else None

        if cp_size == 1:
            if qkv_format == "bshd":
                end = max_seq_len * i + total_length
                start = end - response_length
            else:
                end += total_length
                start = end - response_length
            logits_chunk = logits[start - 1 : end - 1]
            tokens_chunk = tokens[-response_length:]
        else:
            # TODO: this is super ugly... do better abstraction.
            chunk_size, chunks_offset, logits_offset, tokens_offset = get_logits_and_tokens_offset_with_cp(
                total_length, response_length, qkv_format, max_seq_len
            )

            logits_0, logits_1 = logits[end : end + chunk_size], logits[end + chunk_size : end + 2 * chunk_size]
            end += 2 * chunk_size

            logits_0 = logits_0[logits_offset[0][0] - chunks_offset[0][0] : logits_offset[0][1] - chunks_offset[0][0]]
            tokens_0 = tokens[tokens_offset[0][0] : tokens_offset[0][1]]

            logits_1 = logits_1[logits_offset[1][0] - chunks_offset[1][0] : logits_offset[1][1] - chunks_offset[1][0]]
            tokens_1 = tokens[tokens_offset[1][0] : tokens_offset[1][1]]

            assert logits_0.size(0) == tokens_0.size(0), f"{logits_0.size(0)} vs {tokens_0.size(0)}"
            assert logits_1.size(0) == tokens_1.size(0), f"{logits_1.size(0)} vs {tokens_1.size(0)}"

            logits_chunk = torch.cat([logits_0, logits_1], dim=0)
            tokens_chunk = torch.cat([tokens_0, tokens_1], dim=0)

        if apply_temp:
            logits_chunk = logits_chunk / args.rollout_temperature
        yield logits_chunk, tokens_chunk


def get_log_probs_and_entropy(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    with_entropy: bool = False,
    non_loss_data: bool = True,
    max_seq_lens: list[int] | None = None,
) -> dict[str, list[torch.Tensor]]:
    """Compute per-token log-probabilities (and optionally entropy) on responses.

    For each sample, extracts response-aligned logits and tokens, then computes
    log-probabilities via softmax across the tensor-parallel group. Log-probs
    are squeezed from `[R, 1]` to `[R]`. Entropy values are always appended
    (even when `with_entropy=False`), but only included in the result dict
    when requested.

    Args:
        logits: Policy logits with shape `[1, T, V]`.
        args: Configuration (temperature applied in `get_responses`).
        unconcat_tokens: List of token tensors per sample.
        total_lengths: Total sequence lengths per sample.
        response_lengths: Response segment lengths per sample.
        with_entropy: If True, include "entropy" key in result.
        non_loss_data: Unused; kept for API compatibility.

    Returns:
        Dict with key "log_probs" mapping to a list of `[R]` tensors per
        sample. If `with_entropy` is True, also includes "entropy" key with
        a list of `[R]` tensors.
    """
    assert non_loss_data
    log_probs_list = []
    entropy_list = []
    for logits_chunk, tokens_chunk in get_responses(
        logits,
        args=args,
        unconcat_tokens=unconcat_tokens,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        max_seq_lens=max_seq_lens,
    ):
        log_prob, entropy = calculate_log_probs_and_entropy(
            logits_chunk,
            tokens_chunk,
            mpu.get_tensor_model_parallel_group(),
            with_entropy=with_entropy,
            chunk_size=args.log_probs_chunk_size,
        )

        log_probs_list.append(log_prob.squeeze(-1))
        entropy_list.append(entropy)

    res = {
        "log_probs": log_probs_list,
    }
    if with_entropy:
        res["entropy"] = entropy_list
    return torch.empty((0,), device=logits.device), res


def get_values(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    with_entropy: bool = False,
    non_loss_data: bool = True,
    max_seq_lens: list[int] | None = None,
) -> dict[str, list[torch.Tensor]]:
    """Extract per-token value predictions over response tokens.

    For each sample, extracts response-aligned chunks from the value head
    output and squeezes the final dimension from `[R, 1]` to `[R]`.

    Args:
        logits: Value head output with shape `[1, T, 1]`.
        args: Configuration (passed to `get_responses` which uses
            `rollout_temperature` even though values don't need temperature).
        unconcat_tokens: List of token tensors per sample.
        total_lengths: Total sequence lengths per sample.
        response_lengths: Response segment lengths per sample.
        with_entropy: Unused; kept for signature compatibility.
        non_loss_data: Unused; kept for signature compatibility.

    Returns:
        Dict with key "values" mapping to a list of `[R]` value tensors
        per sample.
    """
    value_list = []
    for logits_chunk, _ in get_responses(
        logits,
        args=args,
        unconcat_tokens=unconcat_tokens,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        max_seq_lens=max_seq_lens,
    ):
        assert logits_chunk.size(-1) == 1, f"{logits_chunk.shape}"
        value_list.append(logits_chunk.squeeze(-1))

    return torch.empty((0,), device=logits.device), {
        "values": value_list,
    }


def compute_advantages_and_returns(args: Namespace, rollout_data: RolloutBatch) -> None:
    """Compute advantages and returns in-place based on `args.advantage_estimator`.

    This function extracts rewards, log-probs, values, and masks from
    `rollout_data`, computes KL divergences, then applies the chosen advantage
    estimator. Supported methods: "grpo", "gspo", "step_wise", "ppo",
    "reinforce_plus_plus", and "reinforce_plus_plus_baseline". When
    `args.normalize_advantages` is
    True, advantages are whitened across the data-parallel group using masked
    statistics.

    Early returns if both `log_probs` and `values` are None (intermediate
    pipeline stages).

    Args:
        args: Configuration specifying estimator type, KL coefficient,
            normalization settings, and other hyperparameters.
        rollout_data: Dict containing input lists ("log_probs", "ref_log_probs",
            "rewards", "values", "response_lengths", "loss_masks",
            "total_lengths"). Modified in-place to add "advantages" and
            "returns" keys, each mapping to lists of tensors per sample.
    """
    log_probs: list[torch.Tensor] = rollout_data.get("rollout_log_probs" if args.use_rollout_logprobs else "log_probs")
    ref_log_probs: list[torch.Tensor] = rollout_data.get("ref_log_probs")
    rewards: list[float] = rollout_data.get("rewards")
    values: None | list[torch.Tensor] = rollout_data.get("values")
    response_lengths: list[int] = rollout_data.get("response_lengths")
    loss_masks: list[torch.Tensor] = rollout_data.get("loss_masks")
    total_lengths: list[int] = rollout_data.get("total_lengths")
    max_seq_lens: list[int] | None = rollout_data.get("max_seq_lens", None)

    # return when not the last pp stage.
    if not mpu.is_pipeline_last_stage():
        return

    # Several per-sample tensor lists (loss_masks, log_probs, ref_log_probs)
    # live on CPU after _get_rollout_data / _offload_rollout_data_to_cpu.
    # We need GPU copies for the advantage / KL / normalisation math below
    # (in particular dist.all_reduce requires CUDA tensors on NCCL backend).
    # The original CPU tensors in rollout_data are NOT modified.
    _gpu = torch.cuda.current_device()

    def _ensure_gpu(tensors):
        if tensors and isinstance(tensors[0], torch.Tensor) and not tensors[0].is_cuda:
            return [t.to(device=_gpu) for t in tensors]
        return tensors

    loss_masks = _ensure_gpu(loss_masks)
    if log_probs is not None:
        log_probs = _ensure_gpu(log_probs)
    if ref_log_probs is not None:
        ref_log_probs = _ensure_gpu(ref_log_probs)
    if values is not None:
        values = _ensure_gpu(values)

    if args.kl_coef == 0 or not log_probs:
        # when kl_coef is 0, we won't compute ref_log_prob
        xs = log_probs if log_probs is not None else values
        kl = [torch.zeros_like(x, dtype=torch.float32, device=x.device) for x in xs]
    else:
        kl = [
            compute_approx_kl(
                log_probs[i],
                ref_log_probs[i],
                kl_loss_type=args.kl_loss_type,
            )
            for i in range(len(log_probs))
        ]

    if args.advantage_estimator in ["grpo", "gspo"]:
        # Reward normalization is handled in rollout.py for clarity.
        rewards = torch.tensor(rewards, dtype=torch.float32, device=kl[0].device)
        returns = get_grpo_returns(rewards, kl)
        # TODO: is the copy necessary?
        advantages = [r for r in returns]

    elif args.advantage_estimator == "step_wise":
        # Step-wise rewards are pre-normalized in rollout.py; here we only
        # project each step scalar onto its token span.
        step_rewards_per_sample = rollout_data.get("step_wise_step_rewards")
        step_spans_per_sample = rollout_data.get("step_wise_step_token_spans")
        step_indices_per_sample = rollout_data.get("step_wise_step_indices")
        group_indices = rollout_data.get("group_indices")

        if step_rewards_per_sample is None or step_spans_per_sample is None or group_indices is None:
            raise ValueError(
                "step_wise advantage requires rollout_data keys: "
                "step_wise_step_rewards, step_wise_step_token_spans, group_indices"
            )

        num_samples = len(response_lengths)
        if not (len(step_rewards_per_sample) == len(step_spans_per_sample) == len(group_indices) == num_samples):
            raise ValueError(
                "step_wise metadata length mismatch: "
                f"rewards={len(step_rewards_per_sample)}, spans={len(step_spans_per_sample)}, "
                f"groups={len(group_indices)}, samples={num_samples}"
            )
        if step_indices_per_sample is not None and len(step_indices_per_sample) != num_samples:
            raise ValueError(
                "step_wise metadata length mismatch for step indices: "
                f"indices={len(step_indices_per_sample)}, samples={num_samples}"
            )

        advantages = []
        returns = []
        for i in range(num_samples):
            response_len = int(response_lengths[i])
            total_len = int(total_lengths[i])
            max_seq_len = max_seq_lens[i] if max_seq_lens is not None else None

            # Build full response-space advantage first, then slice with CP.
            full_adv = torch.zeros(response_len, dtype=torch.float32, device=kl[0].device)
            full_mask = loss_masks[i].to(device=full_adv.device, dtype=full_adv.dtype)

            group_idx = int(group_indices[i])
            step_rewards_i = step_rewards_per_sample[i] or []
            step_spans_i = step_spans_per_sample[i] or []
            step_indices_i = (
                (step_indices_per_sample[i] or [])
                if step_indices_per_sample is not None
                else list(range(min(len(step_rewards_i), len(step_spans_i))))
            )
            aligned_len = min(len(step_rewards_i), len(step_spans_i), len(step_indices_i))

            for pos in range(aligned_len):
                span = step_spans_i[pos]
                if not isinstance(span, (list, tuple)) or len(span) != 2:
                    continue
                start, end = int(span[0]), int(span[1])
                if not (0 <= start < end <= response_len):
                    continue

                # Broadcast the same step-level value to all tokens in this step span.
                full_adv[start:end] = float(step_rewards_i[pos])

            # Ensure non-trainable positions remain zero.
            full_adv = full_adv * full_mask

            # Convert full response-space vector to local CP chunk.
            local_adv = slice_log_prob_with_cp(
                full_adv,
                total_len,
                response_len,
                args.qkv_format,
                max_seq_len,
            )
            if isinstance(local_adv, list):
                local_adv = torch.tensor(local_adv, dtype=full_adv.dtype, device=full_adv.device)

            # GRPO-style training path expects returns; for this estimator we set
            # returns == advantages.
            advantages.append(local_adv)
            returns.append(local_adv.clone())

    
    elif args.advantage_estimator == "ppo":
        if values is None:
            raise ValueError("ppo advantage estimator requires rollout_data['values'], but got None.")

        old_rewards = rewards
        rewards = []
        kl_coef = -args.kl_coef
        cp_rank = mpu.get_context_parallel_rank()
        for reward, k in zip(old_rewards, kl, strict=False):
            k *= kl_coef
            if cp_rank == 0:
                k[-1] += reward
            rewards.append(k)
        advantages, returns = get_advantages_and_returns_batch(
            total_lengths, response_lengths, values, rewards, args.gamma, args.lambd
        )

    elif args.advantage_estimator == "reinforce_plus_plus":
        rewards = torch.tensor(rewards, dtype=torch.float32, device=kl[0].device)
        returns = get_reinforce_plus_plus_returns(
            rewards=rewards,
            kl=kl,
            loss_masks=loss_masks,
            response_lengths=response_lengths,
            total_lengths=total_lengths,
            kl_coef=args.kl_coef,
            gamma=args.gamma,
        )
        advantages = [r for r in returns]

    elif args.advantage_estimator == "reinforce_plus_plus_baseline":
        rewards = torch.tensor(rewards, dtype=torch.float32, device=kl[0].device)
        advantages = get_reinforce_plus_plus_baseline_advantages(
            rewards=rewards,
            kl=kl,
            loss_masks=loss_masks,
            kl_coef=args.kl_coef,
        )
        returns = advantages

    elif args.advantage_estimator == "on_policy_distillation":
        student_log_probs = log_probs
        teacher_log_probs = rollout_data.get("teacher_log_probs")
        response_lengths = rollout_data.get("response_lengths")
        device = student_log_probs[0].device
        teacher_log_probs = [t_log_prob.to(device=device) for t_log_prob in teacher_log_probs]
        teacher_log_probs = [
            t_log_prob[-response_length:]
            for t_log_prob, response_length in zip(teacher_log_probs, response_lengths, strict=False)
        ]
        advantages = [
            teacher_log_prob - student_log_prob
            for teacher_log_prob, student_log_prob in zip(teacher_log_probs, student_log_probs, strict=False)
        ]
        returns = advantages

    else:
        raise NotImplementedError(f"advantage_estimator {args.advantage_estimator} is not supported. ")

    # TODO: OpenRLHF always does advantages normalization but veRL doesn't seem to do it.
    if args.normalize_advantages and args.advantage_estimator != "step_wise":
        all_advs = torch.cat(advantages)
        cp_size = mpu.get_context_parallel_world_size()
        if cp_size == 1:
            all_masks = torch.cat(loss_masks)
        else:
            mask_chunks = []
            for i in range(len(advantages)):
                total_len = total_lengths[i]
                response_len = response_lengths[i]
                prompt_len = total_len - response_len
                max_seq_len = max_seq_lens[i] if max_seq_lens is not None else None

                _, _, _, token_offsets = get_logits_and_tokens_offset_with_cp(
                    total_len, response_len, args.qkv_format, max_seq_len
                )

                # Convert global offsets to response-space offsets
                s0, e0 = token_offsets[0]
                s1, e1 = token_offsets[1]
                res_s0, res_e0 = max(0, s0 - prompt_len), max(0, e0 - prompt_len)
                res_s1, res_e1 = max(0, s1 - prompt_len), max(0, e1 - prompt_len)

                local_mask_parts = []
                full_mask = loss_masks[i]
                if res_e0 > res_s0:
                    local_mask_parts.append(full_mask[res_s0:res_e0])
                if res_e1 > res_s1:
                    local_mask_parts.append(full_mask[res_s1:res_e1])

                # Concatenate the parts to form the final mask chunk for this rank and this sequence
                local_mask_chunk = (
                    torch.cat(local_mask_parts)
                    if local_mask_parts
                    else torch.tensor([], device=full_mask.device, dtype=full_mask.dtype)
                )
                mask_chunks.append(local_mask_chunk)

            all_masks = torch.cat(mask_chunks)

        if all_masks.numel() > 0:
            assert (
                all_advs.size() == all_masks.size()
            ), f"Shape mismatch before whitening: advantages {all_advs.size()}, masks {all_masks.size()}"
            dp_group = mpu.get_data_parallel_group()

            whitened_advs_flat = distributed_masked_whiten(
                all_advs,
                all_masks,
                process_group=dp_group,
                shift_mean=True,
            )
            chunk_lengths = [chunk.size(0) for chunk in advantages]
            advantages = list(torch.split(whitened_advs_flat, chunk_lengths))

    rollout_data["advantages"] = advantages
    rollout_data["returns"] = returns


def vanilla_tis_function(
    args,
    *,
    pg_loss: torch.Tensor,
    train_log_probs: list[torch.Tensor],
    rollout_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    **kwargs: Any,
) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]:
    rollout_log_probs = torch.cat(rollout_log_probs, dim=0)
    old_log_probs = torch.cat(train_log_probs, dim=0)
    tis = torch.exp(old_log_probs - rollout_log_probs)
    tis_abs = (torch.exp(old_log_probs - rollout_log_probs) - 1).abs()
    tis_weights = torch.clamp(tis, min=args.tis_clip_low, max=args.tis_clip)
    tis_clipfrac = (tis_weights != tis).float()
    metrics = {
        "tis": tis.clone().detach(),
        "tis_clipfrac": tis_clipfrac.clone().detach(),
        "tis_abs": tis_abs.clone().detach(),
    }
    pg_loss = pg_loss * tis_weights
    return pg_loss, loss_masks, metrics


def icepop_function(
    args,
    *,
    pg_loss: torch.Tensor,
    train_log_probs: list[torch.Tensor],
    rollout_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    **kwargs: Any,
) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]:
    rollout_log_probs = torch.cat(rollout_log_probs, dim=0)
    old_log_probs = torch.cat(train_log_probs, dim=0)
    ice_ratio = torch.exp(old_log_probs - rollout_log_probs)
    ice_abs = (torch.exp(old_log_probs - rollout_log_probs) - 1).abs()
    ice_weight = torch.where(
        (ice_ratio >= args.tis_clip_low) & (ice_ratio <= args.tis_clip), ice_ratio, torch.zeros_like(ice_ratio)
    )
    ice_clipfrac = (ice_weight != ice_ratio).float()
    metrics = {
        "tis": ice_ratio.clone().detach(),
        "tis_clipfrac": ice_clipfrac.clone().detach(),
        "tis_abs": ice_abs.clone().detach(),
    }
    pg_loss = pg_loss * ice_weight
    return pg_loss, loss_masks, metrics


def policy_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute policy loss (PPO/GSPO) and metrics.

    Computes current log-probabilities and entropy from model logits, then
    calculates PPO-style clipped policy gradient loss. For GSPO, gathers
    full sequences via context-parallel all-gather before computing per-sample
    KL. Optionally applies TIS (Truncated Importance Sampling) correction and
    adds KL loss term if configured.

    Args:
        args: Configuration controlling advantage estimator, clipping thresholds,
            entropy/KL coefficients, and TIS settings.
        batch: Mini-batch containing "advantages", "log_probs" (old policy),
            "unconcat_tokens", "response_lengths", "total_lengths", "loss_masks",
            and optionally "ref_log_probs" and "rollout_log_probs".
        logits: Policy logits with shape `[1, T, V]`.
        sum_of_sample_mean: Reduction function that averages per-sample values.

    Returns:
        Tuple of `(loss, metrics)` where `loss` is a scalar tensor and `metrics`
        is a dict containing detached scalars: "loss", "pg_loss",
        "entropy_loss", "pg_clipfrac", "ppo_kl". Additional keys "kl_loss",
        "tis", "ois", "tis_clipfrac" are included when the respective features
        are enabled.
    """
    advantages = torch.cat(batch["advantages"], dim=0)
    old_log_probs = batch["rollout_log_probs"] if args.use_rollout_logprobs else batch["log_probs"]

    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]
    max_seq_lens = batch.get("max_seq_lens", None)

    need_entropy_for_loss = args.entropy_coef != 0.0
    _, log_probs_and_entropy = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=need_entropy_for_loss,
        max_seq_lens=max_seq_lens,
    )

    log_probs = log_probs_and_entropy["log_probs"]

    # Pre-gather log probs if needed by OPSM or GSPO to avoid duplicate gathering
    need_full_log_probs = args.use_opsm or args.advantage_estimator == "gspo"

    full_log_probs = None
    full_old_log_probs = None
    if need_full_log_probs:
        full_log_probs = [
            all_gather_with_cp(log_prob, total_length, response_length)
            for log_prob, total_length, response_length in zip(
                log_probs, total_lengths, response_lengths, strict=False
            )
        ]
        full_old_log_probs = [
            all_gather_with_cp(old_log_prob, total_length, response_length)
            for old_log_prob, total_length, response_length in zip(
                old_log_probs, total_lengths, response_lengths, strict=False
            )
        ]

    # Compute OPSM mask if enabled
    if args.use_opsm:
        opsm_mask, opsm_clipfrac = compute_opsm_mask(
            args=args,
            full_log_probs=full_log_probs,
            full_old_log_probs=full_old_log_probs,
            advantages=batch["advantages"],
            loss_masks=batch["loss_masks"],
        )

    # Compute KL divergence (GSPO uses sequence-level KL, others use per-token KL)
    if args.advantage_estimator == "gspo":
        ppo_kl = compute_gspo_kl(
            full_log_probs=full_log_probs,
            full_old_log_probs=full_old_log_probs,
            local_log_probs=log_probs,
            loss_masks=batch["loss_masks"],
        )
        old_log_probs = torch.cat(old_log_probs, dim=0)
        log_probs = torch.cat(log_probs, dim=0)
    else:
        old_log_probs = torch.cat(old_log_probs, dim=0)
        log_probs = torch.cat(log_probs, dim=0)
        ppo_kl = old_log_probs - log_probs

    pg_loss, pg_clipfrac = compute_policy_loss(ppo_kl, advantages, args.eps_clip, args.eps_clip_high)

    if args.use_opsm:
        pg_loss = pg_loss * opsm_mask

    # Apply off-policy correction using importance sampling if enabled
    if args.get_mismatch_metrics or args.use_tis:
        # NOTE:
        # `tis_func` may apply rejection-sampling style masking (RS) and return `modified_response_masks`.
        # We rebuild `sum_of_sample_mean` with those masks to correct denominators for loss/backprop.
        #
        # However, mismatch/TIS/RS metrics (e.g., "truncate_fraction") are often defined over the
        # *pre-RS* valid tokens. If we aggregate metrics with `modified_response_masks`, the rejected
        # tokens are excluded from the denominator and the metric can be artificially driven to 0.
        # Keep a copy of the original reducer (based on `batch["loss_masks"]`) for metric aggregation.
        sum_of_sample_mean_for_mismatch_metrics = sum_of_sample_mean

        assert "rollout_log_probs" in batch, "rollout_log_probs must be provided for TIS"

        ois = (-ppo_kl).exp()
        tis_kwargs = {
            "args": args,
            "pg_loss": pg_loss,
            "train_log_probs": batch["log_probs"],
            "rollout_log_probs": batch["rollout_log_probs"],
            "loss_masks": batch["loss_masks"],
            "total_lengths": total_lengths,
            "response_lengths": response_lengths,
        }

        if args.custom_tis_function_path is not None:
            tis_func = load_function(args.custom_tis_function_path)
        else:
            tis_func = vanilla_tis_function
        pg_loss, modified_response_masks, tis_metrics = tis_func(**tis_kwargs)

        # [decouple IS and rejection] Rebuild sum_of_sample_mean with modified_response_masks for denominator correction
        # modified_response_masks will be sliced with cp in get_sum_of_sample_mean
        sum_of_sample_mean = get_sum_of_sample_mean(
            total_lengths,
            response_lengths,
            modified_response_masks,
            args.calculate_per_token_loss,
            args.qkv_format,
            max_seq_lens,
        )

    # Determine pg_loss reducer: use custom if specified, otherwise default
    if getattr(args, "custom_pg_loss_reducer_function_path", None) is not None:
        custom_pg_loss_reducer_func = load_function(args.custom_pg_loss_reducer_function_path)
        # Determine which loss_masks to use for pg_loss reducer
        pg_loss_masks = modified_response_masks if (args.get_mismatch_metrics or args.use_tis) else batch["loss_masks"]
        pg_loss_reducer = custom_pg_loss_reducer_func(
            total_lengths, response_lengths, pg_loss_masks, args.calculate_per_token_loss
        )
    else:
        pg_loss_reducer = sum_of_sample_mean

    pg_loss = pg_loss_reducer(pg_loss)
    pg_clipfrac = sum_of_sample_mean(pg_clipfrac)
    ppo_kl = sum_of_sample_mean(ppo_kl)

    # entropy loss:
    # - when entropy contributes to the objective, compute with grad.
    # - when entropy_coef == 0, compute under no_grad only for monitoring.
    if need_entropy_for_loss:
        entropy = log_probs_and_entropy["entropy"]
        entropy = torch.cat(entropy, dim=0)
        entropy_loss = sum_of_sample_mean(entropy)
    else:
        with torch.no_grad():
            _, entropy_for_monitor = get_log_probs_and_entropy(
                logits,
                args=args,
                unconcat_tokens=batch["unconcat_tokens"],
                total_lengths=total_lengths,
                response_lengths=response_lengths,
                with_entropy=True,
                max_seq_lens=max_seq_lens,
            )
            entropy = torch.cat(entropy_for_monitor["entropy"], dim=0)
            entropy_loss = sum_of_sample_mean(entropy)

    loss = pg_loss - args.entropy_coef * entropy_loss

    if args.use_kl_loss:
        ref_log_probs = batch["ref_log_probs"]
        ref_log_probs = torch.cat(ref_log_probs, dim=0)
        importance_ratio = None
        if args.use_unbiased_kl:
            importance_ratio = torch.exp(log_probs - old_log_probs)
        kl = compute_approx_kl(
            log_probs,
            ref_log_probs,
            kl_loss_type=args.kl_loss_type,
            importance_ratio=importance_ratio,
        )
        kl_loss = sum_of_sample_mean(kl)

        loss = loss + args.kl_loss_coef * kl_loss

    # make sure the gradient could backprop correctly.
    if log_probs.numel() == 0:
        loss += 0 * logits.sum()

    train_rollout_logprob_abs_diff = None
    if "rollout_log_probs" in batch and batch["rollout_log_probs"]:
        rollout_log_probs = torch.cat(batch["rollout_log_probs"], dim=0)
        train_rollout_logprob_abs_diff = sum_of_sample_mean((old_log_probs - rollout_log_probs).abs())

    reported_loss = {
        "loss": loss.clone().detach(),
        "pg_loss": pg_loss.clone().detach(),
        "entropy_loss": entropy_loss.clone().detach(),
        "pg_clipfrac": pg_clipfrac.clone().detach(),
        "ppo_kl": ppo_kl.clone().detach(),
    }

    if train_rollout_logprob_abs_diff is not None:
        reported_loss["train_rollout_logprob_abs_diff"] = train_rollout_logprob_abs_diff.clone().detach()

    if args.use_kl_loss:
        reported_loss["kl_loss"] = kl_loss.clone().detach()

    if args.get_mismatch_metrics or args.use_tis:
        # Aggregate mismatch/TIS/RS related metrics with the *pre-RS* masks.
        # See comment above where `sum_of_sample_mean_for_mismatch_metrics` is defined.
        reported_loss["ois"] = sum_of_sample_mean_for_mismatch_metrics(ois).clone().detach()
        # Assume all metrics are already cloned and detached
        for metric_key, metric_value in tis_metrics.items():
            key_name = f"{metric_key}"
            reported_loss[key_name] = sum_of_sample_mean_for_mismatch_metrics(metric_value)

    if args.use_opsm:
        reported_loss["opsm_clipfrac"] = opsm_clipfrac

    # LEWM is deliberately an additive, explicit hook.  With the default
    # disabled/zero-coefficient configuration this is a no-op, so the native
    # GRPO/DAPO logits -> policy-loss path remains independent.
    loss, world_model_metrics = apply_world_model_loss(
        args=args,
        batch=batch,
        logits=logits,
        loss=loss,
        reported_loss={},
    )
    reported_loss.update(world_model_metrics)
    reported_loss["loss"] = loss.clone().detach()

    return loss, reported_loss


def value_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute clipped value loss and metrics.

    Extracts current value predictions from `logits`, compares them against
    stored old values with clipping, and computes the maximum of clipped and
    unclipped squared errors (PPO-style value clipping).

    Args:
        args: Configuration containing `value_clip` threshold.
        batch: Mini-batch with "values" (old predictions), "returns",
            "unconcat_tokens", "total_lengths", and "response_lengths".
        logits: Value head output with shape `[1, T, 1]`.
        sum_of_sample_mean: Reduction function that averages per-sample values.

    Returns:
        Tuple of `(loss, metrics)` where `loss` is a scalar tensor and
        `metrics` contains detached scalars "value_loss" and "value_clipfrac".
    """
    old_values = torch.cat(batch["values"], dim=0)

    _, values = get_values(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
        max_seq_lens=batch.get("max_seq_lens", None),
    )
    values = torch.cat([value.flatten() for value in values["values"]], dim=0)

    returns = torch.cat(batch["returns"], dim=0)

    values_clipfrac = torch.abs(values - old_values) > args.value_clip
    values_clipped = old_values + (values - old_values).clamp(-args.value_clip, args.value_clip)
    surr1 = (values_clipped - returns) ** 2
    surr2 = (values - returns) ** 2
    loss = torch.max(surr1, surr2)

    loss = sum_of_sample_mean(loss)
    values_clipfrac = sum_of_sample_mean(values_clipfrac.float())

    # make sure the gradient could backprop correctly.
    if values.numel() == 0:
        loss += 0 * values.sum()

    reported_loss = {
        "value_loss": loss.clone().detach(),
        "value_clipfrac": values_clipfrac.clone().detach(),
    }

    return loss, reported_loss


def sft_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute supervised fine-tuning loss over response tokens.

    Computes log-probabilities of the ground-truth tokens in the response
    segments and returns the negative log-likelihood as the loss.

    Args:
        args: Configuration (passed through to helpers).
        batch: Mini-batch with "unconcat_tokens", "response_lengths", and
            "total_lengths".
        logits: Policy logits with shape `[1, T, V]`.
        sum_of_sample_mean: Reduction function that averages per-sample values.

    Returns:
        Tuple of `(loss, metrics)` where `metrics` contains a single detached
        scalar "loss".
    """
    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]

    _, log_probs_and_entropy = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=False,
        max_seq_lens=batch.get("max_seq_lens", None),
    )

    log_probs = log_probs_and_entropy["log_probs"]
    log_probs = torch.cat(log_probs, dim=0)
    loss = -sum_of_sample_mean(log_probs)

    # make sure the gradient could backprop correctly.
    if log_probs.numel() == 0:
        loss += 0 * logits.sum()

    return (
        loss,
        {
            "loss": loss.clone().detach(),
        },
    )


def decoupled_policy_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute off-policy decoupled PPO/GRPO loss.

    The objective separates the proximal ratio pi_theta/pi_prox from the
    behavior correction pi_prox/pi_behav, so replayed samples can be trained
    with bounded off-policy correction.
    """
    from slime.utils.offpolicy_utils import (
        aggregate_importance_weights,
        apply_m2po_filtering,
        compute_decoupled_policy_loss,
        compute_offpolicy_importance_weights,
    )

    advantages = torch.cat(batch["advantages"], dim=0)
    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]
    max_seq_lens = batch.get("max_seq_lens", None)

    _, log_probs_and_entropy = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=True,
        max_seq_lens=max_seq_lens,
    )
    log_probs = log_probs_and_entropy["log_probs"]
    entropy = log_probs_and_entropy["entropy"]

    use_proximal_approximation = batch.get("use_proximal_logp_approximation", False)
    if isinstance(use_proximal_approximation, (list, tuple)):
        use_proximal_approximation = any(bool(value) for value in use_proximal_approximation)
    if use_proximal_approximation:
        from slime.utils.proximal_logp_utils import resolve_proximal_logp

        prox_logp_method = getattr(args, "prox_logp_method", "recompute")
        if prox_logp_method == "metrics":
            if is_megatron_main_rank():
                logger.warning(
                    "prox_logp_method=metrics requires recomputed proximal_log_probs; "
                    "using loglinear approximation for training because approximation mode was requested."
                )
            prox_logp_method = "loglinear"

        rollout_log_probs_for_approx = batch.get("rollout_log_probs")
        policy_versions = batch.get("policy_versions")
        current_policy_version = batch.get("current_policy_version")
        if rollout_log_probs_for_approx is None or policy_versions is None or current_policy_version is None:
            raise ValueError(
                "proximal logp approximation requires rollout_log_probs, policy_versions, "
                "and current_policy_version in the batch."
            )
        if isinstance(current_policy_version, list):
            current_policy_version = int(current_policy_version[0])

        proximal_log_probs = []
        for current_logp, behavior_logp, policy_version in zip(
            log_probs,
            rollout_log_probs_for_approx,
            policy_versions,
            strict=False,
        ):
            behavior_logp = behavior_logp.to(device=current_logp.device, dtype=current_logp.dtype)
            token_versions = torch.full(
                (int(current_logp.numel()),),
                int(policy_version),
                dtype=torch.long,
                device=current_logp.device,
            )
            proximal_log_probs.append(
                resolve_proximal_logp(
                    prox_logp_gt=None,
                    prox_logp_method=prox_logp_method,
                    old_logp=behavior_logp,
                    logprobs=current_logp.detach(),
                    versions=token_versions,
                    current_version=int(current_policy_version),
                )
            )
    else:
        if "proximal_log_probs" not in batch or batch["proximal_log_probs"] is None:
            raise ValueError(
                "batch must contain 'proximal_log_probs' for decoupled_policy_loss. "
                "Make sure train_actor() computes proximal log probabilities before training."
            )
        proximal_log_probs = batch["proximal_log_probs"]

    if "behavior_log_probs" in batch and batch["behavior_log_probs"] is not None:
        behavior_log_probs = batch["behavior_log_probs"]
    elif "rollout_log_probs" in batch and batch["rollout_log_probs"] is not None:
        behavior_log_probs = batch["rollout_log_probs"]
    elif "log_probs" in batch and batch["log_probs"] is not None:
        behavior_log_probs = batch["log_probs"]
    else:
        raise ValueError(
            "batch must contain either 'behavior_log_probs', 'rollout_log_probs', or 'log_probs' "
            "for off-policy importance sampling"
        )

    proximal_log_probs_list = list(proximal_log_probs) if isinstance(proximal_log_probs, list) else None
    behavior_log_probs_list = list(behavior_log_probs) if isinstance(behavior_log_probs, list) else None
    per_weight_vector = None

    log_probs = torch.cat(log_probs, dim=0)
    proximal_log_probs = torch.cat(proximal_log_probs, dim=0)
    behavior_log_probs = torch.cat(behavior_log_probs, dim=0)
    loss_masks_list = (
        _local_response_loss_masks(
            list(batch.get("loss_masks", [])),
            total_lengths,
            response_lengths,
            args.qkv_format,
            max_seq_lens,
            device=log_probs.device,
            dtype=log_probs.dtype,
        )
        if batch.get("loss_masks") is not None
        else None
    )

    if log_probs.shape != proximal_log_probs.shape:
        raise RuntimeError(
            f"Shape mismatch between log_probs {log_probs.shape} and proximal_log_probs {proximal_log_probs.shape}."
        )
    if log_probs.shape != behavior_log_probs.shape:
        raise RuntimeError(
            f"Shape mismatch between log_probs {log_probs.shape} and behavior_log_probs {behavior_log_probs.shape}."
        )

    importance_weights = compute_offpolicy_importance_weights(
        proximal_log_probs,
        behavior_log_probs,
        clip_min=getattr(args, "importance_weight_clip_min", None),
        clip_max=getattr(args, "importance_weight_clip_max", None),
    )

    topr_metrics: dict[str, torch.Tensor] = {}
    if (
        getattr(args, "use_topr", False)
        and proximal_log_probs_list is not None
        and behavior_log_probs_list is not None
        and loss_masks_list is not None
    ):
        try:
            from slime.utils.topr_utils import blend_token_and_seq_weights, compute_topr_seq_weights

            w_seq_per_sample, w_seq_token = compute_topr_seq_weights(
                proximal_log_probs_list,
                behavior_log_probs_list,
                loss_masks_list,
                logw_cap=float(getattr(args, "topr_logw_cap", 2.0)),
                w_min=float(getattr(args, "topr_w_min", 0.0)),
                w_max=float(getattr(args, "topr_w_max", 5.0)),
            )
            blend = float(getattr(args, "topr_blend", 1.0))
            importance_weights = blend_token_and_seq_weights(
                importance_weights, w_seq_token.to(importance_weights.dtype), blend
            )
            topr_metrics["topr_w_seq_mean"] = w_seq_per_sample.detach().mean()
            topr_metrics["topr_w_seq_max"] = w_seq_per_sample.detach().max()
            topr_metrics["topr_w_seq_min"] = w_seq_per_sample.detach().min()
            topr_metrics["topr_blend_lambda"] = torch.tensor(
                blend, device=importance_weights.device, dtype=torch.float32
            )
        except Exception as exc:
            if is_megatron_main_rank():
                logger.warning("TOPR disabled due to error: %s", exc)

    per_metrics: dict[str, torch.Tensor] = {}
    per_is_weights = batch.get("per_is_weights", None)
    if per_is_weights is not None and loss_masks_list is not None and len(per_is_weights) == len(loss_masks_list):
        try:
            weights_tensor = torch.tensor(
                list(per_is_weights), dtype=torch.float32, device=importance_weights.device
            )
            per_token_weights = [
                torch.full(
                    (int(token_log_probs.numel()),),
                    float(weight),
                    dtype=importance_weights.dtype,
                    device=importance_weights.device,
                )
                for weight, token_log_probs in zip(per_is_weights, proximal_log_probs_list, strict=False)
            ]
            per_weight_vector = torch.cat(per_token_weights, dim=0) if per_token_weights else None
            if per_weight_vector is not None and per_weight_vector.shape != importance_weights.shape:
                raise RuntimeError(
                    f"PER token weight shape {per_weight_vector.shape} does not match loss shape {importance_weights.shape}"
                )
            per_metrics["per_is_weight_mean"] = weights_tensor.mean()
            per_metrics["per_is_weight_min"] = weights_tensor.min()
            per_metrics["per_is_weight_max"] = weights_tensor.max()
        except Exception as exc:
            if is_megatron_main_rank():
                logger.warning("PER IS weight broadcast failed: %s", exc)
            per_weight_vector = None

    log_ratio_intermediate = proximal_log_probs - log_probs

    m2po_metrics: dict[str, torch.Tensor] = {}
    effective_sum_of_sample_mean = sum_of_sample_mean
    if getattr(args, "enable_m2po_filtering", False):
        if loss_masks_list is None:
            raise ValueError("M2PO filtering requires loss_masks in the batch.")
        delta = behavior_log_probs - proximal_log_probs
        m2 = delta * delta
        total_tokens_before = sum(mask.sum().item() for mask in loss_masks_list)
        modified_loss_masks, num_filtered = apply_m2po_filtering(
            m2=m2,
            loss_masks=loss_masks_list,
            threshold=getattr(args, "m2po_threshold", 0.1),
            policy_version_gaps=None,
            min_gap_for_filtering=0,
        )
        loss_masks_list = modified_loss_masks
        effective_sum_of_sample_mean = _local_sum_of_sample_mean(
            loss_masks_list,
            args.calculate_per_token_loss,
        )
        filter_rate = num_filtered / max(total_tokens_before, 1)
        m2po_metrics["m2po_num_filtered_tokens"] = torch.tensor(
            num_filtered, dtype=torch.float32, device=log_probs.device
        )
        m2po_metrics["m2po_total_tokens"] = torch.tensor(
            total_tokens_before, dtype=torch.float32, device=log_probs.device
        )
        m2po_metrics["m2po_filter_rate"] = torch.tensor(
            filter_rate, dtype=torch.float32, device=log_probs.device
        )

    pg_loss, pg_clipfrac = compute_decoupled_policy_loss(
        log_ratio_intermediate,
        importance_weights,
        advantages,
        args.eps_clip,
        args.eps_clip_high,
        behav_imp_weight_cap=getattr(args, "behav_imp_weight_cap", None),
    )
    if per_weight_vector is not None:
        pg_loss = pg_loss * per_weight_vector
    pg_loss = effective_sum_of_sample_mean(pg_loss)
    pg_clipfrac = effective_sum_of_sample_mean(pg_clipfrac)

    entropy = torch.cat(entropy, dim=0)
    entropy_loss = effective_sum_of_sample_mean(entropy)
    loss = pg_loss - args.entropy_coef * entropy_loss

    if args.use_kl_loss:
        ref_log_probs = torch.cat(batch["ref_log_probs"], dim=0)
        kl = compute_approx_kl(log_probs, ref_log_probs, kl_loss_type=args.kl_loss_type)
        kl_loss = effective_sum_of_sample_mean(kl)
        loss = loss + args.kl_loss_coef * kl_loss

    if log_probs.numel() == 0:
        loss += 0 * logits.sum()

    all_masks = torch.cat(loss_masks_list, dim=0) if loss_masks_list is not None else torch.ones_like(importance_weights)
    reported_loss = {
        "loss": loss.clone().detach(),
        "pg_loss": pg_loss.clone().detach(),
        "entropy_loss": entropy_loss.clone().detach(),
        "pg_clipfrac": pg_clipfrac.clone().detach(),
        "ppo_kl_prox": effective_sum_of_sample_mean(log_ratio_intermediate).clone().detach(),
        "importance_weight_mean": aggregate_importance_weights(
            importance_weights, all_masks, method="mean"
        ).clone().detach(),
        "importance_weight_max": aggregate_importance_weights(
            importance_weights, all_masks, method="max"
        ).clone().detach(),
        "effective_sample_size": aggregate_importance_weights(
            importance_weights, all_masks, method="effective_sample_size"
        ).clone().detach(),
    }

    if args.use_kl_loss:
        reported_loss["kl_loss"] = kl_loss.clone().detach()

    if batch.get("policy_versions") is not None and batch.get("current_policy_version") is not None:
        from slime.utils.offpolicy_utils import compute_policy_version_staleness

        current_version = (
            batch["current_policy_version"][0]
            if isinstance(batch["current_policy_version"], list)
            else batch["current_policy_version"]
        )
        staleness = compute_policy_version_staleness(batch["policy_versions"], current_version)
        reported_loss["mean_staleness"] = staleness.float().mean().clone().detach()
        reported_loss["max_staleness"] = staleness.max().clone().detach()

    for metric_group in (topr_metrics, per_metrics, m2po_metrics):
        for metric_name, metric_value in metric_group.items():
            reported_loss[metric_name] = (
                metric_value.clone().detach()
                if isinstance(metric_value, torch.Tensor)
                else torch.tensor(float(metric_value), device=log_probs.device)
            )

    if use_proximal_approximation:
        reported_loss["prox_logp_approximation_enabled"] = torch.tensor(1.0, device=log_probs.device)

    # Keep the latent-world-model objective available for SPEAR/replay runs as
    # well as the ordinary on-policy policy_loss path.  The hook is inert when
    # its coefficient is zero or no target latent was attached to the batch.
    loss, world_model_metrics = apply_world_model_loss(
        args=args,
        batch=batch,
        logits=logits,
        loss=loss,
        reported_loss={},
    )
    reported_loss.update(world_model_metrics)
    reported_loss["loss"] = loss.clone().detach()

    return loss, reported_loss


def loss_function(
    args: Namespace,
    batch: RolloutBatch,
    num_microbatches: int,
    logits: torch.Tensor,
) -> tuple[torch.Tensor, int | torch.Tensor, dict[str, list[str] | torch.Tensor]]:
    """Dispatch to the configured loss and rescale for Megatron integration.

    Selects one of "policy_loss", "value_loss", "sft_loss", or a custom loss
    function based on `args.loss_type`, computes the loss and metrics, then
    rescales the loss by micro-batch and parallelism factors to integrate with
    Megatron's gradient accumulation.

    Args:
        args: Configuration specifying `loss_type`, `calculate_per_token_loss`,
            `global_batch_size`, and optionally `custom_loss_function_path`.
        batch: Mini-batch with "loss_masks", "response_lengths", and other
            keys required by the selected loss function.
        num_microbatches: Number of gradient accumulation steps.
        logits: Model outputs (policy or value head).

    Returns:
        Tuple of `(scaled_loss, normalizer, logging_dict)` where:
        - `scaled_loss` is the loss tensor (scalar) rescaled for Megatron.
        - `normalizer` is `num_tokens` (scalar tensor) if
          `args.calculate_per_token_loss` is True, else `1` (int).
        - `logging_dict` has keys "keys" (list of str metric names) and
          "values" (1D tensor: [count, metric1, metric2, ...]).
    """
    num_samples = len(batch["response_lengths"])
    sum_of_sample_mean = get_sum_of_sample_mean(
        batch["total_lengths"],
        batch["response_lengths"],
        batch["loss_masks"],
        args.calculate_per_token_loss,
        args.qkv_format,
        batch.get("max_seq_lens", None),
    )

    match args.loss_type:
        case "policy_loss":
            func = policy_loss_function
        case "decoupled_policy_loss":
            func = decoupled_policy_loss_function
        case "value_loss":
            func = value_loss_function
        case "sft_loss":
            func = sft_loss_function
        case "custom_loss":
            func = load_function(args.custom_loss_function_path)
        case _:
            raise ValueError(f"Unknown loss type: {args.loss_type}")

    if args.recompute_loss_function:
        loss, log = checkpoint(func, args, batch, logits, sum_of_sample_mean)
    else:
        loss, log = func(args, batch, logits, sum_of_sample_mean)

    # Megatron's pipeline schedule accumulates this value into an integer
    # `total_num_tokens` tensor.  Count nonzero mask entries here and keep any
    # continuous weighting local to the loss reducer above.
    num_tokens = torch.zeros((), dtype=torch.int, device=logits.device)
    for loss_mask in batch["loss_masks"]:
        valid_tokens = torch.count_nonzero(loss_mask).to(device=logits.device, dtype=torch.int)
        num_tokens = num_tokens + torch.clamp_min(valid_tokens, 1)

    # Here we need to divide by cp_size because to cancel the multiply in Megatron.
    global_batch_size = batch.get("dynamic_global_batch_size", args.global_batch_size)
    if not args.calculate_per_token_loss:
        loss = (
            loss * num_microbatches / global_batch_size * mpu.get_data_parallel_world_size(with_context_parallel=True)
        )
    else:
        loss = loss * mpu.get_context_parallel_world_size()

    return (
        loss,
        (num_tokens if args.calculate_per_token_loss else torch.tensor(1, dtype=torch.int, device=logits.device)),
        {
            "keys": list(log.keys()),
            "values": torch.tensor(
                [
                    num_samples if not args.calculate_per_token_loss else num_tokens,
                ]
                + list(log.values()),
                device=logits.device,
            ),
        },
    )
