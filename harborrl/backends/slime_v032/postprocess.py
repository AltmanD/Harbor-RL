"""Defensive actor-side validation for the Slime v0.3.2 adapter."""
from __future__ import annotations

import math

from harborrl.trajectories.native import finite

REQUIRED_FIELDS = (
    "tokens",
    "response_lengths",
    "rewards",
    "raw_reward",
    "truncated",
    "sample_indices",
    "rollout_ids",
    "loss_masks",
    "rollout_log_probs",
    "rollout_mask_sums",
)


def _finite_scalar(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _integer_scalar(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return int(number) if math.isfinite(number) and number.is_integer() else None


def rollout_data_postprocess(args, rollout_id, rollout_data):
    """Slime ``--rollout-data-postprocess-path`` hook; fail closed on drift."""
    missing = [field for field in REQUIRED_FIELDS if field not in rollout_data]
    if missing:
        raise ValueError(f"native rollout data is missing standard fields: {missing}")
    count = len(rollout_data["tokens"])
    fields = (
        "response_lengths", "rewards", "raw_reward", "truncated", "sample_indices",
        "rollout_ids", "loss_masks", "rollout_log_probs", "rollout_mask_sums",
    )
    if any(len(rollout_data[field]) != count for field in fields):
        raise ValueError("native rollout data fields have mismatched sample counts")
    if count < 1:
        raise ValueError("native rollout data is empty")
    group_size = int(getattr(args, "n_samples_per_prompt", 0))
    if group_size < 2:
        raise ValueError("native training requires a complete group")
    local_weight_sum = 0.0
    for index in range(count):
        response_length = _integer_scalar(rollout_data["response_lengths"][index])
        mask = rollout_data["loss_masks"][index]
        logprobs = rollout_data["rollout_log_probs"][index]
        mask_sum = _integer_scalar(rollout_data["rollout_mask_sums"][index])
        truncated = _integer_scalar(rollout_data["truncated"][index])
        if response_length is None or response_length < 1:
            raise ValueError("invalid native response length")
        if len(mask) != response_length or len(logprobs) != response_length:
            raise ValueError("native response field lengths differ")
        if any(not _finite_scalar(value) for value in logprobs):
            raise ValueError("nonfinite native rollout logprob")
        if (not _finite_scalar(rollout_data["rewards"][index])
                or not _finite_scalar(rollout_data["raw_reward"][index])):
            raise ValueError("nonfinite native reward")
        if mask_sum is None or mask_sum < 1 or mask_sum < sum(1 for flag in mask if flag):
            raise ValueError("invalid native rollout mask sum")
        if truncated is None or truncated not in (0, 1):
            raise ValueError("invalid native truncated flag")
        local_weight_sum += sum(
            1.0 / (group_size * mask_sum) for flag in mask if flag
        )
    if not finite(local_weight_sum) or local_weight_sum > 1.0 + 1e-9:
        raise ValueError(f"native local weights exceed the one-per-group budget: {local_weight_sum}")
