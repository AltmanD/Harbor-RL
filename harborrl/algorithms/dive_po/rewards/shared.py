from __future__ import annotations

import logging
import math
from typing import Any


logger = logging.getLogger(__name__)


# Re-exported from the platform-wide env module so the rewards package keeps
# its historical import path (``rewards.shared.env_int``).
from harborrl.env import env_int, env_str  # noqa: F401


def schedule_multiplier(schedule: str, train_step: Any, decay_steps: int) -> float:
    name = str(schedule or "constant").strip().lower()
    if name in {"", "constant", "none", "off"}:
        return 1.0
    if decay_steps <= 0 or train_step is None:
        return 1.0
    try:
        step = max(0.0, float(train_step))
    except (TypeError, ValueError):
        return 1.0
    progress = min(1.0, step / float(decay_steps))
    if name == "linear":
        return max(0.0, 1.0 - progress)
    if name == "cosine":
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    logger.warning("Unknown reward schedule=%r; using constant", name)
    return 1.0


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def advantage_gate_mode() -> str:
    return env_str("EXPLORE_ADVANTAGE_GATE_MODE", "legacy").lower()


def uses_outcome_status_gate(gate_mode: str | None = None) -> bool:
    mode = advantage_gate_mode() if gate_mode is None else str(gate_mode or "").lower()
    return mode in {
        "outcome",
        "outcome_status",
        "quality",
        "quality_gate",
        "status_quality",
    }


def outcome_candidate_keys() -> list[str]:
    configured = env_str("EXPLORE_ADVANTAGE_OUTCOME_KEY", "raw_score")
    keys = [configured] if configured else []
    keys.extend(
        [
            "raw_score",
            "accuracy",
            "success_score",
            "unit_test_pass_rate",
            "test_acc",
            "pass_rate",
            "base_score",
            "score",
        ]
    )
    return list(dict.fromkeys(key for key in keys if key))


def normalize_outcome_value(key: str, value: float) -> float:
    if key in {"score", "base_score", "task_reward", "raw_reward"} and value < 0.0:
        return clamp01(0.5 * (value + 1.0))
    return clamp01(value)


def normalize_values(values: list[float], use_std: bool) -> list[float]:
    if not values:
        return []
    mean = sum(values) / len(values)
    centered = [value - mean for value in values]
    if not use_std:
        return centered
    if len(values) <= 1:
        return [0.0 for _ in values]
    variance = sum(value * value for value in centered) / max(1, len(values) - 1)
    std = math.sqrt(max(variance, 0.0))
    return [value / (std + 1e-6) for value in centered]


# ── Sample/reward introspection shared by postprocess and rollout_log ──────
# These used to be copy-pasted between algorithms/dive_po/rewards/postprocess.py
# (production reward path) and misc/rollout_log.py (wandb "expected value"
# mirroring); keep exactly one implementation here so the log view cannot
# drift from the training math.  Leading underscore names are kept so existing
# call sites in both modules only change their import line.
from harborrl.env import env_float as _env_float  # noqa: E402


def _sample_train_step(sample: Any) -> Any:
    metadata = getattr(sample, "metadata", None)
    if isinstance(metadata, dict):
        for key in ("train_step", "rollout_step", "rollout_id"):
            if metadata.get(key) is not None:
                return metadata.get(key)
    reward = getattr(sample, "reward", None)
    if isinstance(reward, dict):
        for key in ("train_step", "rollout_step", "rollout_id"):
            if reward.get(key) is not None:
                return reward.get(key)
    return None


def _batch_train_step(samples: list[Any]) -> Any:
    values = [_sample_train_step(sample) for sample in samples]
    numeric: list[float] = []
    for value in values:
        if value is None:
            continue
        try:
            numeric.append(float(value))
        except (TypeError, ValueError):
            continue
    if numeric:
        return max(numeric)
    return next((value for value in values if value is not None), None)


def _status_name(sample: Any) -> str:
    status = getattr(sample, "status", "")
    value = getattr(status, "value", status)
    return str(value).lower()


def _status_intrinsic_scale(sample: Any) -> float:
    status = _status_name(sample)
    if "truncated" in status:
        return max(0.0, _env_float("EXPLORE_ADVANTAGE_TRUNCATED_INTRINSIC_SCALE", 1.0))
    if any(part in status for part in ("failed", "aborted")):
        return max(0.0, _env_float("EXPLORE_ADVANTAGE_FAILED_INTRINSIC_SCALE", 1.0))
    return 1.0


def _status_quality_floor(sample: Any) -> float:
    status = _status_name(sample)
    if "truncated" in status:
        return clamp01(_env_float("EXPLORE_ADVANTAGE_TRUNCATED_FLOOR", 0.15))
    if "aborted" in status:
        return clamp01(_env_float("EXPLORE_ADVANTAGE_ABORTED_FLOOR", 0.0))
    if "failed" in status:
        return clamp01(_env_float("EXPLORE_ADVANTAGE_FAILED_FLOOR", 0.0))
    return clamp01(_env_float("EXPLORE_ADVANTAGE_COMPLETED_FLOOR", 0.5))


def _component_value(sample: Any, key: str) -> float:
    reward = getattr(sample, "reward", None)
    if not isinstance(reward, dict):
        return 0.0
    value = reward.get(key)
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _component_value_or_none(sample: Any, key: str) -> float | None:
    reward = getattr(sample, "reward", None)
    if not isinstance(reward, dict) or key not in reward:
        return None
    value = reward.get(key)
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _outcome_score(sample: Any) -> float:
    for key in outcome_candidate_keys():
        value = _component_value_or_none(sample, key)
        if value is not None:
            return normalize_outcome_value(key, value)
    status = _status_name(sample)
    return 1.0 if "completed" in status else 0.0


def _quality_gate(sample: Any) -> tuple[float, float, float]:
    outcome = _outcome_score(sample)
    floor = _status_quality_floor(sample)
    return clamp01(floor + (1.0 - floor) * outcome), outcome, floor


def _sample_group_key(sample: Any) -> int:
    try:
        return int(sample.group_index) if getattr(sample, "group_index", None) is not None else -1
    except (TypeError, ValueError):
        return -1


def _sample_traj_key(sample: Any, sample_idx: int) -> tuple[int, int]:
    try:
        traj_idx = int(sample.index) if getattr(sample, "index", None) is not None else sample_idx
    except (TypeError, ValueError):
        traj_idx = sample_idx
    return _sample_group_key(sample), traj_idx


def _sample_is_trainable(sample: Any) -> bool:
    """Return whether a sample can contribute gradient to the actor update.

    Failed infrastructure samples remain in the rollout batch for observability,
    but Slime later replaces their loss mask with zeros. They must therefore not
    affect the mean/std used to normalize rewards for trainable samples.
    """
    if bool(getattr(sample, "remove_sample", False)):
        return False
    mask = getattr(sample, "loss_mask", None)
    if mask is None:
        return True
    try:
        return any(float(value) > 0.0 for value in mask)
    except (TypeError, ValueError):
        # Preserve historical behavior for exotic/lazy mask containers; the
        # Slime conversion path performs the definitive validation later.
        return True


def _group_normalize_sample_values(
    args: Any,
    samples: list[Any],
    values: list[float],
) -> list[float]:
    use_std = bool(getattr(args, "grpo_std_normalization", False))
    if getattr(args, "dynamic_history", False):
        value_by_key: dict[tuple[int, int], float] = {}
        group_to_keys: dict[int, list[tuple[int, int]]] = {}
        key_by_sample: list[tuple[int, int]] = []
        for i, sample in enumerate(samples):
            key = _sample_traj_key(sample, i)
            key_by_sample.append(key)
            if _sample_is_trainable(sample) and key not in value_by_key:
                value_by_key[key] = float(values[i])
                group_to_keys.setdefault(key[0], []).append(key)

        normalized_by_key: dict[tuple[int, int], float] = {}
        for keys in group_to_keys.values():
            vals = normalize_values([value_by_key[k] for k in keys], use_std)
            for j, key in enumerate(keys):
                normalized_by_key[key] = float(vals[j])
        return [normalized_by_key.get(key, 0.0) for key in key_by_sample]

    group_to_indices: dict[int, list[int]] = {}
    for i, sample in enumerate(samples):
        if _sample_is_trainable(sample):
            group_to_indices.setdefault(_sample_group_key(sample), []).append(i)

    normalized = [0.0 for _ in values]
    for idxs in group_to_indices.values():
        vals = normalize_values([values[i] for i in idxs], use_std)
        for j, sample_idx in enumerate(idxs):
            normalized[sample_idx] = float(vals[j])
    return normalized


def _sync_reward_aliases(
    reward: dict[str, Any] | None,
    *,
    total_reward: float | None = None,
    extra_exploration_reward: float = 0.0,
) -> None:
    """Add explicit reward component aliases while preserving legacy keys."""
    if not isinstance(reward, dict):
        return
    total = reward.get("score") if total_reward is None else total_reward
    raw = reward.get("raw_score", total)
    task = reward.get("base_score", raw)
    exploration = float(reward.get("explore_total_bonus", 0.0) or 0.0) + extra_exploration_reward
    reward["raw_reward"] = raw
    reward["task_reward"] = task
    reward["exploration_reward"] = exploration
    reward["total_reward"] = total
