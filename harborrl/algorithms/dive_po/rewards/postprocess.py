from __future__ import annotations

import logging
import os
from typing import Any

from harborrl.algorithms.dive_po.rewards.shared import (
    _batch_train_step,
    _component_value,
    _component_value_or_none,
    _group_normalize_sample_values,
    _outcome_score,
    _quality_gate,
    _sample_group_key,
    _sample_is_trainable,
    _sample_train_step,
    _sample_traj_key,
    _status_intrinsic_scale,
    _status_name,
    _status_quality_floor,
    _sync_reward_aliases,
    advantage_gate_mode as _advantage_gate_mode,
    clamp01 as _clamp01,
    env_int as _env_int,
    env_str as _env_str,
    normalize_outcome_value as _normalize_outcome_value,
    normalize_values as _normalize_values,
    outcome_candidate_keys as _outcome_candidate_keys,
    schedule_multiplier as _schedule_multiplier,
    uses_outcome_status_gate as _uses_outcome_status_gate,
)

logger = logging.getLogger(__name__)


from harborrl.env import env_flag as _env_flag, env_float as _env_float





def _reward_value(args: Any, sample: Any) -> float:
    reward = getattr(sample, "reward", None)
    key = getattr(args, "reward_key", None)
    if key:
        if not isinstance(reward, dict):
            return 0.0
        return float(reward.get(key, 0.0) or 0.0)
    return float(reward or 0.0)


def _configured_truncation_penalty() -> float:
    return _env_float(
        "EXPLORE_TRUNCATION_PENALTY",
        _env_float("EXPLORE_ADVANTAGE_TRUNCATION_PENALTY", 0.0),
    )


def _apply_truncation_penalties(
    samples: list[Any],
    adjusted: list[float],
    *,
    exploration_extra: list[float] | None = None,
) -> list[float]:
    penalty_value = _configured_truncation_penalty()
    outcome_aware = _env_flag("EXPLORE_TRUNCATION_PENALTY_OUTCOME_AWARE", "0")
    should_sync_aliases = exploration_extra is not None
    if penalty_value == 0.0 and not should_sync_aliases and not outcome_aware:
        return adjusted

    result = list(adjusted)
    extra = exploration_extra or [0.0 for _ in samples]
    for i, sample in enumerate(samples):
        is_truncated = "truncated" in _status_name(sample)
        outcome = _outcome_score(sample)
        multiplier = (1.0 - outcome) if outcome_aware else 1.0
        if not is_truncated:
            multiplier = 0.0
        penalty = float(penalty_value * multiplier if is_truncated else 0.0)
        result[i] += penalty
        reward = getattr(sample, "reward", None)
        if isinstance(reward, dict):
            if penalty_value != 0.0 or outcome_aware:
                reward["explore_truncation_penalty"] = penalty
                reward["explore_truncation_penalty_coef"] = penalty_value
                reward["explore_truncation_penalty_applied"] = bool(is_truncated)
                reward["explore_truncation_penalty_outcome_aware"] = bool(outcome_aware)
                reward["explore_truncation_penalty_outcome_score"] = outcome
                reward["explore_truncation_penalty_multiplier"] = multiplier
            reward["explore_post_norm_adjusted_reward"] = result[i]
            reward["postprocess_total_reward"] = result[i]
            _sync_reward_aliases(
                reward,
                total_reward=result[i],
                extra_exploration_reward=float(extra[i]) + penalty,
            )
    return result



def _default_post_process(args: Any, samples: list[Any]) -> tuple[list[float], list[float]]:
    """Mirror slime's default reward post-process for GRPO/GSPO.

    This function is only used when EXPLORE_ADVANTAGE_BONUS is enabled; keeping
    the default math here lets us add post-normalization exploration bonuses
    without replacing the rest of slime's behavior.
    """
    raw_rewards = [_reward_value(args, sample) for sample in samples]
    if (
        getattr(args, "advantage_estimator", None) in ["grpo", "gspo"]
        and getattr(args, "rewards_normalization", False)
    ):
        return raw_rewards, _group_normalize_sample_values(args, samples, raw_rewards)

    return raw_rewards, raw_rewards


def _dual_stream_post_process(
    args: Any,
    samples: list[Any],
    base_rewards: list[float],
) -> list[float]:
    intrinsic_key = _env_str(
        "EXPLORE_ADVANTAGE_INTRINSIC_KEY",
        "explore_agent57_intrinsic_signal",
    )
    lambda_coef = _env_float(
        "EXPLORE_ADVANTAGE_LAMBDA",
        _env_float("EXPLORE_ADVANTAGE_BONUS_COEF", 0.1),
    )
    lambda_schedule = _env_str("EXPLORE_ADVANTAGE_LAMBDA_SCHEDULE", "constant")
    lambda_decay_steps = max(0, _env_int("EXPLORE_ADVANTAGE_LAMBDA_DECAY_STEPS", 0))
    train_step = _batch_train_step(samples)
    lambda_multiplier = _schedule_multiplier(
        lambda_schedule,
        train_step,
        lambda_decay_steps,
    )
    effective_lambda = lambda_coef * lambda_multiplier
    arm_weight_mode = _env_str("EXPLORE_ADVANTAGE_ARM_WEIGHT_MODE", "normalized_beta").lower()
    trust_key = _env_str("EXPLORE_ADVANTAGE_TRUST_KEY", "explore_agent57_trust")
    gate_mode = _advantage_gate_mode()
    clip = _env_float("EXPLORE_ADVANTAGE_BONUS_CLIP", 0.0)

    intrinsic_values = [_component_value(sample, intrinsic_key) for sample in samples]
    intrinsic_adv = _group_normalize_sample_values(args, samples, intrinsic_values)

    betas = [_component_value(sample, "explore_agent57_beta") for sample in samples]
    max_beta = max([abs(beta) for beta in betas if beta > 0.0] or [1.0])
    adjusted = list(base_rewards)
    exploration_extra = [0.0 for _ in samples]
    for i, sample in enumerate(samples):
        if arm_weight_mode in {"none", "off", "0"}:
            arm_weight = 1.0
        elif arm_weight_mode in {"raw", "raw_beta"}:
            arm_weight = max(0.0, betas[i])
        else:
            arm_weight = max(0.0, betas[i]) / max(max_beta, 1e-12)
        reward = getattr(sample, "reward", None)
        trust_missing = not isinstance(reward, dict) or trust_key not in reward
        trust = _component_value(sample, trust_key)
        if trust_missing and trust_key == "explore_agent57_trust":
            trust = 1.0
        status_scale = _status_intrinsic_scale(sample)
        quality_gate, outcome, status_floor = _quality_gate(sample)
        if _uses_outcome_status_gate(gate_mode):
            gate = quality_gate
        else:
            gate = trust * status_scale
        raw_bonus = float(effective_lambda * arm_weight * gate * intrinsic_adv[i])
        bonus = max(-clip, min(clip, raw_bonus)) if clip > 0 else raw_bonus
        adjusted[i] += bonus
        exploration_extra[i] = bonus
        if isinstance(reward, dict):
            reward["explore_post_norm_base_reward"] = base_rewards[i]
            reward["explore_post_norm_intrinsic_value"] = intrinsic_values[i]
            reward["explore_post_norm_bonus_raw"] = raw_bonus
            reward["explore_post_norm_bonus"] = bonus
            reward["explore_post_norm_bonus_base_coef"] = lambda_coef
            reward["explore_post_norm_bonus_coef"] = effective_lambda
            reward["explore_post_norm_bonus_schedule"] = lambda_schedule
            reward["explore_post_norm_bonus_decay_steps"] = lambda_decay_steps
            reward["explore_post_norm_bonus_schedule_multiplier"] = lambda_multiplier
            reward["explore_post_norm_train_step"] = train_step
            reward["explore_post_norm_bonus_clip"] = clip
            reward["explore_post_norm_bonus_mode"] = "dual_stream"
            reward["explore_post_norm_intrinsic_key"] = intrinsic_key
            reward["explore_post_norm_intrinsic_advantage"] = intrinsic_adv[i]
            reward["explore_post_norm_arm_weight"] = arm_weight
            reward["explore_post_norm_trust"] = trust
            reward["explore_post_norm_status_intrinsic_scale"] = status_scale
            reward["explore_post_norm_gate_mode"] = gate_mode
            reward["explore_post_norm_effective_gate"] = gate
            reward["explore_post_norm_quality_gate"] = quality_gate
            reward["explore_post_norm_outcome_score"] = outcome
            reward["explore_post_norm_status_floor"] = status_floor
            reward["explore_post_norm_adjusted_reward"] = adjusted[i]
            reward["postprocess_total_reward"] = adjusted[i]
    return _apply_truncation_penalties(
        samples,
        adjusted,
        exploration_extra=exploration_extra,
    )


def post_process_rewards(args: Any, samples: list[Any]) -> tuple[list[float], list[float]]:
    raw_rewards, rewards = _default_post_process(args, samples)
    if not _env_flag("EXPLORE_ADVANTAGE_BONUS_ENABLED", os.getenv("EXPLORE_ADVANTAGE_BONUS", "0")):
        return raw_rewards, _apply_truncation_penalties(samples, rewards)
    mode = _env_str("EXPLORE_ADVANTAGE_BONUS_MODE", "component").lower()
    if mode in {"dual", "dual_stream", "intrinsic_advantage"}:
        return raw_rewards, _dual_stream_post_process(args, samples, rewards)

    component_names = [
        part.strip()
        for part in os.getenv("EXPLORE_ADVANTAGE_BONUS_COMPONENTS", "explore_intrinsic_scaled").split(",")
        if part.strip()
    ]
    coef = _env_float("EXPLORE_ADVANTAGE_BONUS_COEF", 1.0)
    clip = _env_float("EXPLORE_ADVANTAGE_BONUS_CLIP", 0.25)

    adjusted = list(rewards)
    exploration_extra = [0.0 for _ in samples]
    for i, sample in enumerate(samples):
        raw_bonus = sum(_component_value(sample, key) for key in component_names)
        clipped_bonus = max(-clip, min(clip, raw_bonus)) if clip > 0 else raw_bonus
        bonus = coef * clipped_bonus
        adjusted[i] += bonus
        exploration_extra[i] = bonus
        reward = getattr(sample, "reward", None)
        if isinstance(reward, dict):
            reward["explore_post_norm_base_reward"] = rewards[i]
            reward["explore_post_norm_bonus_raw"] = raw_bonus
            reward["explore_post_norm_bonus"] = bonus
            reward["explore_post_norm_bonus_coef"] = coef
            reward["explore_post_norm_bonus_clip"] = clip
            reward["explore_post_norm_bonus_mode"] = "component"
            reward["explore_post_norm_bonus_components"] = ",".join(component_names)
            reward["explore_post_norm_adjusted_reward"] = adjusted[i]
            reward["postprocess_total_reward"] = adjusted[i]
    return raw_rewards, _apply_truncation_penalties(
        samples,
        adjusted,
        exploration_extra=exploration_extra,
    )
