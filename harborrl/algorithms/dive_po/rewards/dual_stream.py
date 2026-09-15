"""DIVE-PO dual-stream reward post-processing (production default).

This is the default ``--custom-reward-post-process-path`` wired by
``platform/slime_train.sh``.  It keeps the episodic estimator, lifelong
estimator, multiplicative NGU-lite signal, and UCB allocator unchanged, and
corrects how the already-computed trajectory intrinsic signal is injected
after DAPO/GRPO reward normalization:

1. normalize one intrinsic value per trajectory inside each prompt group;
2. use the configured beta ladder (not the current batch) for arm scaling;
3. preserve the existing quality gate by default, with an opt-in blend knob;
4. center *after* arm/gate weighting, while keeping beta=0 control arms at 0;
5. use a group-wise scale clip, which preserves the zero-sum property.

When every intrinsic value in the batch is zero (baseline profiles) the
output reduces to the base group normalization plus truncation penalties in
``postprocess.py``.  Set ``DIVE_PO_CENTERED_GATE_ENABLED=0`` to fall back to
``postprocess.post_process_rewards`` entirely.

The public function matches slime's custom reward post-process interface.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any

import harborrl.algorithms.dive_po.rewards.postprocess as base


logger = logging.getLogger(__name__)


from harborrl.env import (
    env_flag as _env_flag,
    env_float as _env_float,
    env_int as _env_int,
)


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def _component(sample: Any, key: str, default: float = 0.0) -> float:
    reward = getattr(sample, "reward", None)
    if not isinstance(reward, dict) or key not in reward:
        return float(default)
    return _finite(reward.get(key), default)


def _component_present(sample: Any, key: str) -> bool:
    reward = getattr(sample, "reward", None)
    if not isinstance(reward, dict) or key not in reward:
        return False
    try:
        return math.isfinite(float(reward.get(key)))
    except (TypeError, ValueError):
        return False


def _configured_max_beta(observed_betas: list[float]) -> float:
    configured: list[float] = []
    for part in os.getenv("EXPLORE_AGENT57_ARM_BETAS", "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            value = abs(float(part))
        except ValueError:
            continue
        if math.isfinite(value):
            configured.append(value)
    # A fixed denominator prevents the same arm from changing strength merely
    # because a high-beta arm is absent from one rollout batch.
    return max(configured or [abs(v) for v in observed_betas] or [1.0], default=1.0)


def _trajectory_key(sample: Any, sample_position: int) -> tuple[int, int]:
    return base._sample_traj_key(sample, sample_position)


def _arm_weight(beta: float, mode: str, max_beta: float) -> float:
    if mode in {"none", "off", "0"}:
        return 1.0
    if mode in {"raw", "raw_beta"}:
        return max(0.0, beta)
    return max(0.0, beta) / max(max_beta, 1e-12)


def _soft_gate(sample: Any, quality_blend: float) -> dict[str, float]:
    reward = getattr(sample, "reward", None)
    trust_key = os.getenv(
        "EXPLORE_ADVANTAGE_TRUST_KEY",
        "explore_agent57_trust",
    ).strip()
    trust_present = isinstance(reward, dict) and trust_key in reward
    trust = _component(sample, trust_key, 1.0 if not trust_present else 0.0)
    trust = max(0.0, min(1.0, trust))
    status_scale = max(0.0, min(1.0, base._status_intrinsic_scale(sample)))
    reliability = trust * status_scale

    quality, outcome, status_floor = base._quality_gate(sample)
    blend = max(0.0, min(1.0, quality_blend))
    gate = (1.0 - blend) * reliability + blend * quality

    eligible_key = os.getenv(
        "DIVE_PO_LIFELONG_ELIGIBLE_KEY",
        "explore_agent57_lifelong_eligible",
    ).strip()
    eligible_present = _component_present(sample, eligible_key)
    eligible = _component(sample, eligible_key, 1.0) if eligible_present else 1.0
    eligible = 1.0 if eligible > 0.0 else 0.0
    gate *= eligible
    return {
        "gate": max(0.0, min(1.0, gate)),
        "trust": trust,
        "status_scale": status_scale,
        "reliability": reliability,
        "quality": quality,
        "outcome": outcome,
        "status_floor": status_floor,
        "eligible": eligible,
    }


def _centered_dual_stream(
    args: Any,
    samples: list[Any],
    base_rewards: list[float],
) -> list[float]:
    intrinsic_key = os.getenv(
        "EXPLORE_ADVANTAGE_INTRINSIC_KEY",
        "explore_agent57_intrinsic_signal",
    ).strip()
    intrinsic_values = [_component(sample, intrinsic_key) for sample in samples]
    intrinsic_adv = base._group_normalize_sample_values(
        args,
        samples,
        intrinsic_values,
    )

    lambda_coef = _env_float(
        "EXPLORE_ADVANTAGE_LAMBDA",
        _env_float("EXPLORE_ADVANTAGE_BONUS_COEF", 0.1),
    )
    schedule = os.getenv("EXPLORE_ADVANTAGE_LAMBDA_SCHEDULE", "constant").strip()
    decay_steps = max(0, _env_int("EXPLORE_ADVANTAGE_LAMBDA_DECAY_STEPS", 0))
    train_step = base._batch_train_step(samples)
    schedule_multiplier = base._schedule_multiplier(schedule, train_step, decay_steps)
    effective_lambda = lambda_coef * schedule_multiplier
    # eta=1 is the correctness-only setting: preserve the proven v0710 quality
    # gate.  Values below 1 are an experimental ablation, not part of this fix.
    quality_blend = _env_float("DIVE_PO_GATE_QUALITY_BLEND", 1.0)
    arm_mode = os.getenv(
        "EXPLORE_ADVANTAGE_ARM_WEIGHT_MODE",
        "normalized_beta",
    ).strip().lower()
    bonus_clip = max(0.0, _env_float("EXPLORE_ADVANTAGE_BONUS_CLIP", 0.0))

    betas = [_component(sample, "explore_agent57_beta") for sample in samples]
    max_beta = _configured_max_beta(betas)

    indices_by_traj: dict[tuple[int, int], list[int]] = {}
    traj_order: list[tuple[int, int]] = []
    for i, sample in enumerate(samples):
        key = _trajectory_key(sample, i)
        if key not in indices_by_traj:
            indices_by_traj[key] = []
            traj_order.append(key)
        indices_by_traj[key].append(i)

    traj_info: dict[tuple[int, int], dict[str, Any]] = {}
    groups: dict[int, list[tuple[int, int]]] = {}
    for key in traj_order:
        idxs = indices_by_traj[key]
        trainable_idxs = [i for i in idxs if base._sample_is_trainable(samples[i])]
        first = trainable_idxs[0] if trainable_idxs else idxs[0]
        # Generated turn samples repeat trajectory-level DiVE-PO metrics.  Use
        # one value per trajectory and the most conservative gate across turns.
        gates = [_soft_gate(samples[i], quality_blend) for i in (trainable_idxs or idxs)]
        gate_info = min(gates, key=lambda item: item["gate"])
        arm_weight = _arm_weight(betas[first], arm_mode, max_beta)
        weight = arm_weight * gate_info["gate"] if trainable_idxs else 0.0
        traj_info[key] = {
            "adv": _finite(intrinsic_adv[first]),
            "intrinsic": intrinsic_values[first],
            "beta": betas[first],
            "arm_weight": arm_weight,
            "weight": weight,
            "gate": gate_info,
            "precenter": weight * _finite(intrinsic_adv[first]),
            "center": 0.0,
            "raw_bonus": 0.0,
            "bonus": 0.0,
            "clip_scale": 1.0,
        }
        groups.setdefault(key[0], []).append(key)

    # Center after all sample-dependent scaling.  The weighted form guarantees
    # sum_i w_i(A_i-mu_w)=0 and leaves beta=0 control trajectories exactly zero.
    for keys in groups.values():
        weight_sum = sum(traj_info[key]["weight"] for key in keys)
        if weight_sum <= 1e-12:
            continue
        weighted_mean = (
            sum(
                traj_info[key]["weight"] * traj_info[key]["adv"]
                for key in keys
            )
            / weight_sum
        )
        for key in keys:
            info = traj_info[key]
            info["weighted_mean"] = weighted_mean
            info["center"] = info["weight"] * (info["adv"] - weighted_mean)
            info["raw_bonus"] = effective_lambda * info["center"]

        # Element-wise clipping would destroy zero mean.  Scale the whole group
        # instead, preserving both the bound and exact trajectory-level zero sum.
        max_abs = max(abs(traj_info[key]["raw_bonus"]) for key in keys)
        clip_scale = (
            min(1.0, bonus_clip / max_abs)
            if bonus_clip > 0.0 and max_abs > 0.0
            else 1.0
        )
        for key in keys:
            traj_info[key]["clip_scale"] = clip_scale
            traj_info[key]["bonus"] = traj_info[key]["raw_bonus"] * clip_scale

    adjusted = list(base_rewards)
    exploration_extra = [0.0 for _ in samples]
    for key in traj_order:
        info = traj_info[key]
        gate = info["gate"]
        for i in indices_by_traj[key]:
            bonus = float(info["bonus"])
            adjusted[i] += bonus
            exploration_extra[i] = bonus
            reward = getattr(samples[i], "reward", None)
            if not isinstance(reward, dict):
                continue
            reward.update(
                {
                    "explore_post_norm_bonus_mode": "dive_po_correctness_fix_v1",
                    "explore_post_norm_base_reward": base_rewards[i],
                    "explore_post_norm_intrinsic_key": intrinsic_key,
                    "explore_post_norm_intrinsic_value": info["intrinsic"],
                    "explore_post_norm_intrinsic_advantage": info["adv"],
                    "explore_post_norm_beta": info["beta"],
                    "explore_post_norm_configured_max_beta": max_beta,
                    "explore_post_norm_arm_weight": info["arm_weight"],
                    "explore_post_norm_trust": gate["trust"],
                    "explore_post_norm_status_intrinsic_scale": gate["status_scale"],
                    "explore_post_norm_reliability_gate": gate["reliability"],
                    "explore_post_norm_quality_gate": gate["quality"],
                    "explore_post_norm_outcome_score": gate["outcome"],
                    "explore_post_norm_status_floor": gate["status_floor"],
                    "explore_post_norm_lifelong_eligible": gate["eligible"],
                    "explore_post_norm_gate_quality_blend": quality_blend,
                    "explore_post_norm_effective_gate": gate["gate"],
                    "explore_post_norm_effective_weight": info["weight"],
                    "explore_post_norm_bonus_precenter": info["precenter"],
                    "explore_post_norm_weighted_mean": info.get("weighted_mean", 0.0),
                    "explore_post_norm_centered_advantage": info["center"],
                    "explore_post_norm_bonus_raw": info["raw_bonus"],
                    "explore_post_norm_group_clip_scale": info["clip_scale"],
                    "explore_post_norm_bonus": bonus,
                    "explore_post_norm_bonus_base_coef": lambda_coef,
                    "explore_post_norm_bonus_coef": effective_lambda,
                    "explore_post_norm_bonus_schedule": schedule,
                    "explore_post_norm_bonus_decay_steps": decay_steps,
                    "explore_post_norm_bonus_schedule_multiplier": schedule_multiplier,
                    "explore_post_norm_train_step": train_step,
                    "explore_post_norm_bonus_clip": bonus_clip,
                    "explore_post_norm_adjusted_reward": adjusted[i],
                    "postprocess_total_reward": adjusted[i],
                }
            )

    return base._apply_truncation_penalties(
        samples,
        adjusted,
        exploration_extra=exploration_extra,
    )


def post_process_rewards(
    args: Any,
    samples: list[Any],
) -> tuple[list[float], list[float]]:
    """Slime-compatible reward hook for the conservative DiVE-PO variant."""
    if not samples:
        return [], []
    if not _env_flag("DIVE_PO_CENTERED_GATE_ENABLED", "1"):
        return base.post_process_rewards(args, samples)
    raw_rewards, base_rewards = base._default_post_process(args, samples)
    return raw_rewards, _centered_dual_stream(args, samples, base_rewards)
