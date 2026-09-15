from __future__ import annotations

import math
from types import SimpleNamespace

import sys
from pathlib import Path

TERMINAL_RL_DIR = Path(__file__).resolve().parents[2] / "harborrl"
if str(TERMINAL_RL_DIR) not in sys.path:
    sys.path.insert(0, str(TERMINAL_RL_DIR))

from harborrl.algorithms.dive_po.rewards import postprocess as reward_postprocess


class DummySample:
    def __init__(
        self,
        *,
        group_index: int,
        index: int,
        score: float,
        intrinsic: float,
        beta: float,
        trust: float = 1.0,
        status: str | None = None,
        train_step: int | None = None,
        raw_score: float | None = None,
    ) -> None:
        self.group_index = group_index
        self.index = index
        self.status = status or "completed"
        self.metadata = {}
        if train_step is not None:
            self.metadata["train_step"] = train_step
        self.reward = {
            "score": score,
            "raw_score": score if raw_score is None else raw_score,
            "base_score": score,
            "explore_agent57_intrinsic_signal": intrinsic,
            "explore_agent57_beta": beta,
            "explore_agent57_trust": trust,
        }


def test_dual_stream_advantage_adds_group_normalized_intrinsic(monkeypatch):
    monkeypatch.setenv("EXPLORE_ADVANTAGE_BONUS", "1")
    monkeypatch.setenv("EXPLORE_ADVANTAGE_BONUS_ENABLED", "1")
    monkeypatch.setenv("EXPLORE_ADVANTAGE_BONUS_MODE", "dual_stream")
    monkeypatch.setenv("EXPLORE_ADVANTAGE_LAMBDA", "0.2")
    monkeypatch.setenv("EXPLORE_ADVANTAGE_ARM_WEIGHT_MODE", "normalized_beta")
    monkeypatch.setenv("EXPLORE_ADVANTAGE_BONUS_CLIP", "0")
    args = SimpleNamespace(
        reward_key="score",
        advantage_estimator="grpo",
        rewards_normalization=True,
        grpo_std_normalization=False,
        dynamic_history=False,
    )
    samples = [
        DummySample(group_index=0, index=0, score=1.0, intrinsic=0.0, beta=0.01),
        DummySample(group_index=0, index=1, score=1.0, intrinsic=1.0, beta=0.02),
    ]

    raw, adjusted = reward_postprocess.post_process_rewards(args, samples)

    assert raw == [1.0, 1.0]
    assert adjusted == [-0.05, 0.1]
    assert samples[0].reward["explore_post_norm_bonus_mode"] == "dual_stream"
    assert samples[1].reward["explore_post_norm_base_reward"] == 0.0
    assert samples[1].reward["explore_post_norm_intrinsic_value"] == 1.0
    assert samples[1].reward["explore_post_norm_intrinsic_advantage"] == 0.5
    assert samples[1].reward["explore_post_norm_trust"] == 1.0
    assert samples[1].reward["explore_post_norm_adjusted_reward"] == 0.1


def test_dual_stream_lambda_schedule_uses_train_step(monkeypatch):
    monkeypatch.setenv("EXPLORE_ADVANTAGE_BONUS", "1")
    monkeypatch.setenv("EXPLORE_ADVANTAGE_BONUS_ENABLED", "1")
    monkeypatch.setenv("EXPLORE_ADVANTAGE_BONUS_MODE", "dual_stream")
    monkeypatch.setenv("EXPLORE_ADVANTAGE_LAMBDA", "0.2")
    monkeypatch.setenv("EXPLORE_ADVANTAGE_LAMBDA_SCHEDULE", "cosine")
    monkeypatch.setenv("EXPLORE_ADVANTAGE_LAMBDA_DECAY_STEPS", "120")
    monkeypatch.setenv("EXPLORE_ADVANTAGE_ARM_WEIGHT_MODE", "normalized_beta")
    monkeypatch.setenv("EXPLORE_ADVANTAGE_BONUS_CLIP", "0")
    args = SimpleNamespace(
        reward_key="score",
        advantage_estimator="grpo",
        rewards_normalization=True,
        grpo_std_normalization=False,
        dynamic_history=False,
    )
    samples = [
        DummySample(group_index=0, index=0, score=1.0, intrinsic=0.0, beta=0.01, train_step=60),
        DummySample(group_index=0, index=1, score=1.0, intrinsic=1.0, beta=0.02, train_step=60),
    ]

    _, adjusted = reward_postprocess.post_process_rewards(args, samples)

    assert math.isclose(adjusted[0], -0.025)
    assert math.isclose(adjusted[1], 0.05)
    assert samples[1].reward["explore_post_norm_bonus_base_coef"] == 0.2
    assert samples[1].reward["explore_post_norm_bonus_coef"] == 0.1
    assert samples[1].reward["explore_post_norm_bonus_schedule"] == "cosine"
    assert math.isclose(
        samples[1].reward["explore_post_norm_bonus_schedule_multiplier"],
        0.5,
    )


def test_dual_stream_can_suppress_truncated_intrinsic(monkeypatch):
    monkeypatch.setenv("EXPLORE_ADVANTAGE_BONUS", "1")
    monkeypatch.setenv("EXPLORE_ADVANTAGE_BONUS_ENABLED", "1")
    monkeypatch.setenv("EXPLORE_ADVANTAGE_BONUS_MODE", "dual_stream")
    monkeypatch.setenv("EXPLORE_ADVANTAGE_LAMBDA", "0.2")
    monkeypatch.setenv("EXPLORE_ADVANTAGE_ARM_WEIGHT_MODE", "normalized_beta")
    monkeypatch.setenv("EXPLORE_ADVANTAGE_BONUS_CLIP", "0")
    monkeypatch.setenv("EXPLORE_ADVANTAGE_TRUNCATED_INTRINSIC_SCALE", "0")
    monkeypatch.setenv("EXPLORE_TRUNCATION_PENALTY", "-0.03")
    args = SimpleNamespace(
        reward_key="score",
        advantage_estimator="grpo",
        rewards_normalization=True,
        grpo_std_normalization=False,
        dynamic_history=False,
    )
    samples = [
        DummySample(group_index=0, index=0, score=1.0, intrinsic=0.0, beta=0.01),
        DummySample(
            group_index=0,
            index=1,
            score=1.0,
            intrinsic=1.0,
            beta=0.02,
            status="truncated",
        ),
    ]

    _, adjusted = reward_postprocess.post_process_rewards(args, samples)

    assert math.isclose(adjusted[0], -0.05)
    assert math.isclose(adjusted[1], -0.03)
    assert samples[1].reward["explore_post_norm_bonus"] == 0.0
    assert samples[1].reward["explore_post_norm_status_intrinsic_scale"] == 0.0
    assert samples[1].reward["explore_truncation_penalty"] == -0.03


def test_dual_stream_outcome_status_gate_rewards_high_quality_truncation(monkeypatch):
    monkeypatch.setenv("EXPLORE_ADVANTAGE_BONUS", "1")
    monkeypatch.setenv("EXPLORE_ADVANTAGE_BONUS_ENABLED", "1")
    monkeypatch.setenv("EXPLORE_ADVANTAGE_BONUS_MODE", "dual_stream")
    monkeypatch.setenv("EXPLORE_ADVANTAGE_LAMBDA", "0.2")
    monkeypatch.setenv("EXPLORE_ADVANTAGE_ARM_WEIGHT_MODE", "none")
    monkeypatch.setenv("EXPLORE_ADVANTAGE_GATE_MODE", "outcome_status")
    monkeypatch.setenv("EXPLORE_ADVANTAGE_COMPLETED_FLOOR", "0.5")
    monkeypatch.setenv("EXPLORE_ADVANTAGE_TRUNCATED_FLOOR", "0.15")
    monkeypatch.setenv("EXPLORE_ADVANTAGE_BONUS_CLIP", "0")
    monkeypatch.setenv("EXPLORE_TRUNCATION_PENALTY", "-0.03")
    monkeypatch.setenv("EXPLORE_TRUNCATION_PENALTY_OUTCOME_AWARE", "1")
    args = SimpleNamespace(
        reward_key="score",
        advantage_estimator="grpo",
        rewards_normalization=True,
        grpo_std_normalization=False,
        dynamic_history=False,
    )
    samples = [
        DummySample(
            group_index=0,
            index=0,
            score=1.0,
            raw_score=0.0,
            intrinsic=0.0,
            beta=0.01,
            trust=0.0,
        ),
        DummySample(
            group_index=0,
            index=1,
            score=1.0,
            raw_score=1.0,
            intrinsic=1.0,
            beta=0.02,
            trust=0.0,
            status="truncated",
        ),
    ]

    _, adjusted = reward_postprocess.post_process_rewards(args, samples)

    assert math.isclose(adjusted[0], -0.05)
    assert math.isclose(adjusted[1], 0.1)
    assert samples[0].reward["explore_post_norm_effective_gate"] == 0.5
    assert samples[1].reward["explore_post_norm_effective_gate"] == 1.0
    assert samples[1].reward["explore_post_norm_quality_gate"] == 1.0
    assert samples[1].reward["explore_post_norm_outcome_score"] == 1.0
    assert samples[1].reward["explore_post_norm_status_floor"] == 0.15
    assert samples[1].reward["explore_truncation_penalty"] == 0.0
    assert samples[1].reward["explore_truncation_penalty_multiplier"] == 0.0


def test_component_postnorm_mode_remains_backward_compatible(monkeypatch):
    monkeypatch.setenv("EXPLORE_ADVANTAGE_BONUS", "1")
    monkeypatch.setenv("EXPLORE_ADVANTAGE_BONUS_ENABLED", "1")
    monkeypatch.delenv("EXPLORE_ADVANTAGE_BONUS_MODE", raising=False)
    monkeypatch.setenv("EXPLORE_ADVANTAGE_BONUS_COMPONENTS", "explore_intrinsic_scaled")
    monkeypatch.setenv("EXPLORE_ADVANTAGE_BONUS_COEF", "1.0")
    monkeypatch.setenv("EXPLORE_ADVANTAGE_BONUS_CLIP", "0.25")
    args = SimpleNamespace(
        reward_key="score",
        advantage_estimator="grpo",
        rewards_normalization=True,
        grpo_std_normalization=False,
        dynamic_history=False,
    )
    sample = DummySample(group_index=0, index=0, score=1.0, intrinsic=0.0, beta=0.0)
    sample.reward["explore_intrinsic_scaled"] = 0.5

    _, adjusted = reward_postprocess.post_process_rewards(args, [sample])

    assert adjusted == [0.25]
    assert sample.reward["explore_post_norm_bonus_mode"] == "component"
    assert sample.reward["explore_post_norm_base_reward"] == 0.0
    assert sample.reward["explore_post_norm_adjusted_reward"] == 0.25


def test_truncated_penalty_applies_without_advantage_bonus(monkeypatch):
    monkeypatch.setenv("EXPLORE_ADVANTAGE_BONUS", "0")
    monkeypatch.setenv("EXPLORE_ADVANTAGE_BONUS_ENABLED", "0")
    monkeypatch.setenv("EXPLORE_TRUNCATION_PENALTY", "-0.03")
    args = SimpleNamespace(
        reward_key="score",
        advantage_estimator="grpo",
        rewards_normalization=False,
        grpo_std_normalization=False,
        dynamic_history=False,
    )
    sample = DummySample(
        group_index=0,
        index=0,
        score=1.0,
        intrinsic=0.0,
        beta=0.0,
        status="truncated",
    )

    raw, adjusted = reward_postprocess.post_process_rewards(args, [sample])

    assert raw == [1.0]
    assert adjusted == [0.97]
    assert sample.reward["explore_truncation_penalty"] == -0.03
    assert sample.reward["explore_truncation_penalty_applied"] is True
    assert sample.reward["exploration_reward"] == -0.03
    assert sample.reward["total_reward"] == 0.97
