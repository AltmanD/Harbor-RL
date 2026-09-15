from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace


TERMINAL_RL = Path(__file__).resolve().parents[2] / "harborrl"
if str(TERMINAL_RL) not in sys.path:
    sys.path.insert(0, str(TERMINAL_RL))

from harborrl.algorithms.dive_po.rewards import dual_stream as centered  # noqa: E402


class DummySample:
    def __init__(
        self,
        *,
        group_index: int,
        index: int,
        score: float,
        intrinsic: float,
        beta: float,
        raw_score: float = 0.0,
        trust: float = 1.0,
        eligible: float = 1.0,
        status: str = "completed",
        turn_idx: int = 0,
        remove_sample: bool = False,
    ) -> None:
        self.group_index = group_index
        self.index = index
        self.status = status
        self.remove_sample = remove_sample
        self.loss_mask = [0] if remove_sample else [1]
        self.metadata = {"turn_idx": turn_idx, "train_step": 10}
        self.reward = {
            "score": score,
            "raw_score": raw_score,
            "base_score": score,
            "explore_agent57_intrinsic_signal": intrinsic,
            "explore_agent57_beta": beta,
            "explore_agent57_trust": trust,
            "explore_agent57_lifelong_eligible": eligible,
        }


def _args(*, dynamic_history: bool = True):
    return SimpleNamespace(
        reward_key="score",
        advantage_estimator="grpo",
        rewards_normalization=True,
        grpo_std_normalization=True,
        dynamic_history=dynamic_history,
    )


def _base_env(monkeypatch):
    monkeypatch.setenv("DIVE_PO_CENTERED_GATE_ENABLED", "1")
    monkeypatch.setenv("DIVE_PO_GATE_QUALITY_BLEND", "1")
    monkeypatch.setenv("EXPLORE_AGENT57_ARM_BETAS", "0,0.01,0.02")
    monkeypatch.setenv("EXPLORE_ADVANTAGE_ARM_WEIGHT_MODE", "normalized_beta")
    monkeypatch.setenv("EXPLORE_ADVANTAGE_LAMBDA", "1")
    monkeypatch.setenv("EXPLORE_ADVANTAGE_BONUS_CLIP", "0")
    monkeypatch.setenv("EXPLORE_ADVANTAGE_COMPLETED_FLOOR", "0.5")
    monkeypatch.setenv("EXPLORE_ADVANTAGE_TRUNCATED_FLOOR", "0.15")
    monkeypatch.setenv("EXPLORE_ADVANTAGE_FAILED_FLOOR", "0")
    monkeypatch.setenv("EXPLORE_ADVANTAGE_ABORTED_FLOOR", "0")
    monkeypatch.setenv("EXPLORE_TRUNCATION_PENALTY", "0")


def test_weighted_center_preserves_zero_sum_and_beta0_control(monkeypatch):
    _base_env(monkeypatch)
    samples = [
        DummySample(group_index=0, index=0, score=0.0, intrinsic=0.0, beta=0.0),
        DummySample(group_index=0, index=1, score=0.0, intrinsic=1.0, beta=0.01),
        DummySample(group_index=0, index=2, score=0.0, intrinsic=2.0, beta=0.02),
    ]

    _, adjusted = centered.post_process_rewards(_args(), samples)
    bonuses = [s.reward["explore_post_norm_bonus"] for s in samples]

    assert bonuses[0] == 0.0
    assert math.isclose(sum(bonuses), 0.0, abs_tol=1e-12)
    assert bonuses[1] < 0.0 < bonuses[2]
    assert adjusted == bonuses


def test_configured_beta_denominator_does_not_depend_on_batch(monkeypatch):
    _base_env(monkeypatch)
    samples = [
        DummySample(group_index=0, index=0, score=0.0, intrinsic=0.0, beta=0.0),
        DummySample(group_index=0, index=1, score=0.0, intrinsic=1.0, beta=0.01),
    ]

    centered.post_process_rewards(_args(), samples)

    assert samples[1].reward["explore_post_norm_configured_max_beta"] == 0.02
    assert samples[1].reward["explore_post_norm_arm_weight"] == 0.5


def test_dynamic_history_uses_one_bonus_per_trajectory(monkeypatch):
    _base_env(monkeypatch)
    samples = [
        DummySample(group_index=0, index=0, score=0.0, intrinsic=0.0, beta=0.01, turn_idx=0),
        DummySample(group_index=0, index=0, score=0.0, intrinsic=0.0, beta=0.01, turn_idx=1),
        DummySample(group_index=0, index=1, score=0.0, intrinsic=2.0, beta=0.02, turn_idx=0),
    ]

    centered.post_process_rewards(_args(), samples)
    bonuses = [s.reward["explore_post_norm_bonus"] for s in samples]

    assert bonuses[0] == bonuses[1]
    assert math.isclose(bonuses[0] + bonuses[2], 0.0, abs_tol=1e-12)


def test_ineligible_trajectory_receives_no_intrinsic_bonus(monkeypatch):
    _base_env(monkeypatch)
    samples = [
        DummySample(
            group_index=0,
            index=0,
            score=0.0,
            intrinsic=0.0,
            beta=0.01,
            eligible=0.0,
        ),
        DummySample(group_index=0, index=1, score=0.0, intrinsic=1.0, beta=0.02),
    ]

    centered.post_process_rewards(_args(), samples)

    assert samples[0].reward["explore_post_norm_bonus"] == 0.0
    # With only one eligible trajectory, weighted centering also correctly
    # produces no within-group learning signal.
    assert samples[1].reward["explore_post_norm_bonus"] == 0.0


def test_non_trainable_trajectory_does_not_shift_valid_bonus(monkeypatch):
    _base_env(monkeypatch)
    samples = [
        DummySample(group_index=0, index=0, score=0.0, intrinsic=0.0, beta=0.01),
        DummySample(
            group_index=0,
            index=1,
            score=0.0,
            intrinsic=100.0,
            beta=0.02,
            remove_sample=True,
            status="failed",
        ),
        DummySample(group_index=0, index=2, score=0.0, intrinsic=2.0, beta=0.02),
    ]

    centered.post_process_rewards(_args(), samples)
    bonuses = [sample.reward["explore_post_norm_bonus"] for sample in samples]

    assert bonuses[1] == 0.0
    assert bonuses[0] < 0.0 < bonuses[2]
    assert math.isclose(bonuses[0] + bonuses[2], 0.0, abs_tol=1e-12)


def test_malformed_trajectory_index_falls_back_to_position(monkeypatch):
    _base_env(monkeypatch)
    samples = [
        DummySample(group_index=0, index=0, score=0.0, intrinsic=0.0, beta=0.01),
        DummySample(group_index=0, index=1, score=0.0, intrinsic=1.0, beta=0.02),
    ]
    samples[1].index = "not-an-integer"

    _, adjusted = centered.post_process_rewards(_args(), samples)

    assert all(math.isfinite(value) for value in adjusted)


def test_group_scale_clip_preserves_bound_and_zero_sum(monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.setenv("EXPLORE_ADVANTAGE_LAMBDA", "100")
    monkeypatch.setenv("EXPLORE_ADVANTAGE_BONUS_CLIP", "0.1")
    samples = [
        DummySample(group_index=0, index=0, score=0.0, intrinsic=0.0, beta=0.01),
        DummySample(group_index=0, index=1, score=0.0, intrinsic=1.0, beta=0.02),
        DummySample(group_index=0, index=2, score=0.0, intrinsic=3.0, beta=0.02),
    ]

    centered.post_process_rewards(_args(), samples)
    bonuses = [s.reward["explore_post_norm_bonus"] for s in samples]

    assert max(abs(value) for value in bonuses) <= 0.1 + 1e-12
    assert math.isclose(sum(bonuses), 0.0, abs_tol=1e-12)


def test_gate_blend_remains_finite_for_nan_components(monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.setenv("DIVE_PO_GATE_QUALITY_BLEND", "0.5")
    samples = [
        DummySample(group_index=0, index=0, score=0.0, intrinsic=float("nan"), beta=0.01),
        DummySample(group_index=0, index=1, score=0.0, intrinsic=1.0, beta=0.02, raw_score=1.0),
    ]

    _, adjusted = centered.post_process_rewards(_args(), samples)

    assert all(math.isfinite(value) for value in adjusted)
    assert all(
        math.isfinite(s.reward["explore_post_norm_effective_gate"])
        for s in samples
    )


def test_eta_one_preserves_quality_gate(monkeypatch):
    _base_env(monkeypatch)
    sample = DummySample(
        group_index=0,
        index=0,
        score=0.0,
        intrinsic=1.0,
        beta=0.02,
        raw_score=0.25,
        trust=0.0,
    )

    centered.post_process_rewards(_args(), [sample])

    # q = completed_floor + (1-completed_floor) * raw_score
    expected_quality = 0.5 + 0.5 * 0.25
    assert sample.reward["explore_post_norm_gate_quality_blend"] == 1.0
    assert sample.reward["explore_post_norm_quality_gate"] == expected_quality
    assert sample.reward["explore_post_norm_effective_gate"] == expected_quality
