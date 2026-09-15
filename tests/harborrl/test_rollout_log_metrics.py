from __future__ import annotations

import importlib
import math
import sys
import types
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

TERMINAL_RL_DIR = Path(__file__).resolve().parents[2] / "harborrl"
REPO_ROOT = TERMINAL_RL_DIR.parent
for path in (REPO_ROOT / "backends" / "slime", TERMINAL_RL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


class _StubSample:
    class Status(Enum):
        COMPLETED = "completed"
        TRUNCATED = "truncated"
        FAILED = "failed"
        ABORTED = "aborted"


def _install_rollout_log_import_stubs() -> dict[str, types.ModuleType | None]:
    previous = {
        name: sys.modules.get(name)
        for name in (
            "wandb",
            "slime",
            "slime.utils",
            "slime.utils.logging_utils",
            "slime.utils.types",
            "slime.ray",
            "slime.ray.rollout",
        )
    }

    wandb = types.ModuleType("wandb")
    wandb.define_metric = lambda *args, **kwargs: None

    slime = types.ModuleType("slime")
    slime.__path__ = []
    slime_utils = types.ModuleType("slime.utils")
    slime_utils.__path__ = []
    logging_utils = types.ModuleType("slime.utils.logging_utils")
    logging_utils.log = lambda *args, **kwargs: None
    slime_types = types.ModuleType("slime.utils.types")
    slime_types.Sample = _StubSample

    slime_ray = types.ModuleType("slime.ray")
    slime_ray.__path__ = []
    rollout = types.ModuleType("slime.ray.rollout")
    rollout.compute_rollout_step = lambda args, rollout_id: rollout_id

    sys.modules["wandb"] = wandb
    sys.modules["slime"] = slime
    sys.modules["slime.utils"] = slime_utils
    sys.modules["slime.utils.logging_utils"] = logging_utils
    sys.modules["slime.utils.types"] = slime_types
    sys.modules["slime.ray"] = slime_ray
    sys.modules["slime.ray.rollout"] = rollout
    return previous


def _restore_import_stubs(previous: dict[str, types.ModuleType | None]) -> None:
    for name, module in previous.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def _import_rollout_log():
    previous = _install_rollout_log_import_stubs()
    try:
        sys.modules.pop("harborrl.misc.rollout_log", None)
        return importlib.import_module("harborrl.misc.rollout_log")
    finally:
        _restore_import_stubs(previous)


rollout_log = _import_rollout_log()


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
            "explore_total_bonus": 0.0,
            "explore_agent57_intrinsic_signal": intrinsic,
            "explore_agent57_ngu_episodic": intrinsic / 2.0,
            "explore_agent57_ngu_life_mod": 2.0,
            "explore_agent57_lifelong_raw": intrinsic / 3.0,
            "explore_agent57_lifelong_bonus": intrinsic / 100.0,
            "explore_agent57_ngu_bonus": intrinsic / 200.0,
            "explore_agent57_beta": beta,
            "explore_agent57_trust": trust,
        }


def test_dual_stream_fallback_matches_scheduled_status_scaled_postprocess(monkeypatch):
    monkeypatch.setenv("EXPLORE_ADVANTAGE_BONUS", "1")
    monkeypatch.setenv("EXPLORE_ADVANTAGE_BONUS_ENABLED", "1")
    monkeypatch.setenv("EXPLORE_ADVANTAGE_BONUS_MODE", "dual_stream")
    monkeypatch.setenv("EXPLORE_ADVANTAGE_LAMBDA", "0.2")
    monkeypatch.setenv("EXPLORE_ADVANTAGE_LAMBDA_SCHEDULE", "cosine")
    monkeypatch.setenv("EXPLORE_ADVANTAGE_LAMBDA_DECAY_STEPS", "120")
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
        DummySample(group_index=0, index=0, score=1.0, intrinsic=0.0, beta=0.01, train_step=60),
        DummySample(
            group_index=0,
            index=1,
            score=1.0,
            intrinsic=1.0,
            beta=0.02,
            status="truncated",
            train_step=60,
        ),
    ]

    values = rollout_log._expected_post_norm_exploration_values(args, samples)

    assert math.isclose(values[0], -0.025)
    assert math.isclose(values[1], -0.03)


def test_component_fallback_includes_truncation_penalty(monkeypatch):
    monkeypatch.setenv("EXPLORE_ADVANTAGE_BONUS", "1")
    monkeypatch.setenv("EXPLORE_ADVANTAGE_BONUS_ENABLED", "1")
    monkeypatch.delenv("EXPLORE_ADVANTAGE_BONUS_MODE", raising=False)
    monkeypatch.setenv("EXPLORE_ADVANTAGE_BONUS_COMPONENTS", "explore_intrinsic_scaled")
    monkeypatch.setenv("EXPLORE_ADVANTAGE_BONUS_COEF", "1.0")
    monkeypatch.setenv("EXPLORE_ADVANTAGE_BONUS_CLIP", "0.25")
    monkeypatch.setenv("EXPLORE_TRUNCATION_PENALTY", "-0.03")
    args = SimpleNamespace(
        reward_key="score",
        advantage_estimator="grpo",
        rewards_normalization=True,
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
    sample.reward["explore_intrinsic_scaled"] = 0.5

    values = rollout_log._expected_post_norm_exploration_values(args, [sample])

    assert values == [0.22]


def test_dual_stream_fallback_uses_outcome_status_gate(monkeypatch):
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

    values = rollout_log._expected_post_norm_exploration_values(args, samples)
    record = rollout_log._metric_record_from_samples(
        args=args,
        phase="train",
        dataset_name="seta",
        source_datasets=["seta"],
        rollout_id=3,
        step=6,
        samples=samples,
    )

    assert values == [-0.05, 0.1]
    assert math.isclose(record["reward/adv_intrinsic"], 0.025)
    assert record["reward/adv_penalty"] == 0.0
    assert record["reward/outcome_score"] == 0.5
    assert record["reward/quality_gate"] == 0.75
    assert record["reward/quality_gate_truncated"] == 1.0
    assert record["reward/truncated_outcome_score"] == 1.0


def test_metric_record_exposes_core_reward_fusion_fields(monkeypatch):
    monkeypatch.setenv("EXPLORE_ADVANTAGE_BONUS", "1")
    monkeypatch.setenv("EXPLORE_ADVANTAGE_BONUS_ENABLED", "1")
    monkeypatch.setenv("EXPLORE_ADVANTAGE_BONUS_MODE", "dual_stream")
    monkeypatch.setenv("EXPLORE_ADVANTAGE_LAMBDA", "0.2")
    monkeypatch.setenv("EXPLORE_ADVANTAGE_LAMBDA_SCHEDULE", "cosine")
    monkeypatch.setenv("EXPLORE_ADVANTAGE_LAMBDA_DECAY_STEPS", "120")
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
        DummySample(
            group_index=0,
            index=0,
            score=1.0,
            intrinsic=0.0,
            beta=0.01,
            trust=1.0,
            train_step=60,
        ),
        DummySample(
            group_index=0,
            index=1,
            score=1.0,
            intrinsic=1.0,
            beta=0.02,
            trust=0.15,
            status="truncated",
            train_step=60,
        ),
    ]

    record = rollout_log._metric_record_from_samples(
        args=args,
        phase="train",
        dataset_name="seta",
        source_datasets=["seta"],
        rollout_id=3,
        step=6,
        samples=samples,
    )

    assert record["reward/task"] == 1.0
    assert record["reward/intrinsic_episodic"] == 0.25
    assert math.isclose(record["reward/intrinsic_lifelong"], 0.005)
    assert record["reward/intrinsic_signal"] == 0.5
    assert math.isclose(record["reward/adv_intrinsic"], -0.0125)
    assert math.isclose(record["reward/adv_penalty"], -0.015)
    assert math.isclose(record["agent57/trust_mean"], 0.575)
    assert record["agent57/trust_completed_mean"] == 1.0
    assert record["agent57/trust_truncated_mean"] == 0.15


def test_step_context_uses_raw_rollout_axis_and_train_axis():
    args = SimpleNamespace(
        num_steps_per_rollout=None,
        rollout_batch_size=64,
        n_samples_per_prompt=8,
        global_batch_size=128,
    )

    context = rollout_log._step_context(args, 3, rollout_step=12)

    assert context == {
        "rollout_id": 3,
        "rollout_step": 3,
        "train_step": 12,
        "steps_per_rollout": 4,
        "legacy_rollout_step": 12,
    }


def test_reward_fusion_axis_metrics_expose_canonical_fields(monkeypatch):
    monkeypatch.setenv("EXPLORE_ADVANTAGE_BONUS", "1")
    monkeypatch.setenv("EXPLORE_ADVANTAGE_BONUS_ENABLED", "1")
    monkeypatch.setenv("EXPLORE_ADVANTAGE_BONUS_MODE", "dual_stream")
    monkeypatch.setenv("EXPLORE_ADVANTAGE_LAMBDA", "0.2")
    monkeypatch.setenv("EXPLORE_ADVANTAGE_ARM_WEIGHT_MODE", "none")
    monkeypatch.setenv("EXPLORE_ADVANTAGE_BONUS_CLIP", "0")
    monkeypatch.setenv("EXPLORE_TRUNCATION_PENALTY", "-0.03")
    args = SimpleNamespace(
        reward_key="score",
        advantage_estimator="grpo",
        rewards_normalization=True,
        grpo_std_normalization=False,
        dynamic_history=False,
    )
    samples = [
        DummySample(group_index=0, index=0, score=0.0, intrinsic=0.0, beta=0.01),
        DummySample(
            group_index=0,
            index=1,
            score=1.0,
            intrinsic=1.0,
            beta=0.02,
            status="truncated",
        ),
    ]
    samples[0].reward["explore_post_norm_base_reward"] = -0.5
    samples[1].reward["explore_post_norm_base_reward"] = 0.5
    samples[0].reward["explore_post_norm_bonus"] = -0.1
    samples[1].reward["explore_post_norm_bonus"] = 0.1
    samples[0].reward["explore_truncation_penalty"] = 0.0
    samples[1].reward["explore_truncation_penalty"] = -0.03
    samples[0].reward["explore_post_norm_adjusted_reward"] = -0.6
    samples[1].reward["explore_post_norm_adjusted_reward"] = 0.57

    metrics = rollout_log._reward_fusion_axis_metrics(args, samples)
    record = rollout_log._metric_record_from_samples(
        args=args,
        phase="train",
        dataset_name="seta",
        source_datasets=["seta"],
        rollout_id=3,
        step=12,
        samples=samples,
    )

    assert metrics["reward/task"] == 0.5
    assert metrics["intrinsic/intra"] == 0.25
    assert math.isclose(metrics["intrinsic/inter"], 0.005)
    assert metrics["intrinsic/fused"] == 0.5
    assert metrics["adv/task"] == 0.0
    assert metrics["adv/intrinsic"] == 0.0
    assert metrics["adv/final_penalty"] == -0.015
    assert math.isclose(metrics["adv/with_penalty"], -0.015)
    assert record["intrinsic/fused"] == metrics["intrinsic/fused"]
    assert record["adv/with_penalty"] == metrics["adv/with_penalty"]


def test_trajectory_store_exports_local_save_decision():
    from harborrl.rollout import trajectory_store

    assert callable(trajectory_store._trajectory_save_decision)


def test_eval_protocol_summary_groups_samples_by_task_for_pass_at_k():
    samples = [
        DummySample(group_index=0, index=0, score=0.0, intrinsic=0.0, beta=0.0),
        DummySample(group_index=0, index=1, score=1.0, intrinsic=0.0, beta=0.0),
        DummySample(group_index=1, index=2, score=0.0, intrinsic=0.0, beta=0.0),
        DummySample(group_index=1, index=3, score=0.0, intrinsic=0.0, beta=0.0),
    ]
    responses = ["inspect logs", "fix service", "read config", "read config"]
    for index, (sample, response) in enumerate(zip(samples, responses, strict=True)):
        sample.metadata.update(
            {
                "eval_prompt_index": index // 2,
                "eval_sample_index": index % 2,
                "eval_sampling_seed": 100 + index,
                "task_name": str(700 + index // 2),
            }
        )
        sample.response = response

    summary, tasks = rollout_log._eval_protocol_summary(samples)

    assert summary["eval/task_count"] == 2
    assert summary["eval/k"] == 2
    assert summary["pass_at_2"] == 0.5
    assert summary["eval/reward_best_at_k"] == 0.5
    assert tasks[0]["rewards"] == [0.0, 1.0]
    assert tasks[0]["pass_at_k"] == 1.0
    assert tasks[1]["response_unique_ratio"] == 0.5


def test_eval_protocol_summary_deduplicates_turns_and_uses_raw_outcome():
    samples = []
    for sample_index in range(8):
        for turn_idx in range(3):
            sample = DummySample(
                group_index=0,
                index=len(samples),
                score=1.0,
                raw_score=0.0,
                intrinsic=0.0,
                beta=0.0,
            )
            sample.metadata.update(
                {
                    "eval_prompt_index": 0,
                    "eval_sample_index": sample_index,
                    "eval_sampling_seed": 200 + sample_index,
                    "task_name": "705",
                    "turn_idx": turn_idx,
                    "uid": f"trajectory-{sample_index}",
                }
            )
            sample.response = f"trajectory {sample_index} turn {turn_idx}"
            samples.append(sample)

    summary, tasks = rollout_log._eval_protocol_summary(samples)

    assert summary["eval/task_count"] == 1
    assert summary["eval/k"] == 8
    assert summary["pass_at_8"] == 0.0
    assert tasks[0]["k"] == 8
    assert len(tasks[0]["responses"]) == 8
