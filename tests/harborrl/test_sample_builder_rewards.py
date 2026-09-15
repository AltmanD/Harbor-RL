"""Pin down the reward shaping math in rollout/sample_builder._build_samples
(binary outcome mapping, per-turn discount, short-response override, DAPO
overlong penalty, PRM/safety fusion, alias syncing) and assert that the
group-normalization helper duplicated in misc/rollout_log stays in lockstep
with the production copy in algorithms/dive_po/rewards/postprocess."""

from __future__ import annotations

import importlib
import math
import sys
import types
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import pytest

TERMINAL_RL_DIR = Path(__file__).resolve().parents[2] / "harborrl"
REPO_ROOT = TERMINAL_RL_DIR.parent
for path in (REPO_ROOT / "backends" / "slime", TERMINAL_RL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from slime.utils.types import Sample

from harborrl.types import Interaction
from harborrl.rollout.sample_builder import (
    _build_samples,
    _dapo_overlong_cfg,
    _dapo_overlong_reward,
    _mark_non_trainable_samples,
    _sync_reward_aliases,
)
from harborrl.rollout import generate_steps
from harborrl.rollout.generate_steps import _agent57_normalized_outcome
from harborrl.algorithms.dive_po.rewards import postprocess


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
            "slime.utils.logging_utils",
            "slime.ray",
            "slime.ray.rollout",
        )
    }

    wandb = types.ModuleType("wandb")
    wandb.define_metric = lambda *args, **kwargs: None
    logging_utils = types.ModuleType("slime.utils.logging_utils")
    logging_utils.log = lambda *args, **kwargs: None

    slime_ray = types.ModuleType("slime.ray")
    slime_ray.__path__ = []
    rollout = types.ModuleType("slime.ray.rollout")
    rollout.compute_rollout_step = lambda args, rollout_id: rollout_id

    sys.modules["wandb"] = wandb
    sys.modules["slime.utils.logging_utils"] = logging_utils
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


def make_interaction(turn_idx: int, n_out: int = 20) -> Interaction:
    return Interaction(
        turn_idx=turn_idx,
        input_ids=[10, 11, 12],
        output_token_ids=list(range(n_out)),
        output_token_logprobs=[-0.1] * n_out,
        output_text=f"turn {turn_idx} response",
        finish_reason="stop",
        latency_ms=12.0,
    )


def make_base_sample() -> Sample:
    sample = Sample()
    sample.group_index = 0
    sample.index = 0
    sample.prompt = "task"
    sample.metadata = {}
    return sample


def test_agent57_seta_alias_uses_raw_outcome_not_shaped_score():
    partial = make_base_sample()
    partial.reward = {"raw_score": 0.5, "score": -1.25}
    full = make_base_sample()
    full.reward = {"raw_score": 1.0, "score": 0.2}

    # Converted SETA data routes rewards under the historical terminal_bench
    # name.  UCB must still see the task-native partial/full credit (mean .75),
    # not the negative score after overlong/truncation shaping.
    assert _agent57_normalized_outcome([partial, full], "terminal_bench") == 0.75
    assert _agent57_normalized_outcome([partial], "seta_env") == 0.5
    assert _agent57_normalized_outcome([partial], "agentharm") is None


def scores(samples):
    return [s.reward["score"] for s in samples]


def test_binary_outcome_maps_to_plus_minus_one():
    base = make_base_sample()
    win = _build_samples([make_interaction(0), make_interaction(1)], base, 1.0, Sample.Status.COMPLETED)
    assert scores(win) == [1.0, 1.0]
    assert win[0].reward["base_score"] == 1.0
    assert win[0].reward["raw_score"] == 1.0

    lose = _build_samples([make_interaction(0)], make_base_sample(), 0.0, Sample.Status.COMPLETED)
    assert scores(lose) == [-1.0]


def test_discount_applies_from_final_turn_backwards():
    samples = _build_samples(
        [make_interaction(0), make_interaction(1), make_interaction(2)],
        make_base_sample(),
        1.0,
        Sample.Status.COMPLETED,
        discount=0.5,
    )
    assert scores(samples) == [0.25, 0.5, 1.0]
    assert samples[0].reward["base_score"] == 0.25


def test_outcome_is_score_passes_value_through():
    samples = _build_samples(
        [make_interaction(0)],
        make_base_sample(),
        0.7,
        Sample.Status.COMPLETED,
        outcome_is_score=True,
    )
    s = samples[0]
    assert s.reward["raw_score"] == 0.7
    assert s.reward["score"] == 0.7
    assert s.reward["outcome_is_score"] is True


def test_short_single_turn_response_is_overridden_to_minus_one():
    samples = _build_samples(
        [make_interaction(0, n_out=5)],
        make_base_sample(),
        1.0,
        Sample.Status.COMPLETED,
    )
    assert scores(samples) == [-1.0]


def test_short_response_penalty_skips_multi_turn():
    samples = _build_samples(
        [make_interaction(0, n_out=5), make_interaction(1, n_out=5)],
        make_base_sample(),
        1.0,
        Sample.Status.COMPLETED,
    )
    assert scores(samples) == [1.0, 1.0]


def test_short_response_penalty_can_be_disabled():
    samples = _build_samples(
        [make_interaction(0, n_out=5)],
        make_base_sample(),
        1.0,
        Sample.Status.COMPLETED,
        penalize_short_response=False,
    )
    assert scores(samples) == [1.0]


def test_dapo_overlong_reward_math():
    cfg = {"max_resp_len": 100, "buffer_len": 20, "penalty_factor": 1.0, "expected_len": 80}
    assert _dapo_overlong_reward(80, cfg) == 0.0
    assert _dapo_overlong_reward(70, cfg) == 0.0
    assert math.isclose(_dapo_overlong_reward(90, cfg), -0.5)
    assert _dapo_overlong_reward(0, None) == 0.0
    assert _dapo_overlong_reward(10, None) == 0.0


def test_dapo_overlong_cfg_env_gating(monkeypatch):
    monkeypatch.setenv("ALGO", "grpo")
    assert _dapo_overlong_cfg(SimpleNamespace(rollout_max_response_len=100)) is None

    monkeypatch.setenv("ALGO", "dapo")
    monkeypatch.delenv("DAPO_OVERLONG_BUFFER_ENABLE", raising=False)
    monkeypatch.delenv("DAPO_MAX_RESPONSE_LEN", raising=False)
    monkeypatch.delenv("DAPO_OVERLONG_BUFFER_LEN", raising=False)
    cfg = _dapo_overlong_cfg(SimpleNamespace(rollout_max_response_len=100))
    assert cfg == {
        "max_resp_len": 100,
        "buffer_len": 100,  # default 4096 buffer is clamped to max_resp_len
        "penalty_factor": 1.0,
        "expected_len": 0,
    }

    monkeypatch.setenv("DAPO_OVERLONG_BUFFER_LEN", "20")
    cfg = _dapo_overlong_cfg(SimpleNamespace(rollout_max_response_len=100))
    assert cfg["buffer_len"] == 20
    assert cfg["expected_len"] == 80

    monkeypatch.setenv("DAPO_OVERLONG_BUFFER_ENABLE", "0")
    assert _dapo_overlong_cfg(SimpleNamespace(rollout_max_response_len=100)) is None


def test_prm_fusion_and_stepwise_metadata():
    samples = _build_samples(
        [make_interaction(0), make_interaction(1)],
        make_base_sample(),
        1.0,
        Sample.Status.COMPLETED,
        prm_turn_scores={1: 0.5},
        prm_coef=2.0,
    )
    assert scores(samples) == [1.0, 2.0]
    assert samples[1].reward["prm_turn_score"] == 0.5
    assert samples[1].metadata["step_wise"] == {
        "step_scores": [0.5],
        "step_scores_with_outcome": [2.0],
        "step_indices": [1],
        "step_token_spans": [[0, 20]],
    }


def test_tokens_logprobs_loss_mask_and_metadata_populated():
    samples = _build_samples([make_interaction(0, n_out=7)], make_base_sample(), 1.0, Sample.Status.COMPLETED)
    s = samples[0]
    assert s.tokens == [10, 11, 12] + list(range(7))
    assert s.response_length == 7
    assert s.loss_mask == [1] * 7
    assert s.rollout_log_probs == [-0.1] * 7
    assert s.response == "turn 0 response"
    assert s.metadata["turn_idx"] == 0
    assert s.metadata["num_turns"] == 1


def test_reward_aliases_synced_after_build():
    samples = _build_samples([make_interaction(0)], make_base_sample(), 1.0, Sample.Status.COMPLETED)
    reward = samples[0].reward
    assert reward["raw_reward"] == reward["raw_score"]
    assert reward["task_reward"] == reward["base_score"]
    assert reward["total_reward"] == reward["score"]
    assert reward["exploration_reward"] == 0.0


def test_sync_reward_aliases_tolerates_missing_keys():
    reward = {"score": 0.3}
    _sync_reward_aliases(reward)
    assert reward["raw_reward"] == 0.3
    assert reward["task_reward"] == 0.3
    assert reward["total_reward"] == 0.3
    _sync_reward_aliases(None)  # no-op, must not raise


def test_mark_non_trainable_samples_flags_removal():
    sample = Sample()
    sample.status = Sample.Status.ABORTED
    sample.reward = None
    keep = Sample()
    keep.status = Sample.Status.COMPLETED
    keep.reward = {"score": 1.0}
    _mark_non_trainable_samples([sample, keep])
    assert sample.remove_sample is True
    assert sample.reward["score"] == 0.0
    assert sample.reward["total_reward"] == 0.0
    assert keep.remove_sample is False


def _norm_samples(group_pairs):
    out = []
    for group_index, index in group_pairs:
        s = SimpleNamespace(group_index=group_index, index=index, metadata={}, status="completed")
        out.append(s)
    return out


@pytest.mark.parametrize("dynamic_history", [False, True])
def test_group_normalization_matches_postprocess(dynamic_history):
    args = SimpleNamespace(grpo_std_normalization=False, dynamic_history=dynamic_history)
    samples = _norm_samples([(0, 0), (0, 1), (1, 0), (1, 1), (1, 2)])
    values = [1.0, 3.0, 0.0, 2.0, 4.0]
    expected = postprocess._group_normalize_sample_values(args, samples, values)
    logged = rollout_log._group_normalize_values_for_log(args, samples, values)
    assert logged == expected


@pytest.mark.parametrize("dynamic_history", [False, True])
def test_group_normalization_excludes_non_trainable_samples(dynamic_history):
    args = SimpleNamespace(grpo_std_normalization=False, dynamic_history=dynamic_history)
    samples = _norm_samples([(0, 0), (0, 1), (0, 2)])
    for sample in samples:
        sample.remove_sample = False
        sample.loss_mask = [1]
    samples[1].remove_sample = True
    samples[1].loss_mask = [0]

    normalized = postprocess._group_normalize_sample_values(
        args,
        samples,
        [1.0, 100.0, 3.0],
    )

    assert normalized == [-1.0, 0.0, 1.0]


def test_evaluation_does_not_mutate_exploration_state(monkeypatch):
    monkeypatch.setattr(generate_steps, "_EXPLORE_INTRINSIC_ENABLED", True)

    def fail_if_called(_turn_records):
        raise AssertionError("evaluation entered exploration reward path")

    monkeypatch.setattr(generate_steps, "_explore_intrinsic_bonus", fail_if_called)
    generate_steps._inject_exploration_bonuses(
        [],
        sample=SimpleNamespace(metadata={}),
        plan=SimpleNamespace(evaluation=True),
        clients=SimpleNamespace(),
        loop=SimpleNamespace(),
        status="completed",
        eval_error=None,
    )
