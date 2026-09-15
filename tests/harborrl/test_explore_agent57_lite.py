from __future__ import annotations

import sys
import sqlite3
import math
from pathlib import Path

TERMINAL_RL_DIR = Path(__file__).resolve().parents[2] / "harborrl"
ROOT_DIR = TERMINAL_RL_DIR.parent
if str(TERMINAL_RL_DIR) not in sys.path:
    sys.path.insert(0, str(TERMINAL_RL_DIR))
if str(ROOT_DIR / "backends" / "slime") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "backends" / "slime"))

from harborrl.algorithms.dive_po.exploration.agent57 import controller as a57
from harborrl.algorithms.dive_po.exploration import rollout_bonus


def _reset_local_agent57_state():
    a57._LOCAL_COUNTS.clear()
    a57._LOCAL_COUNT_LAST_SEEN.clear()
    a57._LOCAL_ARM_EVENTS.clear()
    a57._LOCAL_TRAJ_SEEN = 0
    a57._LOCAL_LIFE_RAW_N = 0
    a57._LOCAL_LIFE_RAW_MEAN = 0.0
    a57._LOCAL_LIFE_RAW_M2 = 0.0
    a57._reset_ucb_rng_for_tests()


def test_rollout_bonus_exposes_latest_episodic_state_without_stale_aliases():
    rollout_bonus._AGENT57_LAST_EPISODIC_STATS = {"action_count": 2.0}
    rollout_bonus._AGENT57_LAST_EPISODIC_BY_TURN = {0: 0.5}

    stats = rollout_bonus._agent57_last_episodic_stats()
    by_turn = rollout_bonus._agent57_last_episodic_by_turn()
    stats["action_count"] = -1.0
    by_turn[0] = -1.0

    assert rollout_bonus._agent57_last_episodic_stats() == {"action_count": 2.0}
    assert rollout_bonus._agent57_last_episodic_by_turn() == {0: 0.5}


def test_agent57_config_defaults_preserve_additive_mode(monkeypatch):
    for name in (
        "EXPLORE_AGENT57_COMBINE_MODE",
        "EXPLORE_AGENT57_NGU_MOD_CLIP",
        "EXPLORE_AGENT57_NGU_EPISODIC_SOURCE",
        "EXPLORE_AGENT57_NGU_EPISODIC_REDUCER",
        "EXPLORE_AGENT57_NGU_LIFE_MOD_MODE",
        "EXPLORE_AGENT57_NGU_LIFE_MOD_STD_CLIP",
        "EXPLORE_AGENT57_MAX_BONUS",
        "EXPLORE_AGENT57_LIFELONG_COUNT_DECAY",
        "EXPLORE_AGENT57_LIFELONG_CAPACITY",
        "EXPLORE_AGENT57_LIFELONG_OBS_MODE",
        "EXPLORE_AGENT57_LIFELONG_HIERARCHICAL",
        "EXPLORE_AGENT57_LIFELONG_TASK_WEIGHT",
        "EXPLORE_AGENT57_LIFELONG_SKILL_WEIGHT",
        "EXPLORE_AGENT57_LIFELONG_GLOBAL_WEIGHT",
        "EXPLORE_AGENT57_SQLITE_BUSY_TIMEOUT_MS",
        "EXPLORE_AGENT57_SQLITE_WAL",
        "EXPLORE_AGENT57_TRUST_GATE",
    ):
        monkeypatch.delenv(name, raising=False)

    config = a57.config_from_env()

    assert config.combine_mode == "add"
    assert config.ngu_mod_clip == 5.0
    assert config.ngu_episodic_source == "signature_intrinsic"
    assert config.ngu_episodic_reducer == "sum"
    assert config.ngu_life_mod_mode == "linear"
    assert config.ngu_life_mod_std_clip == 5.0
    assert config.max_bonus == 0.0
    assert config.lifelong_count_decay == 1.0
    assert config.lifelong_capacity == 0
    assert config.lifelong_obs_mode == "fingerprint"
    assert config.lifelong_hierarchical is False
    assert config.sqlite_busy_timeout_ms == 30000
    assert config.sqlite_wal is False
    assert config.trust_gate_mode == "hard"


def test_ngu_lite_bonus_uses_product_and_clamp(monkeypatch):
    monkeypatch.setenv("EXPLORE_AGENT57_LITE", "1")
    monkeypatch.setenv("EXPLORE_AGENT57_LIFELONG", "1")
    monkeypatch.setenv("EXPLORE_AGENT57_COMBINE_MODE", "ngu_lite")
    monkeypatch.setenv("EXPLORE_AGENT57_ARM_BETAS", "0.02")
    monkeypatch.setenv("EXPLORE_AGENT57_LIFELONG_COEF", "0.5")
    monkeypatch.setenv("EXPLORE_AGENT57_NGU_MOD_CLIP", "3")
    monkeypatch.setenv("EXPLORE_AGENT57_MAX_BONUS", "0.05")
    config = a57.config_from_env()

    metrics = a57.compute_ngu_lite_bonus(
        config=config,
        arm_id=0,
        episodic_novelty=10.0,
        lifelong_raw=5.0,
        lifelong_eligible=True,
        trust_gate=1.0,
    )

    assert metrics["explore_agent57_ngu_life_mod"] == 3.0
    assert metrics["explore_agent57_intrinsic_signal"] == 30.0
    assert metrics["explore_agent57_ngu_bonus_unclipped"] == 0.3
    assert metrics["explore_agent57_ngu_bonus"] == 0.05
    assert metrics["explore_agent57_bonus_clipped"] == 1.0


def test_ngu_lite_bonus_stays_zero_when_lifelong_not_eligible(monkeypatch):
    monkeypatch.setenv("EXPLORE_AGENT57_LITE", "1")
    monkeypatch.setenv("EXPLORE_AGENT57_LIFELONG", "1")
    monkeypatch.setenv("EXPLORE_AGENT57_COMBINE_MODE", "ngu_lite")
    monkeypatch.setenv("EXPLORE_AGENT57_ARM_BETAS", "0.02")
    config = a57.config_from_env()

    metrics = a57.compute_ngu_lite_bonus(
        config=config,
        arm_id=0,
        episodic_novelty=10.0,
        lifelong_raw=1.0,
        lifelong_eligible=False,
    )

    assert metrics["explore_agent57_ngu_bonus"] == 0.0
    assert metrics["explore_agent57_ngu_episodic"] == 10.0


def test_observation_buckets_separate_command_success_from_test_pass():
    command_success = "Command executed successfully (no output)."
    file_success = "Content successfully written to '/tmp/audit.sh' in Docker container."
    test_success = "1 passed in 0.12s"

    assert a57.coarse_observation_fingerprint(command_success) == "success_no_output:lenS"
    assert a57.coarse_observation_label(command_success) == "success_no_output:lenS"
    assert a57.exit_code_bucket(command_success) == "exit0"
    assert a57.coarse_observation_fingerprint(file_success).startswith("operation_success:")
    assert a57.coarse_observation_label(file_success).startswith("operation_success:")
    assert a57.exit_code_bucket(file_success) == "exit0"
    assert a57.coarse_observation_fingerprint(test_success).startswith("test_pass:")
    assert a57.coarse_observation_label(test_success).startswith("test_pass:")


def test_observation_fingerprint_canonicalizes_dynamic_numbers():
    left = "job 1 at Thu Jun 11 07:39:00 2026\n92"
    right = "job 2 at Thu Jun 11 07:41:00 2026\n93"

    assert a57.coarse_observation_fingerprint(left) == a57.coarse_observation_fingerprint(right)


def test_ngu_lite_bonus_soft_trust_scales_product(monkeypatch):
    monkeypatch.setenv("EXPLORE_AGENT57_LITE", "1")
    monkeypatch.setenv("EXPLORE_AGENT57_LIFELONG", "1")
    monkeypatch.setenv("EXPLORE_AGENT57_COMBINE_MODE", "ngu_lite")
    monkeypatch.setenv("EXPLORE_AGENT57_ARM_BETAS", "0.02")
    monkeypatch.setenv("EXPLORE_AGENT57_LIFELONG_COEF", "0.5")
    config = a57.config_from_env()

    metrics = a57.compute_ngu_lite_bonus(
        config=config,
        arm_id=0,
        episodic_novelty=10.0,
        lifelong_raw=1.0,
        lifelong_eligible=True,
        trust_gate=0.1,
    )

    assert math.isclose(metrics["explore_agent57_ngu_bonus_unclipped"], 0.02)
    assert metrics["explore_agent57_trust"] == 0.1


def test_ucb_min_per_arm_prioritizes_under_sampled_arms(monkeypatch):
    _reset_local_agent57_state()
    monkeypatch.setenv("EXPLORE_AGENT57_LITE", "1")
    monkeypatch.setenv("EXPLORE_AGENT57_CONTROLLER", "ucb")
    monkeypatch.setenv("EXPLORE_AGENT57_K", "4")
    monkeypatch.setenv("EXPLORE_AGENT57_LIFELONG_BACKEND", "local")
    monkeypatch.setenv("EXPLORE_AGENT57_UCB_C", "0")
    monkeypatch.setenv("EXPLORE_AGENT57_UCB_MIN_PER_ARM", "2")
    monkeypatch.setenv("EXPLORE_AGENT57_KEEP_BASELINE", "1")
    config = a57.config_from_env()

    for arm_id in (0, 1):
        for _ in range(2):
            a57.record_arm_event(
                config=config,
                arm_id=arm_id,
                base_score=1.0,
                final_score=1.0,
                status="completed",
                parse_error_count=0,
                bonus=0.0,
                dataset="seta",
            )

    arms = a57.assign_group_arms(4, dataset="seta")

    assert arms[:3] == [0, 2, 3]


def test_ucb_dataset_aware_uses_normalized_base_reward(monkeypatch):
    _reset_local_agent57_state()
    monkeypatch.setenv("EXPLORE_AGENT57_LITE", "1")
    monkeypatch.setenv("EXPLORE_AGENT57_CONTROLLER", "ucb")
    monkeypatch.setenv("EXPLORE_AGENT57_K", "3")
    monkeypatch.setenv("EXPLORE_AGENT57_LIFELONG_BACKEND", "local")
    monkeypatch.setenv("EXPLORE_AGENT57_UCB_C", "0")
    monkeypatch.setenv("EXPLORE_AGENT57_UCB_VALUE", "normalized_base")
    monkeypatch.setenv("EXPLORE_AGENT57_UCB_DATASET_AWARE", "1")
    monkeypatch.setenv("EXPLORE_AGENT57_KEEP_BASELINE", "0")
    config = a57.config_from_env()

    for dataset, scores in (
        ("seta", {0: 0.0, 1: 1.0, 2: 0.0}),
        ("agentharm", {0: -1.0, 1: -1.0, 2: 1.0}),
    ):
        for arm_id, score in scores.items():
            a57.record_arm_event(
                config=config,
                arm_id=arm_id,
                base_score=score,
                final_score=score,
                status="completed",
                parse_error_count=0,
                bonus=0.0,
                dataset=dataset,
            )

    assert a57.assign_group_arms(1, dataset="seta") == [1]
    assert a57.assign_group_arms(1, dataset="agentharm") == [2]


def test_ucb_seta_can_use_raw_accuracy_for_normalized_base(monkeypatch):
    _reset_local_agent57_state()
    monkeypatch.setenv("EXPLORE_AGENT57_LITE", "1")
    monkeypatch.setenv("EXPLORE_AGENT57_CONTROLLER", "ucb")
    monkeypatch.setenv("EXPLORE_AGENT57_K", "2")
    monkeypatch.setenv("EXPLORE_AGENT57_LIFELONG_BACKEND", "local")
    monkeypatch.setenv("EXPLORE_AGENT57_UCB_C", "0")
    monkeypatch.setenv("EXPLORE_AGENT57_UCB_VALUE", "normalized_base")
    monkeypatch.setenv("EXPLORE_AGENT57_UCB_DATASET_AWARE", "1")
    monkeypatch.setenv("EXPLORE_AGENT57_KEEP_BASELINE", "0")
    config = a57.config_from_env()

    a57.record_arm_event(
        config=config,
        arm_id=0,
        base_score=-0.5,
        final_score=-0.5,
        status="completed",
        parse_error_count=0,
        bonus=0.0,
        dataset="seta",
        normalized_base_score=0.25,
        success_score=0.25,
    )
    a57.record_arm_event(
        config=config,
        arm_id=1,
        base_score=-0.9,
        final_score=-0.9,
        status="completed",
        parse_error_count=0,
        bonus=0.0,
        dataset="seta",
        normalized_base_score=0.05,
        success_score=0.05,
    )

    assert a57.assign_group_arms(1, dataset="seta") == [0]


def test_ucb_quality_value_uses_outcome_aware_trunc_penalty(monkeypatch):
    _reset_local_agent57_state()
    monkeypatch.setenv("EXPLORE_AGENT57_LITE", "1")
    monkeypatch.setenv("EXPLORE_AGENT57_CONTROLLER", "ucb")
    monkeypatch.setenv("EXPLORE_AGENT57_K", "3")
    monkeypatch.setenv("EXPLORE_AGENT57_LIFELONG_BACKEND", "local")
    monkeypatch.setenv("EXPLORE_AGENT57_UCB_C", "0")
    monkeypatch.setenv("EXPLORE_AGENT57_UCB_VALUE", "quality")
    monkeypatch.setenv("EXPLORE_AGENT57_UCB_PARSE_PENALTY", "0.5")
    monkeypatch.setenv("EXPLORE_AGENT57_UCB_TRUNC_PENALTY", "0.5")
    monkeypatch.setenv("EXPLORE_AGENT57_KEEP_BASELINE", "0")
    config = a57.config_from_env()

    for arm_id, score, status in (
        (0, 0.90, "truncated"),
        (1, 0.60, "completed"),
        (2, 0.40, "truncated"),
    ):
        a57.record_arm_event(
            config=config,
            arm_id=arm_id,
            base_score=score,
            final_score=score,
            status=status,
            parse_error_count=0,
            bonus=0.0,
            dataset="seta",
            normalized_base_score=score,
            success_score=score,
        )

    assert a57.assign_group_arms(3, dataset="seta") == [0, 1, 2]


def test_ucb_can_skip_infra_failure_arm_events(monkeypatch):
    _reset_local_agent57_state()
    monkeypatch.setenv("EXPLORE_AGENT57_UCB_SKIP_INFRA_FAILURES", "1")

    events = [
        {"arm_id": 0, "normalized_base_score": 1.0, "infra_failure": 1.0, "dataset": "seta"},
        {"arm_id": 1, "normalized_base_score": 0.5, "infra_failure": 0.0, "dataset": "seta"},
    ]
    stats = a57._aggregate_arm_stats(
        2,
        events,
        dataset="seta",
        skip_infra_failures=True,
    )

    assert stats[0]["n"] == 0.0
    assert stats[1]["n"] == 1.0


def test_ucb_random_seed_reproduces_tie_break_and_epsilon(monkeypatch):
    _reset_local_agent57_state()
    monkeypatch.setenv("EXPLORE_AGENT57_LITE", "1")
    monkeypatch.setenv("EXPLORE_AGENT57_CONTROLLER", "ucb")
    monkeypatch.setenv("EXPLORE_AGENT57_K", "6")
    monkeypatch.setenv("EXPLORE_AGENT57_LIFELONG_BACKEND", "local")
    monkeypatch.setenv("EXPLORE_AGENT57_KEEP_BASELINE", "0")
    monkeypatch.setenv("EXPLORE_AGENT57_UCB_EPSILON", "0.5")
    monkeypatch.setenv("EXPLORE_AGENT57_UCB_RANDOM_SEED", "123")

    first = [a57.assign_group_arms(6, dataset="seta") for _ in range(4)]

    a57._reset_ucb_rng_for_tests()
    second = [a57.assign_group_arms(6, dataset="seta") for _ in range(4)]

    assert first == second

    monkeypatch.setenv("EXPLORE_AGENT57_UCB_RANDOM_SEED", "456")
    a57._reset_ucb_rng_for_tests()
    third = [a57.assign_group_arms(6, dataset="seta") for _ in range(4)]

    assert third != first


def test_ucb_random_seed_salt_changes_stream(monkeypatch):
    _reset_local_agent57_state()
    monkeypatch.setenv("EXPLORE_AGENT57_LITE", "1")
    monkeypatch.setenv("EXPLORE_AGENT57_CONTROLLER", "ucb")
    monkeypatch.setenv("EXPLORE_AGENT57_K", "6")
    monkeypatch.setenv("EXPLORE_AGENT57_LIFELONG_BACKEND", "local")
    monkeypatch.setenv("EXPLORE_AGENT57_KEEP_BASELINE", "0")
    monkeypatch.setenv("EXPLORE_AGENT57_UCB_EPSILON", "1.0")
    monkeypatch.setenv("EXPLORE_AGENT57_UCB_RANDOM_SEED", "123")
    monkeypatch.setenv("EXPLORE_AGENT57_UCB_SEED_SALT", "worker-a")
    first = [a57.assign_group_arms(6, dataset="seta") for _ in range(4)]

    monkeypatch.setenv("EXPLORE_AGENT57_UCB_SEED_SALT", "worker-a")
    a57._reset_ucb_rng_for_tests()
    second = [a57.assign_group_arms(6, dataset="seta") for _ in range(4)]

    monkeypatch.setenv("EXPLORE_AGENT57_UCB_SEED_SALT", "worker-b")
    a57._reset_ucb_rng_for_tests()
    third = [a57.assign_group_arms(6, dataset="seta") for _ in range(4)]

    assert first == second
    assert third != first


def test_lifelong_key_v1_ignores_context_metadata(monkeypatch):
    monkeypatch.setenv("EXPLORE_AGENT57_LIFELONG_KEY_VERSION", "v1")
    config = a57.config_from_env()
    actions = [{"tool_name": "shell", "signature": "shell|pytest", "raw": "pytest"}]
    turns = [{"turn_idx": 0, "command": "pytest", "result": {"exit_code": 0}}]

    seta_key = a57.lifelong_keys(
        actions,
        turns,
        config=config,
        metadata={"data_source": "seta", "task_path": "seta_env/1"},
    )
    safety_key = a57.lifelong_keys(
        actions,
        turns,
        config=config,
        metadata={"data_source": "agent_safetybench", "task_path": "asb/9"},
    )

    assert seta_key == safety_key


def test_lifelong_key_v2_includes_dataset_by_default(monkeypatch):
    monkeypatch.setenv("EXPLORE_AGENT57_LIFELONG_KEY_VERSION", "v2")
    monkeypatch.setenv("EXPLORE_AGENT57_LIFELONG_INCLUDE_DATASET", "1")
    config = a57.config_from_env()
    actions = [{"tool_name": "shell", "signature": "shell|pytest", "raw": "pytest"}]
    turns = [{"turn_idx": 0, "command": "pytest", "result": {"exit_code": 0}}]

    seta_key = a57.lifelong_keys(
        actions,
        turns,
        config=config,
        metadata={"data_source": "seta", "task_path": "seta_env/1"},
    )
    safety_key = a57.lifelong_keys(
        actions,
        turns,
        config=config,
        metadata={"data_source": "agent_safetybench", "task_path": "asb/9"},
    )

    assert seta_key != safety_key


def test_lifelong_key_v2_label_obs_mode_collapses_generic_text(monkeypatch):
    actions = [{"tool_name": "shell", "signature": "shell|cat", "raw": "cat file"}]
    turns_a = [{"turn_idx": 0, "command": "cat a", "result": {"stdout": "alpha beta"}}]
    turns_b = [{"turn_idx": 0, "command": "cat b", "result": {"stdout": "gamma zeta"}}]
    monkeypatch.setenv("EXPLORE_AGENT57_LIFELONG_KEY_VERSION", "v2")
    monkeypatch.setenv("EXPLORE_AGENT57_LIFELONG_INCLUDE_DATASET", "1")
    monkeypatch.setenv("EXPLORE_AGENT57_LIFELONG_OBS_MODE", "label")
    config = a57.config_from_env()

    key_a = a57.lifelong_keys(actions, turns_a, config=config, metadata={"data_source": "seta"})
    key_b = a57.lifelong_keys(actions, turns_b, config=config, metadata={"data_source": "seta"})

    assert key_a == key_b

    monkeypatch.setenv("EXPLORE_AGENT57_LIFELONG_OBS_MODE", "fingerprint")
    config = a57.config_from_env()
    key_a = a57.lifelong_keys(actions, turns_a, config=config, metadata={"data_source": "seta"})
    key_b = a57.lifelong_keys(actions, turns_b, config=config, metadata={"data_source": "seta"})

    assert key_a != key_b


def test_soft_trust_gate_keeps_failed_lifelong_signal(monkeypatch):
    _reset_local_agent57_state()
    monkeypatch.setenv("EXPLORE_AGENT57_LITE", "1")
    monkeypatch.setenv("EXPLORE_AGENT57_LIFELONG", "1")
    monkeypatch.setenv("EXPLORE_AGENT57_LIFELONG_BACKEND", "local")
    monkeypatch.setenv("EXPLORE_AGENT57_ARM_BETAS", "0.02")
    monkeypatch.setenv("EXPLORE_AGENT57_LIFELONG_COEF", "0.5")
    monkeypatch.setenv("EXPLORE_AGENT57_TRUST_GATE", "soft")
    monkeypatch.setenv("EXPLORE_AGENT57_TRUST_FAILED", "0.1")
    monkeypatch.setenv("EXPLORE_AGENT57_TRUST_PARSE_ERROR", "0.1")
    monkeypatch.setenv("EXPLORE_AGENT57_TRUST_WARMUP", "0.3")
    config = a57.config_from_env()
    actions = [{"tool_name": "shell", "signature": "shell|pytest", "raw": "pytest"}]
    turns = [{"turn_idx": 0, "command": "pytest", "result": {"exit_code": 1}}]

    metrics = a57.compute_lifelong_bonus(
        config=config,
        arm_id=0,
        actions=actions,
        turn_records=turns,
        status="failed",
        parse_error_count=1,
        metadata={"data_source": "seta"},
    )

    assert metrics["explore_agent57_lifelong_eligible"] == 1.0
    assert metrics["explore_agent57_trust"] == 0.1
    assert metrics["explore_agent57_lifelong_bonus"] > 0.0
    assert "parse_error" in metrics["explore_agent57_lifelong_suppressed_reason"]


def test_infra_failure_does_not_pollute_lifelong_state(monkeypatch):
    _reset_local_agent57_state()
    monkeypatch.setenv("EXPLORE_AGENT57_LITE", "1")
    monkeypatch.setenv("EXPLORE_AGENT57_LIFELONG", "1")
    monkeypatch.setenv("EXPLORE_AGENT57_LIFELONG_BACKEND", "local")
    config = a57.config_from_env()

    metrics = a57.compute_lifelong_bonus(
        config=config,
        arm_id=0,
        actions=[{"tool_name": "shell", "signature": "shell|pytest", "raw": "pytest"}],
        turn_records=[{"turn_idx": 0, "result": {"exit_code": 0}}],
        status="failed",
        parse_error_count=0,
        metadata={"data_source": "seta"},
        state_update_allowed=False,
    )

    assert metrics["explore_agent57_lifelong_state_update_allowed"] is False
    assert metrics["explore_agent57_lifelong_suppressed_reason"] == "infra_failure"
    assert metrics["explore_agent57_lifelong_eligible"] == 0.0
    assert a57._LOCAL_TRAJ_SEEN == 0
    assert a57._LOCAL_COUNTS == {}


def test_lifelong_local_counts_support_decay_and_capacity(monkeypatch):
    _reset_local_agent57_state()
    monkeypatch.setenv("EXPLORE_AGENT57_LITE", "1")
    monkeypatch.setenv("EXPLORE_AGENT57_LIFELONG", "1")
    monkeypatch.setenv("EXPLORE_AGENT57_LIFELONG_BACKEND", "local")
    monkeypatch.setenv("EXPLORE_AGENT57_LIFELONG_WARMUP", "0")
    monkeypatch.setenv("EXPLORE_AGENT57_LIFELONG_COUNT_DECAY", "0.5")
    monkeypatch.setenv("EXPLORE_AGENT57_LIFELONG_CAPACITY", "1")
    config = a57.config_from_env()
    turns = [{"turn_idx": 0, "command": "pytest", "result": {"exit_code": 0}}]

    a57.compute_lifelong_bonus(
        config=config,
        arm_id=0,
        actions=[{"tool_name": "shell", "signature": "shell|pytest", "raw": "pytest"}],
        turn_records=turns,
        status="completed",
        parse_error_count=0,
    )
    second = a57.compute_lifelong_bonus(
        config=config,
        arm_id=0,
        actions=[{"tool_name": "shell", "signature": "shell|pytest", "raw": "pytest"}],
        turn_records=turns,
        status="completed",
        parse_error_count=0,
    )
    a57.compute_lifelong_bonus(
        config=config,
        arm_id=0,
        actions=[{"tool_name": "shell", "signature": "shell|ls", "raw": "ls"}],
        turn_records=turns,
        status="completed",
        parse_error_count=0,
    )

    assert math.isclose(second["explore_agent57_lifelong_raw"], 1.0 / math.sqrt(2.0))
    assert len(a57._LOCAL_COUNTS) == 1


def test_lifelong_local_counts_decay_by_last_seen_gap(monkeypatch):
    _reset_local_agent57_state()
    monkeypatch.setenv("EXPLORE_AGENT57_LITE", "1")
    monkeypatch.setenv("EXPLORE_AGENT57_LIFELONG", "1")
    monkeypatch.setenv("EXPLORE_AGENT57_LIFELONG_BACKEND", "local")
    monkeypatch.setenv("EXPLORE_AGENT57_LIFELONG_WARMUP", "0")
    monkeypatch.setenv("EXPLORE_AGENT57_LIFELONG_COUNT_DECAY", "0.5")
    config = a57.config_from_env()
    turns = [{"turn_idx": 0, "command": "pytest", "result": {"exit_code": 0}}]

    action_a = [{"tool_name": "shell", "signature": "shell|pytest", "raw": "pytest"}]
    action_b = [{"tool_name": "shell", "signature": "shell|ls", "raw": "ls"}]
    a57.compute_lifelong_bonus(
        config=config,
        arm_id=0,
        actions=action_a,
        turn_records=turns,
        status="completed",
        parse_error_count=0,
    )
    a57.compute_lifelong_bonus(
        config=config,
        arm_id=0,
        actions=action_b,
        turn_records=turns,
        status="completed",
        parse_error_count=0,
    )
    third = a57.compute_lifelong_bonus(
        config=config,
        arm_id=0,
        actions=action_a,
        turn_records=turns,
        status="completed",
        parse_error_count=0,
    )

    assert math.isclose(third["explore_agent57_lifelong_raw"], 1.0 / math.sqrt(1.5))


def test_lifelong_hierarchical_key_reuses_skill_across_tasks(monkeypatch):
    _reset_local_agent57_state()
    monkeypatch.setenv("EXPLORE_AGENT57_LITE", "1")
    monkeypatch.setenv("EXPLORE_AGENT57_LIFELONG", "1")
    monkeypatch.setenv("EXPLORE_AGENT57_LIFELONG_BACKEND", "local")
    monkeypatch.setenv("EXPLORE_AGENT57_LIFELONG_KEY_VERSION", "v2")
    monkeypatch.setenv("EXPLORE_AGENT57_LIFELONG_INCLUDE_DATASET", "1")
    monkeypatch.setenv("EXPLORE_AGENT57_LIFELONG_INCLUDE_TASK", "1")
    monkeypatch.setenv("EXPLORE_AGENT57_LIFELONG_HIERARCHICAL", "1")
    monkeypatch.setenv("EXPLORE_AGENT57_LIFELONG_TASK_WEIGHT", "0")
    monkeypatch.setenv("EXPLORE_AGENT57_LIFELONG_SKILL_WEIGHT", "1")
    monkeypatch.setenv("EXPLORE_AGENT57_LIFELONG_GLOBAL_WEIGHT", "0")
    monkeypatch.setenv("EXPLORE_AGENT57_LIFELONG_WARMUP", "0")
    config = a57.config_from_env()
    action = [{"tool_name": "shell", "signature": "shell|pgrep|atd", "raw": "pgrep atd"}]
    turns_a = [{"turn_idx": 0, "command": "pgrep atd", "result": {"stdout": "92"}}]
    turns_b = [{"turn_idx": 0, "command": "pgrep atd", "result": {"stdout": "93"}}]

    a57.compute_lifelong_bonus(
        config=config,
        arm_id=0,
        actions=action,
        turn_records=turns_a,
        status="completed",
        parse_error_count=0,
        metadata={"data_source": "seta", "task_path": "seta_env/1"},
    )
    second = a57.compute_lifelong_bonus(
        config=config,
        arm_id=0,
        actions=action,
        turn_records=turns_b,
        status="completed",
        parse_error_count=0,
        metadata={"data_source": "seta", "task_path": "seta_env/2"},
    )

    assert second["explore_agent57_lifelong_task_raw"] == 1.0
    assert math.isclose(second["explore_agent57_lifelong_skill_raw"], 1.0 / math.sqrt(2.0))
    assert math.isclose(second["explore_agent57_lifelong_raw"], 1.0 / math.sqrt(2.0))


def test_lifelong_sqlite_updates_counts_and_stats_in_one_path(tmp_path, monkeypatch):
    db_path = tmp_path / "agent57.sqlite3"
    monkeypatch.setenv("EXPLORE_AGENT57_LITE", "1")
    monkeypatch.setenv("EXPLORE_AGENT57_LIFELONG", "1")
    monkeypatch.setenv("EXPLORE_AGENT57_LIFELONG_BACKEND", "sqlite")
    monkeypatch.setenv("EXPLORE_AGENT57_STATE_PATH", str(db_path))
    monkeypatch.setenv("EXPLORE_AGENT57_LIFELONG_WARMUP", "0")
    monkeypatch.setenv("EXPLORE_AGENT57_ARM_BETAS", "0.02")
    monkeypatch.setenv("EXPLORE_AGENT57_SQLITE_BUSY_TIMEOUT_MS", "12345")
    monkeypatch.setenv("EXPLORE_AGENT57_SQLITE_WAL", "0")
    a57._SQLITE_SCHEMA_INITIALIZED.discard(str(db_path))
    config = a57.config_from_env()
    actions = [{"tool_name": "shell", "signature": "shell|pytest", "raw": "pytest"}]
    turns = [{"turn_idx": 0, "command": "pytest", "result": {"exit_code": 0}}]

    first = a57.compute_lifelong_bonus(
        config=config,
        arm_id=0,
        actions=actions,
        turn_records=turns,
        status="completed",
        parse_error_count=0,
        metadata={"data_source": "seta", "task_path": "seta_env/1"},
    )
    second = a57.compute_lifelong_bonus(
        config=config,
        arm_id=0,
        actions=actions,
        turn_records=turns,
        status="completed",
        parse_error_count=0,
        metadata={"data_source": "seta", "task_path": "seta_env/1"},
    )

    assert first["explore_agent57_sqlite_busy_timeout_ms"] == 12345
    assert first["explore_agent57_sqlite_wal"] is False
    assert first["explore_agent57_lifelong_seen_before"] == 0
    assert second["explore_agent57_lifelong_seen_before"] == 1
    assert second["explore_agent57_lifelong_stat_n"] == 1
    assert math.isclose(second["explore_agent57_lifelong_raw"], 1.0 / math.sqrt(2.0))

    conn = sqlite3.connect(db_path)
    try:
        meta = dict(conn.execute("SELECT name, value FROM meta").fetchall())
        count = conn.execute("SELECT count FROM lifelong_counts").fetchone()[0]
        columns = {
            row[1]: str(row[2]).upper()
            for row in conn.execute("PRAGMA table_info(lifelong_counts)")
        }
    finally:
        conn.close()

    assert int(meta["lifelong_traj_seen"]) == 2
    assert int(meta["lifelong_raw_n"]) == 2
    assert count > 1.0
    assert columns["count"] == "REAL"


def test_standardized_life_mod_override_drives_ngu_product(monkeypatch):
    monkeypatch.setenv("EXPLORE_AGENT57_LITE", "1")
    monkeypatch.setenv("EXPLORE_AGENT57_LIFELONG", "1")
    monkeypatch.setenv("EXPLORE_AGENT57_COMBINE_MODE", "ngu_lite")
    monkeypatch.setenv("EXPLORE_AGENT57_ARM_BETAS", "0.02")
    monkeypatch.setenv("EXPLORE_AGENT57_LIFELONG_COEF", "0.5")
    monkeypatch.setenv("EXPLORE_AGENT57_NGU_LIFE_MOD_MODE", "standardized_softplus")
    config = a57.config_from_env()

    metrics = a57.compute_ngu_lite_bonus(
        config=config,
        arm_id=0,
        episodic_novelty=2.0,
        lifelong_raw=0.0,
        lifelong_eligible=True,
        life_mod_override=1.25,
    )

    assert metrics["explore_agent57_ngu_life_mod"] == 1.25
    assert metrics["explore_agent57_intrinsic_signal"] == 2.5


def test_lifelong_key_v2_task_bucket_is_opt_in(monkeypatch):
    actions = [{"tool_name": "shell", "signature": "shell|pytest", "raw": "pytest"}]
    turns = [{"turn_idx": 0, "command": "pytest", "result": {"exit_code": 0}}]
    monkeypatch.setenv("EXPLORE_AGENT57_LIFELONG_KEY_VERSION", "v2")
    monkeypatch.setenv("EXPLORE_AGENT57_LIFELONG_INCLUDE_DATASET", "1")
    monkeypatch.setenv("EXPLORE_AGENT57_LIFELONG_INCLUDE_TASK", "0")
    config = a57.config_from_env()

    task_a_key = a57.lifelong_keys(
        actions,
        turns,
        config=config,
        metadata={"data_source": "seta", "task_path": "seta_env/1"},
    )
    task_b_key = a57.lifelong_keys(
        actions,
        turns,
        config=config,
        metadata={"data_source": "seta", "task_path": "seta_env/2"},
    )
    assert task_a_key == task_b_key

    monkeypatch.setenv("EXPLORE_AGENT57_LIFELONG_INCLUDE_TASK", "1")
    config = a57.config_from_env()
    task_a_key = a57.lifelong_keys(
        actions,
        turns,
        config=config,
        metadata={"data_source": "seta", "task_path": "seta_env/1"},
    )
    task_b_key = a57.lifelong_keys(
        actions,
        turns,
        config=config,
        metadata={"data_source": "seta", "task_path": "seta_env/2"},
    )
    assert task_a_key != task_b_key


def test_sqlite_arm_event_schema_migration(tmp_path, monkeypatch):
    db_path = tmp_path / "agent57.sqlite3"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "CREATE TABLE arm_events "
            "(id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, "
            "arm_id INTEGER NOT NULL, base_score REAL NOT NULL, "
            "final_score REAL NOT NULL, success INTEGER NOT NULL, "
            "parse_error INTEGER NOT NULL, truncated INTEGER NOT NULL, "
            "bonus REAL NOT NULL)"
        )
        conn.commit()
    finally:
        conn.close()

    monkeypatch.setenv("EXPLORE_AGENT57_LITE", "1")
    monkeypatch.setenv("EXPLORE_AGENT57_CONTROLLER", "ucb")
    monkeypatch.setenv("EXPLORE_AGENT57_LIFELONG_BACKEND", "sqlite")
    monkeypatch.setenv("EXPLORE_AGENT57_STATE_PATH", str(db_path))
    a57._SQLITE_SCHEMA_INITIALIZED.discard(str(db_path))
    config = a57.config_from_env()

    a57.record_arm_event(
        config=config,
        arm_id=1,
        base_score=1.0,
        final_score=1.0,
        status="completed",
        parse_error_count=0,
        bonus=0.0,
        dataset="seta",
        infra_failure=True,
    )

    conn = sqlite3.connect(db_path)
    try:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(arm_events)")}
        row = conn.execute(
            "SELECT dataset, normalized_base_score, infra_failure FROM arm_events ORDER BY id DESC LIMIT 1"
        ).fetchone()
    finally:
        conn.close()

    assert {"dataset", "normalized_base_score", "infra_failure"}.issubset(columns)
    assert row == ("seta", 1.0, 1)
