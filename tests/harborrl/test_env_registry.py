"""Pin the environment registry semantics that used to be spread across
admission/environment_factory/entrypoint/sample_builder/trajectory_store."""

from __future__ import annotations

import sys
from pathlib import Path

TERMINAL_RL_DIR = Path(__file__).resolve().parents[2] / "harborrl"
REPO_ROOT = TERMINAL_RL_DIR.parent
for path in (REPO_ROOT / "backends" / "slime", TERMINAL_RL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from harborrl.environments.registry import (
    canonical_spec,
    direct_score_source,
    interval_candidates_for_slug,
    local_env_spec,
    safety_split_applies,
    slug_for,
    spec_for_source,
    uses_remote_terminal_env,
)


def test_local_env_spec_matches_canonical_sources():
    assert local_env_spec({"data_source": "agent_safetybench"}).data_source == "agent_safetybench"
    assert local_env_spec({"data_source": "agentharm"}).data_source == "agentharm"
    assert local_env_spec({"data_source": "tau2"}).data_source == "tau2"


def test_local_env_spec_rejects_remote_flag_and_unknown(monkeypatch):
    monkeypatch.setenv("AGENTHARM_REMOTE_ENV", "1")
    assert local_env_spec({"data_source": "agentharm"}) is None
    monkeypatch.setenv("TAU2_REMOTE_ENV", "1")
    assert local_env_spec({"data_source": "tau2"}) is None
    assert local_env_spec({"data_source": "swe_verified"}) is None
    assert local_env_spec({"data_source": "terminal_bench"}) is None
    assert local_env_spec(None) is None
    assert local_env_spec({}) is None


def test_uses_remote_terminal_env_default():
    assert uses_remote_terminal_env({"data_source": "terminal_bench"}) is True
    assert uses_remote_terminal_env({"data_source": "agentharm"}) is False


def test_direct_score_source_is_canonical_exact_match():
    assert direct_score_source("agent_safetybench") is True
    assert direct_score_source("agentharm") is True
    assert direct_score_source("tau2") is True
    # Historical aliases resolve for slugs but NOT for behavior decisions.
    assert direct_score_source("safety") is False
    assert direct_score_source("seta") is False
    assert direct_score_source("swe_verified") is False
    assert direct_score_source("") is False


def test_safety_split_applies_only_to_safety_benchmarks():
    assert safety_split_applies("agent_safetybench") is True
    assert safety_split_applies("agentharm") is True
    assert safety_split_applies("tau2") is False
    assert safety_split_applies("seta") is False


def test_slug_for_preserves_historical_aliases():
    assert slug_for(None) == "seta"
    assert slug_for("") == "seta"
    assert slug_for("terminal_bench") == "seta"
    assert slug_for("seta_env") == "seta"
    assert slug_for("safety") == "agent_safetybench"
    assert slug_for("asb") == "agent_safetybench"
    assert slug_for("agent-safety-bench") == "agent_safetybench"
    assert slug_for("agent_harm") == "agentharm"
    assert slug_for("ah") == "agentharm"
    assert slug_for("tau2") == "tau2"
    assert slug_for("swe_verified") == "swe_verified"
    assert slug_for("weird source!") == "weird_source_"


def test_interval_candidates_match_previous_tables():
    args, envs = interval_candidates_for_slug("seta")
    assert args == ("trajectory_save_interval_seta", "trajectory_save_interval_terminal_bench")
    assert envs == ("TRAJECTORY_SAVE_INTERVAL_SETA", "SAVE_INTERVAL_SETA")

    args, envs = interval_candidates_for_slug("agent_safetybench")
    assert args == (
        "trajectory_save_interval_agent_safetybench",
        "trajectory_save_interval_asb",
        "trajectory_save_interval_safety",
    )
    assert envs == (
        "TRAJECTORY_SAVE_INTERVAL_AGENT_SAFETYBENCH",
        "TRAJECTORY_SAVE_INTERVAL_ASB",
        "TRAJECTORY_SAVE_INTERVAL_SAFETY",
        "SAVE_INTERVAL_AGENT_SAFETYBENCH",
        "SAVE_INTERVAL_ASB",
        "SAVE_INTERVAL_SAFETY",
    )

    args, envs = interval_candidates_for_slug("agentharm")
    assert args == ("trajectory_save_interval_agentharm", "trajectory_save_interval_agent_harm")
    assert envs == (
        "TRAJECTORY_SAVE_INTERVAL_AGENTHARM",
        "TRAJECTORY_SAVE_INTERVAL_AGENT_HARM",
        "SAVE_INTERVAL_AGENTHARM",
        "SAVE_INTERVAL_AGENT_HARM",
    )

    args, envs = interval_candidates_for_slug("tau2")
    assert args == ("trajectory_save_interval_tau2",)
    assert envs == ("TRAJECTORY_SAVE_INTERVAL_TAU2",)

    args, envs = interval_candidates_for_slug("my_new_env")
    assert args == ("trajectory_save_interval_my_new_env",)
    assert envs == ("TRAJECTORY_SAVE_INTERVAL_MY_NEW_ENV",)


def test_spec_for_source_derives_remote_spec_for_unknown():
    spec = spec_for_source("brand_new")
    assert spec.slug == "brand_new"
    assert spec.local_runtime is None
    assert canonical_spec("brand_new") is None
