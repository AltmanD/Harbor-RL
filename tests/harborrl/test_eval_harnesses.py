"""Tests for the evaluation harness adapters (no GPU/docker/sglang needed)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harborrl.harnesses.eval import create_eval_harness, normalize_eval_harness_name
from harborrl.harnesses.eval.base import EvalRunSpec, ServingSpec


def _spec(tmp_path: Path, **overrides) -> EvalRunSpec:
    serving = ServingSpec(
        mode="external",
        api_base="http://127.0.0.1:30000/v1",
        model_path="/models/ckpt",
        model_name="my-model",
    )
    kwargs = dict(
        harness="terminus-2",
        job_name="job1",
        dataset_path="/datasets/tasks",
        output_dir=str(tmp_path / "jobs"),
        serving=serving,
    )
    kwargs.update(overrides)
    return EvalRunSpec(**kwargs)


# --- registry ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        ("terminus-2", "terminus2"),
        ("terminus2", "terminus2"),
        ("claude-code", "claude_code_cli"),
        ("camel-agent", "camel_agent"),
        ("camel", "camel_agent"),
    ],
)
def test_normalize_eval_harness_name(alias: str, canonical: str) -> None:
    assert normalize_eval_harness_name(alias) == canonical


def test_normalize_eval_harness_name_unknown() -> None:
    with pytest.raises(ValueError):
        normalize_eval_harness_name("no-such-harness")


def test_create_eval_harness_returns_adapter() -> None:
    assert create_eval_harness("terminus-2").name == "terminus2"
    assert create_eval_harness("claude-code").name == "claude_code_cli"
    assert create_eval_harness("camel-agent").name == "camel_agent"


# --- terminus-2 build_config -------------------------------------------------


def test_terminus2_build_config_structure(tmp_path: Path) -> None:
    spec = _spec(
        tmp_path,
        max_input_tokens=32768,
        max_output_tokens=8192,
        concurrency=16,
        environment={"TZ": "Etc/UTC"},
    )
    config = create_eval_harness("terminus-2").build_config(spec)

    assert config["job_name"] == "job1"
    assert config["jobs_dir"] == str(tmp_path / "jobs")
    assert config["n_concurrent_trials"] == 16
    assert config["retry"] == {"max_retries": 1}

    agent = config["agents"][0]
    assert agent["name"] == "terminus-2"
    assert agent["model_name"] == "openai/my-model"
    assert agent["kwargs"]["api_base"] == "http://127.0.0.1:30000/v1"
    assert agent["kwargs"]["model_info"]["max_input_tokens"] == 32768
    assert agent["kwargs"]["model_info"]["max_output_tokens"] == 8192
    assert agent["env"]["OPENAI_API_KEY"] == "dummy"
    assert agent["env"]["OPENAI_BASE_URL"] == "http://127.0.0.1:30000/v1"
    assert agent["env"]["TZ"] == "Etc/UTC"

    assert config["datasets"] == [{"path": "/datasets/tasks"}]
    assert config["environment"]["type"] == "docker"
    assert config["environment"]["env"] == {"TZ": "Etc/UTC"}


def test_terminus2_build_config_single_task(tmp_path: Path) -> None:
    spec = _spec(tmp_path, task_names=["hello-world"])
    config = create_eval_harness("terminus-2").build_config(spec)
    assert config["datasets"] == [{"path": "/datasets/tasks", "task_names": ["hello-world"]}]


def test_terminus2_launch_command(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    cmd, env = create_eval_harness("terminus-2").launch_command(spec, "/tmp/cfg.json")
    assert cmd == ["harbor", "run", "--config", "/tmp/cfg.json"]
    assert env["OPENAI_API_KEY"] == "dummy"
    assert env["OPENAI_BASE_URL"] == "http://127.0.0.1:30000/v1"
    assert "NO_PROXY" in env


# --- terminus-2 progress / collect -------------------------------------------


def _write_harbor_job(jobs_dir: Path) -> None:
    job = jobs_dir / "job1"
    (job / "task_a__0").mkdir(parents=True)
    (job / "task_b__0").mkdir(parents=True)
    job_result = {
        "finished_at": "2026-01-01T00:10:00",
        "stats": {
            "n_completed_trials": 2,
            "n_running_trials": 0,
            "n_pending_trials": 0,
            "n_errored_trials": 1,
            "evals": {
                "terminus-2__my-model__tasks": {
                    "n_trials": 2,
                    "n_errors": 1,
                    "metrics": [{"mean": 0.5}],
                    "reward_stats": {"reward": {"0.0": ["task_a__0"], "1.0": ["task_b__0"]}},
                    "exception_stats": {"TimeoutError": ["task_a__0"]},
                }
            },
        },
    }
    (job / "result.json").write_text(json.dumps(job_result), encoding="utf-8")
    (job / "task_a__0" / "result.json").write_text(
        json.dumps(
            {
                "verifier_result": {"rewards": {"reward": 0.0}},
                "exception_info": {"exception_type": "TimeoutError"},
                "agent_result": {"n_input_tokens": 100},
            }
        ),
        encoding="utf-8",
    )
    (job / "task_b__0" / "result.json").write_text(
        json.dumps(
            {
                "verifier_result": {"rewards": {"reward": 1.0}},
                "exception_info": None,
                "agent_result": {"n_input_tokens": 200},
            }
        ),
        encoding="utf-8",
    )


def test_terminus2_progress(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    harness = create_eval_harness("terminus-2")
    not_started = harness.progress(spec)
    assert not not_started.finished and not_started.completed == 0

    _write_harbor_job(Path(spec.output_dir))
    progress = harness.progress(spec)
    assert progress.finished
    assert progress.completed == 2
    assert progress.errored == 1


def test_terminus2_collect(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    _write_harbor_job(Path(spec.output_dir))
    result = create_eval_harness("terminus-2").collect(spec)

    assert result.harness == "terminus2"
    assert result.model_name == "my-model"
    assert result.pass_at_1 == 0.5
    assert result.mean_reward == 0.5
    assert result.task_count == 2
    assert result.n_completed == 2
    assert result.n_errored == 1
    assert result.exception_counts == {"TimeoutError": 1}

    assert len(result.task_outcomes) == 2
    by_trial = {o.trial_name: o for o in result.task_outcomes}
    assert by_trial["task_a__0"].task_name == "task_a"
    assert by_trial["task_a__0"].reward == 0.0
    assert by_trial["task_a__0"].exception == "TimeoutError"
    assert by_trial["task_b__0"].reward == 1.0
    assert by_trial["task_b__0"].exception is None


# --- claude-code -------------------------------------------------------------


def test_claude_code_build_agent(tmp_path: Path) -> None:
    spec = _spec(tmp_path, harness="claude-code")
    config = create_eval_harness("claude-code").build_config(spec)
    agent = config["agents"][0]
    assert agent["name"] == "claude-code"
    assert agent["model_name"] == "my-model"  # no openai/ prefix
    assert agent["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:30000/v1"
    assert agent["env"]["ANTHROPIC_AUTH_TOKEN"] == "dummy"

    spec.extra["agent_env"] = {"ANTHROPIC_AUTH_TOKEN": "real-token"}
    config = create_eval_harness("claude-code").build_config(spec)
    assert config["agents"][0]["env"]["ANTHROPIC_AUTH_TOKEN"] == "real-token"


# --- camel-agent collect -----------------------------------------------------


def _write_slime_run(run_dir: Path) -> None:
    step = run_dir / "evaluations" / "seta" / "step_0000"
    step.mkdir(parents=True)
    summary = {
        "dataset": "seta",
        "eval/k": 1,
        "eval/pass_at_k": 0.25,
        "eval/response_pairwise_jaccard_distance": 0.1,
        "eval/response_unique_ratio": 0.9,
        "eval/reward_best_at_k": 0.5,
        "eval/task_count": 4,
        "global_step": 0,
        "pass_at_1": 0.25,
    }
    (step / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    rows = [
        {"group": "prompt:0", "best_reward_at_k": 0.0, "pass_at_k": 0.0},
        {"group": "prompt:1", "best_reward_at_k": 1.0, "pass_at_k": 1.0},
    ]
    (step / "tasks.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )


def test_camel_collect(tmp_path: Path) -> None:
    run_dir = tmp_path / "slime-run"
    _write_slime_run(run_dir)
    spec = _spec(
        tmp_path,
        harness="camel-agent",
        dataset_path="/data/prompts.jsonl",
        extra={"slime_root": "/slime", "run_dir": str(run_dir), "hf_checkpoint": "/models/ckpt"},
    )
    harness = create_eval_harness("camel-agent")

    progress = harness.progress(spec)
    assert progress.finished and progress.completed == 4

    result = harness.collect(spec)
    assert result.harness == "camel_agent"
    assert result.dataset == "seta"
    assert result.pass_at_1 == 0.25
    assert result.reward_best_at_k == 0.5
    assert result.k == 1
    assert result.task_count == 4
    assert result.extras["response_unique_ratio"] == 0.9
    assert result.extras["response_pairwise_jaccard_distance"] == 0.1
    assert len(result.task_outcomes) == 2
    assert result.task_outcomes[1].reward == 1.0


def test_camel_progress_not_started(tmp_path: Path) -> None:
    spec = _spec(
        tmp_path,
        harness="camel-agent",
        extra={"slime_root": "/slime", "run_dir": str(tmp_path / "nope")},
    )
    progress = create_eval_harness("camel-agent").progress(spec)
    assert not progress.finished
