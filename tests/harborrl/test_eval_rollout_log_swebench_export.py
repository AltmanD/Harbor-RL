"""Regression test: eval_rollout_log must import the SWE-bench exporter from its
post-refactor location (tools.evaluation.benchmarks.swebench.report), not the removed
flat ``swebench_report`` module. Before the fix, setting SWEBENCH_RESULTS_DIR
made every eval rollout crash with ModuleNotFoundError."""

from __future__ import annotations

import importlib
import json
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
    def __init__(self, instance_id: str) -> None:
        self.group_index = 0
        self.index = 0
        self.status = "completed"
        self.prompt = ""
        self.metadata = {
            "task_meta": {"swe_instance_id": instance_id},
            "reward_details": {"model_patch": "diff --git a/x b/x"},
        }
        self.reward = {
            "score": 1.0,
            "raw_score": 1.0,
            "base_score": 1.0,
        }


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        num_steps_per_rollout=None,
        rollout_batch_size=64,
        n_samples_per_prompt=8,
        global_batch_size=128,
        use_wandb=False,
        reward_key="score",
        advantage_estimator="grpo",
        rewards_normalization=True,
        grpo_std_normalization=False,
        dynamic_history=False,
    )


def test_eval_rollout_log_exports_swebench_artifacts(monkeypatch, tmp_path):
    instance_id = "django__django-12345"
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        json.dumps({"metadata": {"swe_instance_id": instance_id}}) + "\n",
        encoding="utf-8",
    )
    results_dir = tmp_path / "results"
    monkeypatch.setenv("SWEBENCH_RESULTS_DIR", str(results_dir))
    monkeypatch.setenv("SWEBENCH_EVAL_DATA_PATH", str(dataset_path))
    monkeypatch.setenv("RUN_ID", "pytest_swe_export")
    monkeypatch.setenv("TERMINAL_STRUCTURED_METRICS", "0")

    data = {"swe_verified": {"samples": [DummySample(instance_id)]}}
    result = rollout_log.eval_rollout_log(0, _args(), data)

    assert result is False
    predictions = (results_dir / "predictions.jsonl").read_text(encoding="utf-8")
    assert instance_id in predictions
    summary = json.loads((results_dir / "score_summary.json").read_text(encoding="utf-8"))
    assert summary["submitted"] == 1
    assert summary["run_id"] == "pytest_swe_export"
