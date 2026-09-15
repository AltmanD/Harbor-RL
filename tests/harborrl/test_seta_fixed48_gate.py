from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "compare_seta_fixed_eval",
    ROOT
    / "tools"
    / "evaluation"
    / "benchmarks"
    / "seta"
    / "compare_seta_fixed_eval.py",
)
gate = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(gate)


def _write_stream(run_dir: Path, name: str, step: int, k: int, passes: list[int]) -> None:
    directory = run_dir / "evaluations" / name / f"step_{step:04d}"
    directory.mkdir(parents=True)
    rows = [
        {
            "prompt_index": index,
            "task_id": str(index),
            "k": k,
            "pass_at_k": value,
        }
        for index, value in enumerate(passes)
    ]
    (directory / "tasks.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )


def test_preregistered_gate_accepts_one_advantage_and_one_noninferior(tmp_path):
    dapo, dive = tmp_path / "dapo", tmp_path / "dive"
    _write_stream(dapo, "seta_fixed48_exploit", 100, 1, [0] * 48)
    _write_stream(dive, "seta_fixed48_exploit", 100, 1, [1] * 24 + [0] * 24)
    shared_explore = [1] * 24 + [0] * 24
    _write_stream(dapo, "seta_fixed48_explore", 100, 8, shared_explore)
    _write_stream(dive, "seta_fixed48_explore", 100, 8, shared_explore)

    result = gate.compare_runs(dapo, dive, 100, repetitions=1_000)
    assert result["verdict"] == "优"
    assert result["action"] == "continue_to_1000"
    assert result["metrics"]["exploitation"]["significantly_better"] is True
    assert result["metrics"]["exploration"]["noninferior"] is True


def test_preregistered_gate_rejects_no_significant_advantage(tmp_path):
    dapo, dive = tmp_path / "dapo", tmp_path / "dive"
    values = [1] * 24 + [0] * 24
    for name, k in (("seta_fixed48_exploit", 1), ("seta_fixed48_explore", 8)):
        _write_stream(dapo, name, 200, k, values)
        _write_stream(dive, name, 200, k, values)

    result = gate.compare_runs(dapo, dive, 200, repetitions=1_000)
    assert result["verdict"] == "不优"
    assert result["action"] == "stop_dive_immediately"
