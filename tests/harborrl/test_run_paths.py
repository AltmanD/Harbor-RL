from pathlib import Path

import pytest

from harborrl.platform.paths import RunPaths, normalize_run_category, resolve_run_dir


def test_canonical_run_categories_and_aliases(tmp_path: Path):
    assert normalize_run_category("train") == "training"
    assert normalize_run_category("eval") == "evaluation"
    assert normalize_run_category("test") == "testing"
    assert normalize_run_category("debug") == "testing/debug"
    assert resolve_run_dir("run-1", tmp_path, category="training") == (
        tmp_path / "training" / "run-1"
    )
    assert resolve_run_dir("run-2", tmp_path, category="testing", debug=True) == (
        tmp_path / "testing" / "debug" / "run-2"
    )


def test_run_paths_default_to_training_and_keep_explicit_dir(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("RUN_CATEGORY", raising=False)
    training = RunPaths("train-1", tmp_path / "runs", tmp_path / "ckpt")
    assert training.run_dir == tmp_path / "runs" / "training" / "train-1"

    explicit = tmp_path / "custom" / "run-2"
    resumed = RunPaths(
        "run-2", tmp_path / "runs", tmp_path / "ckpt", category="evaluation", run_dir=explicit
    )
    assert resumed.run_dir == explicit


def test_invalid_category_is_rejected():
    with pytest.raises(ValueError, match="unknown run category"):
        normalize_run_category("scratch")


def test_run_paths_from_env_does_not_rebase_categorized_dir(monkeypatch, tmp_path: Path):
    run_dir = tmp_path / "runs" / "testing" / "debug" / "smoke-1"
    monkeypatch.setenv("RUN_DIR", str(run_dir))
    rp = RunPaths.from_env()
    assert rp is not None
    assert rp.run_dir == run_dir
    assert rp.logs_dir == run_dir / "logs"
