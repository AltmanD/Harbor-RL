from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from slime.utils import checkpoint_utils


def _iteration(root: Path, iteration: int, content: bytes = b"checkpoint") -> Path:
    path = root / f"iter_{iteration:07d}"
    path.mkdir(parents=True)
    (path / "state.bin").write_bytes(content)
    return path


def test_cleanup_uses_commit_marker_and_removes_incomplete_newer_save(tmp_path: Path):
    old = _iteration(tmp_path, 7)
    committed = _iteration(tmp_path, 15)
    incomplete = _iteration(tmp_path, 23)
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("15\n", encoding="utf-8")

    assert checkpoint_utils.cleanup_incomplete_checkpoints(str(tmp_path)) == 1
    assert not incomplete.exists()
    assert checkpoint_utils.cleanup_old_checkpoints(str(tmp_path), max_keep=1) == 1
    assert not old.exists()
    assert committed.exists()


def test_cleanup_refuses_to_delete_without_commit_marker(tmp_path: Path):
    first = _iteration(tmp_path, 7)
    second = _iteration(tmp_path, 15)

    assert checkpoint_utils.cleanup_old_checkpoints(str(tmp_path), max_keep=1) == 0
    assert first.exists()
    assert second.exists()


def test_managed_root_removes_failed_first_save_on_next_preflight(tmp_path: Path):
    (tmp_path / ".lightrl_managed_checkpoint_root").write_text("schema=1\n", encoding="utf-8")
    incomplete = _iteration(tmp_path, 0)

    assert checkpoint_utils.cleanup_incomplete_checkpoints(str(tmp_path), managed_root=True) == 1
    assert not incomplete.exists()


def test_preflight_skips_nonfatally_and_preserves_last_commit(tmp_path: Path, monkeypatch):
    committed = _iteration(tmp_path, 15)
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("15\n", encoding="utf-8")
    monkeypatch.setattr(
        checkpoint_utils.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(total=100, used=99, free=1),
    )

    assert not checkpoint_utils.checkpoint_preflight(
        str(tmp_path),
        min_free_bytes=10,
        expected_bytes=10,
    )
    assert committed.exists()


def test_checkpoint_action_is_nonfatal_by_default():
    def fail():
        raise OSError("disk full")

    assert not checkpoint_utils.run_checkpoint_action("actor", 7, fail)
    with pytest.raises(OSError, match="disk full"):
        checkpoint_utils.run_checkpoint_action("actor", 7, fail, fatal=True)


def test_distributed_false_result_is_propagated():
    assert checkpoint_utils.checkpoint_result_succeeded([True, True])
    assert not checkpoint_utils.checkpoint_result_succeeded([True, False])
