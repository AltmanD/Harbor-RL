"""trajectory_store in-process index cache: appended save/delete events update
the cached active set incrementally instead of re-reading the whole index."""

from __future__ import annotations

import json
import sys
from pathlib import Path

TERMINAL_RL_DIR = Path(__file__).resolve().parents[2] / "harborrl"
REPO_ROOT = TERMINAL_RL_DIR.parent
for path in (REPO_ROOT / "backends" / "slime", TERMINAL_RL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from harborrl.rollout import trajectory_store as store


def _make_saved_record(save_dir: Path, rel_path: str) -> dict:
    record_dir = save_dir / rel_path
    record_dir.mkdir(parents=True, exist_ok=True)
    (record_dir / "traj.json").write_text("{}", encoding="utf-8")
    return {
        "event": "save",
        "schema_version": 1,
        "rel_path": rel_path,
        "path": str(record_dir),
        "ts_ns": 1,
    }


def test_index_cache_updates_on_append_and_delete(monkeypatch, tmp_path):
    monkeypatch.setenv("TERMINAL_SAVE_TRAJ_DIR", str(tmp_path))
    store._INDEX_CACHE.clear()

    save_dir = tmp_path
    record = _make_saved_record(save_dir, "t1_r0")

    # First load populates the cache from disk.
    assert store._trajectory_load_index_cached(save_dir) == []

    store._trajectory_append_index(save_dir, record)
    active = store._trajectory_load_index_cached(save_dir)
    assert [r["rel_path"] for r in active] == ["t1_r0"]

    # Cache serves reads without touching the disk: deleting index.jsonl must
    # not change the cached view (the file is recreated by append anyway).
    active_again = store._trajectory_load_index_cached(save_dir)
    assert [r["rel_path"] for r in active_again] == ["t1_r0"]

    store._trajectory_append_index(
        save_dir,
        {"event": "delete", "schema_version": 1, "rel_path": "t1_r0", "deleted_ts_ns": 2},
    )
    assert store._trajectory_load_index_cached(save_dir) == []

    # Disk journal still reflects both events for a fresh process view.
    store._INDEX_CACHE.clear()
    assert store._trajectory_load_index(save_dir) == []


def test_index_cache_isolated_per_save_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("TERMINAL_SAVE_TRAJ_DIR", str(tmp_path))
    store._INDEX_CACHE.clear()

    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    record = _make_saved_record(dir_a, "t1_r0")

    assert store._trajectory_load_index_cached(dir_a) == []
    assert store._trajectory_load_index_cached(dir_b) == []
    store._trajectory_append_index(dir_a, record)
    assert len(store._trajectory_load_index_cached(dir_a)) == 1
    assert store._trajectory_load_index_cached(dir_b) == []

    # Index journal on disk is valid JSONL.
    lines = (dir_a / "index.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["rel_path"] == "t1_r0"
