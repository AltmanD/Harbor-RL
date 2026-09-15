"""Checkpoint retention and non-fatal save helpers.

Megatron commits a checkpoint by atomically updating
``latest_checkpointed_iteration.txt``.  Directory names alone are therefore
not proof that a checkpoint is complete: a failed save can leave a newer
``iter_*`` directory behind.  All pruning in this module is anchored to that
commit marker so the last known-good checkpoint is never deleted to make room
for an incomplete one.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

_ITER_DIR_RE = re.compile(r"^iter_(\d+)$")
_TRACKER_FILE = "latest_checkpointed_iteration.txt"
_SIZE_CACHE_FILE = ".lightrl_checkpoint_size_bytes"
_OWNERSHIP_MARKER = ".lightrl_managed_checkpoint_root"
_GIB = 1024**3
_DEFAULT_FIRST_CHECKPOINT_BYTES = 128 * _GIB


def _get_iter_dirs(save_dir: str) -> list[tuple[int, str]]:
    """Return sorted ``(iteration, full_path)`` pairs under *save_dir*."""
    results: list[tuple[int, str]] = []
    if not os.path.isdir(save_dir):
        return results
    for name in os.listdir(save_dir):
        match = _ITER_DIR_RE.match(name)
        if match and os.path.isdir(os.path.join(save_dir, name)):
            results.append((int(match.group(1)), os.path.join(save_dir, name)))
    results.sort(key=lambda item: item[0])
    return results


def _dir_size(path: str) -> int:
    """Return total size in bytes of a directory tree."""
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for filename in filenames:
            try:
                total += os.path.getsize(os.path.join(dirpath, filename))
            except OSError:
                pass
    return total


def get_committed_iteration(save_dir: str) -> int | None:
    """Read Megatron's last committed iteration, or return ``None``.

    ``release`` checkpoints are not rollout checkpoints and are deliberately
    ignored by retention logic.
    """
    tracker = os.path.join(save_dir, _TRACKER_FILE)
    try:
        with open(tracker, encoding="utf-8") as stream:
            value = stream.read().strip()
    except OSError:
        return None
    if not value.isdigit():
        return None
    return int(value)


def _remove_checkpoint(path: str, iteration: int, reason: str) -> bool:
    logger.warning(
        "Checkpoint cleanup: removing iter_%07d (%s); reason=%s",
        iteration,
        path,
        reason,
    )
    try:
        shutil.rmtree(path)
    except OSError:
        logger.exception(
            "CHECKPOINT_CLEANUP_FAILED_NONFATAL iteration=%s path=%s",
            iteration,
            path,
        )
        return False
    return True


def cleanup_incomplete_checkpoints(save_dir: str, *, managed_root: bool = False) -> int:
    """Delete directories newer than the last committed iteration.

    A missing tracker is ambiguous, so nothing is removed unless *managed_root*
    proves HarborRL previously initialized this per-run directory.
    Returns the number of directories successfully removed.
    """
    committed = get_committed_iteration(save_dir)
    if committed is None:
        if not managed_root:
            return 0
        removed = 0
        for iteration, path in _get_iter_dirs(save_dir):
            if _remove_checkpoint(path, iteration, "managed-root-without-commit-marker"):
                removed += 1
        return removed
    removed = 0
    for iteration, path in _get_iter_dirs(save_dir):
        if iteration > committed and _remove_checkpoint(path, iteration, "newer-than-commit-marker"):
            removed += 1
    return removed


def cleanup_old_checkpoints(save_dir: str, max_keep: int = 1) -> int:
    """Prune old *committed* checkpoints while preserving the tracker target.

    Directories newer than the commit marker are not counted as valid and are
    left to :func:`cleanup_incomplete_checkpoints`.  Without a readable marker
    this function refuses to delete anything.
    """
    if max_keep < 1:
        max_keep = 1
    committed = get_committed_iteration(save_dir)
    if committed is None:
        if _get_iter_dirs(save_dir):
            logger.warning(
                "Checkpoint cleanup skipped: no valid %s in %s",
                _TRACKER_FILE,
                save_dir,
            )
        return 0

    valid = [(iteration, path) for iteration, path in _get_iter_dirs(save_dir) if iteration <= committed]
    if not any(iteration == committed for iteration, _path in valid):
        logger.error(
            "Checkpoint cleanup skipped: commit marker points to missing iter_%07d in %s",
            committed,
            save_dir,
        )
        return 0

    removed = 0
    for iteration, path in valid[: max(0, len(valid) - max_keep)]:
        if iteration == committed:
            continue
        if _remove_checkpoint(path, iteration, "retention-limit"):
            removed += 1
    if removed:
        logger.info(
            "Checkpoint cleanup: removed %d committed checkpoint(s), kept latest %d",
            removed,
            max_keep,
        )
    return removed


def checkpoint_preflight(
    save_dir: str,
    *,
    max_keep: int = 1,
    min_free_bytes: int | None = None,
    expected_bytes: int | None = None,
    margin_ratio: float = 1.15,
) -> bool:
    """Prepare for a save and return whether it is safe to attempt.

    Cleanup is restricted to this run's *committed* checkpoints.  The last
    committed checkpoint is always retained.  If adequate space still cannot
    be made, the caller should skip this save and continue training.
    """
    try:
        os.makedirs(save_dir, exist_ok=True)
        ownership_marker = os.path.join(save_dir, _OWNERSHIP_MARKER)
        managed_root = os.path.isfile(ownership_marker)
        cleanup_incomplete_checkpoints(save_dir, managed_root=managed_root)
        if not managed_root:
            with open(ownership_marker, "w", encoding="utf-8") as stream:
                stream.write("HarborRL managed per-run checkpoint directory; schema=1\n")
        cleanup_old_checkpoints(save_dir, max_keep=max_keep)

        committed = get_committed_iteration(save_dir)
        committed_path = None
        if committed is not None:
            candidate = os.path.join(save_dir, f"iter_{committed:07d}")
            if os.path.isdir(candidate):
                committed_path = candidate

        if expected_bytes is None:
            configured = os.getenv("SLIME_CHECKPOINT_EXPECTED_BYTES", "").strip()
            expected_bytes = int(configured) if configured else 0
        if expected_bytes <= 0 and committed_path:
            cache_path = os.path.join(save_dir, _SIZE_CACHE_FILE)
            try:
                with open(cache_path, encoding="utf-8") as stream:
                    expected_bytes = int(stream.read().strip())
            except (OSError, ValueError):
                expected_bytes = _dir_size(committed_path)
                try:
                    with open(cache_path, "w", encoding="utf-8") as stream:
                        stream.write(f"{expected_bytes}\n")
                except OSError:
                    logger.warning("Could not write checkpoint size cache: %s", cache_path)
        if expected_bytes <= 0:
            expected_bytes = _DEFAULT_FIRST_CHECKPOINT_BYTES

        configured_min = os.getenv("SLIME_CHECKPOINT_MIN_FREE_BYTES", "").strip()
        if min_free_bytes is None:
            min_free_bytes = int(configured_min) if configured_min else _DEFAULT_FIRST_CHECKPOINT_BYTES
        required = max(int(min_free_bytes), int(expected_bytes * max(1.0, margin_ratio)))

        # Retention may be >1.  Reclaim extra committed versions if free space
        # is low, but never remove the tracker target.
        free = shutil.disk_usage(save_dir).free
        if free < required and committed is not None:
            for iteration, path in _get_iter_dirs(save_dir):
                if iteration >= committed:
                    continue
                if _remove_checkpoint(path, iteration, "disk-space-preflight"):
                    free = shutil.disk_usage(save_dir).free
                if free >= required:
                    break

        free = shutil.disk_usage(save_dir).free
        if free < required:
            logger.error(
                "CHECKPOINT_SAVE_SKIPPED_NONFATAL reason=insufficient-space "
                "path=%s free_gib=%.1f required_gib=%.1f last_committed=%s; training continues",
                save_dir,
                free / _GIB,
                required / _GIB,
                committed,
            )
            return False
        logger.info(
            "Checkpoint preflight passed: path=%s free_gib=%.1f required_gib=%.1f last_committed=%s",
            save_dir,
            free / _GIB,
            required / _GIB,
            committed,
        )
        return True
    except Exception:
        logger.exception(
            "CHECKPOINT_SAVE_SKIPPED_NONFATAL reason=preflight-error path=%s; training continues",
            save_dir,
        )
        return False


def checkpoint_result_succeeded(result: Any) -> bool:
    """Return false if a distributed checkpoint call reports any false result."""
    if result is False:
        return False
    if isinstance(result, (list, tuple)):
        return all(checkpoint_result_succeeded(item) for item in result)
    return True


def run_checkpoint_action(
    label: str,
    rollout_id: int,
    action: Callable[[], Any],
    *,
    fatal: bool = False,
) -> bool:
    """Run one checkpoint action, logging failures without stopping training."""
    try:
        result = action()
        if checkpoint_result_succeeded(result):
            logger.info("Checkpoint action completed: label=%s rollout_id=%s", label, rollout_id)
            return True
        logger.error(
            "CHECKPOINT_SAVE_SKIPPED_NONFATAL label=%s rollout_id=%s "
            "reason=worker-declined-save; training continues",
            label,
            rollout_id,
        )
        if fatal:
            raise RuntimeError(f"checkpoint action declined: label={label} rollout_id={rollout_id}")
    except Exception:
        logger.exception(
            "CHECKPOINT_SAVE_FAILED_NONFATAL label=%s rollout_id=%s; training continues",
            label,
            rollout_id,
        )
        if fatal:
            raise
    return False


# Backward-compatible name used by older integrations.  Unlike the old
# implementation it returns a decision and never deletes the last commit.
def check_disk_space_and_cleanup(save_dir: str, min_free_bytes: int | None = None) -> bool:
    return checkpoint_preflight(save_dir, min_free_bytes=min_free_bytes)
