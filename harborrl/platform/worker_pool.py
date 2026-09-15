from __future__ import annotations

import asyncio
import logging
import math
import os
import time
import traceback
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from harborrl.types import TaskTimeouts
from harborrl.environments.terminal.runtime import (
    TerminalEnv,
    force_remove_orphan_docker_objects,
)
from harborrl.environments.terminal.docker_compose import docker_image_build_status
from harborrl.platform.worker_admission import (
    CapacityError,
    ResetAdmissionBacklogError,
    ResetInProgressError,
    _build_run_ctx,
    _build_task_spec,
    _docker_name_variants,
    _env_bool,
    _env_float,
    _env_int,
    _parse_task_max_runs_overrides,
    _parse_timeout_overrides,
    _split_env_csv,
    _task_id_from_ref,
    _task_key_tokens,
    worker_pressure_stats,
)

logger = logging.getLogger("lightrl.env.worker.pool")

_LIFECYCLE_METRIC_NAMES = (
    "reset_admission_wait",
    "reset",
    "exec_tool",
    "evaluate",
    "close",
)


def _duration_summary(values: list[float]) -> dict[str, float | int]:
    """Return compact nearest-rank latency statistics for worker status."""
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "max": 0.0,
        }
    ordered = sorted(max(0.0, float(value)) for value in values)

    def percentile(fraction: float) -> float:
        index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
        return ordered[index]

    return {
        "count": len(ordered),
        "mean": round(sum(ordered) / len(ordered), 3),
        "p50": round(percentile(0.50), 3),
        "p95": round(percentile(0.95), 3),
        "max": round(ordered[-1], 3),
    }

@dataclass
class RunSlot:
    run_lease_id: str
    task_key: str
    env: TerminalEnv
    created_ts: float = field(default_factory=time.time)
    last_used_ts: float = field(default_factory=time.time)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    phase: str = "allocated"
    in_flight_ops: int = 0
    active_op: str | None = None
    close_requested: bool = False
    close_reason: str | None = None
    close_requested_ts: float | None = None
    reset_started_ts: float | None = None
    reset_completed_ts: float | None = None
    reset_request_id: str | None = None
    reset_future: asyncio.Task | None = None
    reset_result: dict[str, Any] | None = None
    first_step_ts: float | None = None
    evaluate_completed_ts: float | None = None
    drop_scheduled: bool = False  # Flag to prevent double-pop race
    reset_quarantined: bool = False
    reset_quarantine_reason: str | None = None
    reset_quarantine_started_ts: float | None = None
    reset_quarantine_watcher: asyncio.Task | None = None


@dataclass
class TaskSlot:
    task_key: str
    runs: dict[str, RunSlot] = field(default_factory=dict)
    created_ts: float = field(default_factory=time.time)
    last_used_ts: float = field(default_factory=time.time)


class WorkerPool:
    def __init__(
        self,
        *,
        max_tasks: int,
        max_runs_per_task: int,
        run_idle_ttl: int,
        output_root: str,
        default_timeouts: TaskTimeouts,
        max_total_runs: int | None = None,
        idempotency_ttl: int = 300,
        max_concurrent_closes: int = 8,
    ) -> None:
        self.max_tasks = max_tasks
        self.max_runs_per_task = max_runs_per_task
        self.max_total_runs = max(
            1,
            int(
                max_total_runs
                if max_total_runs is not None
                else _env_int(
                    "WORKER_MAX_TOTAL_RUNS", max_tasks * max_runs_per_task
                )
            ),
        )
        self.run_idle_ttl = run_idle_ttl
        self.output_root = Path(output_root).resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.default_timeouts = default_timeouts
        self.idempotency_ttl = idempotency_ttl
        legacy_close_task_timeout = _env_float(
            "WORKER_CLOSE_TASK_TIMEOUT",
            max(30.0, float(default_timeouts.close_session) + 30.0),
        )
        self.close_queue_timeout = _env_float(
            "WORKER_CLOSE_QUEUE_TIMEOUT", legacy_close_task_timeout
        )
        self.close_session_timeout = _env_float(
            "WORKER_CLOSE_SESSION_TIMEOUT",
            max(30.0, float(default_timeouts.close_session)),
        )
        self.close_task_timeout = self.close_queue_timeout + self.close_session_timeout

        self._tasks: dict[str, TaskSlot] = {}
        self._run_to_task: dict[str, str] = {}
        self._idempotency: dict[tuple[str, str], tuple[str, float]] = {}
        self._lock = asyncio.Lock()
        self._shutdown_started = False

        self._close_sem = asyncio.Semaphore(max_concurrent_closes)
        self.max_concurrent_resets = _env_int("WORKER_MAX_CONCURRENT_RESETS", 16)
        self.reset_admission_timeout = _env_float("WORKER_RESET_ADMISSION_TIMEOUT", 30.0)
        self._reset_admission_sem = asyncio.BoundedSemaphore(
            max(1, self.max_concurrent_resets)
        )
        self._reset_admission_waiting = 0
        self._reset_admission_rejected = 0
        self._closing_tasks: set[asyncio.Task] = set()
        self._closing_task_started: dict[asyncio.Task, float] = {}
        self._closing_task_labels: dict[asyncio.Task, str] = {}
        # A lease stops being "active" before its asynchronous Docker cleanup
        # finishes.  Keep a reference-counted reservation until every cleanup
        # task for that lease is done so replacement allocations cannot exceed
        # the daemon's finite network address pools during close/reset churn.
        self._retiring_run_refs: dict[str, int] = {}
        self._force_cleanup_tasks: set[asyncio.Task] = set()
        self._force_cleanup_task_started: dict[asyncio.Task, float] = {}
        self._force_cleanup_task_labels: dict[asyncio.Task, str] = {}
        self._close_requested_release_tasks: dict[str, asyncio.Task] = {}
        self._reset_quarantine_watchers: set[asyncio.Task] = set()
        self._recent_close_failures: dict[str, dict[str, Any]] = {}
        self._close_failure_ttl = max(
            60.0, _env_float("WORKER_CLOSE_FAILURE_TTL", 3600.0)
        )
        self._close_failure_max = max(
            1, _env_int("WORKER_CLOSE_FAILURE_MAX", 256)
        )

        # Track reset count for automatic shim cleanup
        self._reset_count: int = 0
        self._last_shim_cleanup_ts: float = time.time()
        self._last_orphan_sweep_ts: float = 0.0
        self._orphan_sweep_fail_streak: int = 0
        self._orphan_sweep_backoff_until: float = 0.0
        self._serial_task_ids = set(
            _split_env_csv(os.getenv("WORKER_SERIAL_TASK_IDS", "892,1133"))
        )
        self._task_max_runs_overrides = _parse_task_max_runs_overrides(
            os.getenv("WORKER_TASK_MAX_RUNS_OVERRIDES", "")
        )
        self._auto_serialize_unsafe_compose = _env_bool(
            "WORKER_AUTO_SERIALIZE_UNSAFE_COMPOSE", False
        )
        self._unsafe_compose_cache: dict[str, bool] = {}
        self.lifecycle_history_size = max(
            16, _env_int("WORKER_LIFECYCLE_HISTORY_SIZE", 512)
        )
        self._lifecycle_durations: dict[str, deque[float]] = {
            name: deque(maxlen=self.lifecycle_history_size)
            for name in _LIFECYCLE_METRIC_NAMES
        }
        self._lifecycle_outcomes: dict[str, deque[bool]] = {
            name: deque(maxlen=self.lifecycle_history_size)
            for name in _LIFECYCLE_METRIC_NAMES
        }

    def _record_lifecycle_duration(
        self, name: str, started_monotonic: float, *, success: bool
    ) -> None:
        if name not in self._lifecycle_durations:
            return
        duration = max(0.0, time.monotonic() - started_monotonic)
        self._lifecycle_durations[name].append(duration)
        self._lifecycle_outcomes[name].append(success)

    def _lifecycle_latency_snapshot(self) -> dict[str, Any]:
        snapshot: dict[str, Any] = {"history_size": self.lifecycle_history_size}
        for name in _LIFECYCLE_METRIC_NAMES:
            summary = _duration_summary(list(self._lifecycle_durations[name]))
            outcomes = self._lifecycle_outcomes[name]
            success_count = sum(outcomes)
            summary.update(
                {
                    "success": success_count,
                    "failure": len(outcomes) - success_count,
                }
            )
            snapshot[name] = summary
        return snapshot

    def _new_env(self) -> TerminalEnv:
        return TerminalEnv()

    @staticmethod
    def _run_slot_container_info(run_slot: RunSlot) -> dict[str, Any]:
        env = run_slot.env
        terminal = getattr(env, "_terminal", None)
        container = getattr(terminal, "container", None) if terminal is not None else None
        container_id = getattr(container, "id", None)
        short_id = container_id[:12] if isinstance(container_id, str) else None
        container_name = (
            getattr(container, "name", None)
            or getattr(env, "_last_client_container_name", None)
        )
        container_status = getattr(container, "status", None)
        trial_name = getattr(env, "_last_trial_name", None)
        return {
            "id": container_id,
            "short_id": short_id,
            "name": container_name,
            "status": container_status,
            "trial_name": trial_name,
        }

    @classmethod
    def _run_slot_container_ref(cls, run_slot: RunSlot) -> str:
        info = cls._run_slot_container_info(run_slot)
        return (
            f"container_name={info.get('name') or '?'} "
            f"container_id={info.get('short_id') or '?'} "
            f"container_status={info.get('status') or '?'} "
            f"trial={info.get('trial_name') or '?'}"
        )

    def _active_container_names_locked(self) -> set[str]:
        names: set[str] = set()
        for task_slot in self._tasks.values():
            for run_slot in task_slot.runs.values():
                info = self._run_slot_container_info(run_slot)
                name = info.get("name")
                if isinstance(name, str) and name:
                    names.add(name)
        return names

    def _task_uses_unsafe_compose(self, task_key: str) -> bool:
        cached = self._unsafe_compose_cache.get(task_key)
        if cached is not None:
            return cached
        unsafe = False
        dataset_dir = os.getenv("DATASET_DIR", "").strip()
        if dataset_dir and ":" in task_key:
            _task_name, task_path = task_key.split(":", 1)
            compose_path = Path(dataset_dir) / task_path / "docker-compose.yaml"
            try:
                text = compose_path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                text = ""
            if text:
                fixed_non_client_name = False
                current_service = ""
                for raw_line in text.splitlines():
                    stripped = raw_line.strip()
                    if not stripped or stripped.startswith("#"):
                        continue
                    indent = len(raw_line) - len(raw_line.lstrip(" "))
                    if indent == 2 and stripped.endswith(":"):
                        current_service = stripped[:-1]
                    if stripped.startswith("container_name:") and current_service != "client":
                        fixed_non_client_name = True
                unsafe = fixed_non_client_name or "ipam:" in text or "subnet:" in text
        self._unsafe_compose_cache[task_key] = unsafe
        if unsafe:
            logger.warning(
                "Task %s detected as non-parallel-safe compose; "
                "effective max_runs_per_task=1",
                task_key,
            )
        return unsafe

    def _effective_max_runs_per_task(self, task_key: str) -> int:
        tokens = _task_key_tokens(task_key)
        for token in tokens:
            override = self._task_max_runs_overrides.get(token)
            if override is not None:
                return max(1, override)
        if tokens.intersection(self._serial_task_ids):
            return 1
        if self._auto_serialize_unsafe_compose and self._task_uses_unsafe_compose(
            task_key
        ):
            return 1
        return max(1, self.max_runs_per_task)

    def _active_docker_refs_locked(self) -> tuple[set[str], set[str], set[str]]:
        container_names: set[str] = set()
        project_names: set[str] = set()
        task_ids: set[str] = set()
        for task_key, task_slot in self._tasks.items():
            task_id = _task_id_from_ref(task_key)
            if task_id:
                task_ids.add(task_id)
            for run_slot in task_slot.runs.values():
                info = self._run_slot_container_info(run_slot)
                for key in ("name", "trial_name"):
                    value = info.get(key)
                    if not isinstance(value, str) or not value:
                        continue
                    if key == "name":
                        container_names.add(value)
                    project_names.update(_docker_name_variants(value))
                    task_id = _task_id_from_ref(value)
                    if task_id:
                        task_ids.add(task_id)
        return container_names, project_names, task_ids

    def _pop_run_slot_locked(
        self, run_lease_id: str
    ) -> tuple[str, RunSlot] | None:
        task_key = self._run_to_task.pop(run_lease_id, None)
        if task_key is None:
            return None
        task_slot = self._tasks.get(task_key)
        run_slot = task_slot.runs.pop(run_lease_id, None) if task_slot else None
        if task_slot is not None and not task_slot.runs:
            self._tasks.pop(task_key, None)
            logger.info("Removed empty task slot: %s", task_key)
        if run_slot is None:
            return None
        return task_key, run_slot

    def _phase_for_op(self, op_name: str) -> str:
        return {
            "reset": "resetting",
            "exec_tool": "stepping",
            "evaluate": "evaluating",
            "heartbeat": "heartbeat",
        }.get(op_name, op_name)

    async def _begin_run_op(
        self, run_lease_id: str, op_name: str
    ) -> tuple[RunSlot, float]:
        started_monotonic = time.monotonic()
        async with self._lock:
            if self._shutdown_started:
                raise RuntimeError(f"Worker is shutting down; rejecting {op_name}")
            run_slot = self._get_run_slot(run_lease_id)
            if run_slot.reset_quarantined:
                raise RuntimeError(
                    f"Run {run_lease_id} has a quarantined reset; rejecting {op_name}"
                )
            if run_slot.close_requested:
                raise RuntimeError(
                    f"Run {run_lease_id} is closing; rejecting new {op_name} request"
                )
            now = time.time()
            run_slot.in_flight_ops += 1
            run_slot.active_op = op_name
            run_slot.phase = self._phase_for_op(op_name)
            run_slot.last_used_ts = now
            if op_name == "reset":
                run_slot.reset_started_ts = now
            logger.debug(
                "Run op begin: lease=%s task=%s op=%s phase=%s in_flight=%d %s",
                run_lease_id,
                run_slot.task_key,
                op_name,
                run_slot.phase,
                run_slot.in_flight_ops,
                self._run_slot_container_ref(run_slot),
            )
            return run_slot, started_monotonic

    async def _finish_run_op(
        self,
        run_slot: RunSlot,
        op_name: str,
        *,
        success: bool,
        started_monotonic: float,
        is_timeout_drop: bool = False,
    ) -> None:
        close_after: tuple[str, str, RunSlot, str] | None = None
        async with self._lock:
            now = time.time()
            run_slot.in_flight_ops = max(0, run_slot.in_flight_ops - 1)
            run_slot.last_used_ts = now
            if run_slot.reset_quarantined:
                run_slot.phase = "reset_quarantined"
            elif success:
                if op_name == "reset":
                    run_slot.reset_completed_ts = now
                    run_slot.phase = "ready"
                    # Track successful resets for shim cleanup trigger
                    self._reset_count += 1
                elif op_name == "exec_tool":
                    if run_slot.first_step_ts is None:
                        run_slot.first_step_ts = now
                    run_slot.phase = "stepped"
                elif op_name == "evaluate":
                    run_slot.evaluate_completed_ts = now
                    run_slot.phase = "evaluated"
                elif run_slot.in_flight_ops == 0:
                    run_slot.phase = "ready"
            else:
                run_slot.phase = "failed"
            if run_slot.in_flight_ops == 0:
                run_slot.active_op = None

            # Check drop_scheduled flag to prevent double-pop race
            if (
                run_slot.close_requested
                and run_slot.in_flight_ops == 0
                and not run_slot.drop_scheduled
                and not run_slot.reset_quarantined
                and not (
                    run_slot.reset_future is not None
                    and not run_slot.reset_future.done()
                )
            ):
                popped = self._pop_run_slot_locked(run_slot.run_lease_id)
                if popped is not None:
                    task_key, popped_slot = popped
                    close_reason = (
                        "Closing run slot after in-flight "
                        f"{op_name}: {popped_slot.close_reason or 'close_requested'}"
                    )
                    close_after = (
                        task_key,
                        popped_slot.run_lease_id,
                        popped_slot,
                        close_reason,
                    )

        if op_name != "heartbeat":
            self._record_lifecycle_duration(
                op_name, started_monotonic, success=success
            )

        if close_after is not None:
            task_key, run_lease_id, slot_to_close, close_reason = close_after
            self._schedule_close(
                task_key,
                run_lease_id,
                slot_to_close,
                reason=close_reason,
            )

        # Mark a timed-out reset for removal after its outer reset future is done.
        # The public reset path performs the actual pop/cleanup so no Docker
        # cleanup can overlap the tail of _run_reset_once().
        if is_timeout_drop and not run_slot.reset_quarantined:
            logger.info(
                "Timeout drop deferred until after _finish_run_op: lease=%s op=%s",
                run_slot.run_lease_id,
                op_name,
            )
            await self._drop_resetting_run_for_timeout(
                run_slot.run_lease_id, run_slot, timeout=0.0  # timeout already logged earlier
            )

    async def _close_run_slot_under_lock(self, run_slot: RunSlot) -> None:
        async with run_slot.lock:
            run_slot.phase = "closing"
            try:
                await asyncio.wait_for(
                    run_slot.env.close(), timeout=self.close_session_timeout
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "env.close() timed out after %.1fs for lease=%s; proceeding to force_cleanup",
                    self.close_session_timeout,
                    run_slot.run_lease_id,
                )
                raise
            run_slot.phase = "closed"

    def _prune_done_closing_tasks(self) -> int:
        done = {task for task in self._closing_tasks if task.done()}
        self._closing_tasks.difference_update(done)
        for task in done:
            self._closing_task_started.pop(task, None)
            self._closing_task_labels.pop(task, None)
        return len(done)

    def _prune_done_force_cleanup_tasks(self) -> int:
        done = {task for task in self._force_cleanup_tasks if task.done()}
        self._force_cleanup_tasks.difference_update(done)
        for task in done:
            self._force_cleanup_task_started.pop(task, None)
            self._force_cleanup_task_labels.pop(task, None)
        return len(done)

    @staticmethod
    async def _join_task_uncancellable(task: asyncio.Task[Any]) -> None:
        """Join *task* without turning a second cancellation into detachment."""
        while not task.done():
            try:
                # ``shield(task)`` can immediately re-raise CancelledError while
                # a cancellation-resistant child is still unwinding on Python
                # 3.10.  Waiting on the task set observes completion without
                # propagating the child's cancellation state or busy-spinning.
                await asyncio.wait({task}, timeout=0.1)
            except asyncio.CancelledError:
                current = asyncio.current_task()
                if current is not None and hasattr(current, "uncancel"):
                    current.uncancel()
        await asyncio.gather(task, return_exceptions=True)

    def _track_force_cleanup_task(
        self,
        task: asyncio.Task[Any],
        *,
        label: str,
        retiring_run_ids: tuple[str, ...] = (),
    ) -> None:
        for run_lease_id in retiring_run_ids:
            self._retiring_run_refs[run_lease_id] = (
                self._retiring_run_refs.get(run_lease_id, 0) + 1
            )
        self._force_cleanup_tasks.add(task)
        self._force_cleanup_task_started[task] = time.time()
        self._force_cleanup_task_labels[task] = label

        def _on_done(done_task: asyncio.Task[Any]) -> None:
            self._force_cleanup_tasks.discard(done_task)
            self._force_cleanup_task_started.pop(done_task, None)
            self._force_cleanup_task_labels.pop(done_task, None)
            for run_lease_id in retiring_run_ids:
                remaining = self._retiring_run_refs.get(run_lease_id, 0) - 1
                if remaining > 0:
                    self._retiring_run_refs[run_lease_id] = remaining
                else:
                    self._retiring_run_refs.pop(run_lease_id, None)

        task.add_done_callback(_on_done)

    def _record_close_failure(
        self, run_slot: RunSlot, run_lease_id: str, *, reason: str, error: str
    ) -> None:
        self._recent_close_failures[run_lease_id] = {
            "lease_id": run_lease_id,
            "task_key": run_slot.task_key,
            "reason": reason,
            "error": error[:1000],
            "timestamp": time.time(),
        }
        while len(self._recent_close_failures) > self._close_failure_max:
            oldest = next(iter(self._recent_close_failures))
            self._recent_close_failures.pop(oldest, None)

    def _clear_close_failure(self, run_lease_id: str) -> None:
        self._recent_close_failures.pop(run_lease_id, None)

    def _prune_recent_close_failures(self, now: float) -> None:
        expired = [
            lease_id
            for lease_id, failure in self._recent_close_failures.items()
            if now - float(failure.get("timestamp", 0.0)) > self._close_failure_ttl
        ]
        for lease_id in expired:
            self._recent_close_failures.pop(lease_id, None)

    async def _close_run_slot_with_semaphore(self, run_slot: RunSlot) -> None:
        started_monotonic = time.monotonic()
        success = False
        try:
            try:
                await asyncio.wait_for(
                    self._close_sem.acquire(), timeout=self.close_queue_timeout
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Timed out waiting %.1fs for close semaphore lease=%s; "
                    "proceeding to force_cleanup",
                    self.close_queue_timeout,
                    run_slot.run_lease_id,
                )
                raise
            try:
                await self._close_run_slot_under_lock(run_slot)
                success = True
            finally:
                self._close_sem.release()
        finally:
            self._record_lifecycle_duration(
                "close", started_monotonic, success=success
            )

    async def _force_cleanup_after_close_failure(
        self, run_slot: RunSlot, run_lease_id: str, *, reason: str
    ) -> bool:
        # STABILITY FIX: Increase timeout from 30s to 90s to handle Docker operations under load
        # Analysis shows 93 force cleanup timeouts; Docker container removal can take 60-90s under pressure
        timeout = _env_float("WORKER_FORCE_CLEANUP_TIMEOUT", 90.0)
        try:
            logger.warning(
                "Force cleanup starting for run session %s after %s (timeout=%.1fs)",
                run_lease_id,
                reason,
                timeout,
            )
            # Apply timeout here at the caller level; env.force_cleanup should not use nested timeout
            await asyncio.wait_for(run_slot.env.force_cleanup(reason=reason), timeout=timeout)
            logger.warning(
                "Force cleanup finished for run session %s after %s",
                run_lease_id,
                reason,
            )
            self._clear_close_failure(run_lease_id)
            return True
        except asyncio.TimeoutError:
            logger.warning(
                "Force cleanup timed out for run session %s after %s (timeout=%.1fs)",
                run_lease_id,
                reason,
                timeout,
            )
            self._record_close_failure(
                run_slot,
                run_lease_id,
                reason=reason,
                error=f"force cleanup timed out after {timeout:.1f}s",
            )
            return False
        except Exception as exc:
            logger.exception(
                "Force cleanup failed after %s for run session %s",
                reason,
                run_lease_id,
            )
            self._record_close_failure(
                run_slot,
                run_lease_id,
                reason=reason,
                error=f"{type(exc).__name__}: {exc}",
            )
            return False

    async def _close_run_slot(
        self, task_key: str, run_lease_id: str, run_slot: RunSlot, *, reason: str
    ) -> None:
        logger.warning("%s %s (task=%s)", reason, run_lease_id, task_key)
        try:
            await self._close_run_slot_with_semaphore(run_slot)
            self._clear_close_failure(run_lease_id)
        except asyncio.TimeoutError:
            logger.warning(
                "Timed out closing run session %s "
                "(queue_timeout=%.1fs session_timeout=%.1fs); dropping it "
                "from the pool so the close backlog can drain. Watchdog/preflight "
                "cleanup will remove any orphan Docker objects.",
                run_lease_id,
                self.close_queue_timeout,
                self.close_session_timeout,
            )
            await self._force_cleanup_after_close_failure(
                run_slot, run_lease_id, reason="close_timeout"
            )
        except asyncio.CancelledError:
            logger.warning(
                "Close task for run session %s was cancelled; forcing Docker "
                "cleanup before dropping it from the pool.",
                run_lease_id,
            )
            cleanup_task = asyncio.create_task(
                self._force_cleanup_after_close_failure(
                    run_slot, run_lease_id, reason="close_cancelled"
                )
            )
            self._track_force_cleanup_task(
                cleanup_task,
                label=f"close_cancelled lease={run_lease_id} task={task_key}",
            )
            await self._join_task_uncancellable(cleanup_task)
            raise
        except Exception:
            logger.exception("Failed to close run session %s", run_lease_id)
            await self._force_cleanup_after_close_failure(
                run_slot, run_lease_id, reason="close_exception"
            )

    def _schedule_close(
        self, task_key: str, run_lease_id: str, run_slot: RunSlot, *, reason: str
    ) -> None:
        self._retiring_run_refs[run_lease_id] = (
            self._retiring_run_refs.get(run_lease_id, 0) + 1
        )
        task = asyncio.create_task(
            self._close_run_slot(task_key, run_lease_id, run_slot, reason=reason)
        )
        self._closing_tasks.add(task)
        self._closing_task_started[task] = time.time()
        self._closing_task_labels[task] = f"{reason} {run_lease_id} task={task_key}"

        def _on_done(done_task: asyncio.Task) -> None:
            self._closing_tasks.discard(done_task)
            self._closing_task_started.pop(done_task, None)
            self._closing_task_labels.pop(done_task, None)
            remaining = self._retiring_run_refs.get(run_lease_id, 0) - 1
            if remaining > 0:
                self._retiring_run_refs[run_lease_id] = remaining
            else:
                self._retiring_run_refs.pop(run_lease_id, None)

        task.add_done_callback(_on_done)

    def _schedule_force_cleanup_slots(
        self, slots: list[tuple[str, str, RunSlot]], *, reason: str
    ) -> None:
        if not slots:
            return
        task = asyncio.create_task(self._force_cleanup_slots(slots, reason=reason))
        self._track_force_cleanup_task(
            task,
            label=f"{reason} leases={','.join(rid for _tk, rid, _slot in slots[:8])}",
            retiring_run_ids=tuple(rid for _tk, rid, _slot in slots),
        )

    def _schedule_close_requested_force_release(
        self, run_lease_id: str, *, reason: str
    ) -> None:
        if os.getenv("WORKER_CLOSE_REQUESTED_FORCE_RELEASE", "1") != "1":
            return
        existing = self._close_requested_release_tasks.get(run_lease_id)
        if existing is not None and not existing.done():
            return
        delay = max(
            0.0, _env_float("WORKER_CLOSE_REQUESTED_FORCE_RELEASE_AFTER", 30.0)
        )
        task = asyncio.create_task(
            self._force_release_close_requested_after_delay(
                run_lease_id,
                reason=reason,
                delay=delay,
            )
        )
        self._close_requested_release_tasks[run_lease_id] = task

        def _on_done(done_task: asyncio.Task) -> None:
            current = self._close_requested_release_tasks.get(run_lease_id)
            if current is done_task:
                self._close_requested_release_tasks.pop(run_lease_id, None)

        task.add_done_callback(_on_done)

    async def _force_release_close_requested_after_delay(
        self, run_lease_id: str, *, reason: str, delay: float
    ) -> None:
        if delay > 0:
            await asyncio.sleep(delay)
        reset_future: asyncio.Task | None = None
        async with self._lock:
            task_key = self._run_to_task.get(run_lease_id)
            task_slot = self._tasks.get(task_key) if task_key is not None else None
            run_slot = task_slot.runs.get(run_lease_id) if task_slot else None
            if run_slot is not None and run_slot.reset_quarantined:
                return
            if (
                run_slot is not None
                and run_slot.close_requested
                and run_slot.reset_future is not None
                and not run_slot.reset_future.done()
            ):
                reset_future = run_slot.reset_future

        # A reset may still create Docker objects after cancellation begins.
        # Join it before removing the lease or starting cleanup.
        if reset_future is not None:
            reset_future.cancel()
            joined = await self._cancel_and_join_reset_task(reset_future)
            if not joined and run_slot is not None:
                await self._quarantine_reset_run(
                    run_slot,
                    reset_future,
                    reason=f"close_requested_reset_join_timeout:{reason}",
                )

        async with self._lock:
            task_key = self._run_to_task.get(run_lease_id)
            if task_key is None:
                return
            task_slot = self._tasks.get(task_key)
            run_slot = task_slot.runs.get(run_lease_id) if task_slot else None
            if run_slot is None or not run_slot.close_requested:
                return
            if reset_future is not None and run_slot.reset_future is not reset_future:
                return
            if run_slot.reset_quarantined:
                return
            if run_slot.reset_future is not None and not run_slot.reset_future.done():
                return
            if run_slot.in_flight_ops <= 0 and not run_slot.lock.locked():
                popped = self._pop_run_slot_locked(run_lease_id)
                if popped is not None:
                    task_key, run_slot = popped
                    logger.warning(
                        "Force-releasing close_requested idle run lease=%s task=%s "
                        "reason=%s phase=%s",
                        run_lease_id,
                        task_key,
                        reason,
                        run_slot.phase,
                    )
                    self._schedule_close(
                        task_key,
                        run_lease_id,
                        run_slot,
                        reason=f"Force-releasing idle close_requested run: {reason}",
                    )
                return
            logger.warning(
                "Deferring close_requested run lease=%s task=%s until its active "
                "operation finishes: reason=%s phase=%s in_flight=%d active_op=%s %s",
                run_lease_id,
                task_key,
                reason,
                run_slot.phase,
                run_slot.in_flight_ops,
                run_slot.active_op,
                self._run_slot_container_ref(run_slot),
            )
            return

    def _reap_idle_locked(self) -> list[tuple[str, str, RunSlot]]:
        now = time.time()
        expired_slots: list[tuple[str, str, RunSlot]] = []
        allocated_ttl = _env_float("WORKER_ALLOCATED_TTL", 120.0)

        expired_idem = [
            k
            for k, (_, ts) in self._idempotency.items()
            if now - ts > self.idempotency_ttl
        ]
        for k in expired_idem:
            self._idempotency.pop(k, None)

        for task_key, task_slot in list(self._tasks.items()):
            expired_runs: list[str] = []
            for rid, rslot in task_slot.runs.items():
                if rslot.reset_future is not None and not rslot.reset_future.done():
                    continue
                if rslot.in_flight_ops > 0 or rslot.lock.locked():
                    continue
                if rslot.close_requested:
                    continue
                if (
                    rslot.phase == "allocated"
                    and allocated_ttl > 0
                    and now - rslot.created_ts > allocated_ttl
                ):
                    expired_runs.append(rid)
                    continue
                if now - rslot.last_used_ts > self.run_idle_ttl:
                    expired_runs.append(rid)

            for rid in expired_runs:
                rslot = task_slot.runs.pop(rid, None)
                self._run_to_task.pop(rid, None)
                if rslot is not None:
                    expired_slots.append((task_key, rid, rslot))

            if task_slot.runs:
                task_slot.last_used_ts = max(
                    r.last_used_ts for r in task_slot.runs.values()
                )
            else:
                logger.info("Reaping empty task slot: %s", task_key)
                self._tasks.pop(task_key, None)

        return expired_slots

    @staticmethod
    def _stale_reason_for_run_slot(run_slot: RunSlot, now: float) -> tuple[str, float]:
        allocated_ttl = _env_float("WORKER_ALLOCATED_TTL", 120.0)
        # Keep this above WORKER_RESET_OPERATION_TIMEOUT so legitimate reset
        # operations are not reaped before their timeout handler runs.
        resetting_ttl = _env_float("WORKER_RESETTING_TTL", 2100.0)
        closing_ttl = _env_float("WORKER_CLOSING_REQUESTED_TTL", 300.0)
        created_age_sec = now - run_slot.created_ts
        reset_age_sec = (
            now - run_slot.reset_started_ts
            if run_slot.reset_started_ts is not None
            else 0.0
        )
        close_age_sec = (
            now - run_slot.close_requested_ts
            if run_slot.close_requested_ts is not None
            else 0.0
        )
        if run_slot.reset_quarantined:
            quarantine_age_sec = (
                now - run_slot.reset_quarantine_started_ts
                if run_slot.reset_quarantine_started_ts is not None
                else 0.0
            )
            return "reset_quarantined", quarantine_age_sec
        if (
            run_slot.phase == "allocated"
            and allocated_ttl > 0
            and created_age_sec >= allocated_ttl
        ):
            return "allocated_ttl_exceeded", created_age_sec
        if (
            run_slot.phase == "resetting"
            and resetting_ttl > 0
            and reset_age_sec >= resetting_ttl
        ):
            return "resetting_ttl_exceeded", reset_age_sec
        if (
            run_slot.close_requested
            and closing_ttl > 0
            and close_age_sec >= closing_ttl
        ):
            return "closing_requested_ttl_exceeded", close_age_sec
        return "", 0.0

    def _get_run_slot(self, run_lease_id: str) -> RunSlot:
        task_key = self._run_to_task.get(run_lease_id)
        if task_key is None:
            raise KeyError(f"Unknown run_lease_id: {run_lease_id}")
        task_slot = self._tasks.get(task_key)
        if task_slot is None:
            raise KeyError(f"Run {run_lease_id} points to missing task slot")
        run_slot = task_slot.runs.get(run_lease_id)
        if run_slot is None:
            raise KeyError(f"Run {run_lease_id} not found in task slot")
        return run_slot

    async def allocate(
        self, task_key: str, request_id: str | None = None
    ) -> dict[str, Any]:
        async with self._lock:
            if self._shutdown_started:
                raise CapacityError(
                    "WORKER_SHUTTING_DOWN", "Worker is shutting down"
                )
            expired_slots = self._reap_idle_locked()

            if request_id:
                idem_key = (task_key, request_id)
                cached = self._idempotency.get(idem_key)
                if cached is not None:
                    run_lease_id, _ = cached
                    if run_lease_id in self._run_to_task:
                        cached_slot = self._get_run_slot(run_lease_id)
                        if cached_slot.reset_quarantined:
                            raise CapacityError(
                                "TASK_RESET_QUARANTINED",
                                f"Task {task_key} has a quarantined reset",
                            )
                        logger.info(
                            "allocate_ok lease_id=%s task_key=%s request_id=%s reused=%s",
                            run_lease_id,
                            task_key,
                            request_id,
                            True,
                        )
                        return {"lease_id": run_lease_id, "reused": True}

            # max_tasks limits distinct task keys and max_runs_per_task limits
            # fan-out for one key, but neither bounds the number of Docker
            # networks alive across the pool.  Keep a global lease ceiling so
            # compose cannot exhaust the daemon's finite default address pools.
            reserved_run_ids = set(self._run_to_task).union(self._retiring_run_refs)
            if len(reserved_run_ids) >= self.max_total_runs:
                raise CapacityError(
                    "TOTAL_RUN_SLOTS_EXHAUSTED",
                    "Worker at total run capacity: "
                    f"{len(reserved_run_ids)}/{self.max_total_runs} "
                    f"(active={len(self._run_to_task)}, "
                    f"retiring={len(self._retiring_run_refs)})",
                )

            task_slot = self._tasks.get(task_key)
            if task_slot is not None and any(
                run.reset_quarantined for run in task_slot.runs.values()
            ):
                raise CapacityError(
                    "TASK_RESET_QUARANTINED",
                    f"Task {task_key} has a quarantined reset",
                )
            if task_slot is None:
                if len(self._tasks) >= self.max_tasks:
                    raise CapacityError(
                        "TASK_SLOTS_EXHAUSTED",
                        f"Worker at task capacity: {len(self._tasks)}/{self.max_tasks}",
                    )
                task_slot = TaskSlot(task_key=task_key)
                self._tasks[task_key] = task_slot

            effective_max_runs = self._effective_max_runs_per_task(task_key)
            if len(task_slot.runs) >= effective_max_runs:
                raise CapacityError(
                    "RUN_SLOTS_EXHAUSTED",
                    f"Task {task_key} at run capacity: {len(task_slot.runs)}/{effective_max_runs}",
                )

            env = self._new_env()
            run_lease_id = f"run-{uuid.uuid4().hex[:16]}"
            run_slot = RunSlot(run_lease_id=run_lease_id, task_key=task_key, env=env)
            task_slot.runs[run_lease_id] = run_slot
            task_slot.last_used_ts = time.time()
            self._run_to_task[run_lease_id] = task_key

            if request_id:
                self._idempotency[(task_key, request_id)] = (run_lease_id, time.time())

        for tk, rid, rslot in expired_slots:
            self._schedule_close(tk, rid, rslot, reason="Reaping idle run slot")

        logger.info(
            "allocate_ok lease_id=%s task_key=%s request_id=%s reused=%s",
            run_lease_id,
            task_key,
            request_id or "",
            False,
        )
        return {"lease_id": run_lease_id, "reused": False}

    async def heartbeat(self, run_lease_id: str) -> None:
        run_slot, started_monotonic = await self._begin_run_op(
            run_lease_id, "heartbeat"
        )
        success = False
        try:
            async with run_slot.lock:
                success = True
        finally:
            await self._finish_run_op(
                run_slot,
                "heartbeat",
                success=success,
                started_monotonic=started_monotonic,
            )

    @staticmethod
    def _reset_operation_timeout(timeouts: TaskTimeouts) -> float:
        configured = _env_float("WORKER_RESET_OPERATION_TIMEOUT", 0.0)
        if configured > 0:
            return configured
        return max(
            30.0,
            float(timeouts.ensure_image) + float(timeouts.reset_session) + 120.0,
        )

    @staticmethod
    async def _cancel_and_join_reset_task(
        task: asyncio.Task[Any],
        *,
        deadline: float | None = None,
        label: str = "reset",
    ) -> bool:
        """Best-effort join of a cancelled reset within an absolute deadline."""
        def _consume_late_result(done_task: asyncio.Task[Any]) -> None:
            try:
                done_task.exception()
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception("Quarantined %s task failed after its join deadline", label)

        loop = asyncio.get_running_loop()
        if deadline is None:
            timeout = max(
                0.1, _env_float("WORKER_RESET_CANCEL_JOIN_TIMEOUT", 15.0)
            )
            deadline = loop.time() + timeout
        if not task.done():
            task.cancel()
        while not task.done():
            remaining = deadline - loop.time()
            if remaining <= 0:
                task.cancel()
                logger.error(
                    "Cancelled %s task did not stop before its join deadline; "
                    "the caller must retain its lease in quarantine",
                    label,
                )
                task.add_done_callback(_consume_late_result)
                return False
            try:
                done, _ = await asyncio.wait({task}, timeout=remaining)
            except asyncio.CancelledError:
                current = asyncio.current_task()
                if current is not None and hasattr(current, "uncancel"):
                    current.uncancel()
                task.cancel()
                continue
            if task not in done:
                task.cancel()
                logger.error(
                    "Cancelled %s task did not stop before its join deadline; "
                    "the caller must retain its lease in quarantine",
                    label,
                )
                task.add_done_callback(_consume_late_result)
                return False
        await asyncio.gather(task, return_exceptions=True)
        return True

    async def _watch_quarantined_reset(
        self, run_slot: RunSlot, reset_future: asyncio.Task[Any]
    ) -> None:
        while not reset_future.done():
            try:
                # Observe rather than await the reset result.  A reset task may
                # have a pending cancellation while its Docker thread is still
                # exiting; asyncio.wait avoids a tight CancelledError loop.
                await asyncio.wait({reset_future}, timeout=0.1)
            except asyncio.CancelledError:
                current = asyncio.current_task()
                if current is not None and hasattr(current, "uncancel"):
                    current.uncancel()
                continue

        slot_to_cleanup: tuple[str, str, RunSlot] | None = None
        async with self._lock:
            if not reset_future.done() or not run_slot.reset_quarantined:
                return
            current_task_key = self._run_to_task.get(run_slot.run_lease_id)
            current_task_slot = (
                self._tasks.get(current_task_key)
                if current_task_key is not None
                else None
            )
            if (
                current_task_slot is None
                or current_task_slot.runs.get(run_slot.run_lease_id) is not run_slot
            ):
                return
            popped = self._pop_run_slot_locked(run_slot.run_lease_id)
            if popped is not None:
                task_key, popped_slot = popped
                slot_to_cleanup = (task_key, run_slot.run_lease_id, popped_slot)

        if slot_to_cleanup is not None:
            logger.warning(
                "Quarantined reset finished; removing lease=%s and starting Docker cleanup",
                run_slot.run_lease_id,
            )
            await self._force_cleanup_slots(
                [slot_to_cleanup], reason="reset_quarantine_finished"
            )

    async def _quarantine_reset_run(
        self,
        run_slot: RunSlot,
        reset_future: asyncio.Task[Any],
        *,
        reason: str,
    ) -> bool:
        async with self._lock:
            if reset_future.done():
                return False
            task_key = self._run_to_task.get(run_slot.run_lease_id)
            task_slot = self._tasks.get(task_key) if task_key is not None else None
            if task_slot is None or task_slot.runs.get(run_slot.run_lease_id) is not run_slot:
                return False

            now = time.time()
            run_slot.reset_quarantined = True
            run_slot.reset_quarantine_reason = reason
            run_slot.reset_quarantine_started_ts = now
            run_slot.close_requested = True
            run_slot.close_reason = reason
            run_slot.close_requested_ts = now
            run_slot.phase = "reset_quarantined"
            run_slot.last_used_ts = now
            for idem_key, (lease_id, _timestamp) in list(self._idempotency.items()):
                if lease_id == run_slot.run_lease_id:
                    self._idempotency.pop(idem_key, None)

            watcher = run_slot.reset_quarantine_watcher
            if watcher is None or watcher.done():
                watcher = asyncio.create_task(
                    self._watch_quarantined_reset(run_slot, reset_future)
                )
                run_slot.reset_quarantine_watcher = watcher
                self._reset_quarantine_watchers.add(watcher)

                def _on_done(done_task: asyncio.Task[Any]) -> None:
                    self._reset_quarantine_watchers.discard(done_task)

                watcher.add_done_callback(_on_done)

        logger.error(
            "Reset cancellation join deadline expired; quarantined lease=%s task=%s "
            "reason=%s. No lease removal or Docker cleanup will occur until reset exits.",
            run_slot.run_lease_id,
            run_slot.task_key,
            reason,
        )
        return True

    async def _drop_resetting_run_for_timeout(
        self, run_lease_id: str, run_slot: RunSlot, *, timeout: float
    ) -> None:
        async with self._lock:
            if run_slot.reset_quarantined:
                return
            task_key = self._run_to_task.get(run_lease_id)
            if task_key is None:
                return
            current = self._tasks.get(task_key)
            if current is None or current.runs.get(run_lease_id) is not run_slot:
                return
            run_slot.drop_scheduled = True
            run_slot.close_requested = True
            run_slot.close_reason = f"reset_timeout:{timeout:.1f}s"
            run_slot.close_requested_ts = time.time()
            run_slot.phase = "closing_requested"
            logger.warning(
                "Reset timed out; retaining lease=%s task=%s until the outer reset "
                "future exits before Docker cleanup %s",
                run_lease_id,
                task_key,
                self._run_slot_container_ref(run_slot),
            )

    async def _finalize_completed_reset(
        self, run_slot: RunSlot, reset_future: asyncio.Task[Any]
    ) -> None:
        slot_to_close: tuple[str, str, RunSlot] | None = None
        force_cleanup = False
        async with self._lock:
            if not reset_future.done() or run_slot.reset_quarantined:
                return
            task_key = self._run_to_task.get(run_slot.run_lease_id)
            task_slot = self._tasks.get(task_key) if task_key is not None else None
            if (
                task_slot is None
                or task_slot.runs.get(run_slot.run_lease_id) is not run_slot
                or run_slot.reset_future is not reset_future
                or (not run_slot.close_requested and not run_slot.drop_scheduled)
                or run_slot.in_flight_ops > 0
                or run_slot.lock.locked()
            ):
                return
            popped = self._pop_run_slot_locked(run_slot.run_lease_id)
            if popped is not None:
                popped_task_key, popped_slot = popped
                slot_to_close = (
                    popped_task_key,
                    run_slot.run_lease_id,
                    popped_slot,
                )
                force_cleanup = popped_slot.drop_scheduled

        if slot_to_close is None:
            return
        if force_cleanup:
            self._schedule_force_cleanup_slots(
                [slot_to_close], reason=run_slot.close_reason or "reset_failed"
            )
        else:
            task_key, run_lease_id, popped_slot = slot_to_close
            self._schedule_close(
                task_key,
                run_lease_id,
                popped_slot,
                reason=(
                    "Closing run slot after completed reset: "
                    f"{popped_slot.close_reason or 'close_requested'}"
                ),
            )

    async def _acquire_reset_admission(self, run_lease_id: str) -> None:
        started_monotonic = time.monotonic()
        success = False
        timeout = max(0.0, self.reset_admission_timeout)
        async with self._lock:
            self._reset_admission_waiting += 1
        try:
            try:
                if timeout > 0:
                    await asyncio.wait_for(
                        self._reset_admission_sem.acquire(), timeout=timeout
                    )
                else:
                    await self._reset_admission_sem.acquire()
                success = True
            except asyncio.TimeoutError as exc:
                async with self._lock:
                    self._reset_admission_rejected += 1
                raise ResetAdmissionBacklogError(
                    run_lease_id,
                    timeout,
                    self.max_concurrent_resets,
                ) from exc
        finally:
            async with self._lock:
                self._reset_admission_waiting = max(
                    0, self._reset_admission_waiting - 1
                )
            self._record_lifecycle_duration(
                "reset_admission_wait", started_monotonic, success=success
            )

    async def _run_reset_once(
        self,
        run_lease_id: str,
        task_meta: dict[str, Any],
        run_ctx_payload: dict[str, Any] | None,
        task_timeouts: dict[str, Any] | None,
    ) -> dict[str, Any]:
        run_ctx = _build_run_ctx(
            run_ctx_payload, default_log_dir=self.output_root / "AgentRunner_Output"
        )
        timeouts = _parse_timeout_overrides(self.default_timeouts, task_timeouts)
        task_spec = _build_task_spec(task_meta)
        reset_timeout = self._reset_operation_timeout(timeouts)

        # Use a Task instead of a bare coroutine. wait_for() cancels its awaitable
        # on timeout, and bare coroutines cannot be awaited again after that.
        warn_after = _env_float("WORKER_RESET_WARN_AFTER", 300.0)
        warn_timeout = max(0.1, min(reset_timeout / 2.0, warn_after))
        remaining_timeout = max(0.1, reset_timeout - warn_timeout)
        is_timeout_drop = False
        success = False
        reset_task: asyncio.Task[tuple[str, list[dict[str, Any]]]] | None = None
        run_slot: RunSlot | None = None
        reset_admission_acquired = False

        try:
            await self._acquire_reset_admission(run_lease_id)
            reset_admission_acquired = True
            run_slot, started_monotonic = await self._begin_run_op(
                run_lease_id, "reset"
            )
            async with run_slot.lock:
                reset_task = asyncio.create_task(
                    run_slot.env.reset(
                        task_meta=task_meta,
                        task_spec=task_spec,
                        run_ctx=run_ctx,
                        timeouts=timeouts,
                    )
                )
                done, _ = await asyncio.wait({reset_task}, timeout=warn_timeout)

                if reset_task not in done:
                    logger.warning(
                        "Reset exceeds %.1fs (warn threshold), allowing %.1fs more: lease=%s",
                        warn_timeout,
                        remaining_timeout,
                        run_lease_id,
                    )

                try:
                    user_msg, tool_schemas = await asyncio.wait_for(
                        asyncio.shield(reset_task),
                        timeout=remaining_timeout,
                    )
                except asyncio.TimeoutError as exc:
                    if reset_task.done():
                        raise
                    is_timeout_drop = True
                    reset_task.cancel()
                    raise TimeoutError(
                        f"WORKER_RESET_TIMEOUT lease_id={run_lease_id} "
                        f"after {reset_timeout:.1f}s"
                    ) from exc
                success = True
                return {"user_msg": user_msg, "tool_schemas": tool_schemas}
        finally:
            if not success and reset_task is not None and not reset_task.done():
                reset_task.cancel()
                # TerminalEnv.reset may own a non-cancellable Docker thread. Keep
                # this wrapper alive until that thread exits; callers quarantine
                # the outer reset future if their bounded join deadline expires.
                await self._join_task_uncancellable(reset_task)
            if run_slot is not None:
                await self._finish_run_op(
                    run_slot,
                    "reset",
                    success=success,
                    started_monotonic=started_monotonic,
                    is_timeout_drop=is_timeout_drop,
                )
            if reset_admission_acquired:
                self._reset_admission_sem.release()

    async def reset(
        self,
        run_lease_id: str,
        task_meta: dict[str, Any],
        run_ctx_payload: dict[str, Any] | None = None,
        task_timeouts: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        if not isinstance(task_meta, dict):
            raise ValueError("task_meta must be a dict")

        request_id = str(request_id or "")
        future: asyncio.Task

        async with self._lock:
            if self._shutdown_started:
                raise RuntimeError("Worker is shutting down; rejecting reset")
            run_slot = self._get_run_slot(run_lease_id)
            if run_slot.reset_quarantined:
                raise RuntimeError(
                    f"Run {run_lease_id} has a quarantined reset; rejecting reset"
                )
            if run_slot.close_requested:
                raise RuntimeError(
                    f"Run {run_lease_id} is closing; rejecting reset"
                )
            existing = run_slot.reset_future
            if request_id and run_slot.reset_request_id == request_id:
                if run_slot.reset_result is not None:
                    return dict(run_slot.reset_result)
                if existing is not None and not existing.done():
                    future = existing
                elif existing is not None and existing.done():
                    future = existing
                else:
                    future = asyncio.create_task(
                        self._run_reset_once(
                            run_lease_id,
                            task_meta,
                            run_ctx_payload,
                            task_timeouts,
                        )
                    )
                    run_slot.reset_future = future
            else:
                if existing is not None and not existing.done():
                    raise ResetInProgressError(run_lease_id, run_slot.reset_request_id)
                run_slot.reset_request_id = request_id or f"reset-{uuid.uuid4().hex[:16]}"
                run_slot.reset_result = None
                future = asyncio.create_task(
                    self._run_reset_once(
                        run_lease_id,
                        task_meta,
                        run_ctx_payload,
                        task_timeouts,
                    )
                )
                run_slot.reset_future = future

        try:
            result = await asyncio.shield(future)
        except asyncio.CancelledError as exc:
            if not future.done():
                future.cancel()
            joined = await self._cancel_and_join_reset_task(
                future, label=f"reset wrapper lease={run_lease_id}"
            )
            if not joined:
                async with self._lock:
                    try:
                        run_slot = self._get_run_slot(run_lease_id)
                    except KeyError:
                        run_slot = None
                if run_slot is not None:
                    await self._quarantine_reset_run(
                        run_slot,
                        future,
                        reason="reset_request_cancel_join_timeout",
                    )
            else:
                await self._finalize_completed_reset(run_slot, future)
            async with self._lock:
                try:
                    run_slot = self._get_run_slot(run_lease_id)
                except KeyError:
                    pass
                else:
                    if run_slot.reset_future is future:
                        if not run_slot.reset_quarantined and future.done():
                            run_slot.reset_future = None
                            run_slot.reset_result = None
            raise TimeoutError(
                f"WORKER_RESET_CANCELLED lease_id={run_lease_id} request_id={request_id}"
            ) from exc
        except Exception:
            await self._finalize_completed_reset(run_slot, future)
            raise
        else:
            async with self._lock:
                try:
                    run_slot = self._get_run_slot(run_lease_id)
                except KeyError:
                    logger.info(
                        "Reset completed after lease=%s was already removed; "
                        "returning reset result without caching it.",
                        run_lease_id,
                    )
                    return result
                if run_slot.reset_future is future:
                    run_slot.reset_result = dict(result)
            await self._finalize_completed_reset(run_slot, future)
            return result

    async def exec_tool(
        self, run_lease_id: str, tool_name: str, arguments: dict[str, Any] | None = None
    ) -> str:
        run_slot, started_monotonic = await self._begin_run_op(
            run_lease_id, "exec_tool"
        )
        success = False
        try:
            async with run_slot.lock:
                observation = await run_slot.env.exec_tool(tool_name, arguments or {})
                success = True
                return str(observation)
        finally:
            await self._finish_run_op(
                run_slot,
                "exec_tool",
                success=success,
                started_monotonic=started_monotonic,
            )

    async def handle_agent_reply(
        self, run_lease_id: str, assistant_text: str
    ) -> dict[str, Any]:
        async with self._lock:
            run_slot = self._get_run_slot(run_lease_id)
        async with run_slot.lock:
            result = await run_slot.env.handle_agent_reply(assistant_text)
            run_slot.last_used_ts = time.time()
            return dict(result)

    async def evaluate(
        self, run_lease_id: str, trajectory: dict[str, Any] | None = None
    ) -> tuple[float, dict[str, Any] | None]:
        run_slot, started_monotonic = await self._begin_run_op(
            run_lease_id, "evaluate"
        )
        success = False
        try:
            async with run_slot.lock:
                score = await run_slot.env.evaluate(trajectory)
                details = run_slot.env.last_eval_details()
                success = True
                return float(score), details
        finally:
            await self._finish_run_op(
                run_slot,
                "evaluate",
                success=success,
                started_monotonic=started_monotonic,
            )

    async def close_run(self, run_lease_id: str, *, reason: str = "external_close") -> bool:
        close_now: tuple[str, str, RunSlot] | None = None
        async with self._lock:
            task_key = self._run_to_task.get(run_lease_id)
            if task_key is None:
                logger.debug(
                    "close_run: lease %s already gone, nothing to do.", run_lease_id
                )
                return False
            task_slot = self._tasks.get(task_key)
            run_slot = task_slot.runs.get(run_lease_id) if task_slot else None
            if run_slot is None:
                return False

            if run_slot.close_requested:
                logger.info(
                    "close_run: duplicate close ignored lease=%s task=%s phase=%s "
                    "in_flight=%d reason=%s %s",
                    run_lease_id,
                    task_key,
                    run_slot.phase,
                    run_slot.in_flight_ops,
                    run_slot.close_reason,
                    self._run_slot_container_ref(run_slot),
                )
                return True

            run_slot.close_requested = True
            run_slot.close_reason = reason
            run_slot.close_requested_ts = time.time()
            stack = "".join(traceback.format_stack(limit=8))
            logger.warning(
                "close_run requested lease=%s task=%s phase=%s in_flight=%d "
                "first_step=%s evaluate_done=%s reason=%s %s\nClose request stack:\n%s",
                run_lease_id,
                task_key,
                run_slot.phase,
                run_slot.in_flight_ops,
                run_slot.first_step_ts is not None,
                run_slot.evaluate_completed_ts is not None,
                reason,
                self._run_slot_container_ref(run_slot),
                stack,
            )
            if (
                run_slot.in_flight_ops > 0
                or run_slot.lock.locked()
                or (
                    run_slot.reset_future is not None
                    and not run_slot.reset_future.done()
                )
            ):
                run_slot.phase = "closing_requested"
                self._schedule_close_requested_force_release(
                    run_lease_id, reason=reason
                )
                return True

            popped = self._pop_run_slot_locked(run_lease_id)
            if popped is not None:
                task_key, run_slot = popped
                close_now = (task_key, run_lease_id, run_slot)

        if close_now is not None:
            task_key, run_lease_id, run_slot = close_now
            self._schedule_close(task_key, run_lease_id, run_slot, reason="Closing run slot")
        return True

    async def status(self) -> dict[str, Any]:
        async with self._lock:
            self._prune_done_closing_tasks()
            self._prune_done_force_cleanup_tasks()
            now = time.time()
            self._prune_recent_close_failures(now)
            allocated_ttl = _env_float("WORKER_ALLOCATED_TTL", 120.0)
            resetting_ttl = _env_float("WORKER_RESETTING_TTL", 2100.0)
            closing_ttl = _env_float("WORKER_CLOSING_REQUESTED_TTL", 300.0)
            close_ages = [
                now - started for started in self._closing_task_started.values()
            ]
            force_cleanup_ages = [
                now - started for started in self._force_cleanup_task_started.values()
            ]
            pending_close_age_sec = {
                "min": round(min(close_ages), 1) if close_ages else 0.0,
                "max": round(max(close_ages), 1) if close_ages else 0.0,
                "over_close_timeout": sum(
                    1 for age in close_ages if age >= self.close_task_timeout
                ),
            }
            pending_force_cleanup_age_sec = {
                "min": round(min(force_cleanup_ages), 1) if force_cleanup_ages else 0.0,
                "max": round(max(force_cleanup_ages), 1) if force_cleanup_ages else 0.0,
            }
            tasks_info: dict[str, Any] = {}
            active_container_ids: set[str] = set()
            active_container_names: set[str] = set()
            active_trial_names: set[str] = set()
            active_project_names: set[str] = set()
            active_task_ids: set[str] = set()
            phase_counts: dict[str, int] = {}
            stale_runs: list[dict[str, Any]] = []
            reset_ages = []
            total_runs = 0
            in_flight_runs = 0
            closing_requested_runs = 0
            reset_quarantined_runs = 0
            for tk, ts in self._tasks.items():
                task_id = _task_id_from_ref(tk)
                if task_id:
                    active_task_ids.add(task_id)
                run_details = {}
                for rid, rslot in ts.runs.items():
                    phase_counts[rslot.phase] = phase_counts.get(rslot.phase, 0) + 1
                    if rslot.in_flight_ops > 0:
                        in_flight_runs += 1
                    if rslot.close_requested:
                        closing_requested_runs += 1
                    if rslot.reset_quarantined:
                        reset_quarantined_runs += 1
                    container_info = self._run_slot_container_info(rslot)
                    for key in ("id", "short_id"):
                        value = container_info.get(key)
                        if isinstance(value, str) and value:
                            active_container_ids.add(value)
                    container_name = container_info.get("name")
                    if isinstance(container_name, str) and container_name:
                        active_container_names.add(container_name)
                        active_project_names.update(
                            _docker_name_variants(container_name)
                        )
                        task_id = _task_id_from_ref(container_name)
                        if task_id:
                            active_task_ids.add(task_id)
                    trial_name = container_info.get("trial_name")
                    if isinstance(trial_name, str) and trial_name:
                        active_trial_names.add(trial_name)
                        active_project_names.update(_docker_name_variants(trial_name))
                        task_id = _task_id_from_ref(trial_name)
                        if task_id:
                            active_task_ids.add(task_id)
                    created_age_sec = now - rslot.created_ts
                    reset_age_sec = (
                        now - rslot.reset_started_ts
                        if rslot.reset_started_ts is not None
                        else 0.0
                    )
                    if rslot.phase == "resetting":
                        reset_ages.append(reset_age_sec)
                    close_age_sec = (
                        now - rslot.close_requested_ts
                        if rslot.close_requested_ts is not None
                        else 0.0
                    )
                    stale_reason, stale_age_sec = self._stale_reason_for_run_slot(
                        rslot, now
                    )
                    if stale_reason:
                        stale_runs.append(
                            {
                                "lease_id": rid,
                                "task_key": tk,
                                "phase": rslot.phase,
                                "reason": stale_reason,
                                "age_sec": round(stale_age_sec, 1),
                                "in_flight_ops": rslot.in_flight_ops,
                                "active_op": rslot.active_op,
                                "close_requested": rslot.close_requested,
                                "container": container_info,
                            }
                        )
                    run_details[rid] = {
                        "phase": rslot.phase,
                        "in_flight_ops": rslot.in_flight_ops,
                        "active_op": rslot.active_op,
                        "close_requested": rslot.close_requested,
                        "reset_quarantined": rslot.reset_quarantined,
                        "reset_quarantine_reason": rslot.reset_quarantine_reason,
                        "reset_quarantine_age_sec": round(
                            now - rslot.reset_quarantine_started_ts, 1
                        )
                        if rslot.reset_quarantine_started_ts is not None
                        else 0.0,
                        "age_sec": round(now - rslot.last_used_ts, 1),
                        "created_age_sec": round(created_age_sec, 1),
                        "reset_age_sec": round(reset_age_sec, 1),
                        "close_requested_age_sec": round(close_age_sec, 1),
                        "first_step": rslot.first_step_ts is not None,
                        "evaluate_done": rslot.evaluate_completed_ts is not None,
                        "container": container_info,
                    }
                tasks_info[tk] = {
                    "active_runs": len(ts.runs),
                    "max_runs": self._effective_max_runs_per_task(tk),
                    "runs": run_details,
                }
                total_runs += len(ts.runs)

            return {
                "max_tasks": self.max_tasks,
                "active_tasks": len(self._tasks),
                "max_runs_per_task": self.max_runs_per_task,
                "max_total_runs": self.max_total_runs,
                "serial_task_ids": sorted(self._serial_task_ids),
                "task_max_runs_overrides": dict(
                    sorted(self._task_max_runs_overrides.items())
                ),
                "auto_serialize_unsafe_compose": self._auto_serialize_unsafe_compose,
                "total_active_runs": total_runs,
                "total_retiring_runs": len(self._retiring_run_refs),
                "total_reserved_runs": len(
                    set(self._run_to_task).union(self._retiring_run_refs)
                ),
                "retiring_run_ids": sorted(self._retiring_run_refs),
                "in_flight_runs": in_flight_runs,
                "closing_requested_runs": closing_requested_runs,
                "reset_quarantined_runs": reset_quarantined_runs,
                "pending_reset_quarantine_watchers": len(
                    self._reset_quarantine_watchers
                ),
                "pending_closes": len(self._closing_tasks),
                "pending_force_cleanups": len(self._force_cleanup_tasks),
                "pending_close_labels": sorted(
                    self._closing_task_labels.values()
                ),
                "pending_force_cleanup_labels": sorted(
                    self._force_cleanup_task_labels.values()
                ),
                "recent_close_failures": list(
                    self._recent_close_failures.values()
                ),
                "reset_admission": {
                    "max_concurrent": self.max_concurrent_resets,
                    "available": int(getattr(self._reset_admission_sem, "_value", 0)),
                    "waiting": self._reset_admission_waiting,
                    "rejected": self._reset_admission_rejected,
                    "timeout": self.reset_admission_timeout,
                },
                "lifecycle_latency_sec": self._lifecycle_latency_snapshot(),
                "docker_image_build": docker_image_build_status(),
                "close_queue_timeout": self.close_queue_timeout,
                "close_session_timeout": self.close_session_timeout,
                "close_task_timeout": self.close_task_timeout,
                "pending_close_age_sec": pending_close_age_sec,
                "pending_force_cleanup_age_sec": pending_force_cleanup_age_sec,
                "resetting_age_sec": {
                    "min": round(min(reset_ages), 1) if reset_ages else 0.0,
                    "max": round(max(reset_ages), 1) if reset_ages else 0.0,
                },
                "phase_counts": phase_counts,
                "stale_runs": stale_runs,
                "active_container_ids": sorted(active_container_ids),
                "active_container_names": sorted(active_container_names),
                "active_trial_names": sorted(active_trial_names),
                "active_project_names": sorted(active_project_names),
                "active_task_ids": sorted(active_task_ids),
                "tasks": tasks_info,
            }

    async def repair_pending_closes(
        self,
        *,
        reason: str,
        max_active_runs: int = 0,
        cancel_timeout: float = 5.0,
        min_age: float | None = None,
    ) -> dict[str, Any]:
        now = time.time()
        if min_age is None:
            min_age = max(0.0, self.close_task_timeout + 5.0)
        async with self._lock:
            pruned_done = self._prune_done_closing_tasks()
            active_runs = sum(len(ts.runs) for ts in self._tasks.values())
            pending_before_cancel = len(self._closing_tasks)
            if max_active_runs >= 0 and active_runs > max_active_runs:
                return {
                    "repaired": False,
                    "reason": "active_runs_above_limit",
                    "active_runs": active_runs,
                    "max_active_runs": max_active_runs,
                    "pending_closes": pending_before_cancel,
                    "pruned_done": pruned_done,
                }
            tasks_to_cancel = [
                task
                for task in self._closing_tasks
                if now - self._closing_task_started.get(task, now) >= min_age
            ]
            skipped_young = pending_before_cancel - len(tasks_to_cancel)

        # Cancel once, then observe completion without wait_for(gather). A second
        # cancellation would detach the cleanup started by _close_run_slot().
        cancelled = 0
        for task in tasks_to_cancel:
            if not task.done():
                task.cancel()
                cancelled += 1

        if tasks_to_cancel:
            _done, pending = await asyncio.wait(
                set(tasks_to_cancel), timeout=max(0.1, cancel_timeout)
            )
            if pending:
                logger.warning(
                    "Timed out waiting for pending close task cancellation: "
                    "reason=%s pending=%d timeout=%.1fs",
                    reason,
                    len(pending),
                    cancel_timeout,
                )

        async with self._lock:
            pruned_after_cancel = self._prune_done_closing_tasks()
            pending_after = len(self._closing_tasks)

        logger.warning(
            "Repaired pending close tasks: reason=%s active_runs=%d "
            "min_age=%.1fs pruned_done=%d cancelled=%d skipped_young=%d "
            "pruned_after_cancel=%d pending_after=%d",
            reason,
            active_runs,
            min_age,
            pruned_done,
            cancelled,
            skipped_young,
            pruned_after_cancel,
            pending_after,
        )
        return {
            "repaired": True,
            "reason": reason,
            "active_runs": active_runs,
            "max_active_runs": max_active_runs,
            "min_age": min_age,
            "pending_before_cancel": pending_before_cancel,
            "pruned_done": pruned_done,
            "cancelled": cancelled,
            "skipped_young": skipped_young,
            "pruned_after_cancel": pruned_after_cancel,
            "pending_after": pending_after,
        }

    async def repair_stale_runs(
        self,
        *,
        reason: str,
        min_age: float = 0.0,
        max_repairs: int = 20,
        wait_for_cleanup: bool = True,
    ) -> dict[str, Any]:
        now = time.time()
        slots_to_force_cleanup: list[tuple[str, str, RunSlot]] = []
        repaired_runs: list[dict[str, Any]] = []
        async with self._lock:
            self._prune_done_closing_tasks()
            for task_key, task_slot in list(self._tasks.items()):
                for run_lease_id, run_slot in list(task_slot.runs.items()):
                    stale_reason, stale_age_sec = self._stale_reason_for_run_slot(
                        run_slot, now
                    )
                    if not stale_reason or stale_age_sec < min_age:
                        continue
                    if run_slot.reset_quarantined:
                        continue
                    if (
                        run_slot.reset_future is not None
                        and not run_slot.reset_future.done()
                    ):
                        # Live reset cancellation/join belongs to the dedicated
                        # resetting repair path; generic repair must not detach it.
                        continue
                    popped = self._pop_run_slot_locked(run_lease_id)
                    if popped is None:
                        continue
                    popped_task_key, popped_slot = popped
                    slots_to_force_cleanup.append(
                        (popped_task_key, run_lease_id, popped_slot)
                    )
                    repaired_runs.append(
                        {
                            "lease_id": run_lease_id,
                            "task_key": popped_task_key,
                            "phase": popped_slot.phase,
                            "reason": stale_reason,
                            "age_sec": round(stale_age_sec, 1),
                            "in_flight_ops": popped_slot.in_flight_ops,
                            "active_op": popped_slot.active_op,
                            "close_requested": popped_slot.close_requested,
                            "container": self._run_slot_container_info(popped_slot),
                        }
                    )
                    if max_repairs > 0 and len(repaired_runs) >= max_repairs:
                        break
                if max_repairs > 0 and len(repaired_runs) >= max_repairs:
                    break

        if slots_to_force_cleanup and wait_for_cleanup:
            await self._force_cleanup_slots(
                slots_to_force_cleanup,
                reason=f"repair_stale_runs:{reason}",
            )
        elif slots_to_force_cleanup:
            self._schedule_force_cleanup_slots(
                slots_to_force_cleanup,
                reason=f"repair_stale_runs:{reason}",
            )

        return {
            "repaired": bool(repaired_runs),
            "reason": reason,
            "min_age": min_age,
            "max_repairs": max_repairs,
            "wait_for_cleanup": wait_for_cleanup,
            "repaired_count": len(repaired_runs),
            "repaired_runs": repaired_runs,
        }

    async def repair_close_requested_runs(
        self,
        *,
        reason: str,
        min_age: float = 0.0,
        max_repairs: int = 20,
        wait_for_cleanup: bool = False,
    ) -> dict[str, Any]:
        now = time.time()
        slots_to_force_cleanup: list[tuple[str, str, RunSlot]] = []
        candidates: list[
            tuple[str, str, RunSlot, asyncio.Task[Any] | None, float]
        ] = []
        repaired_runs: list[dict[str, Any]] = []
        skipped_active = 0
        async with self._lock:
            self._prune_done_closing_tasks()
            self._prune_done_force_cleanup_tasks()
            for task_key, task_slot in list(self._tasks.items()):
                for run_lease_id, run_slot in list(task_slot.runs.items()):
                    if not run_slot.close_requested:
                        continue
                    if run_slot.reset_quarantined:
                        skipped_active += 1
                        continue
                    close_age_sec = (
                        now - run_slot.close_requested_ts
                        if run_slot.close_requested_ts is not None
                        else now - run_slot.last_used_ts
                    )
                    if close_age_sec < min_age:
                        continue
                    candidates.append(
                        (
                            task_key,
                            run_lease_id,
                            run_slot,
                            run_slot.reset_future,
                            close_age_sec,
                        )
                    )
                    if max_repairs > 0 and len(candidates) >= max_repairs:
                        break
                if max_repairs > 0 and len(candidates) >= max_repairs:
                    break

        reset_join_deadline = asyncio.get_running_loop().time() + max(
            0.1, _env_float("WORKER_RESET_CANCEL_JOIN_TIMEOUT", 15.0)
        )
        for (
            _task_key,
            _run_lease_id,
            run_slot,
            reset_future,
            _close_age_sec,
        ) in candidates:
            if reset_future is not None and not reset_future.done():
                reset_future.cancel()
                joined = await self._cancel_and_join_reset_task(
                    reset_future,
                    deadline=reset_join_deadline,
                    label=f"close-requested reset lease={_run_lease_id}",
                )
                if not joined:
                    await self._quarantine_reset_run(
                        run_slot,
                        reset_future,
                        reason=f"repair_close_requested_join_timeout:{reason}",
                    )

        async with self._lock:
            for (
                task_key,
                run_lease_id,
                run_slot,
                reset_future,
                close_age_sec,
            ) in candidates:
                current_task_key = self._run_to_task.get(run_lease_id)
                current_task_slot = (
                    self._tasks.get(current_task_key)
                    if current_task_key is not None
                    else None
                )
                if (
                    current_task_key != task_key
                    or current_task_slot is None
                    or current_task_slot.runs.get(run_lease_id) is not run_slot
                ):
                    continue
                if run_slot.reset_quarantined:
                    skipped_active += 1
                    continue
                if run_slot.reset_future is not reset_future:
                    skipped_active += 1
                    continue
                if reset_future is not None and not reset_future.done():
                    skipped_active += 1
                    continue
                if run_slot.in_flight_ops > 0 or run_slot.lock.locked():
                    skipped_active += 1
                    continue
                popped = self._pop_run_slot_locked(run_lease_id)
                if popped is None:
                    continue
                popped_task_key, popped_slot = popped
                slots_to_force_cleanup.append(
                    (popped_task_key, run_lease_id, popped_slot)
                )
                repaired_runs.append(
                    {
                        "lease_id": run_lease_id,
                        "task_key": popped_task_key,
                        "phase": popped_slot.phase,
                        "reason": "close_requested_capacity_pressure",
                        "age_sec": round(close_age_sec, 1),
                        "in_flight_ops": popped_slot.in_flight_ops,
                        "active_op": popped_slot.active_op,
                        "close_requested": popped_slot.close_requested,
                        "container": self._run_slot_container_info(popped_slot),
                    }
                )

        if slots_to_force_cleanup and wait_for_cleanup:
            await self._force_cleanup_slots(
                slots_to_force_cleanup,
                reason=f"repair_close_requested_runs:{reason}",
            )
        elif slots_to_force_cleanup:
            self._schedule_force_cleanup_slots(
                slots_to_force_cleanup,
                reason=f"repair_close_requested_runs:{reason}",
            )

        return {
            "repaired": bool(repaired_runs),
            "reason": reason,
            "min_age": min_age,
            "max_repairs": max_repairs,
            "wait_for_cleanup": wait_for_cleanup,
            "repaired_count": len(repaired_runs),
            "repaired_runs": repaired_runs,
            "skipped_active": skipped_active,
        }

    async def repair_resetting_runs(
        self,
        *,
        reason: str,
        min_age: float = 0.0,
        max_repairs: int = 20,
        wait_for_cleanup: bool = False,
    ) -> dict[str, Any]:
        now = time.time()
        slots_to_force_cleanup: list[tuple[str, str, RunSlot]] = []
        candidates: list[
            tuple[str, str, RunSlot, asyncio.Task[Any] | None, float]
        ] = []
        repaired_runs: list[dict[str, Any]] = []
        async with self._lock:
            self._prune_done_closing_tasks()
            self._prune_done_force_cleanup_tasks()
            for task_key, task_slot in list(self._tasks.items()):
                for run_lease_id, run_slot in list(task_slot.runs.items()):
                    if run_slot.phase != "resetting":
                        continue
                    reset_age_sec = (
                        now - run_slot.reset_started_ts
                        if run_slot.reset_started_ts is not None
                        else now - run_slot.last_used_ts
                    )
                    if reset_age_sec < min_age:
                        continue
                    reset_future = run_slot.reset_future
                    run_slot.close_requested = True
                    run_slot.close_reason = f"repair_resetting_runs:{reason}"
                    run_slot.close_requested_ts = now
                    run_slot.drop_scheduled = True
                    run_slot.phase = "closing_requested"
                    candidates.append(
                        (
                            task_key,
                            run_lease_id,
                            run_slot,
                            reset_future,
                            reset_age_sec,
                        )
                    )
                    if max_repairs > 0 and len(candidates) >= max_repairs:
                        break
                if max_repairs > 0 and len(candidates) >= max_repairs:
                    break

        reset_join_deadline = asyncio.get_running_loop().time() + max(
            0.1, _env_float("WORKER_RESET_CANCEL_JOIN_TIMEOUT", 15.0)
        )
        for (
            _task_key,
            _run_lease_id,
            run_slot,
            reset_future,
            _reset_age_sec,
        ) in candidates:
            if reset_future is not None and not reset_future.done():
                reset_future.cancel()
                joined = await self._cancel_and_join_reset_task(
                    reset_future,
                    deadline=reset_join_deadline,
                    label=f"stale reset lease={_run_lease_id}",
                )
                if not joined:
                    await self._quarantine_reset_run(
                        run_slot,
                        reset_future,
                        reason=f"repair_resetting_join_timeout:{reason}",
                    )

        async with self._lock:
            for (
                task_key,
                run_lease_id,
                run_slot,
                reset_future,
                reset_age_sec,
            ) in candidates:
                current_task_key = self._run_to_task.get(run_lease_id)
                current_task_slot = (
                    self._tasks.get(current_task_key)
                    if current_task_key is not None
                    else None
                )
                if (
                    current_task_key != task_key
                    or current_task_slot is None
                    or current_task_slot.runs.get(run_lease_id) is not run_slot
                ):
                    continue
                if run_slot.reset_quarantined:
                    continue
                if run_slot.reset_future is not reset_future:
                    continue
                if reset_future is not None and not reset_future.done():
                    continue
                if run_slot.in_flight_ops > 0 or run_slot.lock.locked():
                    continue
                popped = self._pop_run_slot_locked(run_lease_id)
                if popped is None:
                    continue
                popped_task_key, popped_slot = popped
                slots_to_force_cleanup.append(
                    (popped_task_key, run_lease_id, popped_slot)
                )
                repaired_runs.append(
                    {
                        "lease_id": run_lease_id,
                        "task_key": popped_task_key,
                        "phase": popped_slot.phase,
                        "reason": "resetting_storm_repair",
                        "age_sec": round(reset_age_sec, 1),
                        "in_flight_ops": popped_slot.in_flight_ops,
                        "active_op": popped_slot.active_op,
                        "close_requested": popped_slot.close_requested,
                        "container": self._run_slot_container_info(popped_slot),
                    }
                )

        if slots_to_force_cleanup and wait_for_cleanup:
            await self._force_cleanup_slots(
                slots_to_force_cleanup,
                reason=f"repair_resetting_runs:{reason}",
            )
        elif slots_to_force_cleanup:
            self._schedule_force_cleanup_slots(
                slots_to_force_cleanup,
                reason=f"repair_resetting_runs:{reason}",
            )

        return {
            "repaired": bool(repaired_runs),
            "reason": reason,
            "min_age": min_age,
            "max_repairs": max_repairs,
            "wait_for_cleanup": wait_for_cleanup,
            "repaired_count": len(repaired_runs),
            "repaired_runs": repaired_runs,
        }

    async def _force_cleanup_slots(
        self, slots: list[tuple[str, str, RunSlot]], *, reason: str
    ) -> None:
        if not slots:
            return
        per_slot_timeout = _env_float("WORKER_FORCE_CLEANUP_TIMEOUT", 90.0)
        timeout = _env_float(
            "WORKER_SHUTDOWN_FORCE_CLEANUP_TIMEOUT",
            max(30.0, per_slot_timeout + 10.0),
        )
        logger.warning(
            "Batch force cleanup starting for %d run slot(s), reason=%s timeout=%.1fs",
            len(slots),
            reason,
            timeout,
        )
        cleanup_tasks: dict[asyncio.Task[Any], str] = {}
        for task_key, run_lease_id, run_slot in slots:
            task = asyncio.create_task(
                self._force_cleanup_after_close_failure(
                    run_slot,
                    run_lease_id,
                    reason=reason,
                )
            )
            cleanup_tasks[task] = f"{task_key}:{run_lease_id}"

        done, pending = await asyncio.wait(cleanup_tasks, timeout=timeout)
        if pending:
            logger.warning(
                "Batch force cleanup timed out with %d cleanup task(s) still pending: %s",
                len(pending),
                ",".join(cleanup_tasks[task] for task in pending),
            )
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
        if done:
            await asyncio.gather(*done, return_exceptions=True)
        logger.warning("Batch force cleanup finished for reason=%s", reason)

    async def periodic_reap(self, interval: float = 60.0) -> None:
        while True:
            await asyncio.sleep(interval)
            try:
                async with self._lock:
                    expired_slots = self._reap_idle_locked()
                for tk, rid, rslot in expired_slots:
                    self._schedule_close(
                        tk, rid, rslot, reason="Periodic reaper: idle run slot"
                    )
                if expired_slots:
                    logger.info(
                        "Periodic reaper cleaned up %d idle run slots",
                        len(expired_slots),
                    )
                # Automatic shim cleanup every 50 resets or when pressure detected
                await self._maybe_cleanup_shims()
                await self._maybe_cleanup_orphan_docker_containers()
            except Exception:
                logger.exception("Periodic reaper error")

    async def _maybe_cleanup_orphan_docker_containers(self) -> None:
        if os.getenv("WORKER_ORPHAN_DOCKER_SWEEP", "1") != "1":
            return
        now = time.time()
        if now < self._orphan_sweep_backoff_until:
            return
        interval = max(1.0, _env_float("WORKER_ORPHAN_DOCKER_SWEEP_INTERVAL", 60.0))
        if now - self._last_orphan_sweep_ts < interval:
            return
        self._last_orphan_sweep_ts = now

        async with self._lock:
            (
                active_container_names,
                active_project_names,
                active_task_ids,
            ) = self._active_docker_refs_locked()

        min_age_sec = max(0.0, _env_float("WORKER_ORPHAN_DOCKER_SWEEP_MIN_AGE", 600.0))
        max_remove = _env_int("WORKER_ORPHAN_DOCKER_SWEEP_MAX_REMOVE", 128)
        timeout = max(1.0, _env_float("WORKER_ORPHAN_DOCKER_SWEEP_TIMEOUT", 30.0))
        sweep_task = asyncio.create_task(
            asyncio.to_thread(
                force_remove_orphan_docker_objects,
                active_container_names=active_container_names,
                active_project_names=active_project_names,
                active_task_ids=active_task_ids,
                reason="periodic_reap",
                min_age_sec=min_age_sec,
                max_remove=max_remove,
                cleanup_timeout=timeout,
            )
        )
        try:
            removed = await asyncio.shield(sweep_task)
            if removed < 0:
                self._record_orphan_sweep_failure("docker_ps_failed")
                return
            self._orphan_sweep_fail_streak = 0
            self._orphan_sweep_backoff_until = 0.0
            if removed:
                logger.warning(
                    "Periodic orphan Docker sweep removed %d stale container(s) "
                    "active_containers=%d active_projects=%d active_tasks=%d min_age=%.1fs",
                    removed,
                    len(active_container_names),
                    len(active_project_names),
                    len(active_task_ids),
                    min_age_sec,
                )
        except (asyncio.TimeoutError, TimeoutError):
            logger.warning(
                "Periodic orphan Docker sweep timed out after %.1fs "
                "active_containers=%d active_projects=%d active_tasks=%d",
                timeout,
                len(active_container_names),
                len(active_project_names),
                len(active_task_ids),
            )
            self._record_orphan_sweep_failure(f"timeout_after_{timeout:.1f}s")
        except asyncio.CancelledError:
            # Cancelling asyncio.to_thread does not stop its worker thread.
            # Join the bounded sweep so it cannot mutate Docker state after
            # this reaper invocation has returned.
            await self._join_task_uncancellable(sweep_task)
            raise
        except Exception:
            self._record_orphan_sweep_failure("exception")
            logger.exception("Periodic orphan Docker sweep failed")

    def _record_orphan_sweep_failure(self, reason: str) -> None:
        self._orphan_sweep_fail_streak += 1
        base = max(1.0, _env_float("WORKER_ORPHAN_DOCKER_SWEEP_BACKOFF_BASE", 120.0))
        max_delay = max(base, _env_float("WORKER_ORPHAN_DOCKER_SWEEP_BACKOFF_MAX", 900.0))
        delay = min(max_delay, base * (2 ** min(self._orphan_sweep_fail_streak - 1, 6)))
        self._orphan_sweep_backoff_until = time.time() + delay
        logger.warning(
            "Periodic orphan Docker sweep failed (%s); backoff %.1fs streak=%d",
            reason,
            delay,
            self._orphan_sweep_fail_streak,
        )

    async def _maybe_cleanup_shims(self) -> None:
        """P0 fix: Proactively clean Docker shims to prevent resource exhaustion."""
        if not _env_bool("WORKER_SHIM_CLEANUP_ENABLED", True):
            return
        try:
            now = time.time()
            cleanup_interval = _env_float("WORKER_SHIM_CLEANUP_INTERVAL", 600.0)  # 10 min default
            reset_trigger = _env_int("WORKER_SHIM_CLEANUP_RESET_COUNT", 50)  # every 50 resets
            pressure_threshold = _env_int("WORKER_SHIM_CLEANUP_PRESSURE_THRESHOLD", 140)  # cleanup at 140 shims

            should_cleanup = False
            reason = ""

            # Check if enough time has passed since last cleanup
            if now - self._last_shim_cleanup_ts < cleanup_interval:
                return

            # Trigger 1: Reset count threshold
            if reset_trigger > 0 and self._reset_count >= reset_trigger:
                should_cleanup = True
                reason = f"reset_count={self._reset_count}>={reset_trigger}"

            # Trigger 2: Shim pressure threshold
            pressure = worker_pressure_stats()
            shim_count = int(pressure.get("shim", 0))
            if shim_count >= pressure_threshold:
                should_cleanup = True
                reason = f"shim_pressure={shim_count}>={pressure_threshold}"

            if not should_cleanup:
                return

            logger.warning(
                "Triggering automatic Docker shim cleanup: reason=%s shim_count=%d reset_count=%d",
                reason,
                shim_count,
                self._reset_count,
            )

            # Run docker system prune in background with timeout
            # Add fallback if prune hangs; close stderr pipe immediately to prevent fd leak
            cleanup_timeout = _env_float("WORKER_SHIM_CLEANUP_TIMEOUT", 30.0)
            proc = None
            try:
                proc = await asyncio.wait_for(
                    asyncio.create_subprocess_exec(
                        "docker",
                        "system",
                        "prune",
                        "-f",
                        "--volumes=false",
                        stdout=asyncio.subprocess.DEVNULL,
                        stderr=asyncio.subprocess.DEVNULL,  # Use DEVNULL to avoid fd leak
                    ),
                    timeout=cleanup_timeout,
                )
                await asyncio.wait_for(proc.wait(), timeout=cleanup_timeout)

                # Verify cleanup reduced shim count
                new_pressure = worker_pressure_stats(force=True)
                new_shim_count = int(new_pressure.get("shim", 0))
                logger.warning(
                    "Docker shim cleanup completed: shim_count %d→%d reset_count %d→0",
                    shim_count,
                    new_shim_count,
                    self._reset_count,
                )

                # Reset counters and timestamp
                self._reset_count = 0
                self._last_shim_cleanup_ts = now

            except asyncio.TimeoutError:
                logger.warning(
                    "Docker shim cleanup timed out after %.1fs; skipping and relying on watchdog (non-fatal)",
                    cleanup_timeout,
                )
                # Kill hung subprocess
                if proc is not None:
                    try:
                        proc.kill()
                        await asyncio.wait_for(proc.wait(), timeout=5.0)
                    except Exception:
                        pass
                # Still update timestamp to prevent retry storms
                self._last_shim_cleanup_ts = now
            except Exception as exc:
                logger.warning(
                    "Docker shim cleanup failed: %s (non-fatal, will retry next cycle)",
                    exc,
                )
                # Update timestamp on failure to prevent tight retry loop
                self._last_shim_cleanup_ts = now

        except Exception:
            logger.exception("Error in _maybe_cleanup_shims (non-fatal)")

    async def shutdown(self) -> None:
        async with self._lock:
            self._shutdown_started = True
            reset_entries = [
                (run_slot, run_slot.reset_future)
                for task_slot in self._tasks.values()
                for run_slot in task_slot.runs.values()
                if run_slot.reset_future is not None
                and not run_slot.reset_future.done()
                and not run_slot.reset_quarantined
            ]
            reset_futures = {future for _run_slot, future in reset_entries}

        # A reset may create Docker objects while cancellation propagates. Join
        # every reset before removing leases or starting close/force cleanup.
        reset_join_timeout = max(
            0.1,
            _env_float(
                "WORKER_SHUTDOWN_RESET_JOIN_TIMEOUT",
                _env_float("WORKER_RESET_CANCEL_JOIN_TIMEOUT", 15.0) + 5.0,
            ),
        )
        reset_join_deadline = asyncio.get_running_loop().time() + reset_join_timeout
        for reset_future in reset_futures:
            reset_future.cancel()
        reset_join_failures = 0
        for reset_future in reset_futures:
            joined = await self._cancel_and_join_reset_task(
                reset_future,
                deadline=reset_join_deadline,
                label="reset wrapper during shutdown",
            )
            if not joined:
                quarantined_any = False
                for run_slot, entry_future in reset_entries:
                    if entry_future is reset_future:
                        quarantined_any = (
                            await self._quarantine_reset_run(
                                run_slot,
                                reset_future,
                                reason="shutdown_reset_join_timeout",
                            )
                            or quarantined_any
                        )
                if quarantined_any:
                    reset_join_failures += 1
        if reset_join_failures:
            logger.error(
                "Shutdown reset join deadline %.1fs expired for %d task(s); "
                "their leases remain quarantined and cleanup is deferred until reset exits",
                reset_join_timeout,
                reset_join_failures,
            )

        async with self._lock:
            slots_to_close: list[tuple[str, str, RunSlot]] = []
            all_slots = [
                (task_key, run_lease_id, run_slot)
                for task_key, task_slot in self._tasks.items()
                for run_lease_id, run_slot in task_slot.runs.items()
            ]
            for _task_key, run_lease_id, run_slot in all_slots:
                if run_slot.reset_quarantined:
                    continue
                popped = self._pop_run_slot_locked(run_lease_id)
                if popped is not None:
                    task_key, popped_slot = popped
                    slots_to_close.append((task_key, run_lease_id, popped_slot))
            self._idempotency.clear()

        for task_key, run_lease_id, run_slot in slots_to_close:
            self._schedule_close(
                task_key,
                run_lease_id,
                run_slot,
                reason="Closing run slot during shutdown",
            )

        if self._closing_tasks:
            logger.info(
                "Shutdown: waiting for %d pending close tasks...",
                len(self._closing_tasks),
            )
            shutdown_timeout = _env_float(
                "WORKER_SHUTDOWN_CLOSE_TASKS_TIMEOUT",
                max(5.0, self.close_task_timeout + 5.0),
            )
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._closing_tasks, return_exceptions=True),
                    timeout=shutdown_timeout,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Shutdown timed out after %.1fs with %d pending close tasks; "
                    "cancelling them and forcing Docker cleanup.",
                    shutdown_timeout,
                    len(self._closing_tasks),
                )
                for task in list(self._closing_tasks):
                    task.cancel()
                try:
                    await asyncio.wait_for(
                        asyncio.gather(
                            *self._closing_tasks, return_exceptions=True
                        ),
                        timeout=5.0,
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "Shutdown cancellation wait timed out; exiting with %d "
                        "close task(s) still pending.",
                        len(self._closing_tasks),
                    )
                await self._force_cleanup_slots(
                    slots_to_close,
                    reason="shutdown_close_timeout",
                )
            else:
                await self._force_cleanup_slots(
                    slots_to_close,
                    reason="shutdown_final_sweep",
                )
        if self._force_cleanup_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(
                        *self._force_cleanup_tasks, return_exceptions=True
                    ),
                    timeout=_env_float("WORKER_SHUTDOWN_FORCE_CLEANUP_TASKS_TIMEOUT", 10.0),
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Shutdown timed out waiting for %d background force cleanup task(s)",
                    len(self._force_cleanup_tasks),
                )
