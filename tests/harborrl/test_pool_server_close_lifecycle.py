from __future__ import annotations

import asyncio
import importlib
import sys
import threading
import time
import types
from dataclasses import dataclass
from pathlib import Path

import pytest


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


@dataclass(frozen=True)
class _TaskSpec:
    task_name: str
    task_path: str
    instruction: str


@dataclass(frozen=True)
class _RunContext:
    uid: str
    group_index: int
    sample_index: int
    log_dir: Path
    rollout_id: int | None = None
    train_step: int | None = None
    rollout_step: int | None = None


@dataclass
class _TaskTimeouts:
    ensure_image: float = 300.0
    reset_session: float = 300.0
    close_session: float = 60.0
    eval: float = 600.0


class _FastAPI:
    def get(self, *_args, **_kwargs):
        return lambda fn: fn

    def post(self, *_args, **_kwargs):
        return lambda fn: fn

    def on_event(self, *_args, **_kwargs):
        return lambda fn: fn


class _JSONResponse(dict):
    def __init__(self, content=None, status_code=200, headers=None):
        super().__init__(content or {})
        self.status_code = status_code
        self.headers = headers or {}


class _Response:
    def __init__(self, content="", media_type=None):
        self.body = content
        self.media_type = media_type


class _DummyEnv:
    def __init__(self) -> None:
        self.reset_started = asyncio.Event()
        self.release_reset = asyncio.Event()
        self.close_count = 0

    async def reset(self, **_kwargs):
        self.reset_started.set()
        await self.release_reset.wait()
        return "user", []

    async def exec_tool(self, _tool_name, _arguments):
        return "observation"

    async def evaluate(self, _trajectory=None):
        return 1.0

    def last_eval_details(self):
        return None

    async def close(self):
        self.close_count += 1

    async def force_cleanup(self, reason="external"):
        self.force_cleanup_reason = reason


def _install_import_stubs(monkeypatch):
    fastapi_mod = types.ModuleType("fastapi")
    fastapi_mod.FastAPI = _FastAPI
    fastapi_mod.Request = object
    responses_mod = types.ModuleType("fastapi.responses")
    responses_mod.JSONResponse = _JSONResponse
    responses_mod.Response = _Response

    custom_types_mod = types.ModuleType("harborrl.types")
    custom_types_mod.TaskSpec = _TaskSpec
    custom_types_mod.RunContext = _RunContext
    custom_types_mod.TaskTimeouts = _TaskTimeouts

    request_utils_mod = types.ModuleType("harborrl.http_server")

    async def _json_payload(_request):
        return {}

    request_utils_mod.json_payload = _json_payload

    terminal_env_mod = types.ModuleType("harborrl.environments.terminal.runtime")
    terminal_env_mod.TerminalEnv = _DummyEnv
    terminal_env_mod.force_remove_orphan_docker_objects = lambda **_kwargs: 0

    docker_compose_utils_mod = types.ModuleType("harborrl.environments.terminal.docker_compose")

    class DockerImageBuildError(RuntimeError):
        pass

    class DockerImagePreparationBacklogError(RuntimeError):
        pass

    class DockerImageTransientBuildError(DockerImageBuildError):
        pass

    class TaskImageBlacklistedError(DockerImageBuildError):
        pass

    docker_compose_utils_mod.DockerImageBuildError = DockerImageBuildError
    docker_compose_utils_mod.DockerImagePreparationBacklogError = (
        DockerImagePreparationBacklogError
    )
    docker_compose_utils_mod.DockerImageTransientBuildError = (
        DockerImageTransientBuildError
    )
    docker_compose_utils_mod.TaskImageBlacklistedError = TaskImageBlacklistedError
    docker_compose_utils_mod.docker_image_build_status = lambda: {
        "active": 0,
        "waiting": 0,
    }

    monkeypatch.setitem(sys.modules, "uvicorn", types.ModuleType("uvicorn"))
    monkeypatch.setitem(sys.modules, "fastapi", fastapi_mod)
    monkeypatch.setitem(sys.modules, "fastapi.responses", responses_mod)
    monkeypatch.setitem(sys.modules, "harborrl.types", custom_types_mod)
    monkeypatch.setitem(sys.modules, "harborrl.http_server", request_utils_mod)
    monkeypatch.setitem(sys.modules, "harborrl.environments.terminal.runtime", terminal_env_mod)
    monkeypatch.setitem(
        sys.modules, "harborrl.environments.terminal.docker_compose", docker_compose_utils_mod
    )
    sys.modules.pop("harborrl.platform.worker_app", None)
    return importlib.import_module("harborrl.platform.worker_app")


def _new_pool(pool_server_mod, env: _DummyEnv, tmp_path: Path):
    class TestWorkerPool(pool_server_mod.WorkerPool):
        def _new_env(self):
            return env

    return TestWorkerPool(
        max_tasks=4,
        max_runs_per_task=4,
        run_idle_ttl=1,
        output_root=str(tmp_path),
        default_timeouts=_TaskTimeouts(),
        max_concurrent_closes=2,
    )


def test_pressure_guard_rejects_allocate_on_low_pids_headroom(monkeypatch):
    pool_server = _install_import_stubs(monkeypatch)
    monkeypatch.setenv("WORKER_DISK_GUARD_ENABLED", "0")
    monkeypatch.setenv("WORKER_PIDS_PAUSE_ALLOCATE_PCT", "99")
    monkeypatch.setenv("WORKER_PIDS_MIN_FREE_ALLOCATE", "6000")
    monkeypatch.setattr(
        importlib.import_module("harborrl.platform.worker_admission"),
        "worker_pressure_stats",
        lambda *args, **kwargs: {
            "procs": 100,
            "tasks": 10000,
            "pids_current": 10000,
            "pids_max": 15511,
            "pids_pct": 20.0,
            "zombies": 0,
            "dockerd": 1,
            "containerd": 1,
            "shim": 0,
            "runc": 0,
            "docker_cli_procs": 0,
            "docker_cli_ok": True,
        },
    )

    try:
        pool_server.assert_worker_has_capacity_for_docker(phase="allocate")
    except pool_server.ResourcePressureError as exc:
        assert exc.code == "WORKER_PIDS_HEADROOM_LOW"
        assert exc.details["pids_free"] == 5511
    else:
        raise AssertionError("expected low pids headroom to reject allocate")


def test_pressure_guard_rejects_reset_on_low_pids_headroom(monkeypatch):
    pool_server = _install_import_stubs(monkeypatch)
    monkeypatch.setenv("WORKER_DISK_GUARD_ENABLED", "0")
    monkeypatch.setenv("WORKER_PIDS_REJECT_RESET_PCT", "99")
    monkeypatch.setenv("WORKER_PIDS_MIN_FREE_RESET", "4000")
    monkeypatch.setattr(
        importlib.import_module("harborrl.platform.worker_admission"),
        "worker_pressure_stats",
        lambda *args, **kwargs: {
            "procs": 100,
            "tasks": 12511,
            "pids_current": 12511,
            "pids_max": 15511,
            "pids_pct": 20.0,
            "zombies": 0,
            "dockerd": 1,
            "containerd": 1,
            "shim": 0,
            "runc": 0,
            "docker_cli_procs": 0,
            "docker_cli_ok": True,
        },
    )

    try:
        pool_server.assert_worker_has_capacity_for_docker(phase="reset")
    except pool_server.ResourcePressureError as exc:
        assert exc.code == "WORKER_PIDS_HEADROOM_LOW"
        assert exc.details["pids_free"] == 3000
    else:
        raise AssertionError("expected low pids headroom to reject reset")


def test_close_allocated_run_cleans_up_without_unpack_error(monkeypatch, tmp_path):
    async def _case():
        pool_server = _install_import_stubs(monkeypatch)
        env = _DummyEnv()
        pool = _new_pool(pool_server, env, tmp_path)

        lease = await pool.allocate("task")
        lease_id = lease["lease_id"]

        assert await pool.close_run(lease_id, reason="test_close") is True
        if pool._closing_tasks:
            await asyncio.gather(*pool._closing_tasks, return_exceptions=False)

        assert lease_id not in pool._run_to_task
        assert env.close_count == 1
        assert (await pool.status())["recent_close_failures"] == []

    asyncio.run(_case())


def test_status_reports_worker_lifecycle_latency(monkeypatch, tmp_path):
    async def _case():
        pool_server = _install_import_stubs(monkeypatch)
        env = _DummyEnv()
        pool = _new_pool(pool_server, env, tmp_path)

        lease = await pool.allocate("task")
        lease_id = lease["lease_id"]
        assert await pool.exec_tool(lease_id, "noop", {}) == "observation"
        score, details = await pool.evaluate(lease_id)
        assert score == 1.0
        assert details is None

        status = await pool.status()
        lifecycle = status["lifecycle_latency_sec"]
        assert lifecycle["history_size"] >= 16
        assert lifecycle["exec_tool"]["count"] == 1
        assert lifecycle["exec_tool"]["success"] == 1
        assert lifecycle["evaluate"]["count"] == 1
        assert lifecycle["evaluate"]["success"] == 1
        assert lifecycle["reset"]["count"] == 0

        assert await pool.close_run(lease_id, reason="latency_metrics") is True
        if pool._closing_tasks:
            await asyncio.gather(*pool._closing_tasks, return_exceptions=False)
        lifecycle = (await pool.status())["lifecycle_latency_sec"]
        assert lifecycle["close"]["count"] == 1
        assert lifecycle["close"]["success"] == 1
        for name in ("exec_tool", "evaluate", "close"):
            assert lifecycle[name]["p95"] >= 0.0

    asyncio.run(_case())


def test_global_run_cap_applies_across_task_keys_and_releases_after_close(
    monkeypatch, tmp_path
):
    async def _case():
        pool_server = _install_import_stubs(monkeypatch)
        env = _DummyEnv()

        class TestWorkerPool(pool_server.WorkerPool):
            def _new_env(self):
                return env

        pool = TestWorkerPool(
            max_tasks=8,
            max_runs_per_task=8,
            max_total_runs=2,
            run_idle_ttl=1,
            output_root=str(tmp_path),
            default_timeouts=_TaskTimeouts(),
            max_concurrent_closes=2,
        )
        first = await pool.allocate("task-a")
        await pool.allocate("task-b")

        with pytest.raises(pool_server.CapacityError) as exc_info:
            await pool.allocate("task-c")
        assert exc_info.value.code == "TOTAL_RUN_SLOTS_EXHAUSTED"
        assert (await pool.status())["max_total_runs"] == 2

        assert await pool.close_run(first["lease_id"], reason="release-cap")
        if pool._closing_tasks:
            await asyncio.gather(*pool._closing_tasks, return_exceptions=False)
        third = await pool.allocate("task-c")
        assert third["lease_id"] in pool._run_to_task

    asyncio.run(_case())


def test_global_run_cap_counts_lease_until_async_cleanup_finishes(
    monkeypatch, tmp_path
):
    class BlockingCloseEnv(_DummyEnv):
        def __init__(self):
            super().__init__()
            self.close_started = asyncio.Event()
            self.release_close = asyncio.Event()

        async def close(self):
            self.close_count += 1
            self.close_started.set()
            await self.release_close.wait()

    async def _case():
        pool_server = _install_import_stubs(monkeypatch)
        env = BlockingCloseEnv()

        class TestWorkerPool(pool_server.WorkerPool):
            def _new_env(self):
                return env

        pool = TestWorkerPool(
            max_tasks=2,
            max_runs_per_task=2,
            max_total_runs=1,
            run_idle_ttl=1,
            output_root=str(tmp_path),
            default_timeouts=_TaskTimeouts(),
            max_concurrent_closes=1,
        )
        first = await pool.allocate("task-a")
        assert await pool.close_run(first["lease_id"], reason="replace")
        await env.close_started.wait()

        status = await pool.status()
        assert status["total_active_runs"] == 0
        assert status["total_retiring_runs"] == 1
        assert status["total_reserved_runs"] == 1
        with pytest.raises(pool_server.CapacityError) as exc_info:
            await pool.allocate("task-b")
        assert exc_info.value.code == "TOTAL_RUN_SLOTS_EXHAUSTED"

        env.release_close.set()
        await asyncio.gather(*list(pool._closing_tasks), return_exceptions=False)
        status = await pool.status()
        assert status["total_retiring_runs"] == 0
        assert status["total_reserved_runs"] == 0
        second = await pool.allocate("task-b")
        assert second["lease_id"] in pool._run_to_task

    asyncio.run(_case())


def test_close_cleanup_failure_is_retained_in_worker_status(monkeypatch, tmp_path):
    class FailingCleanupEnv(_DummyEnv):
        async def close(self):
            self.close_count += 1
            raise RuntimeError("close postcondition failed")

        async def force_cleanup(self, reason="external"):
            self.force_cleanup_reason = reason
            raise RuntimeError("force cleanup postcondition failed")

    async def _case():
        pool_server = _install_import_stubs(monkeypatch)
        env = FailingCleanupEnv()
        pool = _new_pool(pool_server, env, tmp_path)
        lease = await pool.allocate("task")
        lease_id = lease["lease_id"]

        assert await pool.close_run(lease_id, reason="test_close") is True
        tasks = list(pool._closing_tasks)
        assert tasks
        await asyncio.gather(*tasks, return_exceptions=False)

        status = await pool.status()
        assert status["pending_closes"] == 0
        assert status["recent_close_failures"] == [
            {
                "lease_id": lease_id,
                "task_key": "task",
                "reason": "close_exception",
                "error": "RuntimeError: force cleanup postcondition failed",
                "timestamp": status["recent_close_failures"][0]["timestamp"],
            }
        ]

    asyncio.run(_case())


def test_idle_reaper_skips_in_flight_reset(monkeypatch, tmp_path):
    async def _case():
        pool_server = _install_import_stubs(monkeypatch)
        env = _DummyEnv()
        pool = _new_pool(pool_server, env, tmp_path)

        lease = await pool.allocate("task")
        lease_id = lease["lease_id"]
        reset_task = asyncio.create_task(
            pool.reset(
                lease_id,
                {"task_name": "task", "task_path": "task", "instruction": "do it"},
                {"uid": "u1", "log_dir": str(tmp_path)},
            )
        )
        await env.reset_started.wait()

        async with pool._lock:
            run_slot = pool._get_run_slot(lease_id)
            run_slot.last_used_ts = time.time() - 100
            expired = pool._reap_idle_locked()

        assert expired == []
        assert lease_id in pool._run_to_task
        assert env.close_count == 0

        env.release_reset.set()
        await reset_task
        assert lease_id in pool._run_to_task
        assert env.close_count == 0

    asyncio.run(_case())


def test_reset_admission_backlog_does_not_mark_waiter_resetting(monkeypatch, tmp_path):
    async def _case():
        monkeypatch.setenv("WORKER_MAX_CONCURRENT_RESETS", "1")
        monkeypatch.setenv("WORKER_RESET_ADMISSION_TIMEOUT", "0.05")
        pool_server = _install_import_stubs(monkeypatch)
        env = _DummyEnv()
        pool = _new_pool(pool_server, env, tmp_path)

        first = await pool.allocate("task-a")
        second = await pool.allocate("task-b")
        first_reset = asyncio.create_task(
            pool.reset(
                first["lease_id"],
                {"task_name": "a", "task_path": "seta_env/a", "instruction": "x"},
            )
        )
        await env.reset_started.wait()

        try:
            await pool.reset(
                second["lease_id"],
                {"task_name": "b", "task_path": "seta_env/b", "instruction": "x"},
            )
        except pool_server.ResetAdmissionBacklogError:
            pass
        else:
            raise AssertionError("expected reset admission backlog")

        status = await pool.status()
        assert status["phase_counts"].get("resetting") == 1
        assert status["phase_counts"].get("allocated") == 1
        assert status["reset_admission"]["rejected"] == 1

        env.release_reset.set()
        await first_reset

    asyncio.run(_case())


def test_cancelled_reset_waiter_leaves_admission_queue(monkeypatch, tmp_path):
    async def _case():
        monkeypatch.setenv("WORKER_MAX_CONCURRENT_RESETS", "1")
        monkeypatch.setenv("WORKER_RESET_ADMISSION_TIMEOUT", "10")
        pool_server = _install_import_stubs(monkeypatch)
        env = _DummyEnv()
        pool = _new_pool(pool_server, env, tmp_path)

        first = await pool.allocate("task-a")
        second = await pool.allocate("task-b")
        first_reset = asyncio.create_task(
            pool.reset(
                first["lease_id"],
                {"task_name": "a", "task_path": "seta_env/a", "instruction": "x"},
            )
        )
        try:
            await env.reset_started.wait()

            second_reset = asyncio.create_task(
                pool.reset(
                    second["lease_id"],
                    {"task_name": "b", "task_path": "seta_env/b", "instruction": "x"},
                )
            )
            for _ in range(100):
                status = await pool.status()
                if status["reset_admission"]["waiting"] == 1:
                    break
                await asyncio.sleep(0.01)
            assert status["reset_admission"]["waiting"] == 1

            second_reset.cancel()
            try:
                await second_reset
            except TimeoutError as exc:
                assert "WORKER_RESET_CANCELLED" in str(exc)
            else:
                raise AssertionError("expected reset cancellation timeout")

            for _ in range(100):
                status = await pool.status()
                if status["reset_admission"]["waiting"] == 0:
                    break
                await asyncio.sleep(0.01)
            assert status["reset_admission"]["waiting"] == 0
            assert status["phase_counts"].get("resetting") == 1
            assert status["phase_counts"].get("allocated") == 1
            async with pool._lock:
                second_slot = pool._get_run_slot(second["lease_id"])
                assert second_slot.reset_future is None
        finally:
            env.release_reset.set()
        await first_reset

    asyncio.run(_case())


def test_admission_waiting_reset_is_not_reaped_and_close_joins_future(
    monkeypatch, tmp_path
):
    async def _case():
        monkeypatch.setenv("WORKER_MAX_CONCURRENT_RESETS", "1")
        monkeypatch.setenv("WORKER_RESET_ADMISSION_TIMEOUT", "10")
        monkeypatch.setenv("WORKER_CLOSE_REQUESTED_FORCE_RELEASE_AFTER", "0")
        pool_server = _install_import_stubs(monkeypatch)
        env = _DummyEnv()
        pool = _new_pool(pool_server, env, tmp_path)
        await pool._reset_admission_sem.acquire()

        lease_id = (await pool.allocate("task"))["lease_id"]
        reset_request = asyncio.create_task(
            pool.reset(
                lease_id,
                {"task_name": "task", "task_path": "task", "instruction": "x"},
            )
        )
        for _ in range(100):
            status = await pool.status()
            if status["reset_admission"]["waiting"] == 1:
                break
            await asyncio.sleep(0.01)
        assert status["reset_admission"]["waiting"] == 1

        async with pool._lock:
            run_slot = pool._get_run_slot(lease_id)
            reset_future = run_slot.reset_future
            run_slot.last_used_ts = time.time() - 100
            assert pool._reap_idle_locked() == []

        assert await pool.close_run(lease_id, reason="test_close") is True
        await asyncio.gather(reset_request, return_exceptions=True)
        release_task = pool._close_requested_release_tasks.get(lease_id)
        if release_task is not None:
            await asyncio.gather(release_task, return_exceptions=False)
        if pool._closing_tasks:
            await asyncio.gather(*pool._closing_tasks, return_exceptions=True)
        if pool._force_cleanup_tasks:
            await asyncio.gather(*pool._force_cleanup_tasks, return_exceptions=True)

        assert reset_future is not None and reset_future.done()
        assert not env.reset_started.is_set()
        assert lease_id not in pool._run_to_task
        pool._reset_admission_sem.release()

    asyncio.run(_case())


def test_finishing_other_op_does_not_drop_admission_waiting_reset(
    monkeypatch, tmp_path
):
    class BlockingExecEnv(_DummyEnv):
        def __init__(self):
            super().__init__()
            self.exec_started = asyncio.Event()
            self.release_exec = asyncio.Event()

        async def exec_tool(self, *_args, **_kwargs):
            self.exec_started.set()
            await self.release_exec.wait()
            return "ok"

    async def _case():
        monkeypatch.setenv("WORKER_MAX_CONCURRENT_RESETS", "1")
        monkeypatch.setenv("WORKER_RESET_ADMISSION_TIMEOUT", "10")
        monkeypatch.setenv("WORKER_CLOSE_REQUESTED_FORCE_RELEASE", "0")
        pool_server = _install_import_stubs(monkeypatch)
        env = BlockingExecEnv()
        pool = _new_pool(pool_server, env, tmp_path)
        await pool._reset_admission_sem.acquire()
        lease_id = (await pool.allocate("task"))["lease_id"]

        reset_request = asyncio.create_task(
            pool.reset(
                lease_id,
                {"task_name": "task", "task_path": "task", "instruction": "x"},
            )
        )
        for _ in range(100):
            status = await pool.status()
            if status["reset_admission"]["waiting"] == 1:
                break
            await asyncio.sleep(0.01)
        assert status["reset_admission"]["waiting"] == 1

        exec_request = asyncio.create_task(pool.exec_tool(lease_id, "noop", {}))
        await env.exec_started.wait()
        assert await pool.close_run(lease_id, reason="test_close") is True
        env.release_exec.set()
        await exec_request

        async with pool._lock:
            run_slot = pool._get_run_slot(lease_id)
            assert run_slot.reset_future is not None
            assert not run_slot.reset_future.done()
        assert lease_id in pool._run_to_task
        assert env.close_count == 0

        reset_request.cancel()
        await asyncio.gather(reset_request, return_exceptions=True)
        if pool._closing_tasks:
            await asyncio.gather(*pool._closing_tasks, return_exceptions=True)
        assert lease_id not in pool._run_to_task
        pool._reset_admission_sem.release()

    asyncio.run(_case())


def test_close_during_reset_is_deferred_until_in_flight_finishes(monkeypatch, tmp_path):
    async def _case():
        pool_server = _install_import_stubs(monkeypatch)
        env = _DummyEnv()
        pool = _new_pool(pool_server, env, tmp_path)

        lease = await pool.allocate("task")
        lease_id = lease["lease_id"]
        reset_task = asyncio.create_task(
            pool.reset(
                lease_id,
                {"task_name": "task", "task_path": "task", "instruction": "do it"},
                {"uid": "u1", "log_dir": str(tmp_path)},
            )
        )
        await env.reset_started.wait()

        assert await pool.close_run(lease_id, reason="test_close") is True
        async with pool._lock:
            run_slot = pool._get_run_slot(lease_id)
            assert run_slot.close_requested is True
            assert run_slot.in_flight_ops == 1
            assert run_slot.phase == "closing_requested"

        env.release_reset.set()
        await reset_task
        if pool._closing_tasks:
            await asyncio.gather(*pool._closing_tasks, return_exceptions=True)

        assert lease_id not in pool._run_to_task
        assert env.close_count == 1

    asyncio.run(_case())


def test_close_during_reset_force_releases_after_delay(monkeypatch, tmp_path):
    class JoinAwareEnv(_DummyEnv):
        def __init__(self):
            super().__init__()
            self.reset_cancelled = asyncio.Event()
            self.release_cancelled_reset = asyncio.Event()

        async def reset(self, **_kwargs):
            self.reset_started.set()
            try:
                await self.release_reset.wait()
            except asyncio.CancelledError:
                self.reset_cancelled.set()
                await self.release_cancelled_reset.wait()
                raise

    async def _case():
        pool_server = _install_import_stubs(monkeypatch)
        monkeypatch.setenv("WORKER_CLOSE_REQUESTED_FORCE_RELEASE_AFTER", "0")
        env = JoinAwareEnv()
        pool = _new_pool(pool_server, env, tmp_path)

        lease = await pool.allocate("task")
        lease_id = lease["lease_id"]
        reset_task = asyncio.create_task(
            pool.reset(
                lease_id,
                {"task_name": "task", "task_path": "task", "instruction": "do it"},
                {"uid": "u1", "log_dir": str(tmp_path)},
            )
        )
        await env.reset_started.wait()

        assert await pool.close_run(lease_id, reason="test_close") is True
        await env.reset_cancelled.wait()

        # Cancellation has reached reset, but cleanup and lease removal must
        # wait until reset itself has completely unwound.
        assert lease_id in pool._run_to_task
        assert not hasattr(env, "force_cleanup_reason")

        env.release_cancelled_reset.set()
        await asyncio.gather(reset_task, return_exceptions=True)
        release_task = pool._close_requested_release_tasks.get(lease_id)
        if release_task is not None:
            await asyncio.gather(release_task, return_exceptions=False)
        if pool._closing_tasks:
            await asyncio.gather(*pool._closing_tasks, return_exceptions=True)
        if pool._force_cleanup_tasks:
            await asyncio.gather(*pool._force_cleanup_tasks, return_exceptions=True)

        assert lease_id not in pool._run_to_task
        assert env.close_count == 1 or env.force_cleanup_reason == (
            "close_requested_force_release:test_close"
        )

    asyncio.run(_case())


def test_reset_timeout_joins_cancelled_reset_before_lease_drop(monkeypatch, tmp_path):
    class JoinAwareEnv(_DummyEnv):
        def __init__(self):
            super().__init__()
            self.reset_cancelled = asyncio.Event()
            self.release_cancelled_reset = asyncio.Event()

        async def reset(self, **_kwargs):
            self.reset_started.set()
            try:
                await self.release_reset.wait()
            except asyncio.CancelledError:
                self.reset_cancelled.set()
                await self.release_cancelled_reset.wait()
                raise

    async def _case():
        monkeypatch.setenv("WORKER_RESET_OPERATION_TIMEOUT", "0.2")
        monkeypatch.setenv("WORKER_RESET_WARN_AFTER", "0.1")
        pool_server = _install_import_stubs(monkeypatch)
        env = JoinAwareEnv()
        pool = _new_pool(pool_server, env, tmp_path)
        lease_id = (await pool.allocate("task"))["lease_id"]

        reset_task = asyncio.create_task(
            pool.reset(
                lease_id,
                {"task_name": "task", "task_path": "task", "instruction": "do it"},
            )
        )
        await env.reset_cancelled.wait()

        assert not reset_task.done()
        assert lease_id in pool._run_to_task
        assert not pool._force_cleanup_tasks

        env.release_cancelled_reset.set()
        try:
            await reset_task
        except TimeoutError as exc:
            assert "WORKER_RESET_TIMEOUT" in str(exc)
        else:
            raise AssertionError("expected reset timeout")

        assert lease_id not in pool._run_to_task
        if pool._force_cleanup_tasks:
            await asyncio.gather(*pool._force_cleanup_tasks, return_exceptions=True)

    asyncio.run(_case())


def test_status_and_readyz_report_stale_allocated_run(monkeypatch, tmp_path):
    async def _case():
        pool_server = _install_import_stubs(monkeypatch)
        monkeypatch.setenv("WORKER_DISK_GUARD_ENABLED", "0")
        monkeypatch.setenv("WORKER_PRESSURE_GUARD_ENABLED", "0")
        monkeypatch.setenv("WORKER_ALLOCATED_TTL", "10")
        env = _DummyEnv()
        pool = _new_pool(pool_server, env, tmp_path)

        lease = await pool.allocate("task")
        lease_id = lease["lease_id"]
        async with pool._lock:
            run_slot = pool._get_run_slot(lease_id)
            run_slot.created_ts = time.time() - 20

        status = await pool.status()
        assert status["phase_counts"] == {"allocated": 1}
        assert status["stale_runs"][0]["lease_id"] == lease_id
        assert status["stale_runs"][0]["reason"] == "allocated_ttl_exceeded"

        pool_server.POOL = pool
        response = await pool_server.readyz()
        assert response.status_code == 503
        assert response["code"] == "WORKER_STALE_RUNS"

    asyncio.run(_case())


def test_allocate_does_not_repair_close_requested_live_operation(monkeypatch, tmp_path):
    async def _case():
        pool_server = _install_import_stubs(monkeypatch)
        monkeypatch.setenv("WORKER_DISK_GUARD_ENABLED", "0")
        monkeypatch.setenv("WORKER_PRESSURE_GUARD_ENABLED", "0")
        monkeypatch.setenv("WORKER_AUTO_REPAIR_ON_CAPACITY", "1")
        env = _DummyEnv()

        class OneTaskPool(pool_server.WorkerPool):
            def _new_env(self):
                return env

        pool = OneTaskPool(
            max_tasks=1,
            max_runs_per_task=4,
            run_idle_ttl=1,
            output_root=str(tmp_path),
            default_timeouts=_TaskTimeouts(),
            max_concurrent_closes=2,
        )
        pool_server.POOL = pool

        old_lease = (await pool.allocate("old-task"))["lease_id"]
        async with pool._lock:
            old_slot = pool._get_run_slot(old_lease)
            old_slot.phase = "closing_requested"
            old_slot.close_requested = True
            old_slot.close_reason = "test"
            old_slot.close_requested_ts = time.time() - 5
            old_slot.in_flight_ops = 1
            old_slot.active_op = "reset"

        async def _payload(_request):
            return {"task_key": "new-task"}

        pool_server.json_payload = _payload
        response = await pool_server.allocate(object())

        assert response.status_code == 429
        assert response["auto_repair"]["close_requested"]["repaired_count"] == 0
        assert old_lease in pool._run_to_task

        async with pool._lock:
            old_slot = pool._get_run_slot(old_lease)
            old_slot.in_flight_ops = 0
            old_slot.active_op = None

        response = await pool_server.allocate(object())
        assert response.status_code == 200
        assert response["ok"] is True
        assert response["auto_repair"]["close_requested"]["repaired_count"] == 1
        assert old_lease not in pool._run_to_task
        assert response["lease_id"] in pool._run_to_task
        if pool._force_cleanup_tasks:
            await asyncio.gather(*pool._force_cleanup_tasks, return_exceptions=True)

    asyncio.run(_case())


def test_repair_stale_run_force_cleans_inflight_close_requested(monkeypatch, tmp_path):
    async def _case():
        pool_server = _install_import_stubs(monkeypatch)
        monkeypatch.setenv("WORKER_CLOSING_REQUESTED_TTL", "10")
        env = _DummyEnv()
        pool = _new_pool(pool_server, env, tmp_path)

        lease = await pool.allocate("task")
        lease_id = lease["lease_id"]
        async with pool._lock:
            run_slot = pool._get_run_slot(lease_id)
            run_slot.phase = "closing_requested"
            run_slot.close_requested = True
            run_slot.close_reason = "test"
            run_slot.close_requested_ts = time.time() - 20
            run_slot.in_flight_ops = 1
            run_slot.active_op = "exec_tool"

        status = await pool.status()
        assert status["stale_runs"][0]["lease_id"] == lease_id
        result = await pool.repair_stale_runs(reason="test", min_age=0, max_repairs=10)

        assert result["repaired"] is True
        assert result["repaired_count"] == 1
        assert result["repaired_runs"][0]["lease_id"] == lease_id
        assert env.force_cleanup_reason == "repair_stale_runs:test"
        status = await pool.status()
        assert status["total_active_runs"] == 0
        assert status["stale_runs"] == []

    asyncio.run(_case())


def test_repair_resetting_rejects_retry_and_checks_reset_generation(
    monkeypatch, tmp_path
):
    async def _case():
        pool_server = _install_import_stubs(monkeypatch)
        env = _DummyEnv()
        pool = _new_pool(pool_server, env, tmp_path)
        lease_id = (await pool.allocate("task"))["lease_id"]
        old_future = asyncio.create_task(asyncio.sleep(3600))
        async with pool._lock:
            run_slot = pool._get_run_slot(lease_id)
            run_slot.phase = "resetting"
            run_slot.reset_started_ts = time.time() - 60
            run_slot.reset_future = old_future

        replacement: list[asyncio.Task] = []

        async def _join_and_replace(task, **_kwargs):
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            with pytest.raises(RuntimeError, match="closing"):
                await pool.reset(
                    lease_id,
                    {"task_name": "task", "task_path": "task", "instruction": "x"},
                )
            new_future = asyncio.create_task(asyncio.sleep(3600))
            replacement.append(new_future)
            async with pool._lock:
                current = pool._get_run_slot(lease_id)
                current.reset_future = new_future
                current.close_requested = False
                current.drop_scheduled = False
                current.phase = "resetting"
            return True

        monkeypatch.setattr(pool, "_cancel_and_join_reset_task", _join_and_replace)
        result = await pool.repair_resetting_runs(
            reason="test", min_age=0, max_repairs=1
        )

        assert result["repaired_count"] == 0
        assert lease_id in pool._run_to_task
        async with pool._lock:
            assert pool._get_run_slot(lease_id).reset_future is replacement[0]

        replacement[0].cancel()
        await asyncio.gather(replacement[0], return_exceptions=True)
        await pool.close_run(lease_id, reason="test_cleanup")
        if pool._closing_tasks:
            await asyncio.gather(*pool._closing_tasks, return_exceptions=True)

    asyncio.run(_case())


def test_reaper_removes_stale_allocated_run(monkeypatch, tmp_path):
    async def _case():
        pool_server = _install_import_stubs(monkeypatch)
        monkeypatch.setenv("WORKER_ALLOCATED_TTL", "10")
        env = _DummyEnv()
        pool = _new_pool(pool_server, env, tmp_path)

        lease = await pool.allocate("task")
        lease_id = lease["lease_id"]
        async with pool._lock:
            run_slot = pool._get_run_slot(lease_id)
            run_slot.created_ts = time.time() - 20
            expired = pool._reap_idle_locked()

        assert len(expired) == 1
        assert expired[0][1] == lease_id
        assert lease_id not in pool._run_to_task

        for task_key, run_id, run_slot in expired:
            pool._schedule_close(task_key, run_id, run_slot, reason="test stale reap")
        if pool._closing_tasks:
            await asyncio.gather(*pool._closing_tasks, return_exceptions=False)
        assert env.close_count == 1

    asyncio.run(_case())

def test_pending_close_repair_allows_negative_active_limit(monkeypatch, tmp_path):
    async def _case():
        pool_server = _install_import_stubs(monkeypatch)
        env = _DummyEnv()
        pool = _new_pool(pool_server, env, tmp_path)
        await pool.allocate("task")

        sleeper = asyncio.create_task(asyncio.sleep(3600))
        pool._closing_tasks.add(sleeper)
        pool._closing_task_started[sleeper] = time.time() - 60
        try:
            result = await pool.repair_pending_closes(
                reason="test",
                max_active_runs=-1,
                cancel_timeout=0.1,
                min_age=0,
            )
        finally:
            sleeper.cancel()
            await asyncio.gather(sleeper, return_exceptions=True)

        assert result["repaired"] is True
        assert result["cancelled"] == 1
        assert result["pending_after"] == 0

    asyncio.run(_case())


def test_rollout_probe_resets_executes_and_closes(monkeypatch, tmp_path):
    async def _case():
        pool_server = _install_import_stubs(monkeypatch)
        monkeypatch.setenv("WORKER_DISK_GUARD_ENABLED", "0")
        monkeypatch.setenv("WORKER_PRESSURE_GUARD_ENABLED", "0")
        env = _DummyEnv()
        env.release_reset.set()
        pool = _new_pool(pool_server, env, tmp_path)
        pool_server.POOL = pool

        async def _payload(_request):
            return {
                "task_key": "probe-task",
                "task_meta": {
                    "task_name": "probe-task",
                    "task_path": "probe-task",
                    "instruction": "probe",
                },
                "run_ctx": {"uid": "probe", "log_dir": str(tmp_path)},
                "tool_call": {"name": "noop", "arguments": {}},
            }

        pool_server.json_payload = _payload
        response = await pool_server.probe_rollout(object())

        assert response.status_code == 200
        assert response["ok"] is True
        assert response["exec"]["tool_name"] == "noop"
        if pool._closing_tasks:
            await asyncio.gather(*pool._closing_tasks, return_exceptions=False)
        assert env.close_count == 1
        assert response["lease_id"] not in pool._run_to_task

    asyncio.run(_case())


def test_shutdown_joins_active_reset_before_removing_lease_or_cleanup(
    monkeypatch, tmp_path
):
    class SlowCancelEnv(_DummyEnv):
        def __init__(self) -> None:
            super().__init__()
            self.cancel_seen = asyncio.Event()
            self.events: list[str] = []

        async def reset(self, **_kwargs):
            self.events.append("reset_started")
            self.reset_started.set()
            try:
                await self.release_reset.wait()
            except asyncio.CancelledError:
                self.cancel_seen.set()
                await self.release_reset.wait()
            self.events.append("reset_finished")
            return "user", []

        async def close(self):
            self.events.append("close")
            await super().close()

        async def force_cleanup(self, reason="external"):
            self.events.append("force_cleanup")
            await super().force_cleanup(reason=reason)

    async def _case():
        pool_server = _install_import_stubs(monkeypatch)
        env = SlowCancelEnv()
        pool = _new_pool(pool_server, env, tmp_path)
        lease = await pool.allocate("task")
        lease_id = lease["lease_id"]
        reset_request = asyncio.create_task(
            pool.reset(
                lease_id,
                {"task_name": "task", "task_path": "task", "instruction": "x"},
            )
        )
        await env.reset_started.wait()

        shutdown_task = asyncio.create_task(pool.shutdown())
        await asyncio.wait_for(env.cancel_seen.wait(), timeout=1.0)
        await asyncio.sleep(0)
        assert not shutdown_task.done()
        assert lease_id in pool._run_to_task
        with pytest.raises(pool_server.CapacityError, match="shutting down"):
            await pool.allocate("new-task")
        with pytest.raises(RuntimeError, match="shutting down"):
            await pool.reset(
                lease_id,
                {"task_name": "task", "task_path": "task", "instruction": "x"},
            )
        with pytest.raises(RuntimeError, match="shutting down"):
            await pool.exec_tool(lease_id, "noop", {})

        env.release_reset.set()
        await asyncio.wait_for(shutdown_task, timeout=2.0)
        await asyncio.gather(reset_request, return_exceptions=True)

        assert reset_request.done()
        assert lease_id not in pool._run_to_task
        assert env.events.index("reset_finished") < env.events.index("close")
        assert env.events.index("reset_finished") < env.events.index("force_cleanup")
        assert env.close_count == 1

    asyncio.run(_case())


def test_shutdown_has_deadline_for_cancellation_resistant_reset(monkeypatch, tmp_path):
    class CancellationResistantEnv(_DummyEnv):
        def __init__(self) -> None:
            super().__init__()
            self.cancel_seen = asyncio.Event()
            self.reset_finished = asyncio.Event()
            self.force_cleanup_count = 0
            self.thread_started = threading.Event()
            self.thread_release = threading.Event()

        async def reset(self, **_kwargs):
            self.reset_started.set()

            def _blocking_reset():
                self.thread_started.set()
                self.thread_release.wait()
                return "user", []

            reset_thread = asyncio.create_task(asyncio.to_thread(_blocking_reset))
            try:
                return await asyncio.shield(reset_thread)
            except asyncio.CancelledError:
                self.cancel_seen.set()
                while not reset_thread.done():
                    try:
                        await asyncio.shield(reset_thread)
                    except asyncio.CancelledError:
                        current = asyncio.current_task()
                        if current is not None and hasattr(current, "uncancel"):
                            current.uncancel()
                        continue
                await asyncio.gather(reset_thread, return_exceptions=True)
                raise
            finally:
                self.reset_finished.set()

        async def force_cleanup(self, reason="external"):
            self.force_cleanup_count += 1
            await super().force_cleanup(reason=reason)

    async def _case():
        pool_server = _install_import_stubs(monkeypatch)
        monkeypatch.setenv("WORKER_RESET_CANCEL_JOIN_TIMEOUT", "0.05")
        monkeypatch.setenv("WORKER_SHUTDOWN_RESET_JOIN_TIMEOUT", "0.10")
        monkeypatch.setenv("WORKER_SHUTDOWN_CLOSE_TASKS_TIMEOUT", "0.05")
        env = CancellationResistantEnv()
        pool = _new_pool(pool_server, env, tmp_path)
        lease = await pool.allocate("task")
        lease_id = lease["lease_id"]
        reset_request = asyncio.create_task(
            pool.reset(
                lease_id,
                {"task_name": "task", "task_path": "task", "instruction": "x"},
            )
        )
        await env.reset_started.wait()
        for _ in range(100):
            if env.thread_started.is_set():
                break
            await asyncio.sleep(0.01)
        assert env.thread_started.is_set()

        await asyncio.wait_for(pool.shutdown(), timeout=5.0)

        assert env.cancel_seen.is_set()
        assert not env.thread_release.is_set()
        assert lease_id in pool._run_to_task
        run_slot = pool._get_run_slot(lease_id)
        assert run_slot.reset_quarantined is True
        assert run_slot.phase == "reset_quarantined"
        assert env.force_cleanup_count == 0

        env.thread_release.set()
        # ThreadPool scheduling can be delayed on loaded Python 3.10 CI hosts.
        await asyncio.wait_for(env.reset_finished.wait(), timeout=5.0)
        await asyncio.gather(reset_request, return_exceptions=True)
        for _ in range(500):
            if lease_id not in pool._run_to_task:
                break
            await asyncio.sleep(0.01)
        assert lease_id not in pool._run_to_task
        assert env.force_cleanup_count == 1
        for _ in range(500):
            if not pool._reset_quarantine_watchers:
                break
            await asyncio.sleep(0.01)
        assert not pool._reset_quarantine_watchers

    asyncio.run(_case())


def test_cancelled_reset_is_quarantined_until_environment_reset_exits(
    monkeypatch, tmp_path
):
    class CancellationResistantEnv(_DummyEnv):
        def __init__(self) -> None:
            super().__init__()
            self.cancel_seen = asyncio.Event()
            self.force_cleanup_count = 0

        async def reset(self, **_kwargs):
            self.reset_started.set()
            try:
                await self.release_reset.wait()
            except asyncio.CancelledError:
                self.cancel_seen.set()
                await self.release_reset.wait()
            return "user", []

        async def force_cleanup(self, reason="external"):
            self.force_cleanup_count += 1
            await super().force_cleanup(reason=reason)

    async def _case():
        pool_server = _install_import_stubs(monkeypatch)
        monkeypatch.setenv("WORKER_RESET_CANCEL_JOIN_TIMEOUT", "0.05")
        env = CancellationResistantEnv()
        pool = _new_pool(pool_server, env, tmp_path)
        lease = await pool.allocate("task")
        lease_id = lease["lease_id"]
        reset_request = asyncio.create_task(
            pool.reset(
                lease_id,
                {"task_name": "task", "task_path": "task", "instruction": "x"},
            )
        )
        await env.reset_started.wait()

        reset_request.cancel()
        with pytest.raises(TimeoutError, match="WORKER_RESET_CANCELLED"):
            await reset_request

        run_slot = pool._get_run_slot(lease_id)
        assert run_slot.reset_quarantined is True
        assert env.force_cleanup_count == 0
        status = await pool.status()
        assert status["reset_quarantined_runs"] == 1
        assert status["stale_runs"][0]["reason"] == "reset_quarantined"
        with pytest.raises(pool_server.CapacityError) as exc_info:
            await pool.allocate("task")
        assert exc_info.value.code == "TASK_RESET_QUARANTINED"
        with pytest.raises(RuntimeError, match="quarantined reset"):
            await pool.exec_tool(lease_id, "noop", {})

        watcher = run_slot.reset_quarantine_watcher
        assert watcher is not None
        watcher.cancel()
        await asyncio.sleep(0)
        assert not watcher.done()
        assert lease_id in pool._run_to_task

        env.release_reset.set()
        for _ in range(100):
            if lease_id not in pool._run_to_task:
                break
            await asyncio.sleep(0.01)
        assert lease_id not in pool._run_to_task
        assert env.force_cleanup_count == 1
        for _ in range(100):
            if not pool._reset_quarantine_watchers:
                break
            await asyncio.sleep(0.01)
        assert not pool._reset_quarantine_watchers

    asyncio.run(_case())


def test_cancelled_orphan_sweep_joins_bounded_worker_thread(monkeypatch, tmp_path):
    async def _case():
        pool_server = _install_import_stubs(monkeypatch)
        monkeypatch.setenv("WORKER_ORPHAN_DOCKER_SWEEP_INTERVAL", "1")
        monkeypatch.setenv("WORKER_ORPHAN_DOCKER_SWEEP_TIMEOUT", "2")
        started = threading.Event()
        release = threading.Event()
        finished = threading.Event()
        observed: dict[str, object] = {}

        def _blocking_sweep(**kwargs):
            observed.update(kwargs)
            started.set()
            release.wait(timeout=5.0)
            finished.set()
            return 0

        monkeypatch.setattr(
            importlib.import_module("harborrl.platform.worker_pool"),
            "force_remove_orphan_docker_objects",
            _blocking_sweep,
        )
        pool = _new_pool(pool_server, _DummyEnv(), tmp_path)
        sweep_request = asyncio.create_task(
            pool._maybe_cleanup_orphan_docker_containers()
        )
        for _ in range(200):
            if started.is_set():
                break
            await asyncio.sleep(0.01)
        assert started.is_set()

        sweep_request.cancel()
        await asyncio.sleep(0.05)
        assert not sweep_request.done()
        assert not finished.is_set()

        release.set()
        with pytest.raises(asyncio.CancelledError):
            await sweep_request
        assert finished.is_set()
        assert observed["cleanup_timeout"] == 2.0

    asyncio.run(_case())


def test_orphan_sweep_builtin_timeout_is_classified_on_python310(
    monkeypatch, tmp_path
):
    async def _case():
        pool_server = _install_import_stubs(monkeypatch)
        monkeypatch.setenv("WORKER_ORPHAN_DOCKER_SWEEP_INTERVAL", "1")
        monkeypatch.setenv("WORKER_ORPHAN_DOCKER_SWEEP_TIMEOUT", "2")
        reasons: list[str] = []

        def _timed_out_sweep(**_kwargs):
            raise TimeoutError("deadline")

        monkeypatch.setattr(
            importlib.import_module("harborrl.platform.worker_pool"),
            "force_remove_orphan_docker_objects",
            _timed_out_sweep,
        )
        pool = _new_pool(pool_server, _DummyEnv(), tmp_path)
        monkeypatch.setattr(pool, "_record_orphan_sweep_failure", reasons.append)

        await pool._maybe_cleanup_orphan_docker_containers()

        assert reasons == ["timeout_after_2.0s"]

    asyncio.run(_case())
