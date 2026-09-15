from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from harborrl.environments.terminal.docker_compose import (
    DockerImageBuildError,
    DockerImagePreparationBacklogError,
    DockerImageTransientBuildError,
    TaskImageBlacklistedError,
)
from harborrl.http_server import json_payload
from harborrl.platform.worker_admission import (
    CapacityError,
    ResetAdmissionBacklogError,
    ResetInProgressError,
    ResourcePressureError,
    _env_float,
    _env_int,
    assert_worker_has_capacity_for_docker,
    docker_data_root_stats,
    worker_pressure_stats,
)
from harborrl.platform.worker_pool import WorkerPool

logger = logging.getLogger("lightrl.env.worker.app")
app = FastAPI()

POOL: WorkerPool | None = None

@app.get("/healthz")
async def healthz() -> JSONResponse:
    try:
        pending_closes = 0
        pool_status: dict[str, Any] | None = None
        if POOL is not None:
            pool_status = await POOL.status()
            pending_closes = int(pool_status.get("pending_closes", 0))
        assert_worker_has_capacity_for_docker(
            phase="health",
            pending_closes=pending_closes,
            pool_status=pool_status if POOL is not None else None,
        )
        return JSONResponse({"ok": True})
    except ResourcePressureError as exc:
        return JSONResponse(
            {
                "ok": False,
                "code": exc.code,
                "error": exc.message,
                "details": exc.details,
            },
            status_code=503,
        )


@app.get("/status")
async def status() -> JSONResponse:
    if POOL is None:
        return JSONResponse(
            {"ok": False, "error": "Pool is not initialized"}, status_code=500
        )
    disk: dict[str, Any] | None = None
    pressure: dict[str, Any] | None = None
    disk_ok = True
    disk_error: str | None = None
    pool_status = await POOL.status()
    try:
        disk = docker_data_root_stats()
        pressure = worker_pressure_stats()
        assert_worker_has_capacity_for_docker(
            phase="health",
            pending_closes=int(pool_status.get("pending_closes", 0)),
            pool_status=pool_status,
        )
    except ResourcePressureError as exc:
        disk_ok = False
        disk_error = exc.message
        pressure = exc.details
    except Exception as exc:
        disk_ok = False
        disk_error = str(exc)
    return JSONResponse(
        {
            "ok": True,
            "pool": pool_status,
            "docker_data_root": disk,
            "resource_pressure": pressure,
            "admission_ok": disk_ok,
            "admission_error": disk_error,
        }
    )


@app.get("/readyz")
async def readyz() -> JSONResponse:
    if POOL is None:
        return JSONResponse(
            {
                "ok": False,
                "code": "POOL_NOT_INITIALIZED",
                "error": "Pool is not initialized",
            },
            status_code=503,
        )

    pool_status = await POOL.status()
    try:
        assert_worker_has_capacity_for_docker(
            phase="health",
            pending_closes=int(pool_status.get("pending_closes", 0)),
            pool_status=pool_status,
        )
    except ResourcePressureError as exc:
        return JSONResponse(
            {
                "ok": False,
                "code": exc.code,
                "error": exc.message,
                "details": exc.details,
                "pool": pool_status,
            },
            status_code=503,
        )

    stale_runs = pool_status.get("stale_runs", [])
    if stale_runs:
        return JSONResponse(
            {
                "ok": False,
                "code": "WORKER_STALE_RUNS",
                "error": f"Worker has {len(stale_runs)} stale run(s)",
                "stale_runs": stale_runs[:20],
                "pool": pool_status,
            },
            status_code=503,
        )

    return JSONResponse({"ok": True, "pool": pool_status})


@app.get("/metrics")
async def metrics() -> Response:
    lines = [
        "# HELP lightrl_worker_up Worker process is serving HTTP.",
        "# TYPE lightrl_worker_up gauge",
        "lightrl_worker_up 1",
    ]
    if POOL is None:
        lines.extend(
            [
                "# HELP lightrl_worker_pool_initialized Worker pool initialization state.",
                "# TYPE lightrl_worker_pool_initialized gauge",
                "lightrl_worker_pool_initialized 0",
            ]
        )
        return Response("\n".join(lines) + "\n", media_type="text/plain")

    pool_status = await POOL.status()
    gauges = {
        "active_tasks": pool_status.get("active_tasks", 0),
        "total_active_runs": pool_status.get("total_active_runs", 0),
        "in_flight_runs": pool_status.get("in_flight_runs", 0),
        "closing_requested_runs": pool_status.get("closing_requested_runs", 0),
        "pending_closes": pool_status.get("pending_closes", 0),
        "stale_runs": len(pool_status.get("stale_runs", []) or []),
    }
    lines.extend(
        [
            "# HELP lightrl_worker_pool_initialized Worker pool initialization state.",
            "# TYPE lightrl_worker_pool_initialized gauge",
            "lightrl_worker_pool_initialized 1",
            "# HELP lightrl_worker_pool_gauge Worker pool gauges.",
            "# TYPE lightrl_worker_pool_gauge gauge",
        ]
    )
    for name, value in gauges.items():
        lines.append(
            f'lightrl_worker_pool_gauge{{name="{name}"}} {int(value or 0)}'
        )

    phase_counts = pool_status.get("phase_counts", {})
    if isinstance(phase_counts, dict):
        lines.extend(
            [
                "# HELP lightrl_worker_run_phase_count Active run count by phase.",
                "# TYPE lightrl_worker_run_phase_count gauge",
            ]
        )
        for phase, count in sorted(phase_counts.items()):
            lines.append(
                f'lightrl_worker_run_phase_count{{phase="{phase}"}} {int(count or 0)}'
            )

    try:
        pressure = worker_pressure_stats()
    except Exception as exc:
        lines.append(f'# worker_pressure_stats_error="{exc}"')
    else:
        lines.extend(
            [
                "# HELP lightrl_worker_pressure Worker pressure gauges.",
                "# TYPE lightrl_worker_pressure gauge",
            ]
        )
        for name in ("tasks", "procs", "zombies", "shim", "runc", "docker_cli_procs"):
            value = pressure.get(name)
            if isinstance(value, (int, float)):
                lines.append(f'lightrl_worker_pressure{{name="{name}"}} {value}')
        pids_pct = pressure.get("pids_pct")
        if isinstance(pids_pct, (int, float)):
            lines.append(f'lightrl_worker_pressure{{name="pids_pct"}} {pids_pct}')
        docker_cli_ok = 1 if pressure.get("docker_cli_ok") else 0
        lines.append(f'lightrl_worker_pressure{{name="docker_cli_ok"}} {docker_cli_ok}')

    return Response("\n".join(lines) + "\n", media_type="text/plain")


@app.post("/probe/rollout")
async def probe_rollout(request: Request) -> JSONResponse:
    if POOL is None:
        return JSONResponse(
            {
                "ok": False,
                "code": "POOL_NOT_INITIALIZED",
                "error": "Pool is not initialized",
            },
            status_code=503,
        )

    data = await json_payload(request)
    task_meta = data.get("task_meta")
    if not isinstance(task_meta, dict):
        return JSONResponse(
            {"ok": False, "error": "task_meta dict is required"},
            status_code=400,
        )

    task_key = str(
        data.get("task_key") or f"probe:{task_meta.get('task_name', 'unknown')}"
    )
    run_ctx_payload = (
        data.get("run_ctx") if isinstance(data.get("run_ctx"), dict) else {}
    )
    task_timeouts = data.get("task_timeouts")
    tool_call = data.get("tool_call")
    request_id = str(data.get("request_id") or f"probe-{uuid.uuid4().hex[:16]}")
    lease_id: str | None = None
    started_ts = time.time()
    exec_result: dict[str, Any] | None = None

    try:
        pool_status = await POOL.status()
        assert_worker_has_capacity_for_docker(
            phase="allocate",
            pending_closes=int(pool_status.get("pending_closes", 0)),
            pool_status=pool_status,
        )
        allocated = await POOL.allocate(task_key=task_key, request_id=request_id)
        lease_id = str(allocated["lease_id"])
        await POOL.reset(
            run_lease_id=lease_id,
            task_meta=task_meta,
            run_ctx_payload=run_ctx_payload,
            task_timeouts=task_timeouts if isinstance(task_timeouts, dict) else None,
        )
        if isinstance(tool_call, dict):
            tool_name = tool_call.get("name")
            arguments = tool_call.get("arguments")
            if isinstance(tool_name, str) and tool_name:
                observation = await POOL.exec_tool(
                    lease_id,
                    tool_name,
                    arguments=arguments if isinstance(arguments, dict) else None,
                )
                exec_result = {
                    "tool_name": tool_name,
                    "observation_len": len(observation),
                }
        return JSONResponse(
            {
                "ok": True,
                "lease_id": lease_id,
                "duration_sec": round(time.time() - started_ts, 3),
                "exec": exec_result,
            }
        )
    except ResourcePressureError as exc:
        return JSONResponse(
            {
                "ok": False,
                "code": exc.code,
                "error": exc.message,
                "details": exc.details,
                "duration_sec": round(time.time() - started_ts, 3),
            },
            status_code=503,
        )
    except ResetAdmissionBacklogError as exc:
        logger.warning(
            "Rollout probe deferred by reset admission backlog lease_id=%s "
            "task_key=%s: %s",
            lease_id,
            task_key,
            exc,
        )
        return JSONResponse(
            {
                "ok": False,
                "code": "WORKER_RESET_ADMISSION_BACKLOG",
                "error": str(exc),
                "task_name": task_meta.get("task_name"),
                "task_path": task_meta.get("task_path"),
                "duration_sec": round(time.time() - started_ts, 3),
            },
            status_code=503,
            headers={"Retry-After": os.getenv("WORKER_RESET_BACKLOG_RETRY_AFTER", "10")},
        )
    except DockerImagePreparationBacklogError as exc:
        logger.warning(
            "Rollout probe deferred by Docker image preparation backlog lease_id=%s "
            "task_key=%s: %s",
            lease_id,
            task_key,
            exc,
        )
        return JSONResponse(
            {
                "ok": False,
                "code": "DOCKER_IMAGE_PREP_BACKLOG",
                "error": str(exc),
                "task_name": task_meta.get("task_name"),
                "task_path": task_meta.get("task_path"),
                "duration_sec": round(time.time() - started_ts, 3),
            },
            status_code=503,
            headers={
                "Retry-After": os.getenv("WORKER_DOCKER_BUILD_BACKLOG_RETRY_AFTER", "15")
            },
        )
    except TaskImageBlacklistedError as exc:
        logger.warning(
            "Rollout probe blocked by task image blacklist lease_id=%s task_key=%s: %s",
            lease_id,
            task_key,
            exc,
        )
        return JSONResponse(
            {
                "ok": False,
                "code": "TASK_IMAGE_BLACKLISTED",
                "error": str(exc),
                "task_name": task_meta.get("task_name"),
                "task_path": task_meta.get("task_path"),
                "duration_sec": round(time.time() - started_ts, 3),
            },
            status_code=503,
            headers={"Retry-After": os.getenv("WORKER_TASK_IMAGE_RETRY_AFTER", "300")},
        )
    except DockerImageTransientBuildError as exc:
        logger.warning(
            "Rollout probe deferred by transient Docker image build failure "
            "lease_id=%s task_key=%s: %s",
            lease_id,
            task_key,
            exc,
        )
        return JSONResponse(
            {
                "ok": False,
                "code": "TASK_IMAGE_BUILD_TRANSIENT",
                "error": str(exc),
                "task_name": task_meta.get("task_name"),
                "task_path": task_meta.get("task_path"),
                "duration_sec": round(time.time() - started_ts, 3),
            },
            status_code=503,
            headers={
                "Retry-After": os.getenv(
                    "WORKER_DOCKER_BUILD_TRANSIENT_RETRY_AFTER", "60"
                )
            },
        )
    except DockerImageBuildError as exc:
        logger.warning(
            "Rollout probe failed on Docker image build lease_id=%s task_key=%s: %s",
            lease_id,
            task_key,
            exc,
        )
        return JSONResponse(
            {
                "ok": False,
                "code": "TASK_IMAGE_BUILD_FAILED",
                "error": str(exc),
                "task_name": task_meta.get("task_name"),
                "task_path": task_meta.get("task_path"),
                "duration_sec": round(time.time() - started_ts, 3),
            },
            status_code=503,
            headers={"Retry-After": os.getenv("WORKER_TASK_IMAGE_RETRY_AFTER", "300")},
        )
    except TimeoutError as exc:
        message = str(exc)
        code = "WORKER_RESET_TIMEOUT"
        if "WORKER_RESET_CANCELLED" in message:
            code = "WORKER_RESET_CANCELLED"
        logger.warning(
            "Rollout probe ended with transient worker timeout lease_id=%s "
            "task_key=%s: %s",
            lease_id,
            task_key,
            exc,
        )
        return JSONResponse(
            {
                "ok": False,
                "code": code,
                "error": message,
                "task_name": task_meta.get("task_name"),
                "task_path": task_meta.get("task_path"),
                "duration_sec": round(time.time() - started_ts, 3),
            },
            status_code=503,
            headers={"Retry-After": os.getenv("WORKER_RESET_TIMEOUT_RETRY_AFTER", "15")},
        )
    except Exception as exc:
        logger.exception(
            "Rollout probe failed lease_id=%s task_key=%s", lease_id, task_key
        )
        return JSONResponse(
            {
                "ok": False,
                "error": str(exc),
                "duration_sec": round(time.time() - started_ts, 3),
            },
            status_code=500,
        )
    finally:
        if lease_id:
            try:
                await POOL.close_run(lease_id, reason="rollout_probe_close")
            except Exception:
                logger.exception("Failed to close rollout probe lease_id=%s", lease_id)


@app.post("/repair/pending_closes")
async def repair_pending_closes(request: Request) -> JSONResponse:
    if POOL is None:
        return JSONResponse(
            {"ok": False, "error": "Pool is not initialized"}, status_code=500
        )
    if os.getenv("WORKER_REPAIR_PENDING_CLOSES", "1") != "1":
        return JSONResponse(
            {
                "ok": False,
                "error": "Pending-close repair endpoint is disabled",
                "code": "REPAIR_DISABLED",
            },
            status_code=403,
        )

    data = await json_payload(request)
    reason = str(data.get("reason") or "manual")
    max_active_runs = _env_int("WORKER_REPAIR_PENDING_CLOSES_MAX_ACTIVE_RUNS", 0)
    cancel_timeout = _env_float("WORKER_REPAIR_PENDING_CLOSES_CANCEL_TIMEOUT", 5.0)
    min_age = _env_float(
        "WORKER_REPAIR_PENDING_CLOSES_MIN_AGE",
        max(0.0, POOL.close_task_timeout + 5.0),
    )
    try:
        if "max_active_runs" in data:
            max_active_runs = int(data["max_active_runs"])
    except (TypeError, ValueError):
        pass
    try:
        if "cancel_timeout" in data:
            cancel_timeout = float(data["cancel_timeout"])
    except (TypeError, ValueError):
        pass
    try:
        if "min_age" in data:
            min_age = float(data["min_age"])
    except (TypeError, ValueError):
        pass

    result = await POOL.repair_pending_closes(
        reason=reason,
        max_active_runs=max_active_runs,
        cancel_timeout=cancel_timeout,
        min_age=min_age,
    )
    return JSONResponse({"ok": True, **result})


@app.post("/repair/stale_runs")
async def repair_stale_runs(request: Request) -> JSONResponse:
    if POOL is None:
        return JSONResponse(
            {"ok": False, "error": "Pool is not initialized"}, status_code=500
        )
    if os.getenv("WORKER_REPAIR_STALE_RUNS", "1") != "1":
        return JSONResponse(
            {
                "ok": False,
                "error": "Stale-run repair endpoint is disabled",
                "code": "REPAIR_DISABLED",
            },
            status_code=403,
        )

    data = await json_payload(request)
    reason = str(data.get("reason") or "manual")
    min_age = _env_float("WORKER_REPAIR_STALE_RUNS_MIN_AGE", 0.0)
    max_repairs = _env_int("WORKER_REPAIR_STALE_RUNS_MAX_REPAIRS", 20)
    try:
        if "min_age" in data:
            min_age = float(data["min_age"])
    except (TypeError, ValueError):
        pass
    try:
        if "max_repairs" in data:
            max_repairs = int(data["max_repairs"])
    except (TypeError, ValueError):
        pass
    wait_for_cleanup = str(data.get("wait_for_cleanup", "1")).lower() in {
        "1",
        "true",
        "yes",
    }

    result = await POOL.repair_stale_runs(
        reason=reason,
        min_age=max(0.0, min_age),
        max_repairs=max(0, max_repairs),
        wait_for_cleanup=wait_for_cleanup,
    )
    return JSONResponse({"ok": True, **result})


@app.post("/repair/close_requested_runs")
async def repair_close_requested_runs(request: Request) -> JSONResponse:
    if POOL is None:
        return JSONResponse(
            {"ok": False, "error": "Pool is not initialized"}, status_code=500
        )
    if os.getenv("WORKER_REPAIR_CLOSE_REQUESTED_RUNS", "1") != "1":
        return JSONResponse(
            {
                "ok": False,
                "error": "Close-requested run repair endpoint is disabled",
                "code": "REPAIR_DISABLED",
            },
            status_code=403,
        )

    data = await json_payload(request)
    reason = str(data.get("reason") or "manual")
    min_age = _env_float("WORKER_REPAIR_CLOSE_REQUESTED_MIN_AGE", 0.0)
    max_repairs = _env_int("WORKER_REPAIR_CLOSE_REQUESTED_MAX_REPAIRS", 20)
    try:
        if "min_age" in data:
            min_age = float(data["min_age"])
    except (TypeError, ValueError):
        pass
    try:
        if "max_repairs" in data:
            max_repairs = int(data["max_repairs"])
    except (TypeError, ValueError):
        pass
    wait_for_cleanup = str(data.get("wait_for_cleanup", "0")).lower() in {
        "1",
        "true",
        "yes",
    }

    result = await POOL.repair_close_requested_runs(
        reason=reason,
        min_age=max(0.0, min_age),
        max_repairs=max(0, max_repairs),
        wait_for_cleanup=wait_for_cleanup,
    )
    return JSONResponse({"ok": True, **result})


@app.post("/repair/resetting_runs")
async def repair_resetting_runs(request: Request) -> JSONResponse:
    if POOL is None:
        return JSONResponse(
            {"ok": False, "error": "Pool is not initialized"}, status_code=500
        )
    if os.getenv("WORKER_REPAIR_RESETTING_RUNS", "1") != "1":
        return JSONResponse(
            {
                "ok": False,
                "error": "Resetting-run repair endpoint is disabled",
                "code": "REPAIR_DISABLED",
            },
            status_code=403,
        )

    data = await json_payload(request)
    reason = str(data.get("reason") or "manual")
    min_age = _env_float("WORKER_REPAIR_RESETTING_MIN_AGE", 2100.0)
    max_repairs = _env_int("WORKER_REPAIR_RESETTING_MAX_REPAIRS", 64)
    try:
        if "min_age" in data:
            min_age = float(data["min_age"])
    except (TypeError, ValueError):
        pass
    try:
        if "max_repairs" in data:
            max_repairs = int(data["max_repairs"])
    except (TypeError, ValueError):
        pass
    wait_for_cleanup = str(data.get("wait_for_cleanup", "0")).lower() in {
        "1",
        "true",
        "yes",
    }

    result = await POOL.repair_resetting_runs(
        reason=reason,
        min_age=max(0.0, min_age),
        max_repairs=max(0, max_repairs),
        wait_for_cleanup=wait_for_cleanup,
    )
    return JSONResponse({"ok": True, **result})


@app.post("/allocate")
async def allocate(request: Request) -> JSONResponse:
    if POOL is None:
        return JSONResponse(
            {"ok": False, "error": "Pool is not initialized"}, status_code=500
        )

    data = await json_payload(request)
    task_key = data.get("task_key", "")
    request_id = data.get("request_id")

    if not task_key:
        return JSONResponse(
            {"ok": False, "error": "task_key is required"}, status_code=400
        )

    auto_repair_result: dict[str, Any] | None = None

    async def _try_allocate_once() -> dict[str, Any]:
        pool_status = await POOL.status()
        assert_worker_has_capacity_for_docker(
            phase="allocate",
            pending_closes=int(pool_status.get("pending_closes", 0)),
            pool_status=pool_status,
        )
        return await POOL.allocate(task_key=str(task_key), request_id=request_id)

    try:
        result = await _try_allocate_once()
        return JSONResponse({"ok": True, **result})
    except ResourcePressureError as exc:
        return JSONResponse(
            {
                "ok": False,
                "error": exc.message,
                "code": exc.code,
                "details": exc.details,
            },
            status_code=503,
            headers={"Retry-After": os.getenv("WORKER_PRESSURE_RETRY_AFTER", "10")},
        )
    except CapacityError as exc:
        if os.getenv("WORKER_AUTO_REPAIR_ON_CAPACITY", "1") == "1":
            max_repairs = _env_int("WORKER_AUTO_REPAIR_MAX_REPAIRS", 20)
            close_min_age = _env_float(
                "WORKER_AUTO_REPAIR_CLOSE_REQUESTED_MIN_AGE", 0.0
            )
            stale_min_age = _env_float("WORKER_AUTO_REPAIR_STALE_MIN_AGE", 0.0)
            close_repair = await POOL.repair_close_requested_runs(
                reason=f"allocate_capacity:{exc.code}",
                min_age=max(0.0, close_min_age),
                max_repairs=max(0, max_repairs),
                wait_for_cleanup=False,
            )
            stale_repair: dict[str, Any] | None = None
            if not close_repair.get("repaired"):
                stale_repair = await POOL.repair_stale_runs(
                    reason=f"allocate_capacity:{exc.code}",
                    min_age=max(0.0, stale_min_age),
                    max_repairs=max(0, max_repairs),
                    wait_for_cleanup=False,
                )
            auto_repair_result = {
                "close_requested": close_repair,
                "stale": stale_repair,
            }
            if close_repair.get("repaired") or (
                stale_repair is not None and stale_repair.get("repaired")
            ):
                try:
                    result = await _try_allocate_once()
                    return JSONResponse(
                        {"ok": True, **result, "auto_repair": auto_repair_result}
                    )
                except CapacityError as retry_exc:
                    exc = retry_exc
        return JSONResponse(
            {
                "ok": False,
                "error": exc.message,
                "code": exc.code,
                "auto_repair": auto_repair_result,
            },
            status_code=429,
            headers={"Retry-After": os.getenv("WORKER_CAPACITY_RETRY_AFTER", "5")},
        )
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)


@app.post("/heartbeat")
async def heartbeat(request: Request) -> JSONResponse:
    if POOL is None:
        return JSONResponse(
            {"ok": False, "error": "Pool is not initialized"}, status_code=500
        )

    data = await json_payload(request)
    lease_id = data.get("lease_id")
    if not lease_id:
        return JSONResponse(
            {"ok": False, "error": "lease_id is required"}, status_code=400
        )

    try:
        await POOL.heartbeat(str(lease_id))
        return JSONResponse({"ok": True})
    except KeyError as exc:
        # FIX-3: Return HTTP 410 Gone for expired lease_id to prevent retry cascades
        logger.warning(
            "Heartbeat lease_id=%s no longer exists (likely timeout cleanup): %s",
            lease_id,
            exc,
        )
        return JSONResponse(
            {
                "ok": False,
                "error": "Lease expired or already cleaned up",
                "code": "LEASE_EXPIRED",
            },
            status_code=410,
        )
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)


@app.post("/reset")
async def reset(request: Request) -> JSONResponse:
    if POOL is None:
        return JSONResponse(
            {"ok": False, "error": "Pool is not initialized"}, status_code=500
        )

    data = await json_payload(request)
    lease_id = data.get("lease_id")
    task_meta = data.get("task_meta")
    run_ctx_payload = data.get("run_ctx")
    task_timeouts = data.get("task_timeouts")
    request_id = data.get("request_id")

    if not lease_id:
        return JSONResponse(
            {"ok": False, "error": "lease_id is required"}, status_code=400
        )
    if not isinstance(task_meta, dict):
        return JSONResponse(
            {"ok": False, "error": "task_meta dict is required"}, status_code=400
        )

    try:
        pool_status = await POOL.status()
        assert_worker_has_capacity_for_docker(
            phase="reset",
            pending_closes=int(pool_status.get("pending_closes", 0)),
            pool_status=pool_status,
        )
        out = await POOL.reset(
            run_lease_id=str(lease_id),
            task_meta=task_meta,
            run_ctx_payload=run_ctx_payload,
            task_timeouts=task_timeouts,
            request_id=str(request_id) if request_id else None,
        )
        return JSONResponse({"ok": True, **out})
    except ResourcePressureError as exc:
        return JSONResponse(
            {
                "ok": False,
                "error": exc.message,
                "code": exc.code,
                "details": exc.details,
            },
            status_code=503,
            headers={"Retry-After": os.getenv("WORKER_PRESSURE_RETRY_AFTER", "10")},
        )
    except ResetAdmissionBacklogError as exc:
        logger.warning("Reset deferred by reset admission backlog lease_id=%s: %s", lease_id, exc)
        try:
            await POOL.close_run(str(lease_id), reason="reset_admission_backlog")
        except Exception:
            logger.exception("Failed to schedule cleanup after reset backlog for %s", lease_id)
        return JSONResponse(
            {
                "ok": False,
                "error": str(exc),
                "code": "WORKER_RESET_ADMISSION_BACKLOG",
                "task_name": task_meta.get("task_name"),
                "task_path": task_meta.get("task_path"),
            },
            status_code=503,
            headers={"Retry-After": os.getenv("WORKER_RESET_BACKLOG_RETRY_AFTER", "10")},
        )
    except DockerImagePreparationBacklogError as exc:
        logger.warning(
            "Reset deferred by Docker image preparation backlog lease_id=%s: %s",
            lease_id,
            exc,
        )
        try:
            await POOL.close_run(str(lease_id), reason="docker_image_prep_backlog")
        except Exception:
            logger.exception("Failed to schedule cleanup after image prep backlog for %s", lease_id)
        return JSONResponse(
            {
                "ok": False,
                "error": str(exc),
                "code": "DOCKER_IMAGE_PREP_BACKLOG",
                "task_name": task_meta.get("task_name"),
                "task_path": task_meta.get("task_path"),
            },
            status_code=503,
            headers={
                "Retry-After": os.getenv("WORKER_DOCKER_BUILD_BACKLOG_RETRY_AFTER", "15")
            },
        )
    except TaskImageBlacklistedError as exc:
        logger.warning("Reset blocked by task image blacklist lease_id=%s: %s", lease_id, exc)
        try:
            await POOL.close_run(str(lease_id), reason="task_image_blacklisted")
        except Exception:
            logger.exception("Failed to schedule cleanup after image blacklist for %s", lease_id)
        return JSONResponse(
            {
                "ok": False,
                "error": str(exc),
                "code": "TASK_IMAGE_BLACKLISTED",
                "task_name": task_meta.get("task_name"),
                "task_path": task_meta.get("task_path"),
            },
            status_code=503,
            headers={"Retry-After": os.getenv("WORKER_TASK_IMAGE_RETRY_AFTER", "300")},
        )
    except DockerImageTransientBuildError as exc:
        logger.warning(
            "Reset deferred by transient Docker image build failure lease_id=%s: %s",
            lease_id,
            exc,
        )
        try:
            await POOL.close_run(str(lease_id), reason="task_image_build_transient")
        except Exception:
            logger.exception(
                "Failed to schedule cleanup after transient image build failure for %s",
                lease_id,
            )
        return JSONResponse(
            {
                "ok": False,
                "error": str(exc),
                "code": "TASK_IMAGE_BUILD_TRANSIENT",
                "task_name": task_meta.get("task_name"),
                "task_path": task_meta.get("task_path"),
            },
            status_code=503,
            headers={
                "Retry-After": os.getenv(
                    "WORKER_DOCKER_BUILD_TRANSIENT_RETRY_AFTER", "60"
                )
            },
        )
    except DockerImageBuildError as exc:
        logger.warning("Reset failed on Docker image build lease_id=%s: %s", lease_id, exc)
        try:
            await POOL.close_run(str(lease_id), reason="task_image_build_failed")
        except Exception:
            logger.exception("Failed to schedule cleanup after image build failure for %s", lease_id)
        return JSONResponse(
            {
                "ok": False,
                "error": str(exc),
                "code": "TASK_IMAGE_BUILD_FAILED",
                "task_name": task_meta.get("task_name"),
                "task_path": task_meta.get("task_path"),
            },
            status_code=503,
            headers={"Retry-After": os.getenv("WORKER_TASK_IMAGE_RETRY_AFTER", "300")},
        )
    except ResetInProgressError as exc:
        return JSONResponse(
            {
                "ok": False,
                "error": str(exc),
                "code": "RESET_IN_PROGRESS",
                "lease_id": exc.run_lease_id,
                "request_id": exc.request_id,
            },
            status_code=429,
            headers={"Retry-After": os.getenv("WORKER_RESET_IN_PROGRESS_RETRY_AFTER", "2")},
        )
    except KeyError as exc:
        # FIX-3: Return HTTP 410 Gone for expired lease_id to prevent retry cascades
        logger.warning(
            "Reset lease_id=%s no longer exists (likely timeout cleanup): %s",
            lease_id,
            exc,
        )
        return JSONResponse(
            {
                "ok": False,
                "error": "Lease expired or already cleaned up",
                "code": "LEASE_EXPIRED",
            },
            status_code=410,
        )
    except TimeoutError as exc:
        message = str(exc)
        code = "WORKER_RESET_TIMEOUT"
        if "WORKER_RESET_CANCELLED" in message:
            code = "WORKER_RESET_CANCELLED"
        logger.warning("Reset ended with transient worker timeout lease_id=%s: %s", lease_id, exc)
        try:
            await POOL.close_run(str(lease_id), reason=code.lower())
        except Exception:
            logger.exception("Failed to schedule cleanup after reset timeout for %s", lease_id)
        return JSONResponse(
            {
                "ok": False,
                "error": message,
                "code": code,
                "task_name": task_meta.get("task_name"),
                "task_path": task_meta.get("task_path"),
            },
            status_code=503,
            headers={"Retry-After": os.getenv("WORKER_RESET_TIMEOUT_RETRY_AFTER", "15")},
        )
    except Exception as exc:
        logger.exception("Reset failed for lease_id=%s", lease_id)
        try:
            await POOL.close_run(str(lease_id), reason="reset_failure")
        except Exception:
            logger.exception("Failed to schedule cleanup after reset failure for %s", lease_id)
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)


@app.post("/exec_tool")
async def exec_tool(request: Request) -> JSONResponse:
    if POOL is None:
        return JSONResponse(
            {"ok": False, "error": "Pool is not initialized"}, status_code=500
        )

    data = await json_payload(request)
    lease_id = data.get("lease_id")
    tool_call = data.get("tool_call")

    if not lease_id:
        return JSONResponse(
            {"ok": False, "error": "lease_id is required"}, status_code=400
        )
    if not isinstance(tool_call, dict):
        return JSONResponse(
            {"ok": False, "error": "tool_call dict is required"}, status_code=400
        )

    tool_name = tool_call.get("name")
    arguments = tool_call.get("arguments")

    if not isinstance(tool_name, str) or not tool_name:
        return JSONResponse(
            {"ok": False, "error": "tool_call.name is required"}, status_code=400
        )
    if arguments is not None and not isinstance(arguments, dict):
        return JSONResponse(
            {"ok": False, "error": "tool_call.arguments must be a dict"},
            status_code=400,
        )

    try:
        observation = await POOL.exec_tool(
            str(lease_id), tool_name, arguments=arguments
        )
        return JSONResponse({"ok": True, "observation": observation})
    except KeyError as exc:
        # FIX-3: Return HTTP 410 Gone for expired lease_id to prevent retry cascades
        logger.warning(
            "Exec_tool lease_id=%s no longer exists (likely timeout cleanup): %s",
            lease_id,
            exc,
        )
        return JSONResponse(
            {
                "ok": False,
                "error": "Lease expired or already cleaned up",
                "code": "LEASE_EXPIRED",
            },
            status_code=410,
        )
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)


@app.post("/agent_reply")
async def agent_reply(request: Request) -> JSONResponse:
    """Handle a non-tool assistant reply for an active environment lease.

    Args:
        request: FastAPI request containing ``lease_id`` and ``assistant_text``.

    Returns:
        JSON response with ``ok=True`` and the environment follow-up payload, or
        an error response when the pool is unavailable or the payload is invalid.
    """
    if POOL is None:
        return JSONResponse(
            {"ok": False, "error": "Pool is not initialized"}, status_code=500
        )

    data = await json_payload(request)
    lease_id = data.get("lease_id")
    assistant_text = data.get("assistant_text")

    if not lease_id:
        return JSONResponse(
            {"ok": False, "error": "lease_id is required"}, status_code=400
        )
    if not isinstance(assistant_text, str):
        return JSONResponse(
            {"ok": False, "error": "assistant_text is required"}, status_code=400
        )

    try:
        result = await POOL.handle_agent_reply(str(lease_id), assistant_text)
        return JSONResponse({"ok": True, **result})
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)


@app.post("/evaluate")
async def evaluate(request: Request) -> JSONResponse:
    if POOL is None:
        return JSONResponse(
            {"ok": False, "error": "Pool is not initialized"}, status_code=500
        )

    data = await json_payload(request)
    lease_id = data.get("lease_id")
    trajectory = data.get("trajectory")

    if not lease_id:
        return JSONResponse(
            {"ok": False, "error": "lease_id is required"}, status_code=400
        )

    try:
        score, details = await POOL.evaluate(
            str(lease_id), trajectory if isinstance(trajectory, dict) else None
        )
        payload: dict[str, Any] = {"ok": True, "score": score}
        if details is not None:
            payload["details"] = details
        return JSONResponse(payload)
    except KeyError as exc:
        # FIX-3: Return HTTP 410 Gone for expired lease_id to prevent retry cascades
        logger.warning(
            "Evaluate lease_id=%s no longer exists (likely timeout cleanup): %s",
            lease_id,
            exc,
        )
        return JSONResponse(
            {
                "ok": False,
                "error": "Lease expired or already cleaned up",
                "code": "LEASE_EXPIRED",
            },
            status_code=410,
        )
    except Exception as exc:
        from harborrl.data.harbor.receipt import HarborVerifierError
        if isinstance(exc, HarborVerifierError):
            # Completed RPC with an explicit verifier failure, not a transport retry.
            return JSONResponse({"ok": False, "error": str(exc), "details": exc.details})
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)


@app.post("/close")
async def close(request: Request) -> JSONResponse:
    if POOL is None:
        return JSONResponse(
            {"ok": False, "error": "Pool is not initialized"}, status_code=500
        )

    data = await json_payload(request)
    lease_id = data.get("lease_id")
    if not lease_id:
        return JSONResponse(
            {"ok": False, "error": "lease_id is required"}, status_code=400
        )

    try:
        found = await POOL.close_run(str(lease_id), reason="http_close")
        return JSONResponse({"ok": True, "found": found})
    except KeyError as exc:
        # FIX-3: Return HTTP 410 Gone for expired lease_id to prevent retry cascades
        logger.warning(
            "Close lease_id=%s no longer exists (likely already cleaned up): %s",
            lease_id,
            exc,
        )
        return JSONResponse(
            {
                "ok": False,
                "error": "Lease expired or already cleaned up",
                "code": "LEASE_EXPIRED",
            },
            status_code=410,
        )
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)


_REAPER_TASK: asyncio.Task | None = None


@app.on_event("startup")
async def _on_startup() -> None:
    global _REAPER_TASK
    if POOL is not None:
        _REAPER_TASK = asyncio.create_task(POOL.periodic_reap(interval=60.0))


@app.on_event("shutdown")
async def _on_shutdown() -> None:
    global POOL, _REAPER_TASK
    if _REAPER_TASK is not None:
        _REAPER_TASK.cancel()
        _REAPER_TASK = None
    if POOL is not None:
        await POOL.shutdown()
        POOL = None
