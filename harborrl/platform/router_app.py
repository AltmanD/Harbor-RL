from __future__ import annotations

import asyncio
import logging
from typing import Any

import aiohttp
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from harborrl.http_server import json_payload
from harborrl.platform.router import (
    Router,
    _env_float,
    _format_error,
    _retryable_allocate_failure,
    _status_from_payload,
)

logger = logging.getLogger("lightrl.env.router.app")
app = FastAPI()
ROUTER: Router | None = None



def _worker_unreachable(
    *,
    worker_idx: int,
    worker_url: str,
    path: str,
    exc: BaseException,
    lease_id: str | None = None,
    task_key: str | None = None,
) -> JSONResponse:
    payload: dict[str, Any] = {
        "ok": False,
        "error": f"Worker unreachable: {_format_error(exc)}",
        "worker_idx": worker_idx,
        "worker_url": worker_url,
        "path": path,
    }
    if lease_id:
        payload["lease_id"] = lease_id
    if task_key:
        payload["task_key"] = task_key
    return JSONResponse(payload, status_code=502)

@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    return {"ok": True}


@app.get("/readyz")
async def readyz() -> JSONResponse:
    if ROUTER is None:
        return JSONResponse(
            {
                "ok": False,
                "code": "ROUTER_NOT_INITIALIZED",
                "error": "Router is not initialized",
            },
            status_code=503,
        )

    await ROUTER.maybe_reload_workers()
    timeout = _env_float("ROUTER_READYZ_WORKER_TIMEOUT", 5.0)

    async def _fetch(idx: int, url: str) -> dict[str, Any]:
        try:
            data, status_code, path = await ROUTER.worker_readiness(url, timeout=timeout)
            ready = 200 <= status_code < 300 and bool(data.get("ok", False))
            return {
                "worker_idx": idx,
                "url": url,
                "ready": ready,
                "status": status_code,
                "path": path,
                "response": data,
            }
        except Exception as exc:
            return {
                "worker_idx": idx,
                "url": url,
                "ready": False,
                "error": _format_error(exc),
            }

    workers = await asyncio.gather(
        *[_fetch(idx, url) for idx, url in enumerate(ROUTER.workers)]
    )
    ready_workers = [worker for worker in workers if worker.get("ready")]
    payload = {
        "ok": bool(ready_workers),
        "num_workers": ROUTER.num_workers,
        "ready_workers": len(ready_workers),
        "workers": workers,
    }
    if not ready_workers:
        payload.update(
            {
                "code": "NO_READY_WORKERS",
                "error": "No env worker is ready",
            }
        )
        return JSONResponse(payload, status_code=503)
    return JSONResponse(payload)


@app.get("/status")
async def status() -> JSONResponse:
    if ROUTER is None:
        return JSONResponse(
            {"ok": False, "error": "Router is not initialized"}, status_code=500
        )

    await ROUTER.maybe_reload_workers()

    async def _fetch(idx: int, url: str) -> dict[str, Any]:
        try:
            data, _ = await ROUTER.worker_status(url, timeout=10)
            return {"worker_idx": idx, "url": url, **data}
        except Exception as exc:
            return {
                "worker_idx": idx,
                "url": url,
                "ok": False,
                "error": _format_error(exc),
            }

    workers = await asyncio.gather(
        *[_fetch(idx, url) for idx, url in enumerate(ROUTER.workers)]
    )
    return JSONResponse(
        {"ok": True, "num_workers": ROUTER.num_workers, "workers": workers}
    )


@app.post("/allocate")
async def allocate(request: Request) -> JSONResponse:
    if ROUTER is None:
        return JSONResponse(
            {"ok": False, "error": "Router is not initialized"}, status_code=500
        )

    await ROUTER.maybe_reload_workers()
    data = await json_payload(request)
    task_key = data.get("task_key", "")
    request_id = data.get("request_id")

    if not task_key:
        return JSONResponse(
            {"ok": False, "error": "task_key is required"}, status_code=400
        )

    try:
        payload = {"task_key": task_key, "request_id": request_id}
        primary_idx, _ = ROUTER.select_worker(str(task_key))
        upstream_errors: list[dict[str, Any]] = []
        candidates = await ROUTER.iter_worker_candidates_for_allocate(primary_idx)
        for worker_idx, worker_url in candidates:
            try:
                result, code = await ROUTER.forward(worker_url, "/allocate", payload)
                if result.get("ok") and "lease_id" in result:
                    if worker_idx != primary_idx:
                        logger.warning(
                            "Allocated on fallback worker for task_key=%s worker_idx=%d url=%s",
                            task_key,
                            worker_idx,
                            worker_url,
                        )
                    result["lease_id"] = Router.encode_lease(
                        worker_idx, str(result["lease_id"])
                    )
                    ROUTER.remember_lease(str(result["lease_id"]), worker_url)
                    result["worker_idx"] = worker_idx
                    return JSONResponse(
                        result, status_code=_status_from_payload(result, code)
                    )

                if _retryable_allocate_failure(result, code):
                    retry_code = str(result.get("code", "") or f"HTTP_{code}")
                    if retry_code != "RUN_SLOTS_EXHAUSTED":
                        ROUTER.mark_worker_unhealthy(worker_idx, retry_code)
                    logger.warning(
                        "Worker pressure for /allocate task_key=%s worker_idx=%d url=%s status=%s code=%s; trying next worker",
                        task_key,
                        worker_idx,
                        worker_url,
                        code,
                        retry_code,
                    )
                    upstream_errors.append(
                        {
                            "worker_idx": worker_idx,
                            "worker_url": worker_url,
                            "status": code,
                            "code": result.get("code"),
                            "error": result.get("error"),
                            "details": result.get("details"),
                        }
                    )
                    continue

                return JSONResponse(result, status_code=_status_from_payload(result, code))
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                ROUTER.mark_worker_unhealthy(worker_idx, "unreachable")
                logger.warning(
                    "Worker unreachable for /allocate task_key=%s worker_idx=%d url=%s err=%s",
                    task_key,
                    worker_idx,
                    worker_url,
                    _format_error(exc),
                )
                upstream_errors.append(
                    {
                        "worker_idx": worker_idx,
                        "worker_url": worker_url,
                        "detail": _format_error(exc),
                    }
                )

        return JSONResponse(
            {
                "ok": False,
                "error": "All worker candidates failed or were under pressure for /allocate",
                "code": "ALL_WORKERS_UNAVAILABLE_OR_PRESSURED",
                "task_key": task_key,
                "primary_worker_idx": primary_idx,
                "upstream_errors": upstream_errors,
            },
            status_code=503,
        )
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)


async def _lease_proxy(path: str, request: Request) -> JSONResponse:
    if ROUTER is None:
        return JSONResponse(
            {"ok": False, "error": "Router is not initialized"}, status_code=500
        )

    data = await json_payload(request)
    global_lease = data.get("lease_id", "")
    if not global_lease:
        return JSONResponse(
            {"ok": False, "error": "lease_id is required"}, status_code=400
        )

    try:
        worker_idx, worker_lease = ROUTER.decode_lease(str(global_lease))
        worker_url = ROUTER.worker_url_for_lease(str(global_lease), worker_idx)
    except (ValueError, IndexError) as exc:
        return JSONResponse(
            {"ok": False, "error": f"Invalid lease_id format: {exc}"}, status_code=400
        )

    payload = dict(data)
    payload["lease_id"] = worker_lease

    try:
        result, code = await ROUTER.forward(worker_url, path, payload)
        if path == "/close":
            ROUTER.forget_lease(str(global_lease))
        return JSONResponse(result, status_code=_status_from_payload(result, code))
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        return _worker_unreachable(
            worker_idx=worker_idx,
            worker_url=worker_url,
            path=path,
            exc=exc,
            lease_id=str(global_lease),
        )
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)


@app.post("/heartbeat")
async def heartbeat(request: Request) -> JSONResponse:
    return await _lease_proxy("/heartbeat", request)


@app.post("/reset")
async def reset(request: Request) -> JSONResponse:
    return await _lease_proxy("/reset", request)


@app.post("/exec_tool")
async def exec_tool(request: Request) -> JSONResponse:
    return await _lease_proxy("/exec_tool", request)


@app.post("/evaluate")
async def evaluate(request: Request) -> JSONResponse:
    return await _lease_proxy("/evaluate", request)


@app.post("/agent_reply")
async def agent_reply(request: Request) -> JSONResponse:
    return await _lease_proxy("/agent_reply", request)


@app.post("/close")
async def close(request: Request) -> JSONResponse:
    return await _lease_proxy("/close", request)


@app.on_event("startup")
async def _on_startup() -> None:
    if ROUTER is not None:
        await ROUTER.startup()


@app.on_event("shutdown")
async def _on_shutdown() -> None:
    if ROUTER is not None:
        await ROUTER.shutdown()
