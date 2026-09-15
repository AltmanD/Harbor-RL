from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from hashlib import sha1
from pathlib import Path
from typing import Any

import aiohttp

logger = logging.getLogger("lightrl.env.router")

RETRYABLE_ALLOCATE_CODES = {
    "WORKER_PENDING_CLOSES_PRESSURE",
    "WORKER_PIDS_PRESSURE",
    "WORKER_SHIM_PRESSURE",
    "WORKER_DOCKER_CLI_UNHEALTHY",
    "WORKER_DOCKER_DISK_PRESSURE",
    "TASK_SLOTS_EXHAUSTED",
    "RUN_SLOTS_EXHAUSTED",
}


def _format_error(exc: BaseException) -> str:
    detail = str(exc).strip()
    if detail:
        return f"{type(exc).__name__}: {detail}"
    return type(exc).__name__


def _status_from_payload(payload: dict[str, Any], default: int) -> int:
    raw = payload.get("status_code")
    if isinstance(raw, int):
        return raw
    return default


from harborrl.env import env_float as _env_float


def _retryable_allocate_failure(payload: dict[str, Any], status: int) -> bool:
    code = str(payload.get("code", "") or "")
    if code in RETRYABLE_ALLOCATE_CODES:
        return True
    return status in {429, 502, 503, 504}


def _parse_worker_urls_text(text: str) -> list[str]:
    chunks: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if line.startswith("WORKER_URLS="):
            line = line.split("=", 1)[1].strip()
        line = line.strip().strip('"').strip("'")
        chunks.extend(part for part in re.split(r"[,\s]+", line) if part)
    return [part.rstrip("/") for part in chunks]


class Router:
    def __init__(
        self,
        worker_urls: list[str],
        forward_timeout: float = 600.0,
        forward_retries: int = 1,
        forward_retry_backoff: float = 0.2,
        pressure_cooldown: float = 60.0,
        workers_file: str | None = None,
        workers_reload_interval: float = 0.0,
    ):
        if not worker_urls:
            raise ValueError("At least one worker URL is required")
        self.workers = [u.rstrip("/") for u in worker_urls]
        self.forward_timeout = float(forward_timeout)
        self.forward_retries = max(0, int(forward_retries))
        self.forward_retry_backoff = max(0.0, float(forward_retry_backoff))
        self.pressure_cooldown = max(0.0, float(pressure_cooldown))
        self.workers_file = str(workers_file or "").strip()
        self.workers_reload_interval = max(0.0, float(workers_reload_interval))
        self._last_workers_reload = 0.0
        self._workers_reload_lock = asyncio.Lock()
        self._lease_worker_urls: dict[str, str] = {}
        self._unhealthy_until: dict[int, float] = {}
        self._status_cache: dict[int, tuple[float, dict[str, Any], int]] = {}
        self._session: aiohttp.ClientSession | None = None

    @property
    def num_workers(self) -> int:
        return len(self.workers)

    async def startup(self) -> None:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.forward_timeout)
            connector = aiohttp.TCPConnector(limit=0, ttl_dns_cache=300)
            self._session = aiohttp.ClientSession(timeout=timeout, connector=connector)

    async def shutdown(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

    async def maybe_reload_workers(self, *, force: bool = False) -> None:
        if not self.workers_file:
            return
        if self.workers_reload_interval <= 0 and not force:
            return
        now = time.monotonic()
        if not force and now - self._last_workers_reload < self.workers_reload_interval:
            return

        async with self._workers_reload_lock:
            now = time.monotonic()
            if not force and now - self._last_workers_reload < self.workers_reload_interval:
                return
            self._last_workers_reload = now
            try:
                text = Path(self.workers_file).read_text(encoding="utf-8")
            except FileNotFoundError:
                logger.warning(
                    "Worker URL file %s is missing; keeping existing workers=%s",
                    self.workers_file,
                    self.workers,
                )
                return
            except OSError as exc:
                logger.warning(
                    "Failed reading worker URL file %s: %s; keeping existing workers=%s",
                    self.workers_file,
                    _format_error(exc),
                    self.workers,
                )
                return

            new_workers = _parse_worker_urls_text(text)
            if not new_workers:
                logger.warning(
                    "Worker URL file %s has no usable URLs; keeping existing workers=%s",
                    self.workers_file,
                    self.workers,
                )
                return
            if new_workers == self.workers:
                return

            old_workers = self.workers
            self.workers = new_workers
            self._unhealthy_until.clear()
            self._status_cache.clear()
            logger.warning(
                "Reloaded worker URLs from %s: old=%s new=%s",
                self.workers_file,
                old_workers,
                new_workers,
            )

    def select_worker(self, task_key: str) -> tuple[int, str]:
        digest = sha1(task_key.encode("utf-8")).digest()
        idx = (
            int.from_bytes(digest[:8], byteorder="big", signed=False) % self.num_workers
        )
        return idx, self.workers[idx]

    @staticmethod
    def encode_lease(worker_idx: int, worker_lease: str) -> str:
        return f"{worker_idx}:{worker_lease}"

    @staticmethod
    def decode_lease(global_lease: str) -> tuple[int, str]:
        sep = global_lease.index(":")
        return int(global_lease[:sep]), global_lease[sep + 1 :]

    def worker_url(self, worker_idx: int) -> str:
        return self.workers[worker_idx]

    def remember_lease(self, global_lease: str, worker_url: str) -> None:
        self._lease_worker_urls[str(global_lease)] = worker_url.rstrip("/")

    def forget_lease(self, global_lease: str) -> None:
        self._lease_worker_urls.pop(str(global_lease), None)

    def worker_url_for_lease(self, global_lease: str, worker_idx: int) -> str:
        return self._lease_worker_urls.get(str(global_lease)) or self.worker_url(worker_idx)

    def iter_worker_candidates(self, start_idx: int) -> list[tuple[int, str]]:
        candidates = [
            (
                (start_idx + offset) % self.num_workers,
                self.workers[(start_idx + offset) % self.num_workers],
            )
            for offset in range(self.num_workers)
        ]
        now = time.monotonic()
        healthy = [
            item for item in candidates if self._unhealthy_until.get(item[0], 0.0) <= now
        ]
        unhealthy = [
            item for item in candidates if self._unhealthy_until.get(item[0], 0.0) > now
        ]
        return healthy + unhealthy if healthy else candidates

    async def _worker_status_cached(
        self, worker_idx: int, worker_url: str
    ) -> tuple[dict[str, Any], int]:
        ttl = _env_float("ROUTER_LOAD_STATUS_CACHE_TTL", 2.0)
        now = time.monotonic()
        cached = self._status_cache.get(worker_idx)
        if cached is not None and now - cached[0] <= ttl:
            return cached[1], cached[2]
        timeout = _env_float("ROUTER_LOAD_STATUS_TIMEOUT", 2.0)
        data, status = await self.worker_status(worker_url, timeout=timeout)
        self._status_cache[worker_idx] = (now, data, status)
        return data, status

    @staticmethod
    def _worker_load_score(data: dict[str, Any], status: int) -> float:
        if status >= 500 or not data.get("ok", False):
            return 1_000_000.0 + status
        pool = data.get("pool", {})
        if not isinstance(pool, dict):
            return 900_000.0
        try:
            max_tasks = int(pool.get("max_tasks", 1) or 1)
            max_runs_per_task = int(pool.get("max_runs_per_task", 1) or 1)
            total_runs = int(pool.get("total_active_runs", 0) or 0)
            pending = int(pool.get("pending_closes", 0) or 0)
            stale = len(pool.get("stale_runs", []) or [])
            phase_counts = pool.get("phase_counts", {})
            resetting = int(phase_counts.get("resetting", 0) or 0) if isinstance(phase_counts, dict) else 0
        except (TypeError, ValueError):
            return 800_000.0
        capacity = max(1, max_tasks * max_runs_per_task)
        util = total_runs / capacity
        reset_ratio = resetting / max(1, total_runs)
        return util * 100.0 + reset_ratio * 100.0 + pending * 2.0 + stale * 50.0

    async def iter_worker_candidates_for_allocate(
        self, start_idx: int
    ) -> list[tuple[int, str]]:
        candidates = self.iter_worker_candidates(start_idx)
        if os.getenv("ROUTER_LOAD_AWARE_ALLOCATE", "1") != "1" or len(candidates) <= 1:
            return candidates

        async def _score(pos: int, item: tuple[int, str]) -> tuple[float, int, tuple[int, str]]:
            worker_idx, worker_url = item
            if self._unhealthy_until.get(worker_idx, 0.0) > time.monotonic():
                return 1_100_000.0, pos, item
            try:
                data, status = await self._worker_status_cached(worker_idx, worker_url)
                return self._worker_load_score(data, status), pos, item
            except Exception as exc:
                logger.warning(
                    "Load-aware status failed for worker_idx=%d url=%s err=%s",
                    worker_idx,
                    worker_url,
                    _format_error(exc),
                )
                return 1_000_000.0, pos, item

        scored = await asyncio.gather(
            *[_score(pos, item) for pos, item in enumerate(candidates)]
        )
        return [item for _score_value, _pos, item in sorted(scored, key=lambda x: (x[0], x[1]))]

    def mark_worker_unhealthy(self, worker_idx: int, reason: str) -> None:
        if self.pressure_cooldown <= 0:
            return
        until = time.monotonic() + self.pressure_cooldown
        self._unhealthy_until[worker_idx] = until
        logger.warning(
            "Marked worker_idx=%d unhealthy for %.1fs due to %s",
            worker_idx,
            self.pressure_cooldown,
            reason,
        )

    async def _request(
        self,
        method: str,
        worker_url: str,
        path: str,
        payload: dict[str, Any] | None,
        timeout: float | None,
        retries: int | None = None,
    ) -> tuple[dict[str, Any], int]:
        if self._session is None:
            raise RuntimeError("Router HTTP session is not initialized")

        kwargs: dict[str, Any] = {}
        if payload is not None:
            kwargs["json"] = payload
        if timeout is not None:
            kwargs["timeout"] = aiohttp.ClientTimeout(total=float(timeout))

        max_retries = self.forward_retries if retries is None else max(0, int(retries))
        max_attempts = max_retries + 1
        for attempt in range(1, max_attempts + 1):
            try:
                async with self._session.request(
                    method, f"{worker_url}{path}", **kwargs
                ) as resp:
                    status = resp.status
                    try:
                        body = await resp.json(content_type=None)
                    except Exception:
                        raw_text = await resp.text()
                        body = {
                            "ok": False,
                            "error": "Worker returned non-JSON response",
                            "raw_text": raw_text,
                            "status_code": status,
                        }
                    return body, status
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                if attempt >= max_attempts:
                    raise
                logger.warning(
                    "Upstream request failed (%s %s) worker=%s attempt=%d/%d err=%s",
                    method,
                    path,
                    worker_url,
                    attempt,
                    max_attempts,
                    _format_error(exc),
                )
                # Exponential backoff with jitter to prevent thundering herd
                backoff = self.forward_retry_backoff * (2 ** (attempt - 1))
                jitter = backoff * 0.2 * (hash(f"{worker_url}{path}{attempt}") % 100) / 100.0
                total_backoff = backoff + jitter
                if total_backoff > 0:
                    await asyncio.sleep(total_backoff)

    async def forward(
        self,
        worker_url: str,
        path: str,
        payload: dict[str, Any],
        timeout: float | None = None,
    ) -> tuple[dict[str, Any], int]:
        return await self._request("POST", worker_url, path, payload, timeout)

    async def forward_by_lease(
        self,
        global_lease: str,
        path: str,
        payload: dict[str, Any],
        timeout: float | None = None,
    ) -> tuple[dict[str, Any], int]:
        worker_idx, worker_lease = self.decode_lease(global_lease)
        url = self.worker_url_for_lease(global_lease, worker_idx)
        forwarded_payload = dict(payload)
        forwarded_payload["lease_id"] = worker_lease
        return await self.forward(url, path, forwarded_payload, timeout)

    async def worker_status(
        self, worker_url: str, timeout: float = 10.0
    ) -> tuple[dict[str, Any], int]:
        return await self._request("GET", worker_url, "/status", None, timeout)

    async def worker_readiness(
        self, worker_url: str, timeout: float = 5.0
    ) -> tuple[dict[str, Any], int, str]:
        data, status = await self._request(
            "GET", worker_url, "/readyz", None, timeout, retries=0
        )
        if status == 404:
            data, status = await self._request(
                "GET", worker_url, "/healthz", None, timeout, retries=0
            )
            return data, status, "/healthz"
        return data, status, "/readyz"
