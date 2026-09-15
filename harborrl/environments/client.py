from __future__ import annotations

import logging
from typing import Any

from harborrl.http_client import post

logger = logging.getLogger(__name__)


from harborrl.env import (
    env_float as _env_float,
    env_int as _env_int,
    env_int_set as _env_status_set,
)


class TerminalEnvClient:

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.default_max_retries = _env_int("ENV_HTTP_MAX_RETRIES", 10)
        self.allocate_max_retries = _env_int("ENV_ALLOCATE_MAX_RETRIES", 100)
        self.allocate_retry_base_delay = _env_float("ENV_ALLOCATE_RETRY_BASE_DELAY", 2.0)
        self.allocate_retry_max_delay = _env_float("ENV_ALLOCATE_RETRY_MAX_DELAY", 30.0)
        self.allocate_retry_backoff = _env_float("ENV_ALLOCATE_RETRY_BACKOFF", 2.0)
        self.allocate_retry_jitter = _env_float("ENV_ALLOCATE_RETRY_JITTER", 0.25)
        self.reset_max_retries = _env_int("ENV_RESET_MAX_RETRIES", 2)
        self.reset_retry_base_delay = _env_float("ENV_RESET_RETRY_BASE_DELAY", 1.0)
        self.reset_retry_max_delay = _env_float("ENV_RESET_RETRY_MAX_DELAY", 5.0)
        self.reset_retry_backoff = _env_float("ENV_RESET_RETRY_BACKOFF", 2.0)
        self.reset_retry_jitter = _env_float("ENV_RESET_RETRY_JITTER", 0.2)
        self.reset_retry_statuses = _env_status_set(
            "ENV_RESET_RETRY_STATUSES", "429,502,503,504"
        )
        self.reset_non_retry_statuses = _env_status_set(
            "ENV_RESET_NON_RETRY_STATUSES", "400,404,409,500"
        )
        self.evaluate_max_retries = _env_int("ENV_EVALUATE_MAX_RETRIES", 1)
        self.close_max_retries = _env_int("ENV_CLOSE_MAX_RETRIES", 3)
        self.exec_tool_max_retries = _env_int("ENV_EXEC_TOOL_MAX_RETRIES", 3)
        self.last_evaluate_details: dict[str, Any] | None = None

    async def allocate(
        self,
        task_key: str,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        out = await post(
            f"{self.base_url}/allocate",
            {"task_key": task_key, "request_id": request_id},
            max_retries=self.allocate_max_retries,
            retry_base_delay=self.allocate_retry_base_delay,
            retry_max_delay=self.allocate_retry_max_delay,
            retry_backoff_factor=self.allocate_retry_backoff,
            retry_jitter=self.allocate_retry_jitter,
        )
        if not out.get("ok", False):
            raise RuntimeError(f"allocate failed: {out}")
        return out

    async def heartbeat(self, lease_id: str) -> None:
        out = await post(
            f"{self.base_url}/heartbeat",
            {"lease_id": lease_id},
            max_retries=self.default_max_retries,
        )
        if not out.get("ok", False):
            raise RuntimeError(f"heartbeat failed: {out}")

    async def reset(
        self,
        lease_id: str,
        task_meta: dict[str, Any],
        run_ctx: dict[str, Any],
        task_timeouts: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        out = await post(
            f"{self.base_url}/reset",
            {
                "lease_id": lease_id,
                "task_meta": task_meta,
                "run_ctx": run_ctx,
                "task_timeouts": task_timeouts,
                "request_id": request_id,
            },
            max_retries=self.reset_max_retries,
            retry_base_delay=self.reset_retry_base_delay,
            retry_max_delay=self.reset_retry_max_delay,
            retry_backoff_factor=self.reset_retry_backoff,
            retry_jitter=self.reset_retry_jitter,
            retry_statuses=self.reset_retry_statuses,
            non_retry_statuses=self.reset_non_retry_statuses,
        )
        if not out.get("ok", False):
            raise RuntimeError(f"reset failed: {out}")
        return out

    async def exec_tool(
        self, lease_id: str, tool_name: str, arguments: dict[str, Any]
    ) -> str:
        out = await post(
            f"{self.base_url}/exec_tool",
            {
                "lease_id": lease_id,
                "tool_call": {"name": tool_name, "arguments": arguments},
            },
            max_retries=self.exec_tool_max_retries,
        )
        if not out.get("ok", False):
            raise RuntimeError(f"exec_tool failed: {out}")
        return str(out.get("observation", ""))

    async def agent_reply(self, lease_id: str, assistant_text: str) -> dict[str, Any]:
        out = await post(
            f"{self.base_url}/agent_reply",
            {"lease_id": lease_id, "assistant_text": assistant_text},
            max_retries=self.default_max_retries,
        )
        if not out.get("ok", False):
            raise RuntimeError(f"agent_reply failed: {out}")
        return {
            "continue": bool(out.get("continue", False)),
            "user_message": str(out.get("user_message", "") or ""),
        }

    async def evaluate(
        self, lease_id: str, trajectory: dict[str, Any] | None = None
    ) -> float:
        payload: dict[str, Any] = {"lease_id": lease_id}
        if trajectory is not None:
            payload["trajectory"] = trajectory
        out = await post(
            f"{self.base_url}/evaluate",
            payload,
            max_retries=self.evaluate_max_retries,
        )
        if not out.get("ok", False):
            raise RuntimeError(f"evaluate failed: {out}")
        details = out.get("details")
        self.last_evaluate_details = details if isinstance(details, dict) else None
        return float(out.get("score", 0.0))

    async def close(self, lease_id: str) -> None:
        try:
            out = await post(
                f"{self.base_url}/close",
                {"lease_id": lease_id},
                max_retries=self.close_max_retries,
            )
        except Exception as exc:
            error_str = str(exc)
            resp_text = ""
            status_code = 0
            # Extract status code to treat 500/502/503 as non-fatal
            if hasattr(exc, "response"):
                try:
                    resp_text = exc.response.text
                    status_code = getattr(exc.response, "status_code", 0)
                except Exception:
                    pass
            combined = f"{error_str} {resp_text}"
            if "Unknown run_lease_id" in combined or "Unknown lease" in combined:
                logger.debug("close(%s): lease already gone, nothing to do.", lease_id)
                return
            # Treat server errors (500/502/503) on close as non-fatal
            # If pool_server crashed mid-close, don't block rollout; watchdog will clean up
            if status_code in (500, 502, 503):
                logger.warning(
                    "close(%s): server error HTTP %d during close, treating as non-fatal: %s",
                    lease_id,
                    status_code,
                    combined[:200],
                )
                return
            raise
        if not out.get("ok", False):
            error_msg = str(out.get("error", ""))
            if "Unknown" in error_msg and "lease" in error_msg.lower():
                logger.debug("close(%s): lease already gone, nothing to do.", lease_id)
                return
            raise RuntimeError(f"close failed: {out}")
