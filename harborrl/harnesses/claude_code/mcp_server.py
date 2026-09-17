from __future__ import annotations

import json
import os
import shlex
import time
import uuid
from pathlib import Path
from typing import Any
from urllib import error, request

from mcp.server.fastmcp import FastMCP


SERVER_NAME = "terminal_rl"


from harborrl.env import env_float as _env_float, env_int as _env_int


def _base_url() -> str:
    value = os.getenv("CLAUDE_CODE_TERMINAL_ENV_SERVER_URL", "").strip().rstrip("/")
    if not value:
        raise RuntimeError("CLAUDE_CODE_TERMINAL_ENV_SERVER_URL is required")
    return value


def _lease_id() -> str:
    value = os.getenv("CLAUDE_CODE_TERMINAL_LEASE_ID", "").strip()
    if not value:
        raise RuntimeError("CLAUDE_CODE_TERMINAL_LEASE_ID is required")
    return value


def _json_post(path: str, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        f"{_base_url()}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    retries = max(1, _env_int("CLAUDE_CODE_HTTP_MAX_RETRIES", 3))
    delay = max(0.0, _env_float("CLAUDE_CODE_HTTP_RETRY_DELAY", 1.0))
    last_exc: BaseException | None = None
    for attempt in range(retries):
        try:
            with request.urlopen(req, timeout=timeout) as resp:
                text = resp.read().decode("utf-8")
                return json.loads(text) if text else {}
        except error.HTTPError as exc:
            last_exc = exc
            if exc.code not in {429, 500, 502, 503, 504} or attempt == retries - 1:
                detail = exc.read().decode("utf-8", errors="replace")[:1000]
                raise RuntimeError(f"HTTP {exc.code} from env server: {detail}") from exc
        except Exception as exc:
            last_exc = exc
            if attempt == retries - 1:
                raise
        if delay > 0:
            time.sleep(delay * (attempt + 1))
    raise RuntimeError(f"env server request failed: {last_exc}")


def _record_tool_call(record: dict[str, Any]) -> None:
    path = os.getenv("CLAUDE_CODE_TOOL_LOG_PATH", "").strip()
    if not path:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, default=str))
        fh.write("\n")


def _clamp_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _exec_terminal_tool(tool_name: str, arguments: dict[str, Any]) -> str:
    if tool_name == "shell_exec" and not arguments.get("id"):
        arguments = {**arguments, "id": "claude-" + uuid.uuid4().hex[:12]}
    timeout = max(1.0, _env_float("CLAUDE_CODE_TOOL_TIMEOUT_SEC", 300.0))
    try:
        _json_post("/heartbeat", {"lease_id": _lease_id()}, timeout=min(timeout, 30.0))
    except Exception:
        # Let the actual tool call surface the terminal-env failure if the lease is bad.
        pass

    call_id = f"claude-code-{uuid.uuid4().hex[:16]}"
    started = time.monotonic()
    record: dict[str, Any] = {
        "tool_call_id": call_id,
        "tool_name": tool_name,
        "args": dict(arguments),
        "source": "claude-code-mcp",
    }
    try:
        out = _json_post(
            "/exec_tool",
            {
                "lease_id": _lease_id(),
                "tool_call": {"name": tool_name, "arguments": arguments},
            },
            timeout=timeout,
        )
        if not out.get("ok", False):
            raise RuntimeError(f"exec_tool failed: {out}")
        observation = str(out.get("observation", ""))
        record["result"] = observation[:4096]
        return observation
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        record["latency_ms"] = (time.monotonic() - started) * 1000.0
        _record_tool_call(record)


mcp = FastMCP(
    SERVER_NAME,
    instructions=(
        "Terminal-RL tools execute inside the current benchmark Docker/container "
        "lease. Use these tools instead of local filesystem or shell tools."
    ),
)


@mcp.tool(description="Execute a shell command inside the terminal-rl task environment.")
def shell_exec(
    command: str,
    id: str = "",
    block: bool = True,
    timeout: int = 20,
) -> str:
    args: dict[str, Any] = {"command": command, "block": block, "timeout": timeout}
    if id:
        args["id"] = id
    return _exec_terminal_tool("shell_exec", args)


@mcp.tool(description="Read output from a running shell session by id.")
def shell_view(id: str) -> str:
    return _exec_terminal_tool("shell_view", {"id": id})


@mcp.tool(description="Write input to a running shell session by id.")
def shell_write_to_process(
    id: str,
    input: str = "",
    press_enter: bool = True,
) -> str:
    return _exec_terminal_tool(
        "shell_write_to_process",
        {"id": id, "input": input, "press_enter": press_enter},
    )


@mcp.tool(description="Write content to a file inside the terminal-rl task environment.")
def shell_write_content_to_file(file_path: str, content: str) -> str:
    return _exec_terminal_tool(
        "shell_write_content_to_file",
        {"file_path": file_path, "content": content},
    )


@mcp.tool(description="Read bytes from a file inside the terminal-rl task environment.")
def read_file(file_path: str, offset: int = 0, max_bytes: int = 20000) -> str:
    offset = _clamp_int(offset, default=0, minimum=0, maximum=10_000_000)
    max_bytes = _clamp_int(max_bytes, default=20_000, minimum=1, maximum=200_000)
    quoted_path = shlex.quote(file_path)
    if offset:
        command = (
            f"dd if={quoted_path} bs=1 skip={offset} count={max_bytes} "
            "2>/dev/null"
        )
    else:
        command = f"head -c {max_bytes} {quoted_path}"
    return _exec_terminal_tool(
        "shell_exec",
        {"command": command, "block": True, "timeout": 20},
    )


@mcp.tool(description="Write text content to a file inside the terminal-rl task environment.")
def write_file(file_path: str, content: str) -> str:
    return _exec_terminal_tool(
        "shell_write_content_to_file",
        {"file_path": file_path, "content": content},
    )


@mcp.tool(description="List a directory inside the terminal-rl task environment.")
def list_dir(path: str = ".", max_entries: int = 200) -> str:
    max_entries = _clamp_int(max_entries, default=200, minimum=1, maximum=1000)
    command = f"ls -la {shlex.quote(path)} | head -n {max_entries}"
    return _exec_terminal_tool(
        "shell_exec",
        {"command": command, "block": True, "timeout": 20},
    )


if __name__ == "__main__":
    mcp.run("stdio")
