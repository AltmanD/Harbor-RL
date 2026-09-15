from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from datetime import date
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List
from urllib import error, request

from harborrl.harnesses.claude_code.qwen_gateway import ClaudeCodeQwenGateway
from harborrl.harnesses.claude_code.prompts import get_developer_agent_prompt
from harborrl.harnesses.protocol import TurnClient
from harborrl.types import Interaction, TurnResult

logger = logging.getLogger(__name__)


DEFAULT_ALLOWED_TOOLS = (
    "mcp__terminal_rl__shell_exec,"
    "mcp__terminal_rl__shell_view,"
    "mcp__terminal_rl__shell_write_to_process,"
    "mcp__terminal_rl__shell_write_content_to_file,"
    "mcp__terminal_rl__read_file,"
    "mcp__terminal_rl__write_file,"
    "mcp__terminal_rl__list_dir"
)


@dataclass
class ClaudeCodeResponse:
    msg: str
    terminated: bool = False
    info: dict[str, Any] = field(default_factory=dict)
    tool_calls: List[dict[str, Any]] = field(default_factory=list)
    tool_calls_count: int = 0
    raw_result: Any = None

    @property
    def text(self) -> str:
        return self.msg


ClaudeCodeFinalResponse = ClaudeCodeResponse
ClaudeCodeModelResponse = ClaudeCodeResponse


from harborrl.env import (
    env_bool as _env_flag,
    env_float as _env_float,
    env_int as _env_int,
)


def _env_mode(name: str, default: str = "auto") -> str:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    text = raw.strip().lower()
    if text in {"1", "true", "yes", "on", "force", "always"}:
        return "force"
    if text in {"0", "false", "no", "off", "never", "skip"}:
        return "skip"
    return text


def _normalize_llm_backend(value: str | None) -> str:
    text = str(value or "sglang").strip().lower().replace("_", "-")
    if text in {"sglang", "qwen", "qwen-sglang", "local", "local-sglang"}:
        return "sglang"
    if text in {"anthropic", "claude", "claude-api", "external"}:
        return "anthropic"
    logger.warning("Invalid CLAUDE_CODE_LLM_BACKEND=%r; using sglang", value)
    return "sglang"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@lru_cache(maxsize=32)
def _claude_cli_help_text(cli_path: str, timeout_sec: float) -> str:
    try:
        with tempfile.TemporaryDirectory() as tmp:
            completed = subprocess.run(
                [cli_path, "--help"],
                text=True,
                cwd=tmp,
                capture_output=True,
                timeout=timeout_sec if timeout_sec > 0 else None,
                check=False,
            )
    except Exception as exc:
        logger.debug("Unable to probe Claude Code CLI help for %s: %s", cli_path, exc)
        return ""
    return f"{completed.stdout or ''}\n{completed.stderr or ''}"


def _text_from_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return "" if content is None else str(content)


def _last_user_text(messages: List[dict[str, Any]] | None) -> str:
    for message in reversed(messages or []):
        if str(message.get("role", "")).lower() == "user":
            text = _text_from_message_content(message.get("content"))
            if text:
                return text
    return ""


def _tokenize(tokenizer: Any, text: str) -> list[int]:
    if tokenizer is None or not text:
        return []
    try:
        encoded = tokenizer(text, add_special_tokens=False)
        if isinstance(encoded, dict):
            return list(encoded.get("input_ids") or [])
    except Exception:
        pass
    try:
        return list(tokenizer.encode(text, add_special_tokens=False))
    except Exception:
        return []


def _sanitize_filename(value: str) -> str:
    text = "".join(c if c.isalnum() or c in "._-" else "-" for c in str(value))
    text = "-".join(part for part in text.split("-") if part)
    return text[:96].strip("._-") or "task"


def _content_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        chunks: list[str] = []
        for item in value:
            if isinstance(item, dict):
                text = item.get("text")
                if text is None:
                    text = item.get("content")
                if text is not None:
                    chunks.append(str(text))
            elif item is not None:
                chunks.append(str(item))
        return "\n".join(chunks)
    if isinstance(value, dict):
        for key in ("text", "content", "result"):
            if key in value:
                return _content_text(value[key])
    return "" if value is None else str(value)


def _extract_result_text(payload: Any) -> str:
    if isinstance(payload, dict):
        for key in ("result", "text", "response", "summary"):
            value = payload.get(key)
            if value:
                return _content_text(value)
        message = payload.get("message")
        if isinstance(message, dict):
            text = _content_text(message.get("content"))
            if text:
                return text
        content = payload.get("content")
        if content:
            return _content_text(content)
    return ""


def _parse_claude_output(stdout: str, output_format: str) -> tuple[str, Any]:
    stripped = stdout.strip()
    if not stripped:
        return "", None
    if output_format == "text":
        return stripped, None

    parsed_events: list[Any] = []
    try:
        payload = json.loads(stripped)
        text = _extract_result_text(payload)
        return text or stripped, payload
    except Exception:
        pass

    for line in stripped.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parsed_events.append(json.loads(line))
        except Exception:
            continue

    for event in reversed(parsed_events):
        text = _extract_result_text(event)
        if text:
            return text, parsed_events
    return stripped, parsed_events or None


class ClaudeCodeAgent:
    """Claude Code CLI harness for terminal-rl rollouts.

    Claude Code owns the agent loop. Tool calls are restricted to an MCP server
    that forwards terminal tools to the current terminal-rl env lease.
    """

    def __init__(
        self,
        *,
        model_type: str,
        sglang_client: TurnClient,
        max_total_tokens: int,
        env_client: Any | None = None,
        lease_id: str | None = None,
        run_context: Any | None = None,
        task_meta: Dict[str, Any] | None = None,
        non_think_mode: bool | None = None,
        max_parse_errors: int | None = None,
    ) -> None:
        _ = (model_type, max_total_tokens)
        self._sglang_client = sglang_client
        self._env_client = env_client
        self._lease_id = lease_id
        self._run_context = run_context
        self._task_meta = task_meta or {}
        self._non_think_mode = True if non_think_mode is None else bool(non_think_mode)
        self.max_parse_errors = max(1, int(max_parse_errors or 3))
        self.parse_error_count = 0
        self._prompt = ""
        self._session_id = ""
        self._last_response: ClaudeCodeResponse | None = None
        self._tmpdir: tempfile.TemporaryDirectory[str] | None = None
        self._workspace = self._resolve_workspace()
        self._local_run_dir = self._workspace
        self._tool_log_path = self._workspace / "terminal_rl_tool_calls.jsonl"
        self._stdout_path = self._workspace / "claude_stdout.log"
        self._stderr_path = self._workspace / "claude_stderr.log"
        self._mcp_config_path = self._workspace / "claude_mcp_config.json"
        self._qwen_records_path = self._workspace / "qwen_gateway_records.jsonl"
        self._qwen_gateway: ClaudeCodeQwenGateway | None = None

        self._cli = os.getenv("CLAUDE_CODE_CLI", "claude").strip() or "claude"
        self._model = os.getenv("CLAUDE_CODE_MODEL", "").strip()
        self._llm_backend = _normalize_llm_backend(os.getenv("CLAUDE_CODE_LLM_BACKEND"))
        self._qwen_gateway_model = (
            os.getenv("CLAUDE_CODE_QWEN_GATEWAY_MODEL", "qwen-8b-sglang").strip()
            or "qwen-8b-sglang"
        )
        self._output_format = os.getenv("CLAUDE_CODE_OUTPUT_FORMAT", "json").strip() or "json"
        self._turn_timeout_sec = _env_float("CLAUDE_CODE_TURN_TIMEOUT_SEC", 900.0)
        self._tool_timeout_ms = _env_int("CLAUDE_CODE_TOOL_TIMEOUT_MS", 300_000)
        self._max_turns = max(1, _env_int("CLAUDE_CODE_MAX_TOOL_ROUNDS", 10))
        self._mcp_python = os.getenv("CLAUDE_CODE_MCP_PYTHON", sys.executable).strip() or sys.executable
        self._cli_help_timeout_sec = _env_float("CLAUDE_CODE_HELP_TIMEOUT_SEC", 5.0)
        self._max_turns_arg_mode = _env_mode("CLAUDE_CODE_MAX_TURNS_ARG", "auto")
        self._strict_mcp_config = _env_flag("CLAUDE_CODE_STRICT_MCP_CONFIG", True)
        self._disable_builtin_tools = _env_flag("CLAUDE_CODE_DISABLE_BUILTIN_TOOLS", True)
        self._bare_mode = _env_flag("CLAUDE_CODE_BARE", self._llm_backend == "sglang")
        builtin_tools = os.getenv("CLAUDE_CODE_BUILTIN_TOOLS")
        self._builtin_tools = "" if builtin_tools is None else builtin_tools
        self._no_session_persistence = _env_flag(
            "CLAUDE_CODE_NO_SESSION_PERSISTENCE",
            True,
        )
        requested_permission_mode = os.getenv(
            "CLAUDE_CODE_PERMISSION_MODE",
            "bypassPermissions",
        ).strip()
        if (
            requested_permission_mode == "bypassPermissions"
            and hasattr(os, "geteuid")
            and os.geteuid() == 0
        ):
            logger.warning(
                "Claude Code forbids bypassPermissions under root; using dontAsk "
                "permission mode with the explicit terminal_rl allowedTools list."
            )
            requested_permission_mode = "dontAsk"
        self._permission_mode = requested_permission_mode
        self._allowed_tools = os.getenv(
            "CLAUDE_CODE_ALLOWED_TOOLS",
            DEFAULT_ALLOWED_TOOLS,
        ).strip()
        self._disallowed_tools = os.getenv("CLAUDE_CODE_DISALLOWED_TOOLS", "").strip()
        self._extra_args = os.getenv("CLAUDE_CODE_EXTRA_ARGS", "").strip()
        self._system_prompt = os.getenv("CLAUDE_CODE_SYSTEM_PROMPT", "").strip()
        self._execute_qwen_tool_bridge = _env_flag(
            "CLAUDE_CODE_EXECUTE_QWEN_TOOL_USES",
            self._uses_sglang_gateway(),
        )
        self._qwen_tool_bridge_max_calls = max(
            0,
            _env_int("CLAUDE_CODE_QWEN_BRIDGE_MAX_TOOL_CALLS", 1),
        )
        self._minimal_system_prompt = _env_flag(
            "CLAUDE_CODE_MINIMAL_SYSTEM_PROMPT",
            self._uses_sglang_gateway(),
        )
        self._accept_qwen_partial_on_timeout = _env_flag(
            "CLAUDE_CODE_ACCEPT_QWEN_PARTIAL_ON_TIMEOUT",
            self._uses_sglang_gateway(),
        )

    def set_max_parse_errors(self, max_parse_errors: int) -> None:
        self.max_parse_errors = max(1, int(max_parse_errors))

    def set_max_iterations(self, max_iterations: int) -> None:
        self._max_turns = max(1, int(max_iterations))

    def start_turn_loop(self, input_message: Any) -> None:
        self.parse_error_count = 0
        self._last_response = None
        self._prompt = _text_from_message_content(input_message)
        uid = getattr(self._run_context, "uid", None) or uuid.uuid4().hex[:8]
        self._session_id = os.getenv("CLAUDE_CODE_SESSION_ID") or (
            f"terminal-rl-claude-{uid}-{uuid.uuid4().hex[:8]}"
        )
        self._tool_log_path.unlink(missing_ok=True)
        self._stdout_path.unlink(missing_ok=True)
        self._stderr_path.unlink(missing_ok=True)
        self._qwen_records_path.unlink(missing_ok=True)

    async def get_turn_context(
        self,
    ) -> tuple[list[dict[str, Any]] | None, ClaudeCodeResponse | None]:
        if self._last_response is not None:
            return None, self._last_response
        return [{"role": "user", "content": self._prompt}], None

    async def consume_completion(
        self, chat_completion: Any
    ) -> tuple[Any | None, list[Any], bool, ClaudeCodeResponse | None]:
        _ = chat_completion
        raise RuntimeError("ClaudeCodeAgent uses the Claude Code CLI run path")

    def record_tool_result(self, tool_call_request: Any, raw_result: Any) -> None:
        _ = (tool_call_request, raw_result)

    async def run_model_turn(
        self,
        context_messages: list[dict[str, Any]] | None = None,
        *,
        sglang_client: TurnClient | None = None,
        tool_schemas: List[Dict[str, Any]] | None = None,
        turn_idx: int = 0,
    ) -> TurnResult:
        _ = (sglang_client, tool_schemas)
        prompt = _last_user_text(context_messages) or self._prompt
        if prompt:
            self._prompt = prompt

        started = time.monotonic()
        completed = await self._run_claude_code_async(self._prompt)
        latency_ms = (time.monotonic() - started) * 1000.0
        text, raw_result = _parse_claude_output(completed.stdout, self._output_format)
        qwen_records = self._load_qwen_gateway_records()
        tool_calls = self._load_tool_calls()
        if self._uses_sglang_gateway() and not tool_calls:
            bridged_tool_calls = self._execute_qwen_tool_uses(qwen_records)
            if bridged_tool_calls:
                tool_calls = self._load_tool_calls()
        if self._uses_sglang_gateway():
            if not qwen_records:
                raise RuntimeError(
                    "claude-code sglang backend produced no Qwen gateway records; "
                    "cannot build trainable GRPO samples"
                )
            interactions = self._interactions_from_qwen_records(qwen_records, turn_idx)
            interaction = interactions[-1]
        else:
            interaction = self._interaction(turn_idx, self._prompt, text, latency_ms)
            interactions = [interaction]
        self._last_response = self._response_from_completed(
            completed,
            text,
            raw_result,
            tool_calls,
            qwen_records=qwen_records,
        )
        return TurnResult(
            interaction=interaction,
            model_response=self._last_response,
            tool_call_requests=[],
            parse_error_recorded=False,
            terminated_response=None,
            interactions=interactions,
        )

    def finalize_response(self, model_response: Any) -> ClaudeCodeResponse:
        if isinstance(model_response, ClaudeCodeResponse):
            return model_response
        return self._last_response or ClaudeCodeResponse(
            msg="",
            terminated=True,
            info={
                "termination_reasons": ["missing_claude_code_response"],
                "harness_option": "claude-code",
            },
        )

    async def close(self) -> None:
        if self._qwen_gateway is not None:
            self._qwen_gateway.close()
            self._qwen_gateway = None
        if self._tmpdir is not None:
            self._tmpdir.cleanup()
            self._tmpdir = None

    def _resolve_workspace(self) -> Path:
        raw = os.getenv("CLAUDE_CODE_LOCAL_RUN_DIR") or os.getenv("CLAUDE_CODE_WORKSPACE")
        if raw:
            path = Path(raw).expanduser()
        else:
            uid = getattr(self._run_context, "uid", None) or uuid.uuid4().hex[:8]
            group_index = getattr(self._run_context, "group_index", None)
            sample_index = getattr(self._run_context, "sample_index", None)
            task_name = str(self._task_meta.get("task_name") or "task")
            root_env = os.getenv("CLAUDE_CODE_LOCAL_RUN_ROOT") or os.getenv(
                "CLAUDE_CODE_WORKSPACE_ROOT"
            )
            if root_env:
                root = Path(root_env).expanduser()
            else:
                run_log_dir = getattr(self._run_context, "log_dir", None)
                if run_log_dir:
                    root = Path(run_log_dir).expanduser() / "claude_code_cli"
                else:
                    root = _repo_root() / "runs" / "claude_code_cli"
            suffix_parts = [str(uid)]
            if group_index is not None:
                suffix_parts.append(f"g{group_index}")
            if sample_index is not None:
                suffix_parts.append(f"s{sample_index}")
            path = root / (
                f"claude-code-{_sanitize_filename(task_name)}-"
                f"{_sanitize_filename('-'.join(suffix_parts))}"
            )
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _run_claude_code(self, prompt: str) -> subprocess.CompletedProcess[str]:
        args, env = self._prepare_claude_command()

        try:
            completed = subprocess.run(
                args,
                input=prompt,
                text=True,
                cwd=str(self._workspace),
                env=env,
                capture_output=True,
                timeout=self._turn_timeout_sec if self._turn_timeout_sec > 0 else None,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            self._stdout_path.write_text(exc.stdout or "", encoding="utf-8")
            self._stderr_path.write_text(exc.stderr or "", encoding="utf-8")
            if self._accept_qwen_partial_on_timeout:
                partial = self._completed_from_qwen_partial_timeout(args, exc)
                if partial is not None:
                    return partial
            raise TimeoutError(
                f"claude-code CLI timed out after {self._turn_timeout_sec:.0f}s"
            ) from exc

        self._check_completed(completed)
        return completed

    def _completed_from_qwen_partial_timeout(
        self,
        args: list[str],
        exc: subprocess.TimeoutExpired,
    ) -> subprocess.CompletedProcess[str] | None:
        records = self._load_qwen_gateway_records()
        if not records:
            return None
        last_text = ""
        for record in reversed(records):
            last_text = str(
                record.get("clean_text")
                or record.get("output_text")
                or ""
            ).strip()
            if last_text:
                break
        if not last_text:
            last_text = "Claude Code timed out after the local Qwen gateway produced records."
        stdout_payload = {
            "type": "terminal_rl_partial_timeout",
            "result": last_text,
            "terminal_rl_partial_timeout": True,
            "timeout_sec": self._turn_timeout_sec,
            "qwen_gateway_turns": len(records),
        }
        stdout = json.dumps(stdout_payload, ensure_ascii=False)
        stderr = exc.stderr or ""
        if stderr:
            stderr += "\n"
        stderr += (
            "[terminal-rl] claude-code CLI timed out; returning captured "
            "Qwen gateway output as a partial trainable sample."
        )
        self._stdout_path.write_text(stdout, encoding="utf-8")
        self._stderr_path.write_text(stderr, encoding="utf-8")
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout=stdout,
            stderr=stderr,
        )

    async def _run_claude_code_async(self, prompt: str) -> subprocess.CompletedProcess[str]:
        result: list[subprocess.CompletedProcess[str]] = []
        errors: list[BaseException] = []

        def target() -> None:
            try:
                result.append(self._run_claude_code(prompt))
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(
            target=target,
            name="terminal-rl-claude-code-cli",
            daemon=True,
        )
        thread.start()
        started = time.monotonic()
        guard_timeout = (
            self._turn_timeout_sec + 5.0 if self._turn_timeout_sec > 0 else None
        )
        while thread.is_alive():
            if guard_timeout is not None and time.monotonic() - started > guard_timeout:
                raise TimeoutError(
                    f"claude-code CLI thread did not finish after {guard_timeout:.0f}s"
                )
            await asyncio.sleep(0.05)
        if errors:
            raise errors[0]
        if not result:
            raise RuntimeError("claude-code CLI finished without a result")
        return result[0]

    def _prepare_claude_command(self) -> tuple[list[str], dict[str, str]]:
        if self._env_client is None or self._lease_id is None:
            raise RuntimeError("terminal env client is required for claude-code tool execution")
        cli_path = self._resolve_cli()
        self._write_mcp_config()
        if self._uses_sglang_gateway():
            self._ensure_qwen_gateway()
        return self._build_cli_args(cli_path), self._build_subprocess_env()

    def _check_completed(self, completed: subprocess.CompletedProcess[str]) -> None:
        self._stdout_path.write_text(completed.stdout or "", encoding="utf-8")
        self._stderr_path.write_text(completed.stderr or "", encoding="utf-8")
        if completed.returncode != 0:
            stderr = (completed.stderr or "").strip()[:2000]
            raise RuntimeError(
                f"claude-code CLI exited with {completed.returncode}: {stderr}"
            )

    def _resolve_cli(self) -> str:
        if os.path.sep in self._cli:
            path = Path(self._cli).expanduser()
            if not path.exists():
                raise RuntimeError(f"CLAUDE_CODE_CLI does not exist: {path}")
            return str(path)
        resolved = shutil.which(self._cli)
        if not resolved:
            raise RuntimeError(
                f"Claude Code CLI {self._cli!r} not found. Set CLAUDE_CODE_CLI."
            )
        return resolved

    def _build_cli_args(self, cli_path: str) -> list[str]:
        args = [cli_path, "-p", "--output-format", self._output_format]
        if self._bare_mode and self._cli_supports_option(cli_path, "--bare"):
            args.append("--bare")
        if self._model:
            args.extend(["--model", self._model])
        if self._should_use_cli_option(cli_path, "--max-turns", self._max_turns_arg_mode):
            args.extend(["--max-turns", str(self._max_turns)])
        args.extend(["--mcp-config", str(self._mcp_config_path)])
        if self._strict_mcp_config and self._cli_supports_option(cli_path, "--strict-mcp-config"):
            args.append("--strict-mcp-config")
        if self._disable_builtin_tools and self._cli_supports_option(cli_path, "--tools"):
            args.extend(["--tools", self._builtin_tools])
        if self._allowed_tools:
            args.extend(["--allowedTools", self._allowed_tools])
        if self._disallowed_tools:
            args.extend(["--disallowedTools", self._disallowed_tools])
        if self._permission_mode:
            args.extend(["--permission-mode", self._permission_mode])
        if (
            self._no_session_persistence
            and self._cli_supports_option(cli_path, "--no-session-persistence")
        ):
            args.append("--no-session-persistence")
        system_prompt = self._system_prompt or self._default_system_prompt()
        if system_prompt:
            args.extend(["--append-system-prompt", system_prompt])
        if self._extra_args:
            args.extend(shlex.split(self._extra_args))
        return args

    def _cli_supports_option(self, cli_path: str, option: str) -> bool:
        help_text = _claude_cli_help_text(cli_path, self._cli_help_timeout_sec)
        return bool(help_text and option in help_text)

    def _should_use_cli_option(self, cli_path: str, option: str, mode: str) -> bool:
        if mode == "skip":
            return False
        if mode == "force":
            return True
        supported = self._cli_supports_option(cli_path, option)
        if not supported:
            logger.info(
                "Skipping Claude Code CLI option %s because it was not found in --help. "
                "Set CLAUDE_CODE_MAX_TURNS_ARG=1 to force it.",
                option,
            )
        return supported

    def _default_system_prompt(self) -> str:
        mcp_boundary = (
            "You are running inside the HarborRL terminal-rl harness. Use only the "
            "terminal_rl MCP tools for task inspection and modification; they execute "
            "inside the remote benchmark Docker/container lease. Use shell_exec, "
            "read_file, write_file, list_dir, shell_view, and shell_write_to_process "
            "through the MCP server for all reads, writes, and commands. Do not rely "
            "on local Read, Write, Edit, or Bash tools for task state. Keep command "
            "output bounded and stop once the task is complete. When using tools, emit "
            "exactly one valid tool call for the next action; do not wrap tool JSON in "
            "prose, markdown, or shell quoting. Keep private reasoning very short. If "
            "you can solve the task with a command or file write, call the appropriate "
            "terminal_rl MCP tool immediately. After the task is solved, return one "
            "brief final answer and do not request more tools."
        )
        if self._minimal_system_prompt:
            return (
                "/no_think\n"
                "You are in HarborRL harborrl. Do not write long reasoning. "
                "Use exactly one terminal_rl MCP tool call for the next concrete action. "
                "Use only mcp__terminal_rl tools, never local filesystem tools. "
                "If the task is complete, answer briefly and stop.\n"
                f"<terminal_rl_mcp_boundary>\n{mcp_boundary}\n</terminal_rl_mcp_boundary>"
            )
        camel_prompt = get_developer_agent_prompt(
            current_date=str(date.today()),
            system="Linux (in Docker)",
            machine=os.getenv("CLAUDE_CODE_MACHINE", "x86_64"),
            is_workforce=False,
            non_think_mode=self._non_think_mode,
        )
        return f"{camel_prompt}\n\n<terminal_rl_mcp_boundary>\n{mcp_boundary}\n</terminal_rl_mcp_boundary>"

    def _build_subprocess_env(self) -> dict[str, str]:
        env = dict(os.environ)
        env.setdefault("CLAUDE_CODE_ENTRYPOINT", "terminal-rl")
        env["TERMINAL_RL_CLAUDE_CODE_SESSION_ID"] = self._session_id
        if self._uses_sglang_gateway():
            gateway = self._ensure_qwen_gateway()
            env["ANTHROPIC_BASE_URL"] = gateway.base_url
            env["ANTHROPIC_AUTH_TOKEN"] = "terminal-rl-qwen"
            env["ANTHROPIC_API_KEY"] = "terminal-rl-qwen"
            env.pop("ANTHROPIC_API_URL", None)
            env["CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS"] = "1"
            env.setdefault("CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY", "0")
        no_proxy = env.get("NO_PROXY") or env.get("no_proxy") or ""
        parts = [p.strip() for p in no_proxy.split(",") if p.strip()]
        for host in ("127.0.0.1", "localhost", "::1"):
            if host not in parts:
                parts.append(host)
        env["NO_PROXY"] = ",".join(parts)
        env["no_proxy"] = env["NO_PROXY"]
        return env

    def _uses_sglang_gateway(self) -> bool:
        return self._llm_backend == "sglang"

    def _default_non_trainable(self) -> bool:
        return not self._uses_sglang_gateway()

    def _ensure_qwen_gateway(self) -> ClaudeCodeQwenGateway:
        if self._qwen_gateway is None:
            self._qwen_gateway = ClaudeCodeQwenGateway(
                sglang_client=self._sglang_client,
                records_path=self._qwen_records_path,
                model_name=self._qwen_gateway_model,
            )
            self._qwen_gateway.start()
        return self._qwen_gateway

    def _write_mcp_config(self) -> None:
        base_url = str(getattr(self._env_client, "base_url", "")).rstrip("/")
        if not base_url:
            raise RuntimeError("env_client.base_url is required for claude-code MCP tools")
        server_path = Path(__file__).with_name("mcp_server.py")
        env = {
            "CLAUDE_CODE_TERMINAL_ENV_SERVER_URL": base_url,
            "CLAUDE_CODE_TERMINAL_LEASE_ID": str(self._lease_id),
            "CLAUDE_CODE_TOOL_TIMEOUT_SEC": str(max(1.0, self._tool_timeout_ms / 1000.0)),
            "CLAUDE_CODE_TOOL_LOG_PATH": str(self._tool_log_path),
            "CLAUDE_CODE_HTTP_MAX_RETRIES": os.getenv("CLAUDE_CODE_HTTP_MAX_RETRIES", "3"),
            "CLAUDE_CODE_HTTP_RETRY_DELAY": os.getenv("CLAUDE_CODE_HTTP_RETRY_DELAY", "1.0"),
        }
        pythonpath = os.getenv("PYTHONPATH", "")
        if pythonpath:
            env["PYTHONPATH"] = pythonpath
        config = {
            "mcpServers": {
                "terminal_rl": {
                    "command": self._mcp_python,
                    "args": [str(server_path)],
                    "env": env,
                }
            }
        }
        self._mcp_config_path.write_text(
            json.dumps(config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _load_tool_calls(self) -> list[dict[str, Any]]:
        if not self._tool_log_path.exists():
            return []
        records: list[dict[str, Any]] = []
        for line in self._tool_log_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            if isinstance(item, dict):
                records.append(item)
        return records

    def _load_qwen_gateway_records(self) -> list[dict[str, Any]]:
        if self._qwen_gateway is not None:
            records = self._qwen_gateway.records()
            if records:
                return records
        if not self._qwen_records_path.exists():
            return []
        records: list[dict[str, Any]] = []
        for line in self._qwen_records_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            if isinstance(item, dict):
                records.append(item)
        return records

    def _execute_qwen_tool_uses(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not self._execute_qwen_tool_bridge or self._qwen_tool_bridge_max_calls <= 0:
            return []
        if self._env_client is None or self._lease_id is None:
            return []

        bridged: list[dict[str, Any]] = []
        for record in records:
            for block in record.get("anthropic_content") or []:
                if not isinstance(block, dict) or block.get("type") != "tool_use":
                    continue
                raw_name = str(block.get("name") or "").strip()
                if not raw_name:
                    continue
                arguments = block.get("input") or {}
                if not isinstance(arguments, dict):
                    arguments = {"value": arguments}
                tool_name = self._normalize_terminal_tool_name(raw_name)
                if not tool_name:
                    continue
                bridged.append(self._exec_terminal_tool_bridge(tool_name, arguments, raw_name))
                if len(bridged) >= self._qwen_tool_bridge_max_calls:
                    return bridged
        return bridged

    def _normalize_terminal_tool_name(self, raw_name: str) -> str | None:
        name = raw_name.strip()
        if name.startswith("mcp__terminal_rl__"):
            name = name[len("mcp__terminal_rl__") :]
        elif name.startswith("terminal_rl__"):
            name = name[len("terminal_rl__") :]
        allowed = {
            "shell_exec",
            "shell_view",
            "shell_write_to_process",
            "shell_write_content_to_file",
            "read_file",
            "write_file",
            "list_dir",
        }
        if name not in allowed:
            logger.warning("Skipping unsupported Qwen terminal tool %r", raw_name)
            return None
        if name == "read_file":
            return "shell_exec"
        if name == "write_file":
            return "shell_write_content_to_file"
        if name == "list_dir":
            return "shell_exec"
        return name

    def _normalize_terminal_tool_args(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        args = dict(arguments)
        if tool_name == "shell_exec":
            args.setdefault("id", "")
            args.setdefault("block", True)
            args.setdefault("timeout", 20)
            if "command" in args:
                return args
            if "file_path" in args:
                return {"id": "", "command": f"head -c 20000 {shlex.quote(str(args['file_path']))}", "block": True, "timeout": 20}
            if "path" in args:
                max_entries = args.get("max_entries", 200)
                return {"id": "", "command": f"ls -la {shlex.quote(str(args['path']))} | head -n {int(max_entries)}", "block": True, "timeout": 20}
        if tool_name == "shell_write_content_to_file" and "content" in args:
            if "file_path" not in args and "path" in args:
                args["file_path"] = args.pop("path")
        return args

    def _env_json_post(self, path: str, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
        base_url = str(getattr(self._env_client, "base_url", "")).rstrip("/")
        if not base_url:
            raise RuntimeError("env_client.base_url is required for Qwen tool bridge")
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            f"{base_url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=timeout) as resp:
            text = resp.read().decode("utf-8")
            return json.loads(text) if text else {}

    def _exec_terminal_tool_bridge(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        raw_name: str,
    ) -> dict[str, Any]:
        args = self._normalize_terminal_tool_args(tool_name, arguments)
        call_id = f"qwen-bridge-{uuid.uuid4().hex[:16]}"
        started = time.monotonic()
        record: dict[str, Any] = {
            "tool_call_id": call_id,
            "tool_name": tool_name,
            "raw_tool_name": raw_name,
            "args": dict(args),
            "source": "qwen-gateway-direct-bridge",
        }
        try:
            try:
                self._env_json_post("/heartbeat", {"lease_id": str(self._lease_id)}, timeout=30.0)
            except Exception:
                pass
            out = self._env_json_post(
                "/exec_tool",
                {
                    "lease_id": str(self._lease_id),
                    "tool_call": {"name": tool_name, "arguments": args},
                },
                timeout=max(1.0, self._tool_timeout_ms / 1000.0),
            )
            if not out.get("ok", False):
                raise RuntimeError(f"exec_tool failed: {out}")
            observation = str(out.get("observation", ""))
            record["result"] = observation[:4096]
            return record
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
            record["error"] = f"HTTPError {exc.code}: {detail}"
            raise RuntimeError(record["error"]) from exc
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            record["latency_ms"] = (time.monotonic() - started) * 1000.0
            self._append_tool_call_record(record)

    def _append_tool_call_record(self, record: dict[str, Any]) -> None:
        self._tool_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._tool_log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str))
            fh.write("\n")

    def _interactions_from_qwen_records(
        self,
        records: list[dict[str, Any]],
        first_turn_idx: int,
    ) -> list[Interaction]:
        interactions: list[Interaction] = []
        for offset, record in enumerate(records):
            output_token_ids = [int(x) for x in (record.get("output_token_ids") or [])]
            output_token_logprobs = [
                float(x) for x in (record.get("output_token_logprobs") or [])
            ]
            if len(output_token_logprobs) != len(output_token_ids):
                raise RuntimeError(
                    "Qwen gateway token/logprob length mismatch: "
                    f"{len(output_token_ids)} tokens vs {len(output_token_logprobs)} logprobs"
                )
            interactions.append(
                Interaction(
                    turn_idx=first_turn_idx + offset,
                    input_ids=[int(x) for x in (record.get("input_ids") or [])],
                    output_token_ids=output_token_ids,
                    output_token_logprobs=output_token_logprobs,
                    output_text=str(record.get("output_text") or ""),
                    finish_reason=str(record.get("finish_reason") or "stop"),
                    messages=list(record.get("messages") or []),
                    latency_ms=float(record.get("latency_ms") or 0.0),
                )
            )
        return interactions

    def _response_from_completed(
        self,
        completed: subprocess.CompletedProcess[str],
        text: str,
        raw_result: Any,
        tool_calls: list[dict[str, Any]],
        *,
        qwen_records: list[dict[str, Any]] | None = None,
    ) -> ClaudeCodeResponse:
        non_trainable = _env_flag(
            "CLAUDE_CODE_MARK_NON_TRAINABLE",
            self._default_non_trainable(),
        )
        info = {
            "termination_reasons": [],
            "harness_option": "claude-code",
            "harness": "claude-code",
            "source": (
                "claude-code-cli+sglang-gateway"
                if self._uses_sglang_gateway()
                else "claude-code-cli"
            ),
            "llm_backend": self._llm_backend,
            "session_id": self._session_id,
            "local_run_dir": str(self._local_run_dir),
            "local_run_dir_kind": "logs_and_cli_control_only",
            "workspace": str(self._local_run_dir),
            "workspace_kind": "logs_and_cli_control_only",
            "task_path": self._task_meta.get("task_path"),
            "returncode": completed.returncode,
            "stdout_path": str(self._stdout_path),
            "stderr_path": str(self._stderr_path),
            "mcp_config_path": str(self._mcp_config_path),
            "output_format": self._output_format,
            "tool_calls_count": len(tool_calls),
            "tool_calls": list(tool_calls),
            "qwen_gateway_records_path": str(self._qwen_records_path),
            "qwen_gateway_turns": len(qwen_records or []),
            "non_trainable": non_trainable,
            "non_trainable_reason": None if not non_trainable else (
                "claude-code CLI uses an external model path without terminal-rl "
                "policy logprobs"
            ),
        }
        return ClaudeCodeResponse(
            msg=text,
            terminated=False,
            info=info,
            tool_calls=list(tool_calls),
            tool_calls_count=len(tool_calls),
            raw_result=raw_result,
        )

    def _interaction(
        self,
        turn_idx: int,
        prompt: str,
        text: str,
        latency_ms: float,
    ) -> Interaction:
        tokenizer = getattr(self._sglang_client, "tokenizer", None)
        input_ids = _tokenize(tokenizer, prompt)
        output_ids = _tokenize(tokenizer, text)
        return Interaction(
            turn_idx=turn_idx,
            input_ids=input_ids,
            output_token_ids=output_ids,
            output_token_logprobs=[0.0] * len(output_ids),
            output_text=text,
            finish_reason="stop",
            messages=[{"role": "user", "content": prompt}],
            latency_ms=latency_ms,
        )
