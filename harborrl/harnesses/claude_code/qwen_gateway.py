from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib import request

from harborrl.harnesses.protocol import TurnClient
from harborrl.harnesses.tool_calls import process_tool_calls

logger = logging.getLogger(__name__)


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except Exception:
        return str(value)


def _content_block_text(block: dict[str, Any]) -> str:
    block_type = str(block.get("type") or "")
    if block_type == "text":
        return str(block.get("text") or "")
    if block_type == "tool_result":
        content = block.get("content")
        if isinstance(content, list):
            content_text = "\n".join(_content_block_text(x) for x in content if isinstance(x, dict))
        else:
            content_text = "" if content is None else str(content)
        return (
            f"<tool_result id={block.get('tool_use_id', '')} "
            f"is_error={block.get('is_error', False)}>\n{content_text}\n</tool_result>"
        )
    if block_type == "tool_use":
        return (
            f"<tool_use id={block.get('id', '')} name={block.get('name', '')}>"
            f"{json.dumps(block.get('input') or {}, ensure_ascii=False)}</tool_use>"
        )
    if "text" in block:
        return str(block.get("text") or "")
    return json.dumps(block, ensure_ascii=False, default=str)


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = _content_block_text(item)
            else:
                text = "" if item is None else str(item)
            if text:
                parts.append(text)
        return "\n".join(parts)
    if isinstance(content, dict):
        return _content_block_text(content)
    return "" if content is None else str(content)


def _anthropic_messages_to_chat(payload: dict[str, Any]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    system = payload.get("system")
    system_text = _content_to_text(system)
    if system_text:
        messages.append({"role": "system", "content": system_text})

    for item in payload.get("messages") or []:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "user")
        if role not in {"system", "user", "assistant", "tool"}:
            role = "user"
        messages.append({"role": role, "content": _content_to_text(item.get("content"))})
    return messages or [{"role": "user", "content": ""}]


def _anthropic_tools_to_openai(tools: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not isinstance(tools, list):
        return out
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        name = str(tool.get("name") or "").strip()
        if not name:
            continue
        out.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": str(tool.get("description") or ""),
                    "parameters": tool.get("input_schema") or {"type": "object"},
                },
            }
        )
    return out


def _openai_tool_calls_to_anthropic(tool_calls: Any) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for call in tool_calls or []:
        function = getattr(call, "function", None)
        name = getattr(function, "name", None)
        arguments = getattr(function, "arguments", "{}")
        if not name:
            continue
        try:
            parsed_args = json.loads(arguments or "{}")
        except Exception:
            parsed_args = {"__raw_arguments__": arguments}
        blocks.append(
            {
                "type": "tool_use",
                "id": f"toolu_{uuid.uuid4().hex[:24]}",
                "name": str(name),
                "input": parsed_args if isinstance(parsed_args, dict) else {"value": parsed_args},
            }
        )
    return blocks


class ClaudeCodeQwenGateway:
    """Minimal Anthropic Messages gateway backed by the rollout SGLang client."""

    def __init__(
        self,
        *,
        sglang_client: TurnClient,
        records_path: Path,
        model_name: str = "qwen-8b-sglang",
    ) -> None:
        self._client = sglang_client
        self._records_path = records_path
        self._model_name = model_name or "qwen-8b-sglang"
        self._records: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.base_url = ""

    def start(self) -> str:
        if self._httpd is not None:
            return self.base_url

        gateway = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args: Any) -> None:
                logger.debug("claude-code qwen gateway: " + fmt, *args)

            def do_GET(self) -> None:
                gateway._handle_get(self)

            def do_POST(self) -> None:
                gateway._handle_post(self)

        self._records_path.parent.mkdir(parents=True, exist_ok=True)
        self._records_path.unlink(missing_ok=True)
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        port = int(self._httpd.server_address[1])
        self.base_url = f"http://127.0.0.1:{port}"
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            name="terminal-rl-claude-code-qwen-gateway",
            daemon=True,
        )
        self._thread.start()
        return self.base_url

    def close(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def records(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(item) for item in self._records]

    def _handle_get(self, handler: BaseHTTPRequestHandler) -> None:
        if handler.path.rstrip("/") == "/v1/models":
            self._write_json(
                handler,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": self._model_name,
                            "object": "model",
                            "created": 0,
                            "owned_by": "terminal-rl",
                            "display_name": "Qwen via terminal-rl SGLang",
                        }
                    ],
                },
            )
            return
        self._write_json(handler, {"error": {"message": "not found"}}, status=404)

    def _handle_post(self, handler: BaseHTTPRequestHandler) -> None:
        try:
            length = int(handler.headers.get("Content-Length") or "0")
            raw = handler.rfile.read(length).decode("utf-8")
            payload = json.loads(raw or "{}")
        except Exception as exc:
            self._write_json(handler, {"error": {"message": f"bad request: {exc}"}}, status=400)
            return

        path = handler.path.split("?", 1)[0].rstrip("/")
        try:
            if path == "/v1/messages/count_tokens":
                self._handle_count_tokens(handler, payload)
            elif path == "/v1/messages":
                response = self._build_message_response(payload)
                if payload.get("stream"):
                    self._write_sse(handler, response)
                else:
                    self._write_json(handler, response)
            else:
                self._write_json(handler, {"error": {"message": "not found"}}, status=404)
        except Exception as exc:
            logger.exception("claude-code qwen gateway request failed")
            self._write_json(
                handler,
                {"error": {"type": "terminal_rl_gateway_error", "message": str(exc)}},
                status=500,
            )

    def _handle_count_tokens(
        self, handler: BaseHTTPRequestHandler, payload: dict[str, Any]
    ) -> None:
        messages = _anthropic_messages_to_chat(payload)
        tools = _anthropic_tools_to_openai(payload.get("tools"))
        input_ids = self._apply_template(messages, tools)
        self._write_json(handler, {"input_tokens": len(input_ids)})

    def _build_message_response(self, payload: dict[str, Any]) -> dict[str, Any]:
        started = time.monotonic()
        messages = _anthropic_messages_to_chat(payload)
        tools = _anthropic_tools_to_openai(payload.get("tools"))
        input_ids = self._apply_template(messages, tools)
        input_ids = self._client.truncate_input_ids(input_ids)

        sampling_params = dict(self._client.sampling_params)
        default_max_new_tokens = sampling_params.get("max_new_tokens")
        requested_max_tokens = payload.get("max_tokens")
        if requested_max_tokens is not None:
            try:
                requested_max_tokens_int = max(1, int(requested_max_tokens))
                if default_max_new_tokens is not None:
                    sampling_params["max_new_tokens"] = min(
                        requested_max_tokens_int, max(1, int(default_max_new_tokens))
                    )
                else:
                    sampling_params["max_new_tokens"] = requested_max_tokens_int
            except (TypeError, ValueError):
                pass
        max_new_tokens_cap = os.getenv("CLAUDE_CODE_QWEN_MAX_NEW_TOKENS", "").strip()
        if max_new_tokens_cap:
            try:
                sampling_params["max_new_tokens"] = min(
                    max(1, int(sampling_params.get("max_new_tokens") or max_new_tokens_cap)),
                    max(1, int(max_new_tokens_cap)),
                )
            except (TypeError, ValueError):
                logger.warning(
                    "Invalid CLAUDE_CODE_QWEN_MAX_NEW_TOKENS=%r; ignoring",
                    max_new_tokens_cap,
                )
        for src_key, dst_key in (("temperature", "temperature"), ("top_p", "top_p"), ("top_k", "top_k")):
            if payload.get(src_key) is not None:
                sampling_params[dst_key] = payload[src_key]

        sglang_output = self._post_sglang(
            {
                "input_ids": input_ids,
                "sampling_params": sampling_params,
                "return_logprob": True,
            }
        )
        raw_text = str(sglang_output.get("text") or "")
        meta_info = sglang_output.get("meta_info") or {}
        raw_finish = meta_info.get("finish_reason") or {}
        finish_reason = str(raw_finish.get("type") if isinstance(raw_finish, dict) else raw_finish)
        if not finish_reason:
            finish_reason = "stop"

        raw_logprobs = meta_info.get("output_token_logprobs") or []
        output_token_ids = [int(x[1]) for x in raw_logprobs if isinstance(x, (list, tuple)) and len(x) >= 2]
        output_token_logprobs = [
            float(x[0]) for x in raw_logprobs if isinstance(x, (list, tuple)) and len(x) >= 2
        ]
        if raw_text and not output_token_ids:
            raise RuntimeError(
                "SGLang response text was non-empty but output_token_logprobs "
                "did not contain token ids; refusing to build trainable claude-code sample"
            )
        if len(output_token_ids) != len(output_token_logprobs):
            raise RuntimeError(
                "SGLang output_token_logprobs token/logprob length mismatch: "
                f"{len(output_token_ids)} tokens vs {len(output_token_logprobs)} logprobs"
            )

        clean_text = raw_text
        tool_blocks: list[dict[str, Any]] = []
        if tools:
            try:
                tool_calls, clean_text, parsed_finish = process_tool_calls(
                    raw_text,
                    tools,
                    self._client.tool_call_parser,
                    finish_reason,
                )
                if tool_calls:
                    finish_reason = parsed_finish
                    tool_blocks = _openai_tool_calls_to_anthropic(tool_calls)
            except Exception:
                logger.exception("Failed to parse Qwen tool call for Claude Code gateway")

        content: list[dict[str, Any]] = []
        if clean_text.strip():
            content.append({"type": "text", "text": clean_text})
        content.extend(tool_blocks)
        if not content:
            content.append({"type": "text", "text": ""})

        stop_reason = "tool_use" if tool_blocks else ("max_tokens" if finish_reason == "length" else "end_turn")
        response = {
            "id": f"msg_{uuid.uuid4().hex[:24]}",
            "type": "message",
            "role": "assistant",
            "model": str(payload.get("model") or self._model_name),
            "content": content,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {
                "input_tokens": len(input_ids),
                "output_tokens": len(output_token_ids),
            },
        }
        latency_ms = (time.monotonic() - started) * 1000.0
        record = {
            "generation_meta": dict(meta_info),
            "auxiliary_request": any(
                m.get("role") == "system"
                and "Generate a concise, sentence-case title (3-7 words)" in str(m.get("content", ""))
                for m in messages
            ),
            "messages": messages,
            "tools_count": len(tools),
            "input_ids": input_ids,
            "output_token_ids": output_token_ids,
            "output_token_logprobs": output_token_logprobs,
            "output_text": raw_text,
            "clean_text": clean_text,
            "finish_reason": finish_reason,
            "stop_reason": stop_reason,
            "anthropic_content": content,
            "latency_ms": latency_ms,
        }
        self._record(record)
        return response

    def _apply_template(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> list[int]:
        try:
            return self._client.apply_chat_template(messages, tools)
        except Exception:
            return self._client.apply_chat_template(messages, None)

    def _post_sglang(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self._client.session_id:
            headers["X-SMG-Routing-Key"] = self._client.session_id
        req = request.Request(self._client.url, data=body, headers=headers, method="POST")
        timeout = self._client.request_timeout
        retries = max(1, int(getattr(self._client, "max_retries", 1) or 1))
        last_exc: BaseException | None = None
        for attempt in range(retries):
            try:
                with request.urlopen(req, timeout=timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except Exception as exc:
                last_exc = exc
                if attempt == retries - 1:
                    raise
                time.sleep(min(2.0, 0.1 * (attempt + 1)))
        raise RuntimeError(f"SGLang request failed: {last_exc}")

    def _record(self, record: dict[str, Any]) -> None:
        with self._lock:
            self._records.append(record)
            with self._records_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False, default=_jsonable))
                fh.write("\n")

    def _write_json(
        self,
        handler: BaseHTTPRequestHandler,
        payload: dict[str, Any],
        *,
        status: int = 200,
    ) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=_jsonable).encode("utf-8")
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)

    def _write_sse(self, handler: BaseHTTPRequestHandler, response: dict[str, Any]) -> None:
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.send_header("Cache-Control", "no-cache")
        handler.send_header("Connection", "close")
        handler.end_headers()

        def event(name: str, data: dict[str, Any]) -> None:
            handler.wfile.write(f"event: {name}\n".encode("utf-8"))
            handler.wfile.write(
                f"data: {json.dumps(data, ensure_ascii=False, default=_jsonable)}\n\n".encode("utf-8")
            )
            handler.wfile.flush()

        event(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    **response,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": response["usage"]["input_tokens"], "output_tokens": 0},
                },
            },
        )
        for idx, block in enumerate(response.get("content") or []):
            if block.get("type") == "tool_use":
                start_block = {
                    "type": "tool_use",
                    "id": block.get("id"),
                    "name": block.get("name"),
                    "input": {},
                }
                event("content_block_start", {"type": "content_block_start", "index": idx, "content_block": start_block})
                event(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": idx,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": json.dumps(block.get("input") or {}, ensure_ascii=False),
                        },
                    },
                )
            else:
                start_block = {"type": "text", "text": ""}
                event("content_block_start", {"type": "content_block_start", "index": idx, "content_block": start_block})
                event(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": idx,
                        "delta": {"type": "text_delta", "text": str(block.get("text") or "")},
                    },
                )
            event("content_block_stop", {"type": "content_block_stop", "index": idx})

        event(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {
                    "stop_reason": response.get("stop_reason"),
                    "stop_sequence": response.get("stop_sequence"),
                },
                "usage": {"output_tokens": response["usage"]["output_tokens"]},
            },
        )
        event("message_stop", {"type": "message_stop"})
        handler.close_connection = True
