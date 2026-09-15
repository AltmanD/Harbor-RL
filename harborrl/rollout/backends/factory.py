from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

from harborrl.rollout.backends.sglang import SGLangTurnClient

logger = logging.getLogger(__name__)


def _infer_completion_budget(sampling_params: Dict[str, Any]) -> int:
    for key in ("max_new_tokens", "max_tokens", "max_completion_tokens"):
        raw_value = sampling_params.get(key)
        if raw_value is None:
            continue
        try:
            parsed = int(raw_value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return 0


def _normalize_tool_schemas(raw_tools: List[Any]) -> List[Dict[str, Any]]:
    schemas: List[Dict[str, Any]] = []
    for tool in raw_tools:
        if hasattr(tool, "get_openai_tool_schema") and callable(
            tool.get_openai_tool_schema
        ):
            schemas.append(tool.get_openai_tool_schema())
        elif isinstance(tool, dict):
            schemas.append(tool)
        else:
            raise TypeError(f"Unsupported tool schema object type: {type(tool)!r}")
    return schemas


def _create_sglang_client(
    args: Any,
    tokenizer: Any,
    sampling_params: Dict[str, Any],
    max_total_tokens: int,
    enable_sglang_non_think: bool,
    *,
    sglang_url: str | None = None,
    max_retries: int = 30,
) -> SGLangTurnClient:
    if not sglang_url:
        sglang_url = (
            f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
        )
    client_template_kwargs = {
        "chat_template_type": getattr(args, "chat_template_type", "hf"),
        "chat_template_kwargs": getattr(args, "chat_template_kwargs", None),
        "messages_delimiter_start": getattr(
            args, "messages_delimiter_start", "<|im_start|>"
        ),
        "messages_delimiter_end": getattr(args, "messages_delimiter_end", "<|im_end|>"),
        "tool_call_parser": getattr(args, "tool_call_parser", "qwen25"),
    }
    if enable_sglang_non_think:
        raw_chat_template_kwargs = client_template_kwargs.get("chat_template_kwargs")
        if isinstance(raw_chat_template_kwargs, dict):
            merged_chat_template_kwargs = dict(raw_chat_template_kwargs)
        else:
            merged_chat_template_kwargs = {}
        merged_chat_template_kwargs["enable_thinking"] = False
        client_template_kwargs["chat_template_kwargs"] = merged_chat_template_kwargs

    completion_budget = _infer_completion_budget(sampling_params)
    effective_context_limit = max_total_tokens
    for maybe_cap in (
        getattr(args, "rollout_max_context_len", None),
        getattr(args, "sglang_max_context_len", None),
    ):
        try:
            parsed_cap = int(maybe_cap)
        except (TypeError, ValueError):
            continue
        if parsed_cap > 0:
            effective_context_limit = min(effective_context_limit, parsed_cap)
    max_input_tokens = max(1, effective_context_limit - completion_budget)
    logger.info(
        "SGLang client: url=%s context_limit=%d, completion_budget=%d, max_input_tokens=%d",
        sglang_url,
        effective_context_limit,
        completion_budget,
        max_input_tokens,
    )
    raw_request_timeout = getattr(args, "sglang_request_timeout", None)
    if raw_request_timeout in (None, "", 0, 0.0):
        raw_request_timeout = os.getenv("SGLANG_REQUEST_TIMEOUT")
    try:
        request_timeout = (
            float(raw_request_timeout) if raw_request_timeout is not None else None
        )
    except (TypeError, ValueError):
        request_timeout = None
    if request_timeout is not None and request_timeout <= 0:
        request_timeout = None

    return SGLangTurnClient(
        model_type=None,
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        url=sglang_url,
        session_id=None,
        max_input_tokens=max_input_tokens,
        request_timeout=request_timeout,
        max_retries=max_retries,
        **client_template_kwargs,
    )
