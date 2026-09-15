"""Shared conversion of SGLang parser output to OpenAI tool calls."""

from __future__ import annotations

import json
import logging
import traceback
import uuid
from typing import Any

from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
    Function,
)

logger = logging.getLogger(__name__)


def _tool_arguments_json(arguments: Any) -> str:
    if isinstance(arguments, dict):
        normalized = arguments
    elif isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            normalized = {"__raw_arguments__": arguments}
        else:
            normalized = (
                parsed if isinstance(parsed, dict) else {"__raw_arguments__": parsed}
            )
    else:
        normalized = {"__raw_arguments__": arguments}
    return json.dumps(normalized, ensure_ascii=False)


def process_tool_calls(
    text: str,
    tools: list[Any],
    tool_call_parser: str | None,
    finish_reason: str,
) -> tuple[
    list[ChatCompletionMessageFunctionToolCall] | None,
    str,
    str,
]:
    from sglang.srt.entrypoints.openai.protocol import Function as SglFunction
    from sglang.srt.entrypoints.openai.protocol import Tool as SglTool
    from sglang.srt.function_call.function_call_parser import FunctionCallParser

    parser_tools = [
        SglTool(type=tool["type"], function=SglFunction(**tool["function"]))
        for tool in tools
    ]
    parser = FunctionCallParser(parser_tools, tool_call_parser)
    if parser.has_tool_call(text):
        if finish_reason == "stop":
            finish_reason = "tool_calls"
        try:
            text, call_info_list = parser.parse_non_stream(text)
            tool_calls = [
                ChatCompletionMessageFunctionToolCall(
                    type="function",
                    id=f"call_{uuid.uuid4().hex[:24]}",
                    function=Function(
                        name=call_info.name,
                        arguments=_tool_arguments_json(call_info.parameters),
                    ),
                )
                for call_info in call_info_list
            ]
            return tool_calls, text, finish_reason
        except Exception as exc:
            logger.error("Tool call parsing error: %s", exc)
            traceback.print_exc()
            return None, text, finish_reason

    return None, text, finish_reason


__all__ = ["process_tool_calls"]
