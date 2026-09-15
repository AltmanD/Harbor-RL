from __future__ import annotations

import asyncio
import datetime
import inspect
import logging
import time
import uuid
from copy import deepcopy
from typing import Any, Dict, List

from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage

from harborrl.harnesses.tool_calls import process_tool_calls
from harborrl.http_client import post as async_post

# async_post is a module-level function with a static signature; probing it
# once here avoids per-request inspect.signature() calls in generate_turn().
_POST_SUPPORTS_HEADERS = "headers" in inspect.signature(async_post).parameters
_POST_SUPPORTS_TIMEOUT = "timeout" in inspect.signature(async_post).parameters

from harborrl.types import Interaction

logger = logging.getLogger(__name__)

_TRACE_META_KEYS = (
    "id",
    "finish_reason",
    "prompt_tokens",
    "completion_tokens",
    "cached_tokens",
    "queue_time",
    "e2e_latency",
    "decode_throughput",
    "pd_prefill_bootstrap_queue_duration",
    "pd_prefill_bootstrap_duration",
    "pd_prefill_alloc_wait_duration",
    "pd_prefill_forward_duration",
    "pd_prefill_transfer_queue_duration",
    "pd_decode_prealloc_duration",
    "pd_decode_bootstrap_duration",
    "pd_decode_alloc_wait_duration",
    "pd_decode_transfer_duration",
    "pd_decode_forward_duration",
    "pd_transfer_speed_gb_s",
    "pd_transfer_total_mb",
    "pd_prefill_retry_count",
)


def _ensure_stop_token_ids(
    tokenizer, sampling_params: Dict[str, Any]
) -> Dict[str, Any]:
    normalized = dict(sampling_params)
    if "stop_token_ids" in normalized:
        return normalized

    stop_ids: set[int] = set()
    if tokenizer.eos_token_id is not None:
        stop_ids.add(tokenizer.eos_token_id)
    if tokenizer.pad_token_id is not None:
        stop_ids.add(tokenizer.pad_token_id)
    if stop_ids:
        normalized["stop_token_ids"] = list(stop_ids)
    return normalized


def _to_positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


class SGLangTurnClient:
    def __init__(
        self,
        *,
        model_type: str | None = None,
        tokenizer,
        sampling_params: Dict[str, Any],
        url: str,
        chat_template_type: str = "hf",
        chat_template_kwargs: Dict[str, Any] | None = None,
        messages_delimiter_start: str = "<|im_start|>",
        messages_delimiter_end: str = "<|im_end|>",
        session_id: str | None = None,
        tool_call_parser: str | None = None,
        max_input_tokens: int | None = None,
        request_timeout: float | None = None,
        max_retries: int = 30,
    ) -> None:
        self.model_type = model_type
        self.tokenizer = tokenizer
        self.sampling_params = _ensure_stop_token_ids(tokenizer, sampling_params)
        self.url = url
        self.chat_template_type = chat_template_type
        self.chat_template_kwargs = chat_template_kwargs or {}
        self.messages_delimiter_start = messages_delimiter_start
        self.messages_delimiter_end = messages_delimiter_end
        self.session_id = session_id
        self.tool_call_parser = tool_call_parser
        self.max_input_tokens = _to_positive_int(max_input_tokens)
        self.request_timeout = (
            request_timeout if (request_timeout and request_timeout > 0) else None
        )
        self.max_retries = max(1, max_retries)

        self.marker = "\n[OMITTED MIDDLE]\n"
        self.sep_ids = self.tokenizer.encode(self.marker, add_special_tokens=False)

    def _truncate_input_ids(self, input_ids: List[int]) -> List[int]:
        max_toks = self.max_input_tokens
        if max_toks is None or len(input_ids) <= max_toks:
            return input_ids

        dropped = len(input_ids) - max_toks
        logger.warning(
            "Prompt is too long for configured budget: input=%d, budget=%d. Truncating %d token(s) from the left and right.",
            len(input_ids),
            max_toks,
            dropped,
        )

        keep_head_ratio = 0.3
        head = max(1, int(max_toks * keep_head_ratio))
        tail = max_toks - head - len(self.sep_ids)
        if tail <= 0:
            logger.warning(
                f"tail is not positive: tail={tail}, head={head}, max_toks={max_toks}, len(self.sep_ids)={len(self.sep_ids)}"
            )
            tail = 1

        return input_ids[:head] + self.sep_ids + input_ids[-tail:]

    async def generate_turn(
        self,
        *,
        messages: List[dict[str, Any]],
        tools: List[dict[str, Any]] | None,
        turn_idx: int,
    ) -> tuple[ChatCompletion, Interaction]:
        input_ids = self._apply_chat_template(messages, tools)
        input_ids = self._truncate_input_ids(input_ids)
        payload: Dict[str, Any] = {
            "input_ids": input_ids,
            "sampling_params": self.sampling_params,
            "return_logprob": True,
        }
        headers: Dict[str, str] | None = None
        if self.session_id:
            headers = {"X-SMG-Routing-Key": self.session_id}

        t0 = time.monotonic()

        async def _do_post():
            post_kwargs: Dict[str, Any] = {"max_retries": self.max_retries}
            if self.request_timeout and _POST_SUPPORTS_TIMEOUT:
                post_kwargs["timeout"] = self.request_timeout
            if headers and _POST_SUPPORTS_HEADERS:
                return await async_post(
                    self.url, payload, headers=headers, **post_kwargs
                )
            else:
                if headers and not _POST_SUPPORTS_HEADERS:
                    logger.warning(
                        "async_post() does not accept headers; routing key will be ignored for this request."
                    )
                return await async_post(self.url, payload, **post_kwargs)

        if self.request_timeout:
            try:
                output = await asyncio.wait_for(
                    _do_post(), timeout=self.request_timeout
                )
            except asyncio.TimeoutError:
                elapsed = (time.monotonic() - t0) * 1000.0
                raise TimeoutError(
                    f"SGLang generate request timed out after {self.request_timeout}s "
                    f"(elapsed={elapsed:.0f}ms, turn_idx={turn_idx})"
                )
        else:
            output = await _do_post()
        latency_ms = (time.monotonic() - t0) * 1000.0

        output_text: str = output["text"]
        raw_output_text = output_text
        meta_info = output["meta_info"]
        finish_reason: str = meta_info["finish_reason"]["type"]

        if "output_token_logprobs" in meta_info:
            raw_logprobs = meta_info["output_token_logprobs"]
            if raw_logprobs and logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "output_token_logprobs sample element: %s", raw_logprobs[0]
                )
            output_token_ids: list[int] = [x[1] for x in raw_logprobs]
            output_token_logprobs: list[float] = [x[0] for x in raw_logprobs]
        else:
            output_token_ids = []
            output_token_logprobs = []

        tool_calls = None
        if tools:
            tool_calls, output_text, finish_reason = process_tool_calls(
                output_text,
                tools,
                self.tool_call_parser,
                finish_reason,
            )

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
        current_time = int(datetime.datetime.now().timestamp())
        chat_completion = ChatCompletion(
            id=completion_id,
            choices=[
                Choice(
                    finish_reason=finish_reason,
                    index=0,
                    logprobs=None,
                    message=ChatCompletionMessage(
                        content=output_text,
                        role="assistant",
                        tool_calls=tool_calls,
                    ),
                )
            ],
            created=current_time,
            model=self.model_type or "unknown",
            object="chat.completion",
            service_tier=None,
            system_fingerprint=None,
            usage=CompletionUsage(
                prompt_tokens=len(input_ids),
                completion_tokens=len(output_token_ids),
                total_tokens=len(input_ids) + len(output_token_ids),
            ),
        )

        interaction = Interaction(
            turn_idx=turn_idx,
            completion=deepcopy(chat_completion),
            input_ids=list(input_ids),
            output_token_ids=output_token_ids,
            output_token_logprobs=output_token_logprobs,
            output_text=raw_output_text,
            finish_reason=finish_reason,
            messages=deepcopy(messages),
            latency_ms=latency_ms,
            # Do not retain output_token_logprobs a second time: only compact
            # serving timings belong in the observability carrier.
            generation_meta={
                key: deepcopy(meta_info[key])
                for key in _TRACE_META_KEYS
                if key in meta_info
            },
        )
        return chat_completion, interaction

    def _apply_chat_template(
        self,
        messages: List[dict[str, Any]],
        tools: List[dict[str, Any]] | None,
    ) -> List[int]:
        if self.chat_template_type == "hf":
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tools=tools or None,
                    add_generation_prompt=True,
                    tokenize=True,
                    **self.chat_template_kwargs,
                )
            except Exception:
                return self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    **self.chat_template_kwargs,
                )

        raise ValueError(f"Unsupported chat_template_type: {self.chat_template_type!r}")
