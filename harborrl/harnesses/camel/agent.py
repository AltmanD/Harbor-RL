from __future__ import annotations

import asyncio
import datetime
import json
import logging
import re
import threading
import time
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type, Union

from openai import AsyncStream, Stream
from openai.lib.streaming.chat import (
    AsyncChatCompletionStreamManager,
    ChatCompletionStreamManager,
)
from openai.types.chat import ChatCompletion
from pydantic import BaseModel

from camel.agents import ChatAgent
from camel.messages import BaseMessage, FunctionCallingMessage, OpenAIMessage
from camel.models import BaseModelBackend
from camel.responses import ChatAgentResponse
from camel.types import ChatCompletionChunk, ModelType, OpenAIBackendRole
from camel.types.agents import ToolCallingRecord
from camel.utils import OpenAITokenCounter
from camel.utils.token_counting import BaseTokenCounter

from harborrl.harnesses.protocol import TurnClient

from .prompts import get_developer_agent_prompt

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from transformers.tokenization_utils_fast import PreTrainedTokenizerFast


class HFTokenCounter(BaseTokenCounter):
    """Token counter backed by the same HF tokenizer used for generation."""

    def __init__(
        self, tokenizer: "PreTrainedTokenizerFast", tokens_per_message: int = 3
    ) -> None:
        self.tokenizer = tokenizer
        self.tokens_per_message = tokens_per_message

    def count_tokens_from_messages(self, messages: List[OpenAIMessage]) -> int:
        num_tokens = 0
        for message in messages:
            num_tokens += self.tokens_per_message
            for _, value in message.items():
                if not isinstance(value, list):
                    num_tokens += len(self.tokenizer.encode(str(value)))
                    continue
                for item in value:
                    if isinstance(item, dict) and item.get("type") == "text":
                        num_tokens += len(
                            self.tokenizer.encode(str(item.get("text", "")))
                        )
                    else:
                        num_tokens += len(self.tokenizer.encode(str(item)))
        num_tokens += 3
        return num_tokens

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)

    def decode(self, token_ids: List[int]) -> str:
        return self.tokenizer.decode(token_ids)


class CamelAgentBackend(BaseModelBackend):
    """Backend adapter that can reuse the external SGLang turn client."""

    def __init__(
        self,
        model_type: ModelType | str,
        *,
        sglang_client: TurnClient | None = None,
        model_config_dict: Dict[str, Any] | None = None,
        token_counter: BaseTokenCounter | None = None,
    ) -> None:
        super().__init__(
            model_type=model_type,
            model_config_dict=model_config_dict or {},
            api_key=None,
            url=None,
            token_counter=token_counter,
            timeout=30.0,
            max_retries=0,
        )
        self._sglang_client = sglang_client
        self._turn_counter = 0
        self.cache: dict[str, Any] = {}

    @property
    def token_counter(self) -> BaseTokenCounter:
        if not self._token_counter:
            hf_tokenizer = None
            if self._sglang_client is not None:
                hf_tokenizer = getattr(self._sglang_client, "tokenizer", None)
            if hf_tokenizer is not None:
                self._token_counter = HFTokenCounter(hf_tokenizer)
            else:
                self._token_counter = OpenAITokenCounter(ModelType.GPT_4O_MINI)
        return self._token_counter

    @property
    def stream(self) -> bool:
        return False

    def _run(
        self,
        messages: list[OpenAIMessage],
        response_format: type[BaseModel] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> (
        ChatCompletion
        | Stream[ChatCompletionChunk]
        | ChatCompletionStreamManager[BaseModel]
    ):
        _ = (messages, response_format, tools)
        raise RuntimeError("CamelAgentBackend._run is not used by AgentRunner.")

    async def _arun(
        self,
        messages: list[OpenAIMessage],
        response_format: type[BaseModel] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> (
        ChatCompletion
        | AsyncStream[ChatCompletionChunk]
        | AsyncChatCompletionStreamManager[BaseModel]
    ):
        _ = response_format
        if self._sglang_client is None:
            raise RuntimeError("CamelAgentBackend has no SGLang client configured.")

        chat_completion, interaction = await self._sglang_client.generate_turn(
            messages=messages,
            tools=tools,
            turn_idx=self._turn_counter,
        )
        self.cache[chat_completion.id] = interaction
        self._turn_counter += 1
        return chat_completion


# Adapted from https://github.com/camel-ai/seta/blob/main/training/tbench_areal_workflow/chat_agent_trace.py
class CamelAgent(ChatAgent):
    """ChatAgent extension used by AgentRunner's rollout loop."""

    def __init__(
        self,
        *,
        model_type: str,
        sglang_client: TurnClient,
        non_think_mode: bool,
        max_total_tokens: int,
        max_parse_errors: int | None = None,
        system: str = "Linux (in Docker)",
        machine: str = "x86_64",
        is_workforce: bool = False,
        current_date: str | None = None,
    ) -> None:
        prompt_date = current_date or str(datetime.date.today())
        system_prompt = get_developer_agent_prompt(
            current_date=prompt_date,
            system=system,
            machine=machine,
            is_workforce=is_workforce,
            non_think_mode=non_think_mode,
        )
        backend = CamelAgentBackend(model_type=model_type, sglang_client=sglang_client)

        super().__init__(
            system_message=BaseMessage.make_assistant_message(
                role_name="Developer Agent",
                content=system_prompt,
            ),
            model=backend,
            tools=[],
            token_limit=max_total_tokens,
        )
        super().reset()

        self.max_parse_errors = max(1, int(max_parse_errors or 3))
        self.parse_error_count = 0
        self._tool_call_records: List[Any] = []
        self._accumulated_context_tokens = 0
        self._step_token_usage = self._create_token_usage_tracker()
        self._original_response_format: Optional[Type[BaseModel]] = None
        self._used_prompt_formatting: bool = False

    def set_max_parse_errors(self, max_parse_errors: int) -> None:
        self.max_parse_errors = max(1, int(max_parse_errors))

    def start_turn_loop(
        self,
        input_message: Union[BaseMessage, str],
        response_format: Optional[Type[BaseModel]] = None,
    ) -> None:
        self.parse_error_count = 0
        self._tool_call_records = []
        self._accumulated_context_tokens = 0
        self._step_token_usage = self._create_token_usage_tracker()

        self._original_response_format = response_format
        input_message, _response_format, used_prompt_formatting = (
            self._handle_response_format_with_non_strict_tools(
                input_message,
                response_format,
            )
        )
        self._used_prompt_formatting = used_prompt_formatting

        if isinstance(input_message, str):
            input_message = BaseMessage.make_user_message(
                role_name="User", content=input_message
            )
        self.update_memory(input_message, OpenAIBackendRole.USER)

    async def _wait_if_paused(self) -> None:
        if self.pause_event is None or self.pause_event.is_set():
            return
        if isinstance(self.pause_event, asyncio.Event):
            await self.pause_event.wait()
            return
        if isinstance(self.pause_event, threading.Event):
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self.pause_event.wait)

    async def get_turn_context(
        self,
    ) -> tuple[Optional[List[dict[str, Any]]], Optional[ChatAgentResponse]]:
        await self._wait_if_paused()
        try:
            context_messages, num_tokens = self.memory.get_context()
            self._accumulated_context_tokens += num_tokens
            return context_messages, None
        except RuntimeError as exc:
            terminated_response = self._step_terminate(
                exc.args[1],
                self._tool_call_records,
                "max_tokens_exceeded",
            )
            return None, terminated_response

    async def consume_completion(
        self, chat_completion: Any
    ) -> tuple[Optional[Any], List[Any], bool, Optional[ChatAgentResponse]]:
        model_response = self._handle_batch_response(chat_completion)
        self._update_token_usage_tracker(
            self._step_token_usage, model_response.usage_dict
        )

        if self.stop_event and self.stop_event.is_set():
            logger.info("Termination triggered by stop_event")
            terminated_response = self._step_terminate(
                self._accumulated_context_tokens,
                self._tool_call_records,
                "termination_triggered",
            )
            return model_response, [], False, terminated_response

        if model_response.tool_call_requests:
            # Recent Camel versions record only the tool result in
            # _record_tool_calling; their normal step loop records the
            # assistant request separately. Our external loop must do so too.
            record_requests = getattr(
                self, "_record_assistant_tool_calls_from_requests", None
            )
            if record_requests is not None:
                content = (
                    (model_response.output_messages[0].content or "")
                    if model_response.output_messages
                    else ""
                )
                record_requests(model_response.tool_call_requests, content=content)
            return model_response, list(model_response.tool_call_requests), False, None

        parse_error_record = await self.adetect_tool_calls_parse_error(model_response)
        if parse_error_record:
            logger.warning(
                f"Detected tool call parse error, prompting model to correct."
            )
            self._tool_call_records.append(parse_error_record)
            return model_response, [], True, None

        return model_response, [], False, None

    async def adetect_tool_calls_parse_error(self, response):
        r"""
        Asynchronously detect tool calls in the response content using Qwen25Detector.
        if the model is Qwen 2.5 or Qwen 3.
        if there's tool call tokens detected, but got json parse failure, format the information into a tool call record,
        so that the agent can handle the error next step.
        add a self.count_parse_error, so that we can limit the number of parse errors we handle in one step. if max reached, just
        break the loop.

        Args:
            response: The model response to check for parse errors

        Returns:
            Optional[ToolCallingRecord]: A tool calling record with error information if parse error detected, None otherwise
        """
        bot_token = "<tool_call>\n"
        eot_token = "\n</tool_call>"

        # Check if we've reached max parse errors
        if self.parse_error_count >= self.max_parse_errors:
            logger.warning(
                f"Max parse errors ({self.max_parse_errors}) reached, stopping error handling"
            )
            return None

        # Extract content from response
        if not response.output_messages:
            return None

        content = response.output_messages[0].content
        if not content or bot_token not in content:
            return None

        # Find all potential tool call blocks
        pattern = rf"{re.escape(bot_token)}(.*?){re.escape(eot_token)}"
        matches = re.findall(pattern, content, re.DOTALL)

        if not matches:
            return None

        # Check each match for JSON parse errors
        for match_text in matches:
            try:
                # Try to parse the JSON
                json.loads(match_text.strip())
                # If successful, no error for this match
                continue
            except json.JSONDecodeError as e:
                # Found a parse error
                self.parse_error_count += 1
                logger.warning(
                    f"Detected JSON parse error (count: {self.parse_error_count}/{self.max_parse_errors}): {str(e)}"
                )
                logger.warning(f"Problematic content: {match_text[:200]}...")

                # Create an error tool calling record
                error_message = (
                    f"JSON Parse Error: {str(e)}\n"
                    f"The tool call format is incorrect. Please ensure:\n"
                    f"1. The JSON is valid and properly formatted\n"
                    f"2. All quotes are properly escaped\n"
                    f"3. The structure matches: {{'name': 'function_name', 'arguments': {{}}}}\n"
                    f"Problematic content (first 200 chars): {match_text[:200]}..."
                )

                # Generate a unique error tool call ID
                error_tool_call_id = f"error_{uuid.uuid4().hex[:8]}"

                # Create the error record
                error_record = ToolCallingRecord(
                    tool_name="json_parse_error",
                    args={"raw_content": match_text, "error": str(e)},
                    result=error_message,
                    tool_call_id=error_tool_call_id,
                )

                # Record this in memory so the model can see the error
                assist_msg = FunctionCallingMessage(
                    role_name=self.role_name,
                    role_type=self.role_type,
                    meta_dict=None,
                    content="",
                    func_name="json_parse_error",
                    args={"raw_content": match_text[:200], "error": str(e)},
                    tool_call_id=error_tool_call_id,
                )

                func_msg = FunctionCallingMessage(
                    role_name=self.role_name,
                    role_type=self.role_type,
                    meta_dict=None,
                    content="",
                    func_name="json_parse_error",
                    result=error_message,
                    tool_call_id=error_tool_call_id,
                )

                # Use precise timestamps
                current_time_ns = time.time_ns()
                base_timestamp = current_time_ns / 1_000_000_000

                self.update_memory(
                    assist_msg, OpenAIBackendRole.ASSISTANT, timestamp=base_timestamp
                )
                self.update_memory(
                    func_msg,
                    OpenAIBackendRole.FUNCTION,
                    timestamp=base_timestamp + 1e-6,
                )

                return error_record

        return None

    def record_tool_result(self, tool_call_request: Any, raw_result: Any) -> None:
        func_name = tool_call_request.tool_name
        args = tool_call_request.args
        tool_call_id = tool_call_request.tool_call_id

        if self.mask_tool_output:
            with self._secure_result_store_lock:
                self._secure_result_store[tool_call_id] = raw_result
            result = (
                "[The tool has been executed successfully, but the "
                "output from the tool is masked. You can move forward]"
            )
        else:
            result = raw_result

        tool_record = self._record_tool_calling(
            func_name,
            args,
            result,
            tool_call_id,
            mask_output=self.mask_tool_output,
            extra_content=tool_call_request.extra_content,
        )
        self._tool_call_records.append(tool_record)

    def record_user_message(self, input_message: Any) -> None:
        if isinstance(input_message, str):
            input_message = BaseMessage.make_user_message(
                role_name="User", content=input_message
            )
        self.update_memory(input_message, OpenAIBackendRole.USER)

    def finalize_response(self, model_response: Any) -> ChatAgentResponse:
        if self._used_prompt_formatting and self._original_response_format:
            self._apply_prompt_based_parsing(
                model_response, self._original_response_format
            )

        self._record_final_output(model_response.output_messages)

        if self.prune_tool_calls_from_memory and self._tool_call_records:
            self.memory.clean_tool_calls()

        return self._convert_to_chatagent_response(
            model_response,
            self._tool_call_records,
            self._accumulated_context_tokens,
            None,
            self._step_token_usage["prompt_tokens"],
            self._step_token_usage["completion_tokens"],
            self._step_token_usage["total_tokens"],
        )
