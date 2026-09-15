from __future__ import annotations

import inspect
from typing import Any, Dict, List, Optional, Protocol

from harborrl.harnesses.protocol import TurnClient
from harborrl.types import TurnContext, TurnResult
from harborrl.harnesses.factory import create_harness, display_harness_name
import logging


logger = logging.getLogger(__name__)


class RolloutAgent(Protocol):
    @property
    def parse_error_count(self) -> int: ...

    def set_max_parse_errors(self, max_parse_errors: int) -> None: ...

    def start_turn_loop(self, input_message: Any) -> None: ...

    async def get_turn_context(
        self,
    ) -> tuple[Optional[List[dict[str, Any]]], Optional[Any]]: ...

    async def consume_completion(
        self, chat_completion: Any
    ) -> tuple[Optional[Any], List[Any], bool, Optional[Any]]: ...

    def record_tool_result(self, tool_call_request: Any, raw_result: Any) -> None: ...
    def record_user_message(self, input_message: Any) -> None: ...

    def finalize_response(self, model_response: Any) -> Any: ...


def normalize_harness_option(value: str | None) -> str:
    """Normalize a harness name to its display form via the extension registry."""
    return display_harness_name(value)


class AgentRunner:
    def __init__(
        self,
        *,
        rollout_agent: RolloutAgent,
        sglang_client: TurnClient,
        tool_schemas: List[Dict[str, Any]],
    ) -> None:
        self._rollout_agent = rollout_agent
        self._sglang_client = sglang_client
        self._tool_schemas = tool_schemas
        self._model_turn_count = 0
        self._max_iterations = 10
        self._max_parse_errors = 3

    @property
    def model_turn_count(self) -> int:
        return self._model_turn_count

    @property
    def parse_error_count(self) -> int:
        return self._rollout_agent.parse_error_count

    @property
    def max_iterations(self) -> int:
        return self._max_iterations

    @property
    def max_parse_errors(self) -> int:
        return self._max_parse_errors

    def reset(self, input_message: Any) -> None:
        self._model_turn_count = 0
        self._rollout_agent.start_turn_loop(input_message)

    def set_max_parse_errors(self, max_parse_errors: int) -> None:
        self._max_parse_errors = max(1, int(max_parse_errors))
        self._rollout_agent.set_max_parse_errors(self._max_parse_errors)

    def set_max_iterations(self, max_iterations: int) -> None:
        self._max_iterations = max(1, int(max_iterations))
        set_agent_max_iterations = getattr(self._rollout_agent, "set_max_iterations", None)
        if callable(set_agent_max_iterations):
            set_agent_max_iterations(self._max_iterations)

    def reached_iteration_limit(self) -> bool:
        return self._model_turn_count >= self._max_iterations

    def reached_parse_error_limit(self) -> bool:
        return self.parse_error_count >= self._max_parse_errors

    async def get_turn_context(self) -> TurnContext:
        messages, terminated = await self._rollout_agent.get_turn_context()
        return TurnContext(context_messages=messages, terminated_response=terminated)

    async def run_model_turn(
        self, context_messages: List[dict[str, Any]]
    ) -> TurnResult:
        agent_turn = getattr(self._rollout_agent, "run_model_turn", None)
        if callable(agent_turn):
            result = self._call_agent_run_model_turn(agent_turn, context_messages)
            if inspect.isawaitable(result):
                result = await result
            interactions = getattr(result, "interactions", None) or [
                result.interaction
            ]
            self._model_turn_count += max(1, len(interactions))
            return result

        chat_completion, interaction = await self._sglang_client.generate_turn(
            messages=context_messages,
            tools=self._tool_schemas,
            turn_idx=self._model_turn_count,
        )
        self._model_turn_count += 1

        model_response, tool_call_requests, parse_error_recorded, terminated = (
            await self._rollout_agent.consume_completion(chat_completion)
        )
        return TurnResult(
            interaction=interaction,
            model_response=model_response,
            tool_call_requests=tool_call_requests,
            parse_error_recorded=parse_error_recorded,
            terminated_response=terminated,
            interactions=[interaction],
        )

    def record_tool_result(self, tool_call_request: Any, raw_result: Any) -> None:
        self._rollout_agent.record_tool_result(tool_call_request, raw_result)

    def record_user_message(self, input_message: Any) -> None:
        self._rollout_agent.record_user_message(input_message)

    def finalize_response(self, model_response: Any) -> Any:
        return self._rollout_agent.finalize_response(model_response)

    async def close(self) -> None:
        close_fn = getattr(self._rollout_agent, "close", None)
        if not callable(close_fn):
            return
        result = close_fn()
        if inspect.isawaitable(result):
            await result

    def _call_agent_run_model_turn(
        self,
        agent_turn: Any,
        context_messages: List[dict[str, Any]],
    ) -> Any:
        signature = inspect.signature(agent_turn)
        params = signature.parameters
        accepts_kwargs = any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
        if accepts_kwargs or "context_messages" in params:
            kwargs: dict[str, Any] = {"context_messages": context_messages}
            if accepts_kwargs or "sglang_client" in params:
                kwargs["sglang_client"] = self._sglang_client
            if accepts_kwargs or "tool_schemas" in params:
                kwargs["tool_schemas"] = self._tool_schemas
            if accepts_kwargs or "turn_idx" in params:
                kwargs["turn_idx"] = self._model_turn_count
            return agent_turn(**kwargs)

        return agent_turn(context_messages)


def create_agent_runner(
    *,
    agent_type: str,
    sglang_client: TurnClient,
    model_type: str,
    tool_schemas: List[Dict[str, Any]],
    non_think_mode: bool,
    max_total_tokens: int,
    env_client: Any | None = None,
    lease_id: str | None = None,
    run_context: Any | None = None,
    task_meta: Dict[str, Any] | None = None,
) -> AgentRunner:
    rollout_agent = create_harness(
        agent_type,
        model_type=model_type,
        sglang_client=sglang_client,
        tool_schemas=tool_schemas,
        env_client=env_client,
        lease_id=lease_id,
        run_context=run_context,
        task_meta=task_meta or {},
        non_think_mode=non_think_mode,
        max_total_tokens=max_total_tokens,
    )

    return AgentRunner(
        rollout_agent=rollout_agent,
        sglang_client=sglang_client,
        tool_schemas=tool_schemas,
    )
