"""Serving contract consumed by agent harnesses."""

from __future__ import annotations

from typing import Any, Protocol

from harborrl.types import Interaction


class RolloutHarness(Protocol):
    """Minimal lifecycle expected by the rollout runner.

    Concrete training harnesses may expose additional constructor and tool
    handling details; this protocol documents the reusable boundary without
    coupling evaluation adapters to training state.
    """

    def start_turn_loop(self, input_message: Any) -> Any: ...

    def finalize_response(self, response: Any) -> Any: ...


class TurnClient(Protocol):
    tokenizer: Any
    sampling_params: dict[str, Any]
    tool_call_parser: str | None
    session_id: str | None
    url: str
    request_timeout: float | None

    async def generate_turn(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        turn_idx: int,
    ) -> tuple[Any, Interaction]: ...

    def truncate_input_ids(self, input_ids: list[int]) -> list[int]: ...

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> list[int]: ...


__all__ = ["RolloutHarness", "TurnClient"]
