"""Structural contract between the rollout orchestrator and environment
clients.  ``TerminalEnvClient`` (remote worker over HTTP) and the in-process
``_LocalEnvClient`` adapters built by ``rollout/environment_factory.py`` both
satisfy this protocol; ``agent_reply`` is only exercised for tau2-style
dual-agent tasks and is guarded by the data source at the call site.
"""

from __future__ import annotations

from typing import Any, Protocol


class EnvClient(Protocol):
    last_evaluate_details: dict[str, Any] | None

    async def reset(
        self,
        lease_id: str,
        task_meta: dict[str, Any],
        run_ctx: dict[str, Any],
        task_timeouts: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]: ...

    async def heartbeat(self, lease_id: str) -> None: ...

    async def exec_tool(
        self, lease_id: str, tool_name: str, arguments: dict[str, Any]
    ) -> str: ...

    async def agent_reply(self, lease_id: str, assistant_text: str) -> dict[str, Any]: ...

    async def evaluate(
        self, lease_id: str, trajectory: dict[str, Any] | None = None
    ) -> float: ...

    async def close(self, lease_id: str) -> None: ...
