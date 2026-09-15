from __future__ import annotations

import logging
import os
from importlib import import_module
from pathlib import Path
from typing import Any, Dict

from harborrl.types import RunContext, TaskSpec
from harborrl.environments.client import TerminalEnvClient
from harborrl.environments.protocol import EnvClient
from harborrl.environments.registry import local_env_spec
from harborrl.rollout.admission import _await_with_optional_timeout
from harborrl.rollout.sample_builder import _make_task_spec

logger = logging.getLogger(__name__)


from harborrl.env import env_float as _env_float


def _normalize_tau2_conversation_mode(raw_mode: Any) -> str:
    mode = str(raw_mode or "solo").strip().lower()
    if mode in {"non_solo", "nonsolo", "non-solo"}:
        return "non_solo"
    return "solo"


class _LocalEnvClient:
    """Adapt an in-process env runtime to the :class:`EnvClient` protocol.

    The runtime only needs ``reset/exec_tool/evaluate/close``; ``agent_reply``
    is exercised solely for tau2-style dual-agent tasks and is guarded by the
    data source upstream.
    """

    def __init__(self, env: Any, *, conversation_mode: str | None = None) -> None:
        self._env = env
        self._conversation_mode = conversation_mode
        self.last_evaluate_details: dict[str, Any] | None = None

    async def reset(
        self,
        lease_id: str,
        task_meta: dict[str, Any],
        run_ctx: dict[str, Any],
        task_timeouts: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        _ = (lease_id, task_timeouts, request_id)
        local_run_ctx = RunContext(
            uid=str(run_ctx.get("uid", "local")),
            group_index=int(run_ctx.get("group_index", 0) or 0),
            sample_index=int(run_ctx.get("sample_index", 0) or 0),
            log_dir=Path(
                str(
                    run_ctx.get(
                        "log_dir",
                        "runs/unscoped/environment_outputs/AgentRunner_Output",
                    )
                )
            ),
        )
        user_msg, tool_schemas = await self._env.reset(
            task_meta=task_meta,
            task_spec=_make_task_spec(task_meta),
            run_ctx=local_run_ctx,
        )
        payload: dict[str, Any] = {"user_msg": user_msg, "tool_schemas": tool_schemas}
        if self._conversation_mode is not None:
            payload["conversation_mode"] = self._conversation_mode
        return payload

    async def heartbeat(self, lease_id: str) -> None:
        _ = lease_id

    async def exec_tool(
        self, lease_id: str, tool_name: str, arguments: dict[str, Any]
    ) -> str:
        _ = lease_id
        return await self._env.exec_tool(tool_name, arguments)

    async def agent_reply(self, lease_id: str, assistant_text: str) -> dict[str, Any]:
        _ = lease_id
        return await self._env.handle_agent_reply(assistant_text)

    async def evaluate(
        self, lease_id: str, trajectory: dict[str, Any] | None = None
    ) -> float:
        _ = lease_id
        score = await self._env.evaluate(trajectory)
        self.last_evaluate_details = getattr(self._env, "_last_eval", None)
        return score

    async def close(self, lease_id: str) -> None:
        _ = lease_id
        await self._env.close()


async def _create_env_client(
    task_spec: TaskSpec,
    run_ctx: RunContext,
    task_meta: Dict[str, Any] | None = None,
) -> tuple[EnvClient, str]:
    spec = local_env_spec(task_meta)
    if spec is not None:
        module_name, class_name = spec.local_runtime
        env_class = getattr(import_module(module_name), class_name)
        conversation_mode = (
            _normalize_tau2_conversation_mode(task_meta.get("tau2_mode"))
            if spec.data_source == "tau2" and isinstance(task_meta, dict)
            else None
        )
        logger.info(
            "Using local %s env backend for task=%s path=%s",
            spec.data_source,
            task_spec.task_name,
            task_spec.task_path,
        )
        return (
            _LocalEnvClient(env_class(), conversation_mode=conversation_mode),
            spec.local_lease_id,
        )

    env_server_url = os.getenv("ENV_SERVER_URL", "")
    if not env_server_url:
        raise RuntimeError("ENV_SERVER_URL is empty.")

    env_client = TerminalEnvClient(env_server_url)
    task_key = f"{task_spec.task_name}:{task_spec.task_path}"
    request_id = (
        f"{task_key}:{run_ctx.uid}:{run_ctx.group_index}:{run_ctx.sample_index}"
    )
    allocate_timeout = _env_float("ENV_ALLOCATE_HTTP_TIMEOUT", 300.0)
    lease = await _await_with_optional_timeout(
        env_client.allocate(task_key=task_key, request_id=request_id),
        allocate_timeout,
        op_name="terminal env allocate",
    )
    lease_id = str(lease["lease_id"])
    logger.info(
        "Using remote terminal env backend lease=%s server=%s", lease_id, env_server_url
    )
    return env_client, lease_id
