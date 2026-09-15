from __future__ import annotations

import json
import os
import uuid
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

from harborrl.data.tau2_support import ensure_tau2_importable, task_instruction


def _repo_default_tau2_root() -> Path:
    return Path(__file__).resolve().parents[2].parent / "tau2-bench"


class Tau2Env:
    """Local solo-mode tau2 backend for harborrl."""

    def __init__(self, root: str | None = None) -> None:
        self.root = Path(
            root or os.getenv("TAU2_BENCH_ROOT", _repo_default_tau2_root())
        ).resolve()
        self._task_meta: dict[str, Any] = {}
        self._task = None
        self._env = None
        self._user = None
        self._user_state = None
        self._run_ctx = None
        self._last_eval: dict[str, Any] | None = None
        self._live_messages: list[Any] | None = None

    def _task_domain(self) -> str:
        return str(
            self._task_meta.get("tau2_domain")
            or os.getenv("TAU2_DOMAIN", "telecom")
        ).strip().lower()

    def _task_split(self) -> str | None:
        split = str(
            self._task_meta.get("tau2_task_split")
            or os.getenv("TAU2_TASK_SPLIT", "train")
        ).strip()
        return split or None

    def _task_id(self) -> str:
        task_id = str(self._task_meta.get("tau2_task_id") or "").strip()
        if not task_id:
            raise ValueError("tau2_task_id is required for tau2 samples")
        return task_id

    def _telecom_policy_type(self) -> str:
        return str(
            self._task_meta.get("tau2_policy_type")
            or os.getenv("TAU2_POLICY_TYPE", "manual")
        ).strip().lower()

    def _env_kwargs(self) -> dict[str, Any]:
        if self._task_domain() == "telecom":
            return {"policy_type": self._telecom_policy_type()}
        return {}

    def _mode(self) -> str:
        mode = str(
            self._task_meta.get("tau2_mode") or os.getenv("TAU2_MODE", "solo")
        ).strip().lower()
        if mode in {"non_solo", "nonsolo", "non-solo"}:
            return "non_solo"
        return "solo"

    def _is_solo_mode(self) -> bool:
        return self._mode() == "solo"

    def _user_llm(self) -> str:
        return str(
            self._task_meta.get("tau2_user_llm")
            or os.getenv("TAU2_USER_LLM", "openai/Qwen3.6-27B-FP8")
        ).strip()

    def _user_llm_args(self) -> dict[str, Any]:
        raw = self._task_meta.get("tau2_user_llm_args")
        if isinstance(raw, dict):
            return deepcopy(raw)
        raw_text = str(os.getenv("TAU2_USER_LLM_ARGS", "")).strip()
        if not raw_text:
            timeout_raw = str(os.getenv("TAU2_USER_LLM_TIMEOUT", "15")).strip()
            try:
                timeout = float(timeout_raw)
            except ValueError:
                timeout = 15.0
            return {
                "api_base": os.getenv(
                    "TAU2_USER_LLM_API_BASE",
                    "http://s-20260523131729-dtntr.ailab-pj.pjh-service.org.cn/v1",
                ).strip(),
                "api_key": os.getenv("VLLM_API_KEY", "dummy").strip() or "dummy",
                "timeout": timeout,
            }
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError:
            return {}
        return deepcopy(parsed) if isinstance(parsed, dict) else {}

    def _load_task(self) -> Any:
        from tau2.runner.helpers import get_tasks

        tasks = get_tasks(
            task_set_name=self._task_domain(),
            task_split_name=self._task_split(),
            task_ids=[self._task_id()],
        )
        if not tasks:
            raise ValueError(
                "tau2 task not found: "
                f"domain={self._task_domain()} split={self._task_split()} task_id={self._task_id()}"
            )
        return tasks[0]

    def _load_env(self) -> Any:
        from tau2.registry import registry

        return registry.get_env_constructor(self._task_domain())(
            solo_mode=self._is_solo_mode(), **self._env_kwargs()
        )

    def _build_user(self, env: Any, task: Any) -> Any:
        from tau2.runner.build import build_user

        return build_user(
            "user_simulator",
            env,
            task,
            llm=self._user_llm(),
            llm_args=self._user_llm_args(),
            solo_mode=False,
        )

    def _tool_schemas(self, env: Any) -> list[dict[str, Any]]:
        tools = list(env.get_tools())
        if self._is_solo_mode():
            try:
                tools.extend(env.get_user_tools())
            except Exception:
                pass

        schemas: list[dict[str, Any]] = []
        seen_names: set[str] = set()
        for tool in tools:
            schema = deepcopy(tool.openai_schema)
            name = str((schema.get("function") or {}).get("name") or "")
            if name and name in seen_names:
                continue
            if name:
                seen_names.add(name)
            schemas.append(schema)
        return schemas

    def _user_message(self, env: Any, task: Any) -> str:
        if not self._is_solo_mode():
            return (
                "<instructions>\n"
                "You are solving a tau2-bench task in non-solo mode.\n"
                "A simulated user may reply after your non-tool messages.\n"
                "Solve the ticket by talking to the simulated user and calling the provided tools.\n"
                "Only make one tool call at a time.\n"
                "When the issue is resolved or escalation is necessary, give a brief final answer to the user.\n"
                "</instructions>\n"
                f"<policy>\n{env.get_policy()}\n</policy>\n"
                f"<ticket>\n{task_instruction(task)}\n</ticket>"
            )
        return (
            "<instructions>\n"
            "You are solving a tau2-bench task in solo mode.\n"
            "You cannot interact with the user. Solve the ticket by calling the provided tools.\n"
            "Only make one tool call at a time.\n"
            "When the issue is resolved or escalation is necessary, give a brief final answer and stop making tool calls.\n"
            "</instructions>\n"
            f"<policy>\n{env.get_policy()}\n</policy>\n"
            f"<ticket>\n{task_instruction(task)}\n</ticket>"
        )

    async def reset(
        self,
        *,
        task_meta: dict[str, Any],
        task_spec: Any,
        run_ctx: Any,
    ) -> tuple[str, list[dict[str, Any]]]:
        _ = task_spec
        ensure_tau2_importable(self.root)
        self._task_meta = deepcopy(task_meta)
        self._run_ctx = run_ctx
        self._last_eval = None
        self._user = None
        self._user_state = None

        task = self._load_task()
        env = self._load_env()
        initial_state = getattr(task, "initial_state", None)
        message_history = (
            deepcopy(getattr(initial_state, "message_history", None) or [])
            if initial_state is not None
            else []
        )
        env.set_state(
            initialization_data=(
                getattr(initial_state, "initialization_data", None)
                if initial_state is not None
                else None
            ),
            initialization_actions=(
                getattr(initial_state, "initialization_actions", None)
                if initial_state is not None
                else None
            ),
            message_history=message_history,
        )

        self._task = task
        self._env = env
        self._live_messages = deepcopy(message_history)
        if not self._is_solo_mode():
            self._user = self._build_user(env, task)
            self._user_state = self._user.get_init_state(message_history=message_history)
        return self._user_message(env, task), self._tool_schemas(env)

    async def exec_tool(self, name: str, arguments: dict[str, Any]) -> str:
        if self._env is None:
            raise RuntimeError("tau2 env is not initialized; call reset first")

        from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage

        tool_call = ToolCall(
            id=f"call_{uuid.uuid4().hex[:12]}",
            name=name,
            arguments=deepcopy(arguments or {}),
            requestor="assistant",
        )
        tool_result = self._env.get_response(tool_call)
        if self._live_messages is not None and not self._is_solo_mode():
            self._live_messages.append(
                AssistantMessage(role="assistant", tool_calls=[tool_call])
            )
            self._live_messages.append(
                ToolMessage(
                    role="tool",
                    id=tool_call.id,
                    requestor="assistant",
                    content=str(tool_result.content),
                    error=False,
                )
            )
        return str(tool_result.content)

    async def handle_agent_reply(self, assistant_text: str) -> dict[str, Any]:
        if self._is_solo_mode() or self._env is None or self._user is None:
            return {"continue": False, "user_message": ""}

        from tau2.data_model.message import AssistantMessage, MultiToolMessage, ToolMessage

        current_message: Any = AssistantMessage(role="assistant", content=assistant_text)
        if self._live_messages is not None:
            self._live_messages.append(current_message)

        while True:
            user_message, self._user_state = self._user.generate_next_message(
                current_message, self._user_state
            )
            if self._live_messages is not None:
                self._live_messages.append(user_message)

            if self._is_stop_message(user_message):
                return {"continue": False, "user_message": ""}

            if not user_message.is_tool_call():
                return {
                    "continue": True,
                    "user_message": str(getattr(user_message, "content", "") or ""),
                }

            tool_messages: list[ToolMessage] = []
            for tool_call in list(getattr(user_message, "tool_calls", []) or []):
                tool_response = self._env.get_response(tool_call)
                tool_message = ToolMessage(
                    role="tool",
                    id=tool_call.id,
                    requestor="user",
                    content=str(tool_response.content),
                    error=False,
                )
                tool_messages.append(tool_message)
                if self._live_messages is not None:
                    self._live_messages.append(tool_message)

            current_message = (
                tool_messages[0]
                if len(tool_messages) == 1
                else MultiToolMessage(tool_messages=tool_messages)
            )

    @staticmethod
    def _is_stop_message(message: Any) -> bool:
        if getattr(message, "is_tool_call", lambda: False)():
            return False
        content = getattr(message, "content", None)
        if content is None:
            return False
        text = str(content).strip().upper()
        stop_markers = (
            "<STOP>",
            "###STOP###",
            "<TRANSFER>",
            "###TRANSFER###",
            "<OUT_OF_SCOPE>",
            "###OUT_OF_SCOPE###",
        )
        return any(marker in text for marker in stop_markers)

    def _messages_from_payload(self, trajectory: dict[str, Any] | None) -> list[Any]:
        from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage

        initial_state = (
            getattr(self._task, "initial_state", None) if self._task is not None else None
        )
        messages = deepcopy(getattr(initial_state, "message_history", None) or [])
        payload = trajectory if isinstance(trajectory, dict) else {}

        for idx, turn in enumerate(payload.get("turn_records") or []):
            tool_calls = [
                call for call in (turn.get("tool_calls") or []) if isinstance(call, dict)
            ]
            if tool_calls:
                tau2_calls: list[ToolCall] = []
                tool_messages: list[ToolMessage] = []
                for call_idx, call in enumerate(tool_calls):
                    tool_call_id = str(
                        call.get("tool_call_id")
                        or f"call_{turn.get('turn_idx', idx)}_{call_idx}"
                    )
                    tau2_calls.append(
                        ToolCall(
                            id=tool_call_id,
                            name=str(call.get("tool_name") or ""),
                            arguments=deepcopy(call.get("args") or {}),
                            requestor="assistant",
                        )
                    )
                    tool_messages.append(
                        ToolMessage(
                            role="tool",
                            id=tool_call_id,
                            requestor="assistant",
                            content=str(call.get("result") or ""),
                            error=False,
                        )
                    )
                messages.append(
                    AssistantMessage(
                        role="assistant",
                        tool_calls=tau2_calls,
                        turn_idx=int(turn.get("turn_idx") or idx),
                    )
                )
                messages.extend(tool_messages)
                continue

            assistant_output = str(turn.get("assistant_output") or "").strip()
            if assistant_output:
                messages.append(
                    AssistantMessage(
                        role="assistant",
                        content=assistant_output,
                        turn_idx=int(turn.get("turn_idx") or idx),
                    )
                )

        final_text = str(payload.get("final_response") or "").strip()
        if final_text:
            if not messages or getattr(messages[-1], "content", None) != final_text:
                messages.append(AssistantMessage(role="assistant", content=final_text))
        return messages

    async def evaluate(self, trajectory: dict[str, Any] | None = None) -> float:
        if self._task is None:
            raise RuntimeError("tau2 env is not initialized; call reset first")

        from tau2.data_model.simulation import SimulationRun, TerminationReason
        from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation

        payload = trajectory if isinstance(trajectory, dict) else {}
        status = str(payload.get("status") or "").strip().lower()
        termination_reason = {
            "completed": TerminationReason.AGENT_STOP,
            "truncated": TerminationReason.MAX_STEPS,
            "failed": TerminationReason.AGENT_ERROR,
            "aborted": TerminationReason.INFRASTRUCTURE_ERROR,
        }.get(status, TerminationReason.AGENT_STOP)

        now = datetime.now().isoformat()
        sim = SimulationRun(
            id=str(self._run_ctx.uid if self._run_ctx is not None else uuid.uuid4().hex[:8]),
            task_id=str(self._task.id),
            start_time=now,
            end_time=now,
            duration=0.0,
            termination_reason=termination_reason,
            messages=(
                deepcopy(self._live_messages)
                if (self._live_messages is not None and not self._is_solo_mode())
                else self._messages_from_payload(payload)
            ),
        )
        reward_info = evaluate_simulation(
            simulation=sim,
            task=self._task,
            evaluation_type=EvaluationType.ALL,
            solo_mode=self._is_solo_mode(),
            domain=self._task_domain(),
            env_kwargs=self._env_kwargs(),
        )
        self._last_eval = {
            "mode": "tau2",
            "score": float(reward_info.reward),
            "reward_basis": [str(x) for x in (reward_info.reward_basis or [])],
            "reward_breakdown": {
                str(k): float(v) for k, v in (reward_info.reward_breakdown or {}).items()
            },
            "info": deepcopy(reward_info.info or {}),
            "task_id": str(self._task.id),
            "tau2_domain": self._task_domain(),
            "tau2_task_split": self._task_split(),
        }
        return float(reward_info.reward)

    async def close(self) -> None:
        self._task_meta = {}
        self._task = None
        self._env = None
        self._user = None
        self._user_state = None
        self._run_ctx = None
        self._live_messages = None
