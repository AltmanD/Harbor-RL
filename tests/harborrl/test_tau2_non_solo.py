from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path

TEST_FILE = Path(__file__).resolve()
TERMINAL_RL_ROOT = TEST_FILE.parents[2] / "harborrl"
REPO_ROOT = TERMINAL_RL_ROOT.parent
SLIME_ROOT = REPO_ROOT / "backends" / "slime"

for path in (str(TERMINAL_RL_ROOT), str(REPO_ROOT), str(SLIME_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from harborrl.rollout.runner import AgentRunner
from harborrl.environments.client import TerminalEnvClient
from harborrl.environments.tau2.runtime import Tau2Env


class DummyRolloutAgent:
    parse_error_count = 0

    def __init__(self) -> None:
        self.started: list[str] = []
        self.user_messages: list[str] = []

    def set_max_parse_errors(self, max_parse_errors: int) -> None:
        self.max_parse_errors = max_parse_errors

    def start_turn_loop(self, input_message: str) -> None:
        self.started.append(input_message)

    async def get_turn_context(self):
        return [], None

    async def consume_completion(self, chat_completion):
        return None, [], False, None

    def record_tool_result(self, tool_call_request, raw_result) -> None:
        pass

    def record_user_message(self, input_message) -> None:
        self.user_messages.append(input_message)

    def finalize_response(self, model_response):
        return model_response


class DummySGLangClient:
    async def generate_turn(self, messages, tools, turn_idx):
        raise AssertionError("not used in this test")


def test_agent_runner_records_follow_up_user_message():
    runner = AgentRunner(
        rollout_agent=DummyRolloutAgent(),
        sglang_client=DummySGLangClient(),
        tool_schemas=[],
    )

    runner.reset("first user message")
    runner.record_user_message("second user message")

    assert runner._rollout_agent.started == ["first user message"]
    assert runner._rollout_agent.user_messages == ["second user message"]


def test_env_client_exposes_agent_reply():
    client = TerminalEnvClient("http://example.com")
    assert hasattr(client, "agent_reply")


class FakeUser:
    def __init__(self) -> None:
        self.calls: list[object] = []

    def get_init_state(self, message_history=None):
        return {"history": list(message_history or [])}

    def generate_next_message(self, message, state):
        self.calls.append(message)
        return (
            types.SimpleNamespace(
                role="user",
                content="follow-up from user",
                tool_calls=None,
                is_tool_call=lambda: False,
            ),
            state,
        )


class FakeTool:
    def __init__(self, name: str) -> None:
        self.openai_schema = {
            "type": "function",
            "function": {
                "name": name,
                "parameters": {"type": "object", "properties": {}},
            },
        }


class FakeTau2ToolEnv:
    def __init__(self) -> None:
        self.agent_tools = [FakeTool("get_customer_by_phone")]
        self.user_tools = [FakeTool("check_network_status"), FakeTool("reseat_sim_card")]

    def get_tools(self):
        return self.agent_tools

    def get_user_tools(self):
        return self.user_tools


def _schema_names(tool_schemas):
    return [schema["function"]["name"] for schema in tool_schemas]


def test_tau2_env_solo_exposes_agent_and_user_tools():
    env = Tau2Env()
    env._task_meta = {"tau2_mode": "solo"}

    assert _schema_names(env._tool_schemas(FakeTau2ToolEnv())) == [
        "get_customer_by_phone",
        "check_network_status",
        "reseat_sim_card",
    ]


def test_tau2_env_non_solo_only_exposes_agent_tools():
    env = Tau2Env()
    env._task_meta = {"tau2_mode": "non_solo"}

    assert _schema_names(env._tool_schemas(FakeTau2ToolEnv())) == [
        "get_customer_by_phone",
    ]


def test_tau2_env_non_solo_generates_follow_up_user_message(monkeypatch):
    message_mod = types.ModuleType("tau2.data_model.message")
    message_mod.AssistantMessage = lambda **kwargs: types.SimpleNamespace(**kwargs)
    message_mod.MultiToolMessage = lambda **kwargs: types.SimpleNamespace(**kwargs)
    message_mod.ToolMessage = lambda **kwargs: types.SimpleNamespace(**kwargs)
    monkeypatch.setitem(sys.modules, "tau2", types.ModuleType("tau2"))
    monkeypatch.setitem(sys.modules, "tau2.data_model", types.ModuleType("tau2.data_model"))
    monkeypatch.setitem(sys.modules, "tau2.data_model.message", message_mod)

    env = Tau2Env()
    env._task_meta = {"tau2_mode": "non_solo"}
    env._task = types.SimpleNamespace(initial_state=None, id="task_1")
    env._env = types.SimpleNamespace(solo_mode=False)
    env._user = FakeUser()
    env._user_state = env._user.get_init_state()

    result = asyncio.run(env.handle_agent_reply("I updated the task."))

    assert result == {
        "continue": True,
        "user_message": "follow-up from user",
    }


def test_tau2_env_recognizes_hash_stop_message():
    message = types.SimpleNamespace(
        content="###STOP###",
        tool_calls=None,
        is_tool_call=lambda: False,
    )

    assert Tau2Env._is_stop_message(message) is True

def test_tau2_env_uses_local_openai_compatible_user_llm_defaults(monkeypatch):
    monkeypatch.delenv("TAU2_USER_LLM", raising=False)
    monkeypatch.delenv("TAU2_USER_LLM_ARGS", raising=False)
    monkeypatch.delenv("TAU2_USER_LLM_TIMEOUT", raising=False)
    monkeypatch.setenv("VLLM_API_KEY", "dummy")

    env = Tau2Env()
    env._task_meta = {"tau2_mode": "non_solo"}

    assert env._user_llm() == "openai/Qwen3.6-27B-FP8"
    assert env._user_llm_args() == {
        "api_base": "http://s-20260523131729-dtntr.ailab-pj.pjh-service.org.cn/v1",
        "api_key": "dummy",
        "timeout": 15.0,
    }
