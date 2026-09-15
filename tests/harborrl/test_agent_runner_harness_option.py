from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path

AGENTIC_RL_DIR = Path(__file__).resolve().parents[2] / "harborrl"
ROOT_DIR = AGENTIC_RL_DIR.parent
if str(AGENTIC_RL_DIR) not in sys.path:
    sys.path.insert(0, str(AGENTIC_RL_DIR))
if str(ROOT_DIR / "backends" / "slime") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "backends" / "slime"))

from harborrl.rollout.runner import AgentRunner, create_agent_runner, normalize_harness_option
from harborrl.types import Interaction, TurnResult


def test_normalize_harness_option_aliases():
    assert normalize_harness_option(None) == "camel-agent"
    assert normalize_harness_option("camel-agent") == "camel-agent"
    assert normalize_harness_option("camel_agent") == "camel-agent"
    assert normalize_harness_option("claude_code") == "claude-code"
    assert normalize_harness_option("claude-code") == "claude-code"



def test_create_agent_runner_keeps_camel_agent_env_optional(monkeypatch):
    class FakeCamelAgent:
        parse_error_count = 0

        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def set_max_parse_errors(self, max_parse_errors):
            self.max_parse_errors = max_parse_errors

        def start_turn_loop(self, input_message):
            self.input_message = input_message

        async def get_turn_context(self):
            return [], None

        async def consume_completion(self, chat_completion):
            return chat_completion, [], False, None

        def record_tool_result(self, tool_call_request, raw_result):
            pass

        def finalize_response(self, model_response):
            return model_response

    fake_module = types.ModuleType("harborrl.harnesses.camel.agent")
    fake_module.CamelAgent = FakeCamelAgent
    monkeypatch.setitem(sys.modules, "harborrl.harnesses.camel.agent", fake_module)

    runner = create_agent_runner(
        agent_type="camel_agent",
        sglang_client=object(),
        model_type="Qwen3",
        tool_schemas=[],
        non_think_mode=True,
        max_total_tokens=1024,
        env_client=None,
        lease_id=None,
    )
    assert isinstance(runner, AgentRunner)


def test_create_agent_runner_routes_claude_code(monkeypatch):
    class FakeClaudeCodeAgent:
        parse_error_count = 0

        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def set_max_parse_errors(self, max_parse_errors):
            self.max_parse_errors = max_parse_errors

        def start_turn_loop(self, input_message):
            self.input_message = input_message

        async def get_turn_context(self):
            return [], None

        async def consume_completion(self, chat_completion):
            return chat_completion, [], False, None

        def record_tool_result(self, tool_call_request, raw_result):
            pass

        def finalize_response(self, model_response):
            return model_response

    fake_module = types.ModuleType("harborrl.harnesses.claude_code.agent")
    fake_module.ClaudeCodeAgent = FakeClaudeCodeAgent
    monkeypatch.setitem(sys.modules, "harborrl.harnesses.claude_code.agent", fake_module)

    runner = create_agent_runner(
        agent_type="claude_code",
        sglang_client=object(),
        model_type="Qwen3",
        tool_schemas=[],
        non_think_mode=True,
        max_total_tokens=1024,
        env_client=object(),
        lease_id="lease-1",
        task_meta={"task_name": "task"},
    )
    assert isinstance(runner, AgentRunner)
    assert runner._rollout_agent.kwargs["lease_id"] == "lease-1"


def test_agent_runner_prefers_custom_run_model_turn():
    class CustomAgent:
        parse_error_count = 0

        def set_max_parse_errors(self, max_parse_errors):
            pass

        def start_turn_loop(self, input_message):
            pass

        async def get_turn_context(self):
            return [], None

        async def run_model_turn(self, **kwargs):
            assert kwargs["turn_idx"] == 0
            interaction = Interaction(turn_idx=0, output_text="custom")
            return TurnResult(
                interaction=interaction,
                interactions=[interaction],
                model_response=types.SimpleNamespace(text="custom"),
                tool_call_requests=[],
                parse_error_recorded=False,
            )

        async def consume_completion(self, chat_completion):
            raise AssertionError("fallback path should not be called")

        def record_tool_result(self, tool_call_request, raw_result):
            pass

        def finalize_response(self, model_response):
            return model_response

    runner = AgentRunner(
        rollout_agent=CustomAgent(),
        sglang_client=object(),
        tool_schemas=[],
    )
    result = asyncio.run(runner.run_model_turn([{"role": "user", "content": "x"}]))
    assert result.interaction.output_text == "custom"
    assert runner.model_turn_count == 1


def test_agent_runner_fallback_path_for_camel_style_agent():
    class FakeClient:
        async def generate_turn(self, messages, tools, turn_idx):
            assert turn_idx == 0
            return "completion", Interaction(turn_idx=0, output_text="fallback")

    class CamelStyleAgent:
        parse_error_count = 0

        def set_max_parse_errors(self, max_parse_errors):
            pass

        def start_turn_loop(self, input_message):
            pass

        async def get_turn_context(self):
            return [], None

        async def consume_completion(self, chat_completion):
            assert chat_completion == "completion"
            return types.SimpleNamespace(text="fallback"), [], False, None

        def record_tool_result(self, tool_call_request, raw_result):
            pass

        def finalize_response(self, model_response):
            return model_response

    runner = AgentRunner(
        rollout_agent=CamelStyleAgent(),
        sglang_client=FakeClient(),
        tool_schemas=[],
    )
    result = asyncio.run(runner.run_model_turn([{"role": "user", "content": "x"}]))
    assert result.interaction.output_text == "fallback"
    assert result.interactions == [result.interaction]
    assert runner.model_turn_count == 1
