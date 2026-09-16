"""Exercise the real Camel memory boundary without model or Docker calls."""

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("camel")
from openai.types.chat import ChatCompletion

from harborrl.harnesses.camel.agent import CamelAgent


class ByteTokenizer:
    def encode(self, text):
        return list(text.encode())

    def decode(self, tokens):
        return bytes(tokens).decode()


def test_camel_tool_request_precedes_result():
    async def run():
        agent = CamelAgent(
            model_type="Qwen3",
            sglang_client=SimpleNamespace(tokenizer=ByteTokenizer()),
            non_think_mode=True,
            max_total_tokens=16384,
        )
        agent.start_turn_loop("Inspect the task directory")
        # Two calls must be preserved together before their matching results.
        completion = ChatCompletion(
            id="test",
            created=0,
            model="Qwen3",
            object="chat.completion",
            choices=[{
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": "Inspecting the workspace.",
                    "tool_calls": [{
                        "id": f"call_{index}",
                        "type": "function",
                        "function": {
                            "name": "shell_exec",
                            "arguments": '{"command":"pwd"}',
                        },
                    } for index in range(2)],
                },
            }],
            usage={"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        )
        _, calls, _, _ = await agent.consume_completion(completion)
        for call in calls:
            agent.record_tool_result(call, "/workspace")
        context, terminated = await agent.get_turn_context()
        assert terminated is None
        assert [message["role"] for message in context] == [
            "system", "user", "assistant", "tool", "tool",
        ]
        assert context[2]["content"] == "Inspecting the workspace."
        assert [call["id"] for call in context[2]["tool_calls"]] == [
            message["tool_call_id"] for message in context[3:]
        ] == ["call_0", "call_1"]

    asyncio.run(run())
