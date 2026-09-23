import json
import threading
from urllib import request, error
from types import SimpleNamespace

import pytest

from harborrl.gateway.messages import ProtocolError, convert, parse_qwen_output, sse_events
from harborrl.gateway.server import Gateway, make_server
from harborrl.gateway.trace import TraceRegistry
from harborrl.rollout.harbor_job.audit import audit_session
from test_native_contracts import identity


def payload(**changes):
    return {"model": "policy", "max_tokens": 64, "messages": [{"role": "user", "content": "hello"}], **changes}


class Backend:
    """Test-only synthetic generation; no model quality or GPU evidence."""
    def count_tokens(self, converted):
        return 2

    def generate(self, converted, ident, response_id):
        return {"input_ids": [1, 2], "output_ids": [3], "logprobs": [-.2], "logprob_semantics": "raw_model",
                "policy_version": ident.policy_version, "tokenizer_digest": "tok", "template_digest": "tpl",
                "engine_id": "synthetic", "serving_input": {"input_ids": [1, 2]}, "sampling": converted["sampling"],
                "finish_reason": "end_turn", "evidence_kind": "synthetic", "content": [{"type": "text", "text": "done"}]}


def setup(tmp_path):
    registry = TraceRegistry(tmp_path / "traces")
    ident = identity()
    token = registry.register(ident, "trial")
    return registry, ident, token, Gateway(registry, Backend(), model="policy")


@pytest.mark.parametrize("change", [{"model": "external"}, {"thinking": {"type": "enabled"}},
    {"temperature": .7}, {"top_p": .9}, {"top_k": 1}, {"max_tokens": True}, {"stream": 1},
    {"messages": []}, {"messages": [{"role": "assistant", "content": [{"type": "image"}]}]},
    {"tool_choice": {"type": "any"}}])
def test_unsupported_request(change):
    with pytest.raises(ProtocolError):
        convert(payload(**change), model="policy")


def test_tool_roundtrip():
    tools = [{"name": "bash", "input_schema": {"type": "object"}}]
    p = payload(tools=tools, messages=[
        {"role": "user", "content": "run"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "bash", "input": {"command": "false"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "exit 1", "is_error": True}]}])
    converted = convert(p, model="policy")
    assert converted["messages"][-1]["content"] == "[tool_error]\nexit 1"
    out = parse_qwen_output('<tool_call>{"name":"bash","arguments":{"command":"pwd"}}</tool_call>', response_id="m", tools=converted["tools"])
    assert out[0]["input"] == {"command": "pwd"}
    assert out[0]["id"] == "toolu_m_0"
    p["messages"][-1]["content"][0]["tool_use_id"] = "other"
    with pytest.raises(ProtocolError, match="orphan"):
        convert(p, model="policy")


def test_claude_2_1_220_control_turns_are_accepted_without_hidden_sampling():
    converted = convert(payload(
        output_config={"effort": "high"},
        thinking={"type": "adaptive"},
        context_management={"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]},
        system=[{"type": "text", "text": "locked system prompt"}],
        messages=[
            {"role": "user", "content": "run"},
            {"role": "system", "content": [{"type": "text", "text": "session update",
                                            "cache_control": {"type": "ephemeral"}}]},
        ]), model="policy")
    assert converted["messages"][0] == {"role": "system", "content": "locked system prompt"}
    assert converted["messages"][1] == {"role": "user", "content": "run"}
    assert converted["messages"][2] == {"role": "system", "content": "session update"}
    assert converted["sampling"]["temperature"] == 1
    with pytest.raises(ProtocolError, match="thinking"):
        convert(payload(thinking={"type": "enabled", "budget_tokens": 1}), model="policy")
    with pytest.raises(ProtocolError, match="context management"):
        convert(payload(context_management={"edits": []}), model="policy")


@pytest.mark.parametrize("text", ['<tool_call>{broken}</tool_call>', '<tool_call>{}', '<think>x</think>', '<tool_call>{"name":"unknown","arguments":{}}</tool_call>'])
def test_model_output_never_repaired(text):
    with pytest.raises(ProtocolError):
        parse_qwen_output(text, response_id="m", tools=[])


def test_trace_requires_actual_consumption_and_is_immutable(tmp_path):
    registry, ident, token, gateway = setup(tmp_path)
    aid, message = gateway.generate(token, payload())
    registry.delivery(aid, message["id"], "sent")
    registry.close(aid)
    turns, seal = registry.seal(aid)
    assert seal["errors"] == ["unconfirmed_consumption"]
    with pytest.raises(ValueError, match="sealed"):
        registry.consume(aid, message["id"], message["content"], "session")
    with pytest.raises(PermissionError):
        gateway.generate(token, payload())


def test_session_audit_covers_final_response(tmp_path):
    registry, ident, token, gateway = setup(tmp_path)
    aid, message = gateway.generate(token, payload())
    registry.delivery(aid, message["id"], "sent")
    session = tmp_path / "session.jsonl"
    session.write_text(json.dumps({"sessionId": "session", "type": "assistant", "message": message}) + "\n")
    audited = audit_session(session, registry, aid)
    assert audited["consumed_response_ids"] == [message["id"]]
    registry.close(aid)
    turns, seal = registry.seal(aid)
    assert seal["errors"] == []
    assert turns[0]["delivery"] == "consumed_confirmed"


def test_session_audit_merges_split_assistant_blocks(tmp_path):
    class MultiBlockBackend(Backend):
        def generate(self, converted, ident, response_id):
            turn = super().generate(converted, ident, response_id)
            turn["content"] = [*turn["content"], {"type": "text", "text": "tail"}]
            return turn

    registry, ident, token, gateway = setup(tmp_path)
    gateway.backend = MultiBlockBackend()
    aid, message = gateway.generate(token, payload())
    registry.delivery(aid, message["id"], "sent")
    assert len(message["content"]) == 2
    session = tmp_path / "session.jsonl"
    session.write_text("".join(
        json.dumps({"sessionId": "session", "type": "assistant", "message": part}) + "\n"
        for part in ({**message, "content": [block]} for block in message["content"])))
    audited = audit_session(session, registry, aid)
    assert audited["consumed_response_ids"] == [message["id"]]
    registry.close(aid)
    assert registry.seal(aid)[1]["errors"] == []


def test_session_audit_rejects_blocks_the_gateway_never_served(tmp_path):
    registry, ident, token, gateway = setup(tmp_path)
    aid, message = gateway.generate(token, payload())
    registry.delivery(aid, message["id"], "sent")
    forged = {**message, "content": [{"type": "text", "text": "different"}]}
    session = tmp_path / "session.jsonl"
    session.write_text("".join(
        json.dumps({"sessionId": "session", "type": "assistant", "message": part}) + "\n"
        for part in (message, forged)))
    with pytest.raises(ValueError, match="differs"):
        audit_session(session, registry, aid)


def test_unknown_auxiliary_request_poison_seal(tmp_path):
    registry, ident, token, gateway = setup(tmp_path)
    with pytest.raises(ProtocolError):
        gateway.generate(token, payload(model="haiku"))
    registry.close(ident.attempt_id)
    assert registry.seal(ident.attempt_id)[1]["errors"] == ["ProtocolError"]


def test_pending_generation_blocks_version_change(tmp_path):
    registry, ident, token, gateway = setup(tmp_path)
    aid, rid, _, _ = registry.begin(token)
    with pytest.raises(TimeoutError):
        registry.drain(.01)
    with pytest.raises(ValueError):
        registry.publish_version([{"healthy": True, "weight_version": "2"}], "2")
    registry.finish(aid, rid, error="cancelled")
    registry.seal(aid)
    registry.publish_version([{"healthy": True, "weight_version": "2", "endpoint": "a"},
                              {"healthy": True, "weight_version": "2", "endpoint": "b"}], "2")
    with pytest.raises(ValueError, match="version"):
        registry.register(identity("1"), "trial2")


def test_pool_mismatch_keeps_admission_closed(tmp_path):
    registry = TraceRegistry(tmp_path)
    registry.drain(.1)
    with pytest.raises(RuntimeError):
        registry.publish_version([{"healthy": True, "weight_version": "1"}, {"healthy": True, "weight_version": "2"}], "2")
    assert not registry.accepting


def test_http_sse_and_auth(tmp_path):
    registry, ident, token, gateway = setup(tmp_path)
    server = make_server(gateway)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/v1/messages"
    ready_url = f"http://127.0.0.1:{server.server_port}/readyz"
    opener = request.build_opener(request.ProxyHandler({}))
    try:
        with opener.open(ready_url, timeout=5) as response:
            assert json.load(response) == {"ok": True, "model": "policy"}
        req = request.Request(url, json.dumps(payload(stream=True)).encode(), {"x-api-key": token})
        with opener.open(req, timeout=5) as response:
            events = [json.loads(line[6:]) for line in response.read().decode().splitlines() if line.startswith("data: ")]
        assert events[0]["type"] == "message_start"
        assert events[-1]["type"] == "message_stop"
        assert events[2]["delta"]["text"] == "done"
        with pytest.raises(error.HTTPError) as exc:
            opener.open(request.Request(url, b'{}', {"x-api-key": "bad"}), timeout=5)
        assert exc.value.code == 401
        query = request.Request(url + "?beta=true", json.dumps(payload()).encode(),
                                {"x-api-key": token})
        with opener.open(query, timeout=5) as response:
            assert json.load(response)["role"] == "assistant"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)


def test_sse_tool_parameters_are_json_delta():
    message = {"id": "m", "role": "assistant", "type": "message", "model": "p", "content": [
        {"type": "tool_use", "id": "t", "name": "bash", "input": {"command": "pwd"}}],
        "stop_reason": "tool_use", "stop_sequence": None, "usage": {"input_tokens": 2, "output_tokens": 3}}
    events = list(sse_events(message))
    assert json.loads(events[2][1]["delta"]["partial_json"]) == {"command": "pwd"}
    assert events[-2][1]["delta"]["stop_reason"] == "tool_use"


@pytest.mark.parametrize("tools", [None, {}, "tool"])
def test_invalid_tool_list_is_protocol_error(tools):
    with pytest.raises(ProtocolError):
        convert(payload(tools=tools), model="policy")


def test_nonfinite_generated_tool_is_rejected():
    with pytest.raises(ProtocolError):
        parse_qwen_output('<tool_call>{"name":"bash","arguments":{"x":NaN}}</tool_call>',
                          response_id="m", tools=[{"function": {"name": "bash"}}])


def test_sglang_generation_version_and_logprob_evidence():
    from harborrl.gateway.sglang_backend import SGLangBackend
    class Tokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return [11, 12]
        def decode(self, token_ids, skip_special_tokens):
            return "done"
    backend = SGLangBackend("http://engine", Tokenizer(), tokenizer_digest="tok", template_digest="tpl")
    converted = convert(payload(), model="policy")
    serving = {"input_ids": backend.prepare(converted)}
    output = {"text": "done", "meta_info": {"weight_version": "1", "output_token_logprobs": [[-.2, 13]],
                                               "finish_reason": {"type": "stop"}}}
    decoded = backend.decode(output, serving, converted, identity(), "msg")
    assert decoded["output_ids"] == [13]
    assert decoded["logprob_semantics"] == "unverified"
    output["meta_info"]["weight_version"] = "2"
    with pytest.raises(ValueError, match="version"):
        backend.decode(output, serving, converted, identity(), "msg")
    output["meta_info"]["weight_version"] = "1"
    output["meta_info"]["output_token_logprobs"] = [[None, 13]]
    with pytest.raises(ValueError, match="logprobs"):
        backend.decode(output, serving, converted, identity(), "msg")
    output["meta_info"]["output_token_logprobs"] = [[-.2, 13]]
    output["output_ids"] = [14]
    with pytest.raises(ValueError, match="token IDs"):
        backend.decode(output, serving, converted, identity(), "msg")
    output["output_ids"] = [13]
    output["text"] = "different"
    with pytest.raises(ValueError, match="text"):
        backend.decode(output, serving, converted, identity(), "msg")
    backend.max_context = 10
    with pytest.raises(ProtocolError, match="context"):
        backend.prepare(converted)


def test_sglang_weight_version_is_explicit():
    from harborrl.gateway.sglang_backend import SGLangBackend

    class Response:
        def read(self):
            return json.dumps({"weight_version": "7"}).encode()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class Opener:
        def open(self, request, timeout):
            assert request.full_url == "http://engine/get_weight_version"
            assert timeout == 30
            return Response()

    backend = SGLangBackend("http://engine", SimpleNamespace(), tokenizer_digest="tok",
                            template_digest="tpl", timeout=30)
    backend.opener = Opener()
    assert backend.weight_version() == "7"
