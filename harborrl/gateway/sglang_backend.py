"""Direct /generate adapter with explicit tokenizer and generation-time evidence.

The live engine must return weight_version in meta_info and be exclusively
updated through the caller's drained pool barrier. Before/after probes alone
are deliberately insufficient. This adapter does not load models on import.
"""
import json
from urllib import request
from harborrl.trajectories.native import digest, finite
from .messages import ProtocolError, parse_qwen_output


class SGLangBackend:
    def __init__(self, endpoint, tokenizer, *, tokenizer_digest, template_digest,
                 timeout=120, max_context=32768, audited_raw_logprobs=False):
        self.endpoint = endpoint.rstrip("/")
        self.tokenizer = tokenizer
        self.tokenizer_digest, self.template_digest = tokenizer_digest, template_digest
        self.timeout, self.max_context = timeout, max_context
        self.audited_raw_logprobs = audited_raw_logprobs
        self.opener = request.build_opener(request.ProxyHandler({}))

    def prepare(self, converted):
        ids = self.tokenizer.apply_chat_template(converted["messages"], tools=converted["tools"] or None,
                                                tokenize=True, add_generation_prompt=True, enable_thinking=False)
        if not isinstance(ids, list) or not ids or any(type(i) is not int or i < 0 for i in ids):
            raise ProtocolError("tokenizer did not produce actual input IDs")
        if len(ids) + converted["sampling"]["max_new_tokens"] > self.max_context:
            raise ProtocolError("context budget exceeded; implicit truncation is forbidden")
        return ids

    def count_tokens(self, converted):
        return len(self.prepare(converted))

    def generate(self, converted, identity, response_id):
        ids = self.prepare(converted)
        payload = {"input_ids": ids, "sampling_params": converted["sampling"],
                   "return_logprob": True, "stream": False}
        req = request.Request(self.endpoint + "/generate", data=json.dumps(payload).encode(),
                              headers={"Content-Type": "application/json"})
        with self.opener.open(req, timeout=self.timeout) as response:
            output = json.load(response)
        return self.decode(output, payload, converted, identity, response_id)

    def decode(self, output, payload, converted, identity, response_id):
        meta = output.get("meta_info", {})
        if str(meta.get("weight_version", "")) != identity.policy_version:
            raise ValueError("missing or mismatched generation-time weight version")
        rows = meta.get("output_token_logprobs")
        if not isinstance(rows, list) or not rows or any(
            not isinstance(r, (list, tuple)) or len(r) < 2 or not finite(r[0]) or r[0] > 0
            or type(r[1]) is not int or r[1] < 0 for r in rows
        ):
            raise ValueError("missing or malformed generation logprobs")
        text = output.get("text")
        if not isinstance(text, str):
            raise ValueError("missing generation text")
        finish = meta.get("finish_reason", {})
        if finish.get("type") not in ("stop", "length"):
            raise ValueError("unsupported serving finish reason")
        blocks = parse_qwen_output(text, response_id=response_id, tools=converted["tools"])
        has_tools = any(b["type"] == "tool_use" for b in blocks)
        stop_sequence = finish.get("matched") if isinstance(finish.get("matched"), str) else None
        if stop_sequence not in converted["sampling"]["stop"]:
            stop_sequence = None
        reason = "max_tokens" if finish["type"] == "length" else (
            "tool_use" if has_tools else "stop_sequence" if stop_sequence else "end_turn")
        # Until a live probe confirms raw probability semantics, serve only EVAL_ONLY evidence.
        return {"input_ids": payload["input_ids"], "output_ids": [r[1] for r in rows],
                "logprobs": [r[0] for r in rows], "logprob_semantics": "raw_model" if self.audited_raw_logprobs else "unverified",
                "engine_id": self.endpoint, "policy_version": str(meta["weight_version"]),
                "tokenizer_digest": self.tokenizer_digest, "template_digest": self.template_digest,
                "serving_input": payload, "evidence_kind": "serving", "content": blocks,
                "finish_reason": reason, "stop_sequence": stop_sequence, "sampling": converted["sampling"],
                "serving_output_digest": digest(output)}
