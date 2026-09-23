"""Audit persisted Claude native session JSONL, including final assistant output.

Never infer consumption from HTTP write success or matching request timestamps.
Unknown auxiliary sessions, summaries and conflicting duplicate messages reject
an attempt. Claude Code 2.1.141 may persist one logical assistant message as
multiple session lines (one content block per line, parallel tool calls); the
adapter merges those splits before the immutable trace consumption check. This
parser must be probed against the locked real CLI version.
"""
import hashlib
import json
from pathlib import Path


def audit_session(path, registry, attempt_id):
    path = Path(path)
    raw = path.read_bytes()
    evidence = f"{path}#sha256={hashlib.sha256(raw).hexdigest()}"
    sessions = set()
    merged = {}  # message id -> merged content blocks, in first-seen order
    for line in raw.splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("isSidechain") or event.get("type") == "summary" or event.get("subtype") == "compact_boundary":
            raise ValueError("auxiliary generation or compact is not enabled")
        if event.get("sessionId"):
            sessions.add(event["sessionId"])
        if event.get("type") != "assistant":
            continue
        message = event.get("message", {})
        if message.get("role") != "assistant" or not isinstance(message.get("content"), list) or not message.get("id"):
            raise ValueError("assistant consumption lacks ID or content")
        rid, content = message["id"], message["content"]
        if rid not in merged:
            merged[rid] = list(content)
            continue
        current = merged[rid]
        if content == current or current[:len(content)] == content:
            continue  # identical rewrite or a shorter recap of the same message
        if content[:len(current)] == current:
            merged[rid] = list(content)  # growing incremental update of one message
        else:
            current.extend(content)  # per-block split of one logical message
    records = [(rid, content) for rid, content in merged.items()]
    if len(sessions) != 1 or not records:
        raise ValueError("expected one complete native session")
    # Validate the complete file before mutating the registry; seal catches omitted responses.
    for rid, content in records:
        registry.consume(attempt_id, rid, content, evidence)
    return {"session_id": next(iter(sessions)), "consumed_response_ids": list(merged), "evidence_ref": evidence}
