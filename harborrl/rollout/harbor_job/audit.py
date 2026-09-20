"""Audit persisted Claude native session JSONL, including final assistant output.

Never infer consumption from HTTP write success or matching request timestamps.
Unknown auxiliary sessions, summaries and conflicting duplicate messages reject
an attempt. This parser must be probed against the locked real CLI version.
"""
import hashlib
import json
from pathlib import Path
from harborrl.trajectories.native import digest


def audit_session(path, registry, attempt_id):
    path = Path(path)
    raw = path.read_bytes()
    evidence = f"{path}#sha256={hashlib.sha256(raw).hexdigest()}"
    seen, sessions = {}, set()
    records = []
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
        if rid in seen and seen[rid] != digest(content):
            raise ValueError("conflicting or incremental session message; CLI profile needs an adapter")
        if rid not in seen:
            records.append((rid, content))
            seen[rid] = digest(content)
    if len(sessions) != 1 or not records:
        raise ValueError("expected one complete native session")
    # Validate the complete file before mutating the registry; seal catches omitted responses.
    for rid, content in records:
        registry.consume(attempt_id, rid, content, evidence)
    return {"session_id": next(iter(sessions)), "consumed_response_ids": list(seen), "evidence_ref": evidence}
