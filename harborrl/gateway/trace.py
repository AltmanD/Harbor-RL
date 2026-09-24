"""Attempt-scoped credentials, request draining and immutable trace sealing."""
from copy import deepcopy
import hashlib
from pathlib import Path
import secrets
import threading
import time
from harborrl.trajectories.native import digest, publish


class TraceRegistry:
    def __init__(self, root):
        self.root = Path(root)
        self.lock = threading.RLock()
        self.changed = threading.Condition(self.lock)
        self.attempts = {}
        self.credentials = {}
        self.accepting = True
        self.version = None

    def register(self, identity, trial_id, *, credential=None):
        with self.lock:
            if not self.accepting or (self.version is not None and self.version != identity.policy_version):
                raise ValueError("policy admission closed or version mismatch")
            if identity.attempt_id in self.attempts:
                raise ValueError("attempt already registered; restart requires a new attempt")
            if not isinstance(trial_id, str) or not trial_id:
                raise ValueError("trial identity required")
            token = secrets.token_urlsafe(32) if credential is None else credential
            if not isinstance(token, str) or not token:
                raise ValueError("nonempty credential required")
            key = hashlib.sha256(token.encode()).hexdigest()
            if key in self.credentials or any(s["credential"] == key for s in self.attempts.values()):
                raise ValueError("attempt credential must be unique")
            root = self.root / digest(identity.to_dict())
            publish(root / "manifest.json", {"identity": identity.to_dict(), "harbor_trial_id": trial_id})
            # Never resurrect a persisted attempt after a process restart.
            claim = root / "writer.claim"
            with claim.open("x") as f:
                f.write("single writer\n")
            self.version = identity.policy_version
            self.attempts[identity.attempt_id] = {"identity": identity, "trial_id": trial_id, "root": root,
                "pending": set(), "turns": [], "errors": [], "closed": False, "sealed": False, "credential": key}
            self.credentials[key] = identity.attempt_id
            return token

    def authenticate(self, token):
        with self.lock:
            aid = self.credentials.get(hashlib.sha256(token.encode()).hexdigest())
            if aid is None:
                raise PermissionError("invalid attempt credential")
            state = self.attempts[aid]
            if not self.accepting or state["closed"]:
                raise PermissionError("attempt admission closed")
            return aid

    def begin(self, token):
        with self.lock:
            aid = self.authenticate(token)
            state = self.attempts[aid]
            if state["pending"]:
                state["errors"].append("concurrent_or_auxiliary_generation")
                raise ValueError("only one policy generation per attempt may be in flight")
            request_id = secrets.token_hex(16)
            publish(state["root"] / "requests" / f"{request_id}.json", {"request_id": request_id, "state": "registered"})
            state["pending"].add(request_id)
            return aid, request_id, state["identity"], state["trial_id"]

    def finish(self, aid, request_id, turn=None, error=None):
        with self.changed:
            state = self.attempts[aid]
            if request_id not in state["pending"]:
                raise ValueError("request not pending")
            if error:
                state["errors"].append(error)
            if turn is not None:
                state["turns"].append(deepcopy(turn))
            publish(state["root"] / "generated" / f"{request_id}.json", {"turn": turn, "error": error})
            if error:
                state["pending"].remove(request_id)
                self.changed.notify_all()

    def delivery(self, aid, response_id, status):
        with self.changed:
            state = self.attempts[aid]
            if state["sealed"]:
                raise ValueError("trace already sealed")
            matches = [t for t in state["turns"] if t["response_id"] == response_id]
            if len(matches) != 1 or status not in ("sent", "delivery_unknown"):
                raise ValueError("invalid delivery record")
            matches[0]["delivery"] = status
            publish(state["root"] / "delivery" / f"{response_id}.json", {"status": status})
            state["pending"].discard(matches[0]["request_id"])
            self.changed.notify_all()

    def consume(self, aid, response_id, content, evidence_ref):
        """Called only by trusted native-session audit, never by an HTTP policy client."""
        with self.lock:
            state = self.attempts[aid]
            if state["sealed"]:
                raise ValueError("trace already sealed")
            matches = [t for t in state["turns"] if t["response_id"] == response_id]
            if len(matches) != 1 or not evidence_ref:
                raise ValueError("unknown consumed response")
            turn = matches[0]
            if turn["response"]["content"] != content or turn["delivery"] not in ("sent", "consumed_confirmed"):
                raise ValueError("consumed response differs or delivery is unknown")
            publish(state["root"] / "consumption" / f"{response_id}.json", {"response_id": response_id, "content_digest": digest(content), "evidence_ref": evidence_ref})
            turn.update(delivery="consumed_confirmed", consumption_ref=evidence_ref)

    def close(self, aid):
        with self.lock:
            state = self.attempts[aid]
            state["closed"] = True
            self.credentials.pop(state["credential"], None)

    def seal(self, aid):
        with self.lock:
            state = self.attempts[aid]
            if not state["closed"] or state["pending"]:
                raise ValueError("close admission and drain requests before sealing")
            turns = deepcopy(state["turns"])
            errors = list(state["errors"])
            if any(t.get("delivery") != "consumed_confirmed" for t in turns):
                errors.append("unconfirmed_consumption")
            seal = {"identity": state["identity"].to_dict(), "harbor_trial_id": state["trial_id"],
                    "status": "sealed", "pending_requests": 0, "request_ids": [t["request_id"] for t in turns],
                    "turns_digest": digest(turns), "errors": sorted(set(errors))}
            publish(state["root"] / "trace.json", {"turns": turns, "seal": seal})
            state["sealed"] = True
            return turns, seal

    def drain(self, timeout):
        deadline = time.monotonic() + timeout
        with self.changed:
            self.accepting = False
            for aid in self.attempts:
                self.close(aid)
            while any(s["pending"] for s in self.attempts.values()):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("generation still in flight; weights must not change")
                self.changed.wait(remaining)

    def publish_version(self, states, expected):
        from harborrl.platform.policy_pool import validate_pool
        with self.lock:
            if self.accepting or any(not s["sealed"] for s in self.attempts.values()):
                raise ValueError("all attempts must be sealed behind a closed admission barrier")
            self.version = validate_pool(states, expected)
            self.accepting = True
