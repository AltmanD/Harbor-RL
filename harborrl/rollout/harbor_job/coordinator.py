"""Bounded rollout groups and an isolated Runner client.

No low-reward retry, no incomplete group export, no automatic weight updates.
"""
from __future__ import annotations
import asyncio
from dataclasses import replace
import json
import os
from pathlib import Path
import secrets

from harborrl.trajectories.native import publish, assemble
from .audit import audit_session
from .bindings import sanitized_runner_env
from .collector import collect


class GroupSlots:
    def __init__(self, identities, *, max_attempts=2):
        if len(identities) < 2 or len({i.group_key for i in identities}) != 1:
            raise ValueError("one group with at least two slots required")
        if any(len({getattr(i, key) for i in identities}) != len(identities)
               for key in ("slot_id", "attempt_id", "trajectory_id")) or type(max_attempts) is not int or max_attempts < 1:
            raise ValueError("invalid slots or retry limit")
        self.identities = {i.slot_id: i for i in identities}
        self.max_attempts = max_attempts
        self.history = {i.slot_id: [] for i in identities}
        self.completed = {}

    def start(self, slot):
        if slot in self.completed or len(self.history[slot]) >= self.max_attempts:
            raise ValueError("slot complete or retry budget exhausted")
        previous = self.history[slot]
        if previous and previous[-1]["status"] == "running":
            raise ValueError("slot already running")
        original = self.identities[slot]
        current = original if not previous else replace(original, attempt_id=secrets.token_hex(16), trajectory_id=secrets.token_hex(16))
        previous.append({"identity": current.to_dict(), "status": "running",
                         "parent_attempt_id": previous[-1]["identity"]["attempt_id"] if previous else None})
        return current

    def finish(self, slot, ir=None, *, error=None):
        from harborrl.trajectories.native import require_ready
        history = self.history[slot]
        if not history or history[-1]["status"] != "running":
            raise ValueError("no active attempt")
        if ir is not None:
            require_ready(ir)
            if ir["identity"] != history[-1]["identity"]:
                raise ValueError("result is not from the active attempt")
            self.completed[slot] = ir
            history[-1]["status"] = "valid"  # zero rewards are complete, never retried
        else:
            if not error:
                raise ValueError("failed attempt requires a reason")
            history[-1].update(status="invalid", error=error)

    def export(self):
        from harborrl.rollout.exporters.native import export_group
        if set(self.completed) != set(self.identities):
            raise ValueError("incomplete group cannot enter training")
        return export_group([self.completed[slot] for slot in self.identities], len(self.identities))


async def run_attempt(identity, *, task_path, profile, gateway_url, runner_python,
                      root, registry, reward_profile, timeout=1800, cleanup_timeout=120):
    """One real trial. CPU-only tests use a separate fake worker, never this path."""
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    credential = secrets.token_urlsafe(32)
    project = str(Path(__file__).resolve().parents[3])
    env = sanitized_runner_env(os.environ)
    env["PYTHONPATH"] = project
    stderr = (root / "runner.stderr").open("wb")
    try:
        process = await asyncio.create_subprocess_exec(runner_python, "-m", "harborrl.rollout.harbor_job.runner",
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=stderr, env=env)
    except BaseException:
        stderr.close()
        publish(root / "launch-failed.json", {"identity": identity.to_dict(), "reason": "Runner process did not start"})
        raise
    registered, trial_id = False, None
    messages = []
    try:
        req = {"identity": identity.to_dict(), "task_path": str(Path(task_path).resolve()), "profile": profile,
               "gateway_url": gateway_url, "credential": credential, "output": str(root / "runner")}
        process.stdin.write(json.dumps(req).encode() + b"\n")
        await process.stdin.drain()

        async def lifecycle():
            nonlocal registered, trial_id
            while line := await process.stdout.readline():
                event = json.loads(line)
                messages.append(event)
                if event["type"] == "prepared":
                    if registered:
                        raise ValueError("duplicate prepared event")
                    trial_id = event["trial_id"]
                    registry.register(identity, trial_id, credential=credential)
                    registered = True
                    process.stdin.write(b'{"command":"run"}\n')
                    await process.stdin.drain()
                elif event["type"] == "event":
                    if not registered or event.get("trial_id") != trial_id:
                        raise ValueError("event trial identity mismatch")
                elif event["type"] in ("terminal", "cancelled"):
                    if not registered or event["trial_id"] != trial_id:
                        raise ValueError("terminal trial identity mismatch")
                    return event
                else:
                    raise ValueError("unknown Runner message")
            raise RuntimeError("Runner exited before terminal result")

        terminal = await asyncio.wait_for(lifecycle(), timeout)
        registry.close(identity.attempt_id)
        # Do not kill a serving thread: drain safely or leave a blocking cleanup record.
        await asyncio.wait_for(process.wait(), cleanup_timeout)
        if process.returncode != 0 or terminal["type"] != "terminal":
            raise RuntimeError("native trial did not finish normally")
        trial_dir = Path(terminal["trial_dir"]).resolve()
        result_path = Path(terminal["result_path"]).resolve()
        if not trial_dir.is_relative_to(root / "runner" / "harbor") or result_path != trial_dir / "result.json":
            raise ValueError("Runner result path is outside this attempt")
        sessions = [p for p in (trial_dir / "agent" / "sessions" / "projects").rglob("*.jsonl")]
        if len(sessions) != 1:
            raise ValueError("expected exactly one native Claude session; auxiliary sessions require review")
        audit_session(sessions[0], registry, identity.attempt_id)
        turns, seal = registry.seal(identity.attempt_id)
        receipt = collect(identity, trial_id, terminal["result_path"], trial_dir / "verifier", reward_profile)
        publish(root / "evaluation.json", receipt)
        ir = assemble(identity, trial_id, turns, seal, receipt)
        publish(root / "trajectory.v2.json", ir)
        return ir
    except BaseException as exc:
        publish(root / "failure.json", {"identity": identity.to_dict(), "error_type": type(exc).__name__})
        raise
    finally:
        if registered:
            registry.close(identity.attempt_id)
        if process.returncode is None:
            try:
                process.stdin.write(b'{"command":"cancel"}\n')
                await process.stdin.drain()
                await asyncio.wait_for(process.wait(), cleanup_timeout)
            except (BrokenPipeError, ConnectionResetError, asyncio.TimeoutError):
                process.kill()
                await process.wait()
                publish(root / "cleanup-required.json", {"identity": identity.to_dict(), "trial_id": trial_id,
                    "reason": "Runner cleanup deadline exceeded; inspect Harbor resources before retry"})
        if process.returncode != 0 and not (root / "cleanup-required.json").exists():
            publish(root / "cleanup-required.json", {"identity": identity.to_dict(), "trial_id": trial_id,
                "reason": "Runner exited abnormally; inspect Harbor resources before retry"})
        if registered:
            try:
                registry.seal(identity.attempt_id)
            except ValueError:
                publish(root / "trace-unsealed.json", {"identity": identity.to_dict(), "reason": "in-flight response or conflicting seal"})
        stderr.close()
        publish(root / "runner-events.json", messages)
