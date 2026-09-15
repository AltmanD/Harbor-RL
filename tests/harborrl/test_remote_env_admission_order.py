import asyncio
from pathlib import Path
from types import SimpleNamespace

from harborrl.types import RunContext, TaskTimeouts
from harborrl.rollout import generate_steps


def test_remote_admission_precedes_worker_allocate(monkeypatch):
    events = []

    class FakeClient:
        async def reset(self, **kwargs):
            events.append("reset")
            return {"user_msg": "ready", "tool_schemas": []}

    async def acquire(task_key, *, log_tag):
        events.append("admission")
        return task_key

    async def create_env_client(task_spec, run_ctx, task_meta):
        assert events == ["admission"]
        events.append("allocate")
        return FakeClient(), "lease-1"

    monkeypatch.setattr(generate_steps, "_uses_remote_terminal_env", lambda _: True)
    monkeypatch.setattr(generate_steps, "_task_circuit_open_reason", lambda _: None)
    monkeypatch.setattr(generate_steps, "_acquire_remote_env_admission", acquire)
    monkeypatch.setattr(generate_steps, "_create_env_client", create_env_client)
    monkeypatch.setattr(generate_steps, "_make_task_spec", lambda _: object())
    monkeypatch.setenv("ENV_HEARTBEAT_INTERVAL", "0")

    run_ctx = RunContext("uid", -1, 0, Path("/tmp"))
    plan = SimpleNamespace(
        task_meta={"data_source": "seta"},
        task_key="705:seta_env/705",
        log_tag="[test]",
        run_ctx=run_ctx,
        run_ctx_payload=run_ctx.to_payload(),
        timeouts=TaskTimeouts(),
    )
    session = generate_steps._EnvSession()

    asyncio.run(generate_steps._open_env_session(plan, session))

    assert events == ["admission", "allocate", "reset"]
    assert session.admission_key == plan.task_key
    assert session.lease_id == "lease-1"
