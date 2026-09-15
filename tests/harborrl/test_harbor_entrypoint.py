"""Exercise the real hook's Harbor branches with external services replaced."""

import importlib.util
from pathlib import Path
import sys
import types
from types import SimpleNamespace as NS

import pytest


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode",
    [
        "success",
        "open_failure",
        "no_response",
        "no_tokens",
        "eval_failure",
        "version_change",
    ],
)
async def test_terminal_ir_and_cleanup(tmp_path, monkeypatch, mode):
    pytest.importorskip("torch")
    root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(root / "backends/slime"))
    from slime.utils.types import Sample
    from harborrl.types import Interaction, RunContext
    from harborrl.trajectories.store import load
    from harborrl.rollout import harbor_bridge

    # Only the external serving scheduler is substituted; execute the actual hook.
    stub = types.ModuleType("slime.rollout.sglang_rollout")
    stub.GenerateState = lambda args: NS(tokenizer=NS(eos_token_id=0))
    monkeypatch.setitem(sys.modules, "slime.rollout.sglang_rollout", stub)
    spec = importlib.util.spec_from_file_location(
        "_harbor_entrypoint_test", root / "harborrl/rollout/entrypoint.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    plan = NS(
        data_source="harbor_terminal",
        task_meta={
            "data_source": "harbor_terminal",
            "harbor_task_ref": {
                "dataset": "data",
                "task": "task",
                "source_digest": "digest",
            },
        },
        run_ctx=RunContext("trajectory", 3, 9, tmp_path),
        log_tag="test",
        task_key="task",
        task_spec=NS(),
        prm_coef=0.0,
        traj_save_interval=0,
    )
    monkeypatch.setattr(module, "_prepare_rollout_plan", lambda *a: plan)
    monkeypatch.setattr(module, "_uses_remote_terminal_env", lambda *a: False)
    closed = []

    async def open_env(*args):
        if mode == "open_failure":
            raise RuntimeError("cannot start")

    async def close_env(*args):
        closed.append(True)

    monkeypatch.setattr(module, "_open_env_session", open_env)
    monkeypatch.setattr(module, "_close_rollout_session", close_env)

    def clients(args, state, plan, session, clients, sampling):
        clients.sglang_client = NS(
            tokenizer=NS(
                name_or_path="tok", chat_template="template", get_vocab=lambda: {"a": 1}
            ),
            sampling_params=sampling,
            chat_template_type="hf",
        )

    monkeypatch.setattr(module, "_build_turn_clients", clients)

    async def turn_loop(plan, session, clients, loop):
        if mode != "no_response":
            loop.final_response = NS(msg="done")
        if mode != "no_tokens":
            loop.interactions.append(
                Interaction(
                    input_ids=[1],
                    output_token_ids=[2] * 12,
                    output_token_logprobs=[-0.1] * 12,
                    generation_meta={"weight_version": "7"},
                )
            )

    monkeypatch.setattr(module, "_run_turn_loop", turn_loop)
    monkeypatch.setattr(
        module, "_decide_status", lambda *a: (Sample.Status.COMPLETED, False)
    )

    async def evaluate(*args):
        if mode == "eval_failure":
            return (
                0.0,
                {"exception_stage": "reward_parse"},
                "missing reward",
                Sample.Status.FAILED,
            )
        return 0.0, {"raw_reward": 0.0}, None, Sample.Status.COMPLETED

    monkeypatch.setattr(module, "_evaluate_outcome", evaluate)
    monkeypatch.setattr(module, "_summarize_turn_uncertainty", lambda *a, **kw: {})
    monkeypatch.setattr(module, "_finalize_sample_metadata", lambda *a, **kw: None)
    monkeypatch.setattr(module, "_save_rollout_artifacts", lambda *a, **kw: None)
    calls = []

    async def version(client):
        calls.append(True)
        return "8" if mode == "version_change" and len(calls) > 1 else "7"

    monkeypatch.setattr(harbor_bridge, "serving_version", version)
    monkeypatch.setenv("HARBOR_LOGPROB_SOURCE", "sglang-audited")
    monkeypatch.setenv("HARBOR_LOGPROB_SEMANTICS", "raw_model")
    monkeypatch.setenv("HARBOR_IR_ROOT", str(tmp_path / "ir"))
    result = await module.generate(NS(), Sample(group_index=3, index=9), {})
    assert closed == [True]
    assert len(list((tmp_path / "ir/started").glob("*.json"))) == 1
    records = list((tmp_path / "ir").glob("*.json"))
    assert len(records) == 1
    ir = load(records[0])
    if mode == "success":
        assert ir.readiness == "RL_READY"
        assert result[0].reward["accuracy"] == 0.0
        assert not result[0].remove_sample
    else:
        assert ir.readiness == "EVAL_ONLY"
        assert result[0].remove_sample
