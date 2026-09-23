"""CPU contracts for schema-2 launch and Slime tensor boundaries."""
from __future__ import annotations

import json
import asyncio
from pathlib import Path
import os
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

from harborrl.config import launch_plan, load_config


def native_config(tmp_path, **overrides):
    task = tmp_path / "task"
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "task.toml").write_text('version = "1.0"\n')
    (task / "instruction.md").write_text("original instruction\n")
    (task / "environment/Dockerfile").write_text("FROM debian:bookworm\n")
    (task / "tests/test.sh").write_text("true\n")
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps([{
        "id": "task", "revision": "1", "path": str(task),
        "task_digest": "0" * 64, "reward_profile": {"key": "reward"},
    }]))
    profile = tmp_path / "claude.json"
    profile.write_text(json.dumps({
        "cli_version": "2.1.0", "model": "policy", "max_turns": 4,
        "agent_timeout_sec": 60, "setup_timeout_sec": 60, "verifier_timeout_sec": 60,
    }))
    values = {
        "schema_version": 2,
        "execution": {"backend": "harbor_job"},
        "tasks": {"catalog": str(catalog)},
        "harness": {"name": "claude_code", "profile": str(profile)},
        "harbor": {"python": "/opt/harbor/bin/python", "version": "0.23.0",
                    "workers": ["worker-a", "worker-b"]},
        "gateway": {"host": "gateway", "port": 49183,
                    "advertised_url": "http://gateway:49183",
                    "tokenizer_digest": "1" * 64, "template_digest": "2" * 64,
                    "audited_raw_logprobs": True},
        "model": {"checkpoint": "/model", "reference": "/reference", "args_file": "qwen3-8B"},
        "training": {"num_rollout": 1, "learning_rate": 0.0001, "save_interval": 1},
        "sampling": {"group_size": 2, "groups_per_batch": 1, "max_attempts": 2,
                     "max_tokens": 128, "max_context": 1024},
        "deployment": {"layout": "split", "num_gpus": 8, "actor_gpus": 4,
                       "rollout_gpus": 4, "actor_tensor_parallel_size": 4,
                       "rollout_gpus_per_engine": 4},
        "output": {"root": str(tmp_path / "runs")},
    }
    values.update(overrides)
    path = tmp_path / "native.yaml"
    lines = ["schema_version: 2"]
    for section, fields in values.items():
        if section == "schema_version":
            continue
        lines.append(f"{section}: {json.dumps(fields)}")
    path.write_text("\n".join(lines) + "\n")
    return path, task


def test_schema2_launch_plan_and_strict_override(tmp_path, capsys):
    path, task = native_config(tmp_path)
    from harborrl.data.harbor.native_inspector import inspect_native
    digest = inspect_native(task)["task_digest"]
    catalog_path = json.loads(path.read_text().split("tasks: ", 1)[1].split("\n", 1)[0])["catalog"]
    catalog = json.loads(Path(catalog_path).read_text())
    catalog[0]["task_digest"] = digest
    Path(catalog_path).write_text(json.dumps(catalog))

    runner = tmp_path / "venv" / "python"
    runner.parent.mkdir()
    runner.symlink_to("/opt/absolute-base/python")
    path.write_text(path.read_text().replace("/opt/harbor/bin/python", str(runner)))
    assert load_config(path)["harbor"]["python"] == str(runner)

    plan = launch_plan(load_config(path))
    assert plan["environment"]["HARBORRL_NATIVE_ROLLOUT"] == "1"
    assert plan["environment"]["HARBORRL_NATIVE_SLIME_CONTRACT"] == "slime-v032"
    with pytest.raises(ValueError, match="backend_contract"):
        load_config(path, ["training.backend_contract=unknown"])
    assert plan["environment"]["HARBORRL_NATIVE_AUDITED_LOGPROBS"] == "1"
    assert json.loads(plan["environment"]["HARBORRL_NATIVE_RUNNER_WORKERS"]) == ["worker-a", "worker-b"]
    assert plan["environment"]["N_SAMPLES"] == "2"
    assert plan["training_command"][0] == "bash"
    assert plan["launcher"].startswith("native Claude Code")
    from harborrl.cli import main
    assert main(["train", "--config", str(path), "--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)["config"]["schema_version"] == 2

    config = load_config(path, ["sampling.group_size=3"])
    assert config["sampling"]["group_size"] == 3
    with pytest.raises(ValueError, match="unknown native override"):
        load_config(path, ["sampling.unknown=3"])
    with pytest.raises(ValueError, match="GPU budget"):
        load_config(path, ["deployment.rollout_gpus=5"])
    config["harbor"]["workers"] = ["bad worker"]
    path.write_text(path.read_text().replace(
        json.dumps(["worker-a", "worker-b"]), json.dumps(["bad worker"])))
    with pytest.raises(ValueError, match="harbor.workers"):
        load_config(path)


def test_native_shell_dry_run_uses_official_v032_hooks(tmp_path):
    native_config(tmp_path)
    prompt = tmp_path / "native.jsonl"
    prompt.write_text('{"task":"fixture","metadata":{"harborrl_native":true}}\n')
    root = tmp_path / "run"
    slime = tmp_path / "slime"
    model_args = slime / "scripts" / "models"
    model_args.mkdir(parents=True)
    (slime / "train.py").write_text("# official Slime v0.3.2 entrypoint\n")
    (model_args / "qwen3-8B.sh").write_text("MODEL_ARGS=(--hidden-size 4096)\n")
    env = {key: value for key, value in os.environ.items()
           if key in {"PATH", "HOME", "USER", "LANG", "LC_ALL", "TMPDIR"}}
    env.update({
        "DRY_RUN": "1", "DATASET": "native", "ALGO": "grpo",
        "HARNESS_OPTION": "native-claude-code", "HARBORRL_NATIVE_ROLLOUT": "1",
        "HARBORRL_SKIP_GLOBAL_CLEANUP": "1", "ROLLOUT_PROMPT_DATA": str(prompt),
        "HF_CKPT": "/model", "REF_LOAD": "/reference", "MODEL_ARGS_FILE": "qwen3-8B",
        "MODEL_TAG": "qwen3-8B", "NUM_GPUS": "8", "ACTOR_GPUS": "4",
        "ROLLOUT_GPUS": "4", "TP_SIZE": "4", "ROLLOUT_NUM_GPUS_PER_ENGINE": "4",
        "HARBORRL_GPU_LAYOUT": "split", "NUM_ROLLOUT": "1", "SAVE_INTERVAL": "1",
        "ROLLOUT_BATCH_SIZE": "1", "N_SAMPLES": "2", "RUN_DIR": str(root), "RUN_ID": "run",
        "RUNS_ROOT": str(tmp_path), "CKPT_ROOT": str(root / "checkpoints"),
        "HARBORRL_NATIVE_MAX_ATTEMPTS": "2", "HARBORRL_NATIVE_DRAIN_TIMEOUT": "1",
        "HARBORRL_NATIVE_RUNNER_WORKERS": '["worker-a", "worker-b"]',
        "HARBORRL_NATIVE_GATEWAY_HOST": "0.0.0.0",
        "HARBORRL_NATIVE_SLIME_CONTRACT": "slime-v032",
        "SLIME_DIR": str(slime),
    })
    script = Path(__file__).resolve().parents[2] / "harborrl/platform/slime_train.sh"
    output = tmp_path / "shell.log"
    with output.open("w") as stream:
        result = subprocess.run(["bash", str(script)], stdout=stream, stderr=subprocess.STDOUT,
                                env=env, check=False, timeout=30)
    result.stdout = output.read_text()
    assert result.returncode == 0, result.stdout
    command = next(line for line in result.stdout.splitlines() if line.startswith("[dry-run] "))
    assert "--loss-type custom_loss" in command
    for argument in (
        "--rollout-function-path harborrl.backends.slime_v032.rollout.generate_rollout",
        "--custom-convert-samples-to-train-data-path harborrl.backends.slime_v032.converter.convert_samples_to_train_data",
        "--custom-advantage-function-path harborrl.backends.slime_v032.advantage.compute_advantages_and_returns",
        "--custom-loss-function-path harborrl.backends.slime_v032.loss.loss_function",
        "--rollout-data-postprocess-path harborrl.backends.slime_v032.postprocess.rollout_data_postprocess",
    ):
        assert argument in command
    assert "native_slime" not in command
    assert "--use-rollout-logprobs" in command
    assert "--megatron-to-hf-mode bridge" in command
    assert "--load /model" in command
    assert command.count(" --load ") == 1
    assert "--num-steps-per-rollout 1" in command
    assert "--use-kl-loss" not in command
    assert "harborrl.rollout.entrypoint.generate" not in command


def test_native_runtime_binds_gateway_with_registry_and_backend(tmp_path, monkeypatch):
    from harborrl.gateway import server as gateway_server
    from harborrl.gateway import sglang_backend
    from harborrl.gateway import trace
    from harborrl.rollout import native_generate

    catalog = tmp_path / "catalog.json"
    catalog.write_text("[]")
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"model": "policy"}))
    tokenizer = SimpleNamespace(
        apply_chat_template=lambda *args, **kwargs: [1, 2],
        get_vocab=lambda: {"token": 1},
    )
    tokenizer_digest, template_digest = native_generate.tokenizer_fingerprint(tokenizer)
    environment = {
        "RUN_DIR": str(tmp_path), "HARBORRL_NATIVE_CATALOG": str(catalog),
        "HARBORRL_NATIVE_PROFILE": str(profile),
        "HARBORRL_NATIVE_TOKENIZER_DIGEST": tokenizer_digest,
        "HARBORRL_NATIVE_TEMPLATE_DIGEST": template_digest,
        "HARBORRL_NATIVE_GATEWAY_HOST": "127.0.0.1",
        "HARBORRL_NATIVE_GATEWAY_PORT": "49183",
    }
    for key, value in environment.items():
        monkeypatch.setenv(key, value)

    calls = {}

    class FakeGateway:
        def __init__(self, registry, backend, *, model, max_output_tokens):
            calls["gateway"] = (registry, backend, model, max_output_tokens)
            calls["gateway_object"] = self

    class FakeServer:
        def serve_forever(self):
            pass

        def shutdown(self):
            pass

        def server_close(self):
            pass

    class FakeTraceRegistry:
        def __init__(self, path):
            calls["registry_path"] = path

    class FakeBackend:
        def __init__(self, *args, **kwargs):
            calls["backend_args"] = (args, kwargs)

        def weight_version(self):
            return "7"

    def fake_make_server(gateway, host, port):
        calls["server"] = (gateway, host, port)
        return FakeServer()

    slime = ModuleType("slime")
    utils = ModuleType("slime.utils")
    processing = ModuleType("slime.utils.processing_utils")
    processing.load_tokenizer = lambda *args, **kwargs: tokenizer
    slime.utils = utils
    utils.processing_utils = processing
    monkeypatch.setitem(sys.modules, "slime", slime)
    monkeypatch.setitem(sys.modules, "slime.utils", utils)
    monkeypatch.setitem(sys.modules, "slime.utils.processing_utils", processing)
    monkeypatch.setattr(gateway_server, "Gateway", FakeGateway)
    monkeypatch.setattr(gateway_server, "make_server", fake_make_server)
    monkeypatch.setattr(sglang_backend, "SGLangBackend", FakeBackend)
    monkeypatch.setattr(trace, "TraceRegistry", FakeTraceRegistry)
    monkeypatch.setattr(native_generate, "_RUNTIME", None)

    args = SimpleNamespace(rollout_num_gpus=4, rollout_num_gpus_per_engine=4,
                           sglang_server_concurrency=1, hf_checkpoint="/model",
                           sglang_router_ip="127.0.0.1", sglang_router_port=30000,
                           rollout_max_context_len=1024, rollout_max_response_len=128)
    runtime = asyncio.run(native_generate.runtime(args))
    try:
        registry = calls["gateway"][0]
        backend = calls["gateway"][1]
        assert calls["gateway"] == (registry, backend, "policy", 128)
        assert calls["server"][0] is calls["gateway_object"]
        assert calls["server"][1:] == ("127.0.0.1", 49183)
        assert runtime.backend.weight_version() == "7"
    finally:
        runtime.close()
        native_generate._RUNTIME = None

def test_native_policy_identity_uses_synchronized_pool_version(tmp_path):
    from harborrl.rollout import native_generate

    runtime = SimpleNamespace(run_root=tmp_path)
    (tmp_path / "policy-pool.json").write_text(json.dumps({"weight_version": "1"}))
    assert native_generate.current_policy_version(runtime) == "1"
    (tmp_path / "policy-pool.json").write_text(json.dumps({"weight_version": 1}))
    with pytest.raises(ValueError, match="versioned policy pool"):
        native_generate.current_policy_version(runtime)


def test_native_slots_spread_workers_and_shift_retries(tmp_path, monkeypatch):
    from harborrl.rollout import native_generate

    entry = {"id": "task", "revision": "1", "path": "/task", "task_digest": "d" * 64,
             "reward_profile": {"key": "reward", "scale": 1.0, "offset": 0.0, "raw_range": [0.0, 1.0]}}
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"model": "policy"}))
    monkeypatch.setenv("RUN_ID", "run")
    monkeypatch.setenv("HARBORRL_NATIVE_PROFILE", str(profile))
    monkeypatch.setenv("HARBORRL_NATIVE_RUNNER_WORKERS", json.dumps(["worker-a", "worker-b"]))
    monkeypatch.setenv("HARBORRL_NATIVE_MAX_ATTEMPTS", "2")
    monkeypatch.setenv("HARBORRL_NATIVE_GATEWAY_URL", "http://gateway:32123")
    monkeypatch.setenv("HARBORRL_NATIVE_RUNNER_PYTHON", "/python")

    class FakeSlots:
        max_attempts = 2

        def __init__(self):
            self.started = {}

        def start(self, slot):
            self.started[slot] = self.started.get(slot, 0) + 1
            return SimpleNamespace(attempt_id=f"{slot}-{self.started[slot]}", group_id="group-0",
                                   slot_id=slot, attempt_no=self.started[slot])

        def finish(self, slot, ir=None, error=None):
            assert (ir is None) != (error is None)

    def fake_group_slots(identities, *, max_attempts=2):
        return FakeSlots()

    calls = []

    async def fake_run_attempt(identity, *, runner_host, **kwargs):
        calls.append((identity.slot_id, runner_host))
        if identity.slot_id == "1" and identity.attempt_no == 1:
            raise RuntimeError("simulated first-attempt failure")
        return {"identity": identity.__dict__}

    monkeypatch.setattr(native_generate, "GroupSlots", fake_group_slots)
    monkeypatch.setattr(native_generate, "run_attempt", fake_run_attempt)

    args = SimpleNamespace(n_samples_per_prompt=4, rollout_max_response_len=64)
    runtime_ = SimpleNamespace(rollout_root=tmp_path, semaphore=asyncio.Semaphore(8),
                               registry=SimpleNamespace(), profile={"model": "policy"})
    sample = SimpleNamespace(metadata={"rollout_id": 0, "native_task": entry}, group_index=0)

    async def gather_slots():
        return await asyncio.gather(*[
            native_generate._run_slot(args, runtime_, sample, slot, entry, "sampling", "1")
            for slot in range(4)])

    results = asyncio.run(gather_slots())
    assert len(results) == 4
    by_slot = {}
    for slot, host in calls:
        by_slot.setdefault(slot, []).append(host)
    assert by_slot == {"0": ["worker-a"], "1": ["worker-b", "worker-a"],
                       "2": ["worker-a"], "3": ["worker-b"]}


def test_native_dispatch_passes_materialized_prompt_data(tmp_path, monkeypatch):
    path, task = native_config(tmp_path)
    from harborrl.data.harbor.native_inspector import inspect_native
    digest = inspect_native(task)["task_digest"]
    catalog_path = json.loads(path.read_text().split("tasks: ", 1)[1].split("\n", 1)[0])["catalog"]
    catalog = json.loads(Path(catalog_path).read_text())
    catalog[0]["task_digest"] = digest
    Path(catalog_path).write_text(json.dumps(catalog))

    from harborrl.platform import native_train
    plan = launch_plan(load_config(path))
    output = tmp_path / "prompt-data-env"
    path_output = tmp_path / "path-env"
    workers_output = tmp_path / "workers-env"
    gateway_output = tmp_path / "gateway-env"
    backend_output = tmp_path / "backend-env"
    plan["training_command"] = [
        "bash", "-c",
        f"printf %s \"$ROLLOUT_PROMPT_DATA\" > {output}; "
        f"printf %s \"$PATH\" > {path_output}; "
        f"printf %s \"$HARBORRL_NATIVE_RUNNER_WORKERS\" > {workers_output}; "
        f"printf %s \"$HARBORRL_NATIVE_GATEWAY_HOST\" > {gateway_output}; "
        f"printf %s \"$SLIME_DIR:$MEGATRON_DIR:$SGLANG_IMAGE\" > {backend_output}",
    ]
    monkeypatch.setenv("SLIME_DIR", "/opt/slime-v0.3.2")
    monkeypatch.setenv("MEGATRON_DIR", "/opt/megatron")
    monkeypatch.setenv("SGLANG_IMAGE", "sglang:v0.5.15.post1-cu129")
    monkeypatch.setattr(native_train, "doctor", lambda *_args, **_kwargs: [])
    assert native_train.dispatch(plan) == 0
    launch_path = next((tmp_path / "runs" / "training").glob("*/launch.json"))
    launch = json.loads(launch_path.read_text())
    assert launch["prompt_data"] == output.read_text()
    assert path_output.read_text().split(os.pathsep, 1)[0] == str(Path(plan["command"][0]).parent)
    assert json.loads(workers_output.read_text()) == ["worker-a", "worker-b"]
    assert gateway_output.read_text() == "gateway"
    assert backend_output.read_text() == (
        "/opt/slime-v0.3.2:/opt/megatron:sglang:v0.5.15.post1-cu129")
