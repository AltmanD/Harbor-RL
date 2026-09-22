"""Schema-2 native training dispatch and preflight.

The dispatcher freezes a run directory and materializes the original-task lock
into Slime prompt rows before handing lifecycle control to the existing Slime
launcher. It never rewrites task tests or rewards.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
from uuid import uuid4

from harborrl.config import ROOT
from harborrl.trajectories.native import publish, read_json


def _run_root(plan):
    name = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8]
    return Path(plan["config"]["output"]["root"]) / "training" / name


def doctor(plan):
    config = plan["config"]
    checks = []

    runner = Path(config["harbor"]["python"])
    checks.append({"path": "harbor.python", "ok": runner.is_file()})
    from harborrl.rollout.harbor_job.coordinator import runner_command
    if runner.is_file():
        for worker in config["harbor"]["workers"]:
            try:
                probe = subprocess.run(
                    runner_command(str(runner), ROOT, worker) + ["--probe"],
                    capture_output=True, text=True, timeout=30, check=False,
                )
            except (subprocess.SubprocessError, OSError) as exc:
                probe = SimpleNamespace(returncode=1, stdout="", stderr=str(exc))
            try:
                payload = json.loads(probe.stdout)
                runtime_version = tuple(payload.get("python", "").split(".")[:2])
                version_ok = payload.get("harbor_version") == config["harbor"]["version"]
                runtime_ok = len(runtime_version) == 2 and all(part.isdigit() for part in runtime_version)
            except (ValueError, AttributeError, TypeError):
                runtime_version, version_ok, runtime_ok = (), False, False
            checks.append({
                "service": "external Harbor worker",
                "worker": worker,
                "ok": probe.returncode == 0 and runtime_ok
                      and tuple(map(int, runtime_version)) >= (3, 12) and version_ok,
                "detail": probe.stdout.strip() or probe.stderr.strip()[-1000:],
            })

    for key, path in (("tasks.catalog", config["tasks"]["catalog"]),
                      ("harness.profile", config["harness"]["profile"]),
                      ("model.checkpoint", config["model"]["checkpoint"]),
                      ("model.reference", config["model"]["reference"])):
        checks.append({"path": key, "ok": Path(path).exists()})

    model_config = Path(config["model"]["checkpoint"]) / "config.json"
    if model_config.is_file():
        try:
            model = json.loads(model_config.read_text())
            heads = model.get("num_attention_heads", 0)
            kv_heads = model.get("num_key_value_heads", heads)
            for tp in {config["deployment"]["actor_tensor_parallel_size"],
                       config["deployment"]["rollout_gpus_per_engine"]}:
                checks.append({"model_tp": tp, "ok": bool(
                    heads and kv_heads and heads % tp == 0 and (kv_heads % tp == 0 or tp % kv_heads == 0))})
        except (OSError, ValueError, TypeError) as exc:
            checks.append({"path": "model.config", "ok": False, "detail": str(exc)})

    from harborrl.config.native import catalog
    try:
        entries = catalog(config)
        for entry in entries:
            from harborrl.data.harbor.native_inspector import inspect_native
            report = inspect_native(entry["path"], expected_digest=entry["task_digest"])
            checks.append({"task": entry["id"], "ok": report["status"] == "NEEDS_PROBE",
                           "detail": report["reasons"]})
    except (ValueError, OSError, TypeError) as exc:
        checks.append({"service": "native catalog", "ok": False, "detail": str(exc)})

    for package in ("ray", "torch", "sglang", "transformers", "mbridge",
                    "megatron-bridge", "nvidia-modelopt"):
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = None
        checks.append({"dependency": package, "version": version, "ok": version is not None})
    checks.append({"dependency": "bundled Megatron-LM",
                   "ok": (ROOT / "backends/Megatron-LM/megatron/core/__init__.py").is_file()})

    runtime_env = os.environ.copy()
    runtime_pythonpath = os.pathsep.join([
        str(ROOT / "backends/Megatron-LM"), str(ROOT), str(ROOT / "backends/slime"),
        runtime_env.get("PYTHONPATH", ""),
    ])
    runtime_env["PYTHONPATH"] = runtime_pythonpath
    for module in ("megatron.bridge", "mbridge", "slime_plugins.megatron_bridge",
                   "modelopt.torch.distill.plugins.megatron"):
        probe = subprocess.run([sys.executable, "-c", f"import {module}"],
                               capture_output=True, text=True, timeout=30, check=False,
                               env=runtime_env)
        checks.append({"dependency": module, "ok": probe.returncode == 0,
                       "detail": probe.stderr.strip()[-1000:]})

    if shutil.which("nvidia-smi"):
        try:
            result = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                                    capture_output=True, text=True, timeout=10, check=True)
            visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            count = (len([item for item in visible.split(",") if item.strip() and item.strip() != "-1"])
                     if visible is not None else len(result.stdout.splitlines()))
            checks.append({"resource": "visible GPU budget", "available": count,
                           "ok": count >= config["deployment"]["num_gpus"]})
        except (subprocess.SubprocessError, OSError) as exc:
            checks.append({"resource": "visible GPU budget", "ok": False, "detail": str(exc)})
    else:
        checks.append({"resource": "visible GPU budget", "ok": False, "detail": "nvidia-smi unavailable"})

    checks.append({
        "service": "raw-model logprob audit",
        "ok": True,
        "detail": ("enabled by lock; live serving semantic probe remains a GPU gate"
                   if config["gateway"]["audited_raw_logprobs"] else
                   "disabled; generated evidence remains EVAL_ONLY and training will fail closed"),
    })
    return checks


def _materialize(plan, root):
    from harborrl.config.native import catalog

    config = plan["config"]
    entries = catalog(config)
    normalized = root / "config"
    normalized.mkdir(parents=True)
    catalog_path = normalized / "native_catalog.json"
    publish(catalog_path, entries)
    prompt_path = normalized / "native_tasks.jsonl"
    rows = []
    for entry in entries:
        instruction = (Path(entry["path"]) / "instruction.md").read_text()
        rows.append({"task": instruction, "metadata": {"harborrl_native": True, "native_task": entry}})
    prompt_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))
    return catalog_path, prompt_path


def _source_audit(root):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT)

    files = git("ls-files", "-z", "--cached", "--others", "--exclude-standard", "--",
                "harborrl", "backends", "pyproject.toml").decode().split("\0")
    untracked = set(git("ls-files", "-z", "--others", "--exclude-standard", "--",
                        "harborrl", "backends").decode().split("\0"))
    (root / "source.patch").write_bytes(git("diff", "--binary", "HEAD", "--",
                                             "harborrl", "backends", "pyproject.toml"))
    hashes = {}
    for name in files:
        source = ROOT / name
        if name and source.is_file():
            hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
            if name in untracked:
                target = root / "source-new" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read_bytes())
    payload = {
        "commit": git("rev-parse", "HEAD").decode().strip(),
        "dirty": bool(git("status", "--porcelain")),
        "source_hashes": hashes,
        "patch": "source.patch",
    }
    publish(root / "source-manifest.json", payload)
    return payload


def dispatch(plan, *, doctor_only=False):
    if doctor_only:
        checks = doctor(plan)
        print(json.dumps({"checks": checks,
                          "scope": "native config, original tasks, isolated Harbor, training dependencies and GPU budget; protocol/token semantics require the deferred live probe"},
                         indent=2))
        return int(any(not check["ok"] for check in checks))

    checks = doctor(plan)
    failed = [check for check in checks if not check["ok"]]
    if failed:
        print(json.dumps({"failed_checks": failed}, indent=2), file=sys.stderr)
        return 1

    root = _run_root(plan)
    root.mkdir(parents=True)
    catalog_path, prompt_path = _materialize(plan, root)
    source = _source_audit(root)
    environment = dict(plan["environment"])
    environment.update({
        "RUN_ID": root.name, "RUN_NAME": root.name, "RUN_DIR": str(root),
        "RUNS_ROOT": str(Path(plan["config"]["output"]["root"]).resolve()),
        "CKPT_ROOT": str(root / "checkpoints"), "WANDB_DIR": str(root / "metrics/wandb"),
        "HARBORRL_NATIVE_CATALOG": str(catalog_path),
        "ROLLOUT_PROMPT_DATA": str(prompt_path),
        "HARBORRL_NATIVE_PROFILE": plan["config"]["harness"]["profile"],
        "HARBOR_IR_ROOT": str(root / "ir"),
    })
    launch = {"plan": plan, "run_dir": str(root), "source": source,
              "materialized_catalog": str(catalog_path), "prompt_data": str(prompt_path)}
    publish(root / "launch.json", launch)

    allowed = {"PATH", "HOME", "USER", "LANG", "LC_ALL", "TMPDIR", "LD_LIBRARY_PATH",
               "CUDA_VISIBLE_DEVICES", "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY",
               "http_proxy", "https_proxy", "no_proxy"}
    child_env = {key: value for key, value in os.environ.items() if key in allowed}
    child_env["PATH"] = os.pathsep.join(
        [str(Path(sys.executable).parent), child_env.get("PATH", os.defpath)])
    child_env.update(environment)
    child = subprocess.Popen(plan["training_command"], cwd=ROOT, env=child_env, start_new_session=True)

    def forward_signal(signum, frame):
        if child.poll() is None:
            try:
                child.send_signal(signum)
            except ProcessLookupError:
                pass

    previous = {sig: signal.signal(sig, forward_signal) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        return child.wait()
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m harborrl.platform.native_train PLAN_JSON")
    raise SystemExit(dispatch(read_json(Path(sys.argv[1]))))
