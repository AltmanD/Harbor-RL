from __future__ import annotations

import ast
import importlib.util
import json
import os
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
AGENTIC_RL = ROOT / "harborrl"
CONVERTER_PATH = (
    AGENTIC_RL / "data" / "convert_sweverified.py"
)
REPORT_PATH = AGENTIC_RL / "evaluation" / "swebench" / "report.py"
SMOKE_PATH = ROOT / "tools" / "dev" / "smoke_swe_worker.py"
SWE_UTILS_PATH = AGENTIC_RL / "environments" / "terminal" / "swe_utils.py"
WORKER = ROOT / "deploy" / "workers" / "run_pool_server_sweverified.sh"
OFFICIAL_HARNESS = (
    ROOT
    / "tools"
    / "evaluation"
    / "benchmarks"
    / "swebench"
    / "run_swebench_verified_official_harness.sh"
)
TERMINAL_ENV = AGENTIC_RL / "environments" / "terminal" / "runtime.py"
WATCHDOG = ROOT / "deploy" / "workers" / "docker_watchdog.sh"
POOL_LAUNCHER = ROOT / "deploy" / "workers" / "run_pool_server.sh"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


converter = _load_module("convert_sweverified", CONVERTER_PATH)
report = _load_module("swebench_report", REPORT_PATH)
swe_utils = _load_module("swe_task_utils", SWE_UTILS_PATH)


def _official_row(instance_id: str = "django__django-11099") -> dict:
    return {
        "repo": "django/django",
        "instance_id": instance_id,
        "base_commit": "a" * 40,
        "patch": "diff --git a/a b/a\n",
        "test_patch": "diff --git a/t b/t\n",
        "problem_statement": "Fix the reported regression.",
        "hints_text": "",
        "created_at": "2024-01-01",
        "version": "3.0",
        "FAIL_TO_PASS": json.dumps(["tests.test_case"]),
        "PASS_TO_PASS": json.dumps(["tests.test_existing"]),
        "environment_setup_commit": "b" * 40,
    }


def test_converter_emits_official_provenance_and_prediction_only_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    item = converter.convert_sample(_official_row())
    meta = item["metadata"]
    assert meta["source_dataset"] == "princeton-nlp/SWE-bench_Verified"
    assert meta["source_revision"] == converter.DATASET_REVISION
    assert meta["swebench_harness_commit"] == converter.SWEBENCH_COMMIT
    assert meta["task_format_version"] == converter.TASK_FORMAT_VERSION
    assert meta["task_path"] == "sweverified_env/django__django-11099"
    assert "/testbed" in item["task"][0]["content"]
    assert "FAIL_TO_PASS" not in item["task"][0]["content"]
    assert "PASS_TO_PASS" not in item["task"][0]["content"]

    assert converter.create_task_dir(item, dataset_root=tmp_path, overwrite=False)
    task_dir = tmp_path / meta["task_path"]
    compose = yaml.safe_load(
        (task_dir / "docker-compose.yaml").read_text(encoding="utf-8")
    )
    service = compose["services"]["client"]
    assert service["labels"] == [
        "harborrl.pool-namespace=${TERMINAL_RL_POOL_NAMESPACE:-default}"
    ]
    assert compose["networks"]["default"]["labels"] == [
        "harborrl.pool-namespace=${TERMINAL_RL_POOL_NAMESPACE:-default}"
    ]
    assert converter.official_image_name(meta["swe_instance_id"]) in (
        task_dir / "Dockerfile"
    ).read_text(encoding="utf-8")
    run_tests = (task_dir / "run-tests.sh").read_text(encoding="utf-8")
    assert "official harness" in run_tests
    assert "exit 2" in run_tests
    assert converter.validate_task_dir_fingerprint(item, task_dir)

    tree = ast.parse(TERMINAL_ENV.read_text(encoding="utf-8"))
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_current_pool_namespace", "_compose_declares_pool_namespace"}
    ]
    namespace = {
        "os": os,
        "Path": Path,
        "_POOL_NAMESPACE_RE": re.compile(r"^[a-z0-9][a-z0-9_-]{0,62}$"),
    }
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), str(TERMINAL_ENV), "exec"),
        namespace,
    )
    monkeypatch.setenv("TERMINAL_RL_POOL_NAMESPACE", "sweverified")
    assert namespace["_compose_declares_pool_namespace"](
        task_dir / "docker-compose.yaml"
    )
    (task_dir / "Dockerfile").write_text("FROM untrusted\n", encoding="utf-8")
    assert not converter.validate_task_dir_fingerprint(item, task_dir)
    assert converter.create_task_dir(item, dataset_root=tmp_path, overwrite=False)
    assert converter.validate_task_dir_fingerprint(item, task_dir)


def test_formal_conversion_rejects_local_or_truncated_sources(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(
        json.dumps(_official_row()) + "\n",
        encoding="utf-8",
    )
    args = SimpleNamespace(
        input_jsonl=str(input_path),
        output_dir=str(tmp_path / "out"),
        output_name="test.jsonl",
        stats_name="convert_stats.json",
        env_dir=str(tmp_path / "sweverified_env"),
        max_samples=None,
        create_env_dirs=True,
        overwrite_env_dirs=False,
        overwrite_output=False,
        formal=True,
        hf_cache_dir=str(tmp_path / "cache"),
        hf_endpoint="https://huggingface.co",
    )
    with pytest.raises(SystemExit, match="pinned Hugging Face"):
        converter.convert(args)

    args.formal = False
    args.max_samples = 1
    stats = converter.convert(args)
    assert stats["converted"] == 1
    assert stats["unique_instance_ids"] == 1


def test_prediction_artifacts_have_official_schema_and_complete_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "test.jsonl"
    ids = ["owner__repo-1", "owner__repo-2"]
    dataset.write_text(
        "".join(
            json.dumps(
                {
                    "metadata": {
                        "swe_instance_id": instance_id,
                        "task_name": instance_id,
                    }
                }
            )
            + "\n"
            for instance_id in ids
        ),
        encoding="utf-8",
    )
    results = tmp_path / "official"
    monkeypatch.setenv("SWEBENCH_RESULTS_DIR", str(results))
    monkeypatch.setenv("SWEBENCH_EVAL_DATA_PATH", str(dataset))
    monkeypatch.setenv("SWEBENCH_MODEL_NAME_OR_PATH", "Qwen/Qwen3-8B")
    monkeypatch.setenv("RUN_ID", "unit")

    samples = []
    for instance_id in ids:
        samples.append(
            SimpleNamespace(
                status="completed",
                remove_sample=False,
                prompt={},
                metadata={
                    "task_meta": {"swe_instance_id": instance_id},
                    "reward_details": {
                        "instance_id": instance_id,
                        "grader": "swebench_prediction_export",
                        "grading_deferred": True,
                        "model_patch": (
                            f"diff --git a/{instance_id} b/{instance_id}\n"
                        ),
                    },
                },
            )
        )

    summary = report.write_official_artifacts(samples)
    assert summary is not None
    assert summary["submitted"] == 2
    assert summary["incomplete"] == 0
    assert summary["technical_failures"] == 0
    assert summary["pending_official_grading"] == 2
    predictions = [
        json.loads(line)
        for line in (results / "predictions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["instance_id"] for row in predictions] == ids
    assert all(
        set(row) == {"instance_id", "model_name_or_path", "model_patch"}
        for row in predictions
    )
    with pytest.raises(RuntimeError, match="Duplicate SWE-bench prediction"):
        report.write_official_artifacts([samples[0], samples[0]])


def test_worker_and_harness_pin_official_inputs() -> None:
    worker = WORKER.read_text(encoding="utf-8")
    harness = OFFICIAL_HARNESS.read_text(encoding="utf-8")

    assert "OFFICIAL_INSTANCE_COUNT" in worker
    assert "for line_no, instance_id, task_path, row in task_paths" in worker
    assert "validate_task_dir_fingerprint(row, task_dir)" in worker
    assert "WORKER_PREFLIGHT_ONLY" in worker
    assert "SWEVERIFIED_REQUIRE_PINNED_WORKER_DEPS" in worker
    assert "requirements-swesmith-worker.txt" in worker
    assert ".venv-swesmith-worker/bin/python" in worker
    assert 'import_module("harborrl.platform.worker_app")' in worker
    assert worker.count('import_module("harborrl.platform.worker_app")') == 2
    assert "pool_server Python dependency preflight failed" in worker
    assert converter.SWEBENCH_COMMIT in harness
    assert converter.DATASET_NAME in harness
    assert "swebench.harness.run_evaluation" in harness
    assert "len(rows) != 500" in harness
    assert "HARNESS_PREFLIGHT_ONLY" in harness
    assert "pip install --editable" in harness
    assert "from swebench.harness.run_evaluation import main" in harness


def test_worker_rejects_explicit_python_that_cannot_import_pool_server() -> None:
    env = os.environ.copy()
    env["POOL_SERVER_PYTHON"] = "/bin/false"
    proc = subprocess.run(
        ["bash", str(WORKER)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc.returncode == 2
    assert "pool_server Python dependency preflight failed" in proc.stderr


def test_network_pruning_and_compose_lifecycle_share_host_lock() -> None:
    terminal_env = TERMINAL_ENV.read_text(encoding="utf-8")
    watchdog = WATCHDOG.read_text(encoding="utf-8")
    pool = POOL_LAUNCHER.read_text(encoding="utf-8")
    lock_name = "lightrl_docker_network_lifecycle.lock"

    assert lock_name in terminal_env
    assert lock_name in watchdog
    assert lock_name in pool
    assert "with _docker_network_lifecycle_lock():" in terminal_env
    assert "docker_network_prune_safe" in watchdog
    assert "docker_network_rm_safe" in pool


def test_swe_workspace_prompt_is_shared_by_smith_and_verified() -> None:
    for task_path in (
        "swesmith_env/owner__repo.instance",
        "sweverified_env/owner__repo-1",
    ):
        message = swe_utils.build_swe_user_message(
            task_name="owner__repo",
            task_path=task_path,
            instruction="Fix the issue.",
        )
        assert "already checked out at /testbed" in message
        assert "Do not clone" in message
        assert message.endswith("Task instruction: Fix the issue.")
