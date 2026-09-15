from __future__ import annotations

import ast
import os
import re
import time
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
AGENTIC_RL = ROOT / "harborrl"
TERMINAL_ENV = AGENTIC_RL / "environments" / "terminal" / "runtime.py"
POOL_SERVER = AGENTIC_RL / "platform" / "worker_pool.py"
COMPOSE_UTILS = AGENTIC_RL / "environments" / "terminal" / "docker_compose.py"
SETA_LAUNCHER = ROOT / "deploy" / "workers" / "run_pool_server.sh"
SWE_LAUNCHER = ROOT / "deploy" / "workers" / "run_pool_server_swesmith.sh"
TRAIN_LAUNCHER = AGENTIC_RL / "platform" / "slime_train.sh"
DOWNLOADER = AGENTIC_RL / "data" / "download_swesmith.sh"
SMOKE_CLIENT = ROOT / "tools" / "dev" / "smoke_swesmith_worker.py"
WORKER_REQUIREMENTS = ROOT / "deploy" / "runtime" / "requirements-swesmith-worker.txt"


COMPOSE = """services:
  client:
    build:
      context: .
      dockerfile: Dockerfile
    image: ${T_BENCH_TASK_DOCKER_CLIENT_IMAGE_NAME}
    container_name: ${T_BENCH_TASK_DOCKER_CLIENT_CONTAINER_NAME}
    command: [ "sh", "-c", "sleep infinity" ]
    environment:
      - TEST_DIR=${T_BENCH_TEST_DIR}
      - SWESMITH_RUN_PASS_TO_PASS
    labels:
      - harborrl.pool-namespace=${TERMINAL_RL_POOL_NAMESPACE:-default}
    volumes:
      - ${T_BENCH_TASK_LOGS_PATH}:${T_BENCH_CONTAINER_LOGS_PATH}
      - ${T_BENCH_TASK_AGENT_LOGS_PATH}:${T_BENCH_CONTAINER_AGENT_LOGS_PATH}
networks:
  default:
    labels:
      - harborrl.pool-namespace=${TERMINAL_RL_POOL_NAMESPACE:-default}
"""


def _compose_validator():
    tree = ast.parse(TERMINAL_ENV.read_text(encoding="utf-8"))
    wanted = {"_current_pool_namespace", "_compose_declares_pool_namespace"}
    nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in wanted
    ]
    namespace = {
        "os": os,
        "Path": Path,
        "_POOL_NAMESPACE_RE": re.compile(r"^[a-z0-9][a-z0-9_-]{0,62}$"),
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(TERMINAL_ENV), "exec"), namespace)
    return namespace["_compose_declares_pool_namespace"]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda model: model["services"].update({"helper": dict(model["services"]["client"])}),
        lambda model: model["services"]["client"].update({"image": "foreign:latest"}),
        lambda model: model["services"]["client"].update({"container_name": "fixed"}),
        lambda model: model["services"]["client"]["build"].update({"context": ".."}),
        lambda model: model["services"]["client"]["build"].update({"dockerfile": "Otherfile"}),
        lambda model: model["services"]["client"]["build"].update({"network": "host"}),
        lambda model: model.update({"name": "global-project"}),
        lambda model: model["networks"]["default"].update({"name": "global-network"}),
        lambda model: model["networks"]["default"].update({"driver_opts": {"x": "y"}}),
        lambda model: model["services"]["client"].update({"network_mode": "host"}),
        lambda model: model["services"]["client"]["volumes"].append("/host:/foreign"),
    ],
)
def test_non_default_compose_accepts_only_generated_swesmith_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutate
) -> None:
    monkeypatch.setenv("TERMINAL_RL_POOL_NAMESPACE", "swesmith")
    validator = _compose_validator()
    compose_path = tmp_path / "docker-compose.yaml"
    compose_path.write_text(COMPOSE, encoding="utf-8")
    assert validator(compose_path)

    model = yaml.safe_load(COMPOSE)
    mutate(model)
    compose_path.write_text(yaml.safe_dump(model, sort_keys=False), encoding="utf-8")
    assert not validator(compose_path)


def test_non_default_compose_validation_precedes_image_prepare() -> None:
    tree = ast.parse(TERMINAL_ENV.read_text(encoding="utf-8"))
    reset = next(
        node
        for node in tree.body[-1].body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "_reset_locked"
    )
    references = {
        name: min(
            node.lineno
            for node in ast.walk(reset)
            if isinstance(node, ast.Name) and node.id == name
        )
        for name in {
            "validate_task_dir_fingerprint",
            "_compose_declares_pool_namespace",
            "prepare_task_docker_image",
        }
    }
    assert references["validate_task_dir_fingerprint"] < references["prepare_task_docker_image"]
    assert references["_compose_declares_pool_namespace"] < references["prepare_task_docker_image"]


def test_reset_lifecycle_has_no_detached_thread_or_pool_cleanup() -> None:
    terminal_source = TERMINAL_ENV.read_text(encoding="utf-8")
    pool_source = POOL_SERVER.read_text(encoding="utf-8")

    assert "self._lifecycle_lock = asyncio.Lock()" in terminal_source
    assert terminal_source.count("await _join_async_task(") >= 3
    assert "Thread may still be running" not in terminal_source
    assert pool_source.count("await self._cancel_and_join_reset_task(") >= 5
    assert "await self._join_task_uncancellable(reset_task)" in pool_source
    assert "reset_quarantined" in pool_source
    assert "No lease removal or Docker cleanup will occur until reset exits" in pool_source
    assert "WORKER_RESET_CANCEL_JOIN_TIMEOUT" in pool_source
    assert "WORKER_SHUTDOWN_RESET_JOIN_TIMEOUT" in pool_source
    assert "deadline=reset_join_deadline" in pool_source
    assert "if not joined:" in pool_source
    assert "cleanup_timeout=timeout" in pool_source
    assert "removed = await asyncio.shield(sweep_task)" in pool_source
    assert "await self._join_task_uncancellable(sweep_task)" in pool_source


def test_docker_build_and_start_paths_are_bounded() -> None:
    terminal_source = TERMINAL_ENV.read_text(encoding="utf-8")
    compose_source = COMPOSE_UTILS.read_text(encoding="utf-8")

    assert "self._terminal.start(" not in terminal_source
    assert "compose_manager.build(" not in compose_source
    assert "subprocess.run(" in compose_source
    assert "timeout=timeout" in compose_source


def test_terminal_honors_task_recording_configuration() -> None:
    tree = ast.parse(TERMINAL_ENV.read_text(encoding="utf-8"))
    terminal_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "Terminal"
    ]
    assert len(terminal_calls) == 1
    disable_recording = next(
        (
            keyword.value
            for keyword in terminal_calls[0].keywords
            if keyword.arg == "disable_recording"
        ),
        None,
    )
    assert disable_recording is not None
    assert ast.unparse(disable_recording) == "task_config.disable_asciinema"


def test_docker_build_passes_configured_proxy_as_value_free_build_args() -> None:
    tree = ast.parse(COMPOSE_UTILS.read_text(encoding="utf-8"))
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_docker_build_args_with_proxy"
    )
    namespace: dict[str, object] = {}
    exec(
        compile(ast.Module(body=[helper], type_ignores=[]), str(COMPOSE_UTILS), "exec"),
        namespace,
    )
    build_args = namespace["_docker_build_args_with_proxy"](
        {"HTTP_PROXY": "sensitive", "NO_PROXY": "localhost"}
    )
    assert build_args == [
        "build",
        "--build-arg",
        "HTTP_PROXY",
        "--build-arg",
        "NO_PROXY",
    ]
    assert "sensitive" not in build_args


def test_docker_build_failure_cache_separates_transient_registry_errors() -> None:
    source = COMPOSE_UTILS.read_text(encoding="utf-8")
    tree = ast.parse(source)
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_is_transient_build_failure"
    )
    namespace: dict[str, object] = {"re": re}
    exec(
        compile(ast.Module(body=[helper], type_ignores=[]), str(COMPOSE_UTILS), "exec"),
        namespace,
    )
    is_transient = namespace["_is_transient_build_failure"]
    assert is_transient("429 Too Many Requests from registry-1.docker.io")
    assert is_transient("HEAD manifest returned 503 Service Unavailable")
    assert is_transient("net/http: TLS handshake timeout")
    assert not is_transient("Dockerfile parse error line 7: unknown instruction")

    assert "_BUILD_TRANSIENT_FAILED" in source
    assert "DockerImageTransientBuildError" in source
    assert '"cached_transient_failed"' in source


def test_seta_worker_pid_limit_has_uv_thread_pool_headroom() -> None:
    launcher = SETA_LAUNCHER.read_text(encoding="utf-8")
    compose_source = COMPOSE_UTILS.read_text(encoding="utf-8")

    assert 'CONTAINER_PIDS_LIMIT="${CONTAINER_PIDS_LIMIT:-512}"' in launcher
    assert 'visible_cpu_count="$(nproc 2>/dev/null || echo 1)"' in launcher
    assert "CONTAINER_PIDS_LIMIT <= visible_cpu_count" in launcher
    assert 'os.getenv("CONTAINER_PIDS_LIMIT", "512")' in compose_source


def test_swesmith_artifact_publication_and_consumers_share_a_lock() -> None:
    downloader = DOWNLOADER.read_text(encoding="utf-8")
    smoke_client = SMOKE_CLIENT.read_text(encoding="utf-8")
    worker = SWE_LAUNCHER.read_text(encoding="utf-8")
    trainer = "\n".join(
        part.read_text(encoding="utf-8")
        for part in [TRAIN_LAUNCHER]
        + sorted((AGENTIC_RL / "platform" / "slime_train").glob("lib_*.sh"))
    )

    assert "flock -n 9" in downloader
    assert "PUBLISH_STARTED=1" in downloader
    assert "PUBLISH_COMPLETE=1" in downloader
    assert "TMP_ENV_DIR" in downloader
    assert 'mv "${TMP_ENV_DIR}" "${ENV_PATH}"' in downloader
    assert "dataset/swesmith_smoke/swesmith_convert" in downloader
    assert '"swesmith_smoke"' in smoke_client
    assert '"swesmith_convert"' in smoke_client
    assert '"smoke.jsonl"' in smoke_client
    assert "flock -s -n 8" in worker
    assert "flock -s -n 8" in trainer
    assert "SWESMITH_AUTO_CREATE_ENV" not in trainer
    assert "training preflight is read-only" in trainer
    assert 'chmod a-w "${SWESMITH_FROZEN_PROMPT_DATA}"' in trainer


def test_force_cleanup_uses_one_absolute_deadline() -> None:
    source = TERMINAL_ENV.read_text(encoding="utf-8")
    tree = ast.parse(source)
    bounded_functions = {
        "_docker_object_pool_namespace_state",
        "_compose_project_pool_namespace_state",
        "_container_compose_project_state",
        "_docker_compose_down_projects",
        "_remove_fixed_task_services_without_running_clients",
        "_remove_inactive_compose_resources",
        "_force_remove_docker_objects_impl",
        "force_remove_orphan_docker_objects",
    }
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in bounded_functions
    }
    assert set(functions) == bounded_functions
    for name, function in functions.items():
        run_calls = [
            call
            for call in ast.walk(function)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "subprocess"
            and call.func.attr == "run"
        ]
        assert run_calls, name
        for call in run_calls:
            timeout = next(
                (keyword.value for keyword in call.keywords if keyword.arg == "timeout"),
                None,
            )
            assert timeout is not None, name
            assert "_docker_cleanup_command_timeout" in ast.unparse(timeout), name

    helper_nodes = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.ClassDef)
            and node.name == "_DockerCleanupDeadlineExceeded"
        )
        or (
            isinstance(node, ast.FunctionDef)
            and node.name == "_docker_cleanup_command_timeout"
        )
    ]
    namespace = {"time": time}
    exec(
        compile(ast.Module(body=helper_nodes, type_ignores=[]), str(TERMINAL_ENV), "exec"),
        namespace,
    )
    remaining_timeout = namespace["_docker_cleanup_command_timeout"]
    deadline_error = namespace["_DockerCleanupDeadlineExceeded"]
    deadline = time.monotonic() + 0.02
    command_timeout = remaining_timeout(10.0, deadline)
    assert 0 < command_timeout <= 0.02
    time.sleep(0.03)
    with pytest.raises(deadline_error):
        remaining_timeout(10.0, deadline)

    compact = " ".join(source.split())
    assert "cleanup_deadline = time.monotonic() + cleanup_timeout" in compact
    assert "cleanup_deadline=cleanup_deadline" in compact


def test_worker_dependencies_are_pinned() -> None:
    requirements = WORKER_REQUIREMENTS.read_text(encoding="utf-8")
    launcher = SWE_LAUNCHER.read_text(encoding="utf-8")

    assert "camel-ai==0.2.90" in requirements
    assert "terminal-bench @ git+https://github.com/laude-institute/terminal-bench.git@" in requirements
    assert "d28711d0da2675d0bb1d56de45ae5df6082438a3" in requirements
    assert 'SWESMITH_REQUIRE_PINNED_WORKER_DEPS:-1' in launcher


def test_launchers_export_namespace_and_seta_cleanup_filters_ownership() -> None:
    seta = SETA_LAUNCHER.read_text(encoding="utf-8")
    swe = SWE_LAUNCHER.read_text(encoding="utf-8")

    assert 'export TERMINAL_RL_POOL_NAMESPACE="${TERMINAL_RL_POOL_NAMESPACE:-default}"' in seta
    assert 'export TERMINAL_RL_POOL_NAMESPACE="${TERMINAL_RL_POOL_NAMESPACE:-swesmith}"' in swe
    assert 'export WORKER_SHIM_CLEANUP_ENABLED="${WORKER_SHIM_CLEANUP_ENABLED:-0}"' in seta
    assert "harborrl.pool-namespace" in seta
    assert "{{.Names}}" in seta
    assert "{{.Image}}" in seta
    assert "com.docker.compose.project" in seta
    assert 'WORKER_CLEANUP_LEGACY_UNLABELED=1' in seta
    assert 'WORKER_CLEANUP_LEGACY_UNLABELED=0' in seta
    assert 'export FINAL_DOCKER_CLEANUP="${FINAL_DOCKER_CLEANUP:-1}"' in swe
    assert (
        'export POOL_SERVER_CHILD_EXIT_CLEANUP="${POOL_SERVER_CHILD_EXIT_CLEANUP:-1}"'
        in swe
    )
    assert 'legacy == "1" && $4 == ""' in seta
    assert 'legacy == "1" && $2 == ""' in seta
    assert '(ns != "default" && $4 == ns)' in seta
    assert 'ns != "default" && $2 == ns' in seta
    assert "docker container prune" not in seta
    assert "docker network prune" not in seta


def test_seta_worker_launcher_uses_repository_data_roots() -> None:
    launcher = SETA_LAUNCHER.read_text(encoding="utf-8")

    assert "${REPO_ROOT}/benchmarks" in launcher
    assert "${OPENCLAW_REMOTE_LOG_DIR}/task_outputs" in launcher
    assert "${TERMINAL_RL}/dataset" not in launcher
