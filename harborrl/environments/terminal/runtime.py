from __future__ import annotations

import asyncio
import fcntl
import importlib
import json
import logging
import os
import re
import shlex
import subprocess
import inspect
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from camel.toolkits import FunctionTool, TerminalToolkit
    from terminal_bench.handlers.trial_handler import TrialHandler
    from terminal_bench.parsers.base_parser import UnitTestStatus
    from terminal_bench.parsers.parser_factory import ParserFactory
    from terminal_bench.terminal.docker_compose_manager import DockerComposeManager
    from terminal_bench.terminal.terminal import Terminal

# Heavy third-party packages (camel.toolkits pulls in openai/litellm; importing
# any terminal_bench submodule runs an __init__ that pulls in the harness stack:
# litellm/openai/mcp/boto3/streamlit/sqlalchemy — tens of seconds on shared
# storage). They are only needed when a terminal task actually resets/evaluates,
# so resolve them lazily to keep pool-server startup and preflight import checks
# fast.
_LAZY_IMPORT_MODULES = {
    "FunctionTool": "camel.toolkits",
    "TerminalToolkit": "camel.toolkits",
    "TrialHandler": "terminal_bench.handlers.trial_handler",
    "UnitTestStatus": "terminal_bench.parsers.base_parser",
    "ParserFactory": "terminal_bench.parsers.parser_factory",
    "DockerComposeManager": "terminal_bench.terminal.docker_compose_manager",
    "Terminal": "terminal_bench.terminal.terminal",
}


def _lazy_import(name: str) -> Any:
    """Resolve a lazily imported name for in-function use.

    The module-level __getattr__ (PEP 562) only intercepts attribute access on
    the module object; a bare global reference inside a function raises
    NameError instead.  Call sites inside functions must resolve lazy names
    through this helper (assigning a same-named local keeps the diff minimal).
    """
    value = getattr(importlib.import_module(_LAZY_IMPORT_MODULES[name]), name)
    globals()[name] = value
    return value


def __getattr__(name: str) -> Any:
    if name not in _LAZY_IMPORT_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return _lazy_import(name)

from harborrl.types import RunContext, TaskSpec, TaskTimeouts

from harborrl.environments.agentharm.runtime import AgentHarmEnv
from harborrl.environments.agent_safetybench.runtime import AgentSafetyBenchEnv
from harborrl.environments.terminal.docker_compose import compose_up_no_build, prepare_task_docker_image

from harborrl.environments.terminal.swe_utils import build_swe_user_message, is_swe_task_path

from harborrl.environments.tau2.runtime import Tau2Env


logger = logging.getLogger("terminal.env.worker.terminal_env")
logger.setLevel(logging.INFO)

_TASK_CONTAINER_RE = re.compile(r"^[0-9]+-[A-Za-z0-9]{8}-slime-run$")
_TASK_ID_PREFIX_RE = re.compile(r"^([0-9]+)(?:[-_.:]|$)")
_FIXED_TASK_SERVICE_RE = re.compile(r"^tb__([0-9]+)__.*")
_TEST_EXIT_CODE_RE = re.compile(r"__TERMINAL_RL_TEST_EXIT_CODE__=(\d+)\b")
_GIT_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40,64}$")
_POOL_NAMESPACE_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,62}$")
_DOCKER_CLEANUP_EXECUTOR: ThreadPoolExecutor | None = None


@contextmanager
def _docker_network_lifecycle_lock():
    """Serialize Compose network start/stop with host watchdog pruning."""

    lock_path = Path(
        os.getenv(
            "DOCKER_NETWORK_LIFECYCLE_LOCK",
            "/tmp/lightrl_docker_network_lifecycle.lock",
        )
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = os.open(lock_path, os.O_RDONLY | os.O_CREAT, 0o666)
    with os.fdopen(lock_fd, "r", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


class _DockerCleanupDeadlineExceeded(TimeoutError):
    pass


from harborrl.env import env_float as _env_float, env_int as _env_int


# Legacy "anything not explicitly falsey is true" semantics (empty string is
# *true* here, unlike platform.env.env_bool).  Kept local on purpose: worker
# timeouts/limits rely on it; do not swap for env_bool without auditing every
# call site.
def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _docker_cleanup_command_timeout(
    command_timeout: float, deadline: float | None
) -> float:
    """Return one command's share of a single cleanup wall-clock budget."""
    if deadline is None:
        return command_timeout
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise _DockerCleanupDeadlineExceeded("Docker cleanup deadline exceeded")
    return min(command_timeout, remaining)


def _terminal_test_reward(
    parser_results: dict[str, UnitTestStatus], data_source: str
) -> tuple[float, int]:
    UnitTestStatus = _lazy_import("UnitTestStatus")  # noqa: F811 (lazy global)
    passed = sum(
        1 for status in parser_results.values() if status == UnitTestStatus.PASSED
    )
    if data_source == "swesmith":
        return float(passed == len(parser_results)), passed
    return float(passed / len(parser_results)), passed


def _docker_cleanup_executor() -> ThreadPoolExecutor:
    global _DOCKER_CLEANUP_EXECUTOR
    if _DOCKER_CLEANUP_EXECUTOR is None:
        workers = max(1, _env_int("TERMINAL_ENV_DOCKER_CLEANUP_WORKERS", 8))
        _DOCKER_CLEANUP_EXECUTOR = ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="lightrl-docker-cleanup",
        )
    return _DOCKER_CLEANUP_EXECUTOR


def _docker_name_variants(value: str | None) -> set[str]:
    if not value:
        return set()
    raw = value.strip()
    if not raw:
        return set()
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", raw).strip("-_.")
    variants = {
        raw,
        cleaned,
        cleaned.replace(".", "-"),
        cleaned.replace("_", "-"),
        cleaned.replace(".", "_"),
    }
    return {v for v in variants if v and "slime-run" in v}


def _matches_project_name(name: str, project_names: set[str], *, broad: bool) -> bool:
    if not name:
        return False
    for project in project_names:
        if name == project:
            return True
        if broad and (
            name.startswith(f"{project}-")
            or name.startswith(f"{project}_")
            or name.startswith(project)
        ):
            return True
    return False


def _docker_image_prefixes(*values: str | None) -> set[str]:
    prefixes: set[str] = set()
    for value in values:
        if not value:
            continue
        raw = value.strip()
        if raw:
            prefixes.add(raw)
        task_match = re.match(r"^([0-9]+)[-_.]", raw)
        if task_match:
            prefixes.add(f"tb__{task_match.group(1)}__")
    return {prefix for prefix in prefixes if prefix.startswith("tb__")}


def _task_id_from_ref(value: str | None) -> str | None:
    raw = (value or "").strip()
    if not raw:
        return None
    fixed = _FIXED_TASK_SERVICE_RE.match(raw)
    if fixed:
        return fixed.group(1)
    prefixed = _TASK_ID_PREFIX_RE.match(raw)
    if prefixed:
        return prefixed.group(1)
    return None


def _fixed_task_service_id(name: str, image: str = "") -> str | None:
    for value in (name, image):
        match = _FIXED_TASK_SERVICE_RE.match(value or "")
        if match:
            return match.group(1)
    return None


def _compose_project_candidates(
    trial_name: str | None, client_container_name: str | None
) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()
    for value in (client_container_name, trial_name):
        variants = sorted(_docker_name_variants(value))
        raw = (value or "").strip()
        if raw:
            variants.insert(0, raw)
        for variant in variants:
            if variant and variant not in seen:
                candidates.append(variant)
                seen.add(variant)
    return candidates[:6]


def _docker_status_age_seconds(status: str) -> float | None:
    text = (status or "").strip().lower()
    if not text:
        return None
    if "less than a second" in text:
        return 0.0
    if "about a minute" in text or "a minute" in text:
        return 60.0
    match = re.search(
        r"(\d+)\s+"
        r"(second|seconds|minute|minutes|hour|hours|day|days|week|weeks|month|months)",
        text,
    )
    if match is None:
        return None
    value = int(match.group(1))
    unit = match.group(2)
    if unit.startswith("second"):
        return float(value)
    if unit.startswith("minute"):
        return float(value * 60)
    if unit.startswith("hour"):
        return float(value * 3600)
    if unit.startswith("day"):
        return float(value * 86400)
    if unit.startswith("week"):
        return float(value * 7 * 86400)
    if unit.startswith("month"):
        return float(value * 30 * 86400)
    return None


def _is_task_container(name: str, image: str) -> bool:
    if _TASK_CONTAINER_RE.match(name or ""):
        return True
    return bool((name or "").endswith("-slime-run") and (image or "").startswith("tb__"))


def _clean_docker_label(value: str | None) -> str:
    raw = (value or "").strip()
    return "" if raw == "<no value>" else raw


def _current_pool_namespace() -> str:
    namespace = os.getenv("TERMINAL_RL_POOL_NAMESPACE", "default").strip() or "default"
    if not _POOL_NAMESPACE_RE.fullmatch(namespace):
        raise ValueError(
            "TERMINAL_RL_POOL_NAMESPACE must match "
            f"{_POOL_NAMESPACE_RE.pattern!r}; got {namespace!r}"
        )
    return namespace


def _pool_scoped_trial_name(task_name: str, uid: str) -> str:
    base = f"{task_name}.{uid}.slime-run"
    namespace = _current_pool_namespace()
    if namespace == "default":
        return base
    return f"{namespace}.{base}"


def _matches_pool_namespace(value: str | None) -> bool:
    """Keep orphan cleanup inside the pool that owns a Docker object."""
    current = _current_pool_namespace()
    observed = _clean_docker_label(value)
    if current == "default":
        return observed in {"", "default"}
    return observed == current


def _docker_object_pool_namespace_state(
    object_kind: str,
    object_ref: str,
    *,
    timeout: float,
    deadline: float | None = None,
) -> str:
    """Return absent, owned, foreign, or unknown for a Docker object."""
    if object_kind == "container":
        command = [
            "docker",
            "inspect",
            "--type",
            "container",
            "--format",
            '{{ index .Config.Labels "harborrl.pool-namespace" }}',
            object_ref,
        ]
    elif object_kind == "network":
        command = [
            "docker",
            "network",
            "inspect",
            "--format",
            '{{ index .Labels "harborrl.pool-namespace" }}',
            object_ref,
        ]
    elif object_kind == "volume":
        command = [
            "docker",
            "volume",
            "inspect",
            "--format",
            '{{ index .Labels "harborrl.pool-namespace" }}',
            object_ref,
        ]
    else:
        return "unknown"
    try:
        inspected = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=_docker_cleanup_command_timeout(timeout, deadline),
            check=False,
        )
    except _DockerCleanupDeadlineExceeded:
        raise
    except Exception:
        return "unknown"
    if inspected.returncode == 0:
        return "owned" if _matches_pool_namespace(inspected.stdout) else "foreign"
    missing = (inspected.stderr or "").lower()
    if any(
        marker in missing
        for marker in ("no such object", "no such container", "not found")
    ):
        return "absent"
    return "unknown"


def _docker_object_matches_pool_namespace(
    object_kind: str,
    object_ref: str,
    *,
    timeout: float,
    deadline: float | None = None,
) -> bool:
    """Preserve legacy default cleanup; prove non-default ownership."""
    if _current_pool_namespace() == "default":
        return True
    return (
        _docker_object_pool_namespace_state(
            object_kind, object_ref, timeout=timeout, deadline=deadline
        )
        == "owned"
    )


def _compose_project_pool_namespace_state(
    project: str, *, timeout: float, deadline: float | None = None
) -> str:
    """Return absent, owned, foreign, or unknown for a Compose project."""
    commands = (
        [
            "docker",
            "ps",
            "-a",
            "--filter",
            f"label=com.docker.compose.project={project}",
            "--format",
            '{{.Label "harborrl.pool-namespace"}}',
        ],
        [
            "docker",
            "network",
            "ls",
            "--filter",
            f"label=com.docker.compose.project={project}",
            "--format",
            '{{.Label "harborrl.pool-namespace"}}',
        ],
        [
            "docker",
            "volume",
            "ls",
            "--filter",
            f"label=com.docker.compose.project={project}",
            "--format",
            '{{.Label "harborrl.pool-namespace"}}',
        ],
    )
    observed: list[str] = []
    try:
        for command in commands:
            listed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=_docker_cleanup_command_timeout(timeout, deadline),
                check=False,
            )
            if listed.returncode != 0:
                return "unknown"
            observed.extend(listed.stdout.splitlines())
    except _DockerCleanupDeadlineExceeded:
        raise
    except Exception:
        return "unknown"
    if not observed:
        return "absent"
    return (
        "owned"
        if all(_matches_pool_namespace(value) for value in observed)
        else "foreign"
    )


def _compose_project_matches_pool_namespace(
    project: str, *, timeout: float, deadline: float | None = None
) -> bool:
    """Preserve legacy default cleanup; prove non-default project ownership."""
    if _current_pool_namespace() == "default":
        return True
    return (
        _compose_project_pool_namespace_state(
            project, timeout=timeout, deadline=deadline
        )
        == "owned"
    )


def _container_compose_project_state(
    container_ref: str, *, timeout: float, deadline: float | None = None
) -> tuple[str, str]:
    """Return container ownership state and its exact Compose project label."""
    try:
        inspected = subprocess.run(
            [
                "docker",
                "inspect",
                "--type",
                "container",
                "--format",
                '{{ index .Config.Labels "harborrl.pool-namespace" }}\t'
                '{{ index .Config.Labels "com.docker.compose.project" }}',
                container_ref,
            ],
            capture_output=True,
            text=True,
            timeout=_docker_cleanup_command_timeout(timeout, deadline),
            check=False,
        )
    except _DockerCleanupDeadlineExceeded:
        raise
    except Exception:
        return "unknown", ""
    if inspected.returncode != 0:
        missing = (inspected.stderr or "").lower()
        if any(
            marker in missing
            for marker in ("no such object", "no such container", "not found")
        ):
            return "absent", ""
        return "unknown", ""
    labels = inspected.stdout.rstrip("\r\n").split("\t", 1)
    namespace = _clean_docker_label(labels[0] if labels else "")
    project = _clean_docker_label(labels[1] if len(labels) > 1 else "")
    state = "owned" if _matches_pool_namespace(namespace) else "foreign"
    return state, project


def _remove_owned_container_for_reset(container_name: str, *, timeout: float) -> bool:
    """Remove a stale owned container by immutable ID and verify its name is free."""
    try:
        inspected = subprocess.run(
            [
                "docker",
                "inspect",
                "--type",
                "container",
                "--format",
                '{{ index .Config.Labels "harborrl.pool-namespace" }}\t{{.Id}}',
                container_name,
            ],
            timeout=timeout,
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as exc:
        raise RuntimeError(
            f"could not inspect stale container {container_name}: {exc}"
        ) from exc
    if inspected.returncode != 0:
        missing = (inspected.stderr or "").lower()
        if any(
            marker in missing
            for marker in ("no such object", "no such container", "not found")
        ):
            return False
        raise RuntimeError(
            f"could not inspect stale container {container_name}: "
            f"{(inspected.stderr or inspected.stdout).strip()[-1000:]}"
        )
    labels = inspected.stdout.rstrip("\r\n").split("\t", 1)
    state = "owned" if _matches_pool_namespace(labels[0] if labels else "") else "foreign"
    container_id = _clean_docker_label(labels[1] if len(labels) > 1 else "")
    if state != "owned" or not container_id:
        raise RuntimeError(
            f"refusing reset pre-cleanup for {container_name}: pool state={state}"
        )
    removed = subprocess.run(
        ["docker", "rm", "-f", container_id],
        timeout=timeout,
        capture_output=True,
        text=True,
        check=False,
    )
    if removed.returncode != 0:
        detail = (removed.stderr or removed.stdout or "no output").strip()[-1000:]
        raise RuntimeError(
            f"failed to remove owned stale container {container_name}: "
            f"rc={removed.returncode} detail={detail}"
        )
    post_state = _docker_object_pool_namespace_state(
        "container", container_name, timeout=timeout
    )
    if post_state != "absent":
        raise RuntimeError(
            f"container name {container_name} was recreated during reset cleanup: "
            f"state={post_state}"
        )
    return True


def _terminal_stop_ownership_verified(container_name: str, *, timeout: float) -> bool:
    """Verify both the exact container and its project before Terminal.stop."""
    state, project = _container_compose_project_state(container_name, timeout=timeout)
    if state == "absent":
        return False
    if state != "owned" or not project:
        raise RuntimeError(
            f"refusing Terminal.stop for {container_name}: "
            f"container state={state} compose_project={project!r}"
        )
    project_state = _compose_project_pool_namespace_state(project, timeout=timeout)
    if project_state != "owned":
        raise RuntimeError(
            f"refusing Terminal.stop for {container_name}: "
            f"compose project {project!r} state={project_state}"
        )
    return True


def _compose_declares_pool_namespace(compose_path: Path) -> bool:
    """Validate the static Compose model used by a non-default pool.

    The SWE-smith worker intentionally does not accept an override file: a
    separately merged model could introduce an unlabeled service, network, or
    volume after this check. Generated task files use bind mounts only and an
    explicitly labeled default network.
    """
    namespace = _current_pool_namespace()
    if namespace == "default":
        return True
    if os.getenv("COMPOSE_OVERRIDE_PATH", "").strip():
        return False
    try:
        import yaml

        document = yaml.safe_load(compose_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(document, dict):
        return False
    # Generated SWE-smith Compose files have an intentionally tiny, static
    # model.  Reject every top-level selector outside that model so a newer
    # Compose feature cannot silently escape namespace ownership checks.
    if set(document) != {"services", "networks"}:
        return False

    expected_values = {
        namespace,
        "${TERMINAL_RL_POOL_NAMESPACE}",
        "${TERMINAL_RL_POOL_NAMESPACE:-default}",
    }

    def _has_namespace_label(config: object) -> bool:
        if not isinstance(config, dict):
            return False
        labels = config.get("labels")
        if isinstance(labels, dict):
            if set(labels) != {"harborrl.pool-namespace"}:
                return False
            value = labels.get("harborrl.pool-namespace")
            return str(value) in expected_values
        if isinstance(labels, list):
            prefix = "harborrl.pool-namespace="
            return len(labels) == 1 and any(
                isinstance(label, str)
                and label.startswith(prefix)
                and label[len(prefix) :] in expected_values
                for label in labels
            )
        return False

    services = document.get("services")
    if not isinstance(services, dict) or set(services) != {"client"}:
        return False
    if not all(_has_namespace_label(config) for config in services.values()):
        return False
    client_config = services.get("client")
    if (
        not isinstance(client_config, dict)
        or set(client_config)
        != {
            "build",
            "image",
            "container_name",
            "command",
            "environment",
            "labels",
            "volumes",
        }
        or client_config.get("image")
        != "${T_BENCH_TASK_DOCKER_CLIENT_IMAGE_NAME}"
        or client_config.get("container_name")
        != "${T_BENCH_TASK_DOCKER_CLIENT_CONTAINER_NAME}"
    ):
        return False
    build = client_config.get("build")
    if not isinstance(build, dict) or build != {
        "context": ".",
        "dockerfile": "Dockerfile",
    }:
        # This also rejects build.network and all other build selectors.
        return False

    networks = document.get("networks")
    if (
        not isinstance(networks, dict)
        or set(networks) != {"default"}
        or not all(_has_namespace_label(config) for config in networks.values())
    ):
        return False
    default_network = networks["default"]
    if not isinstance(default_network, dict) or set(default_network) != {"labels"}:
        return False

    allowed_binds = {
        "${T_BENCH_TASK_LOGS_PATH}:${T_BENCH_CONTAINER_LOGS_PATH}",
        "${T_BENCH_TASK_AGENT_LOGS_PATH}:${T_BENCH_CONTAINER_AGENT_LOGS_PATH}",
    }

    def _service_mounts_are_scoped(config: object) -> bool:
        if not isinstance(config, dict) or any(
            key in config
            for key in (
                "extends",
                "external_links",
                "ipc",
                "links",
                "network_mode",
                "pid",
                "uts",
                "volumes_from",
            )
        ):
            return False
        container_name = config.get("container_name")
        if container_name is not None and (
            container_name != "${T_BENCH_TASK_DOCKER_CLIENT_CONTAINER_NAME}"
        ):
            return False
        service_networks = config.get("networks")
        if service_networks is None:
            network_names = {"default"}
        elif isinstance(service_networks, list):
            network_names = {str(value) for value in service_networks}
        elif isinstance(service_networks, dict):
            network_names = {str(value) for value in service_networks}
        else:
            return False
        if network_names != {"default"}:
            return False

        mounts = config.get("volumes")
        return (
            isinstance(mounts, list)
            and len(mounts) == len(allowed_binds)
            and set(mounts) == allowed_binds
        )

    return all(_service_mounts_are_scoped(config) for config in services.values())


async def _join_async_task(task: asyncio.Future[Any]) -> None:
    """Wait until *task* is finished, even if the waiter is cancelled again."""
    while not task.done():
        try:
            # Observe completion without propagating a child's pending
            # cancellation state into a tight CancelledError loop.
            await asyncio.wait({task}, timeout=0.1)
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if current is not None and hasattr(current, "uncancel"):
                current.uncancel()
    await asyncio.gather(task, return_exceptions=True)


def _capture_swesmith_stage_commits(
    container_name: str, timeout: float = 30.0
) -> tuple[str, str]:
    """Capture trusted task/bug commits before the agent receives shell access."""
    command = r"""
set -eu
repo_dir=""
for candidate in . /testbed /workspace; do
  if git -C "${candidate}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    repo_dir="${candidate}"
    break
  fi
done
test -n "${repo_dir}"
task_commit="$(git -C "${repo_dir}" rev-parse refs/agentic_rl/swesmith-task-stage^{commit})"
bug_commit="$(git -C "${repo_dir}" rev-parse refs/agentic_rl/swesmith-bug-stage^{commit})"
test "$(git -C "${repo_dir}" rev-parse "${task_commit}^")" = "${bug_commit}"
test "$(git -C "${repo_dir}" diff --name-only --diff-filter=D "${bug_commit}" "${task_commit}" | wc -l)" -gt 0
printf '__SWESMITH_COMMITS__=%s %s\n' "${task_commit}" "${bug_commit}"
"""
    result = subprocess.run(
        ["docker", "exec", "-u", "root", container_name, "sh", "-lc", command],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    marker = "__SWESMITH_COMMITS__="
    commit_line = next(
        (
            line[len(marker) :]
            for line in reversed(result.stdout.splitlines())
            if line.startswith(marker)
        ),
        "",
    )
    parts = commit_line.split()
    if result.returncode != 0 or len(parts) != 2 or not all(
        _GIT_COMMIT_RE.fullmatch(value) for value in parts
    ):
        detail = (result.stderr or result.stdout or "no output").strip()[-1000:]
        raise RuntimeError(
            "Could not capture trusted SWE-smith task/bug commits before agent access: "
            f"rc={result.returncode} detail={detail}"
        )
    return parts[0], parts[1]


def _run_container_shell(
    container_name: str, command: str, *, timeout: float
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", "exec", "-u", "root", container_name, "sh", "-lc", command],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _snapshot_sweverified_workspace(
    container_name: str, task_meta: dict[str, Any]
) -> str:
    """Snapshot the image's pre-agent worktree without changing its checkout."""

    base_commit = str(task_meta.get("base_commit") or "").strip()
    if not _GIT_COMMIT_RE.fullmatch(base_commit):
        raise RuntimeError(
            f"SWE-Verified task has invalid base_commit={base_commit!r}"
        )
    git = "git -c safe.directory=/testbed -C /testbed"
    command = f"""
set -eu
base={shlex.quote(base_commit)}
{git} cat-file -e "$base^{{commit}}"
original_index_tree=$({git} write-tree)
{git} add -A
baseline_tree=$({git} write-tree)
baseline_commit=$(printf '%s\\n' terminal-rl-sweverified-baseline | \
  {git} -c user.name=terminal-rl -c user.email=terminal-rl@localhost \
  commit-tree "$baseline_tree" -p "$base")
{git} read-tree "$original_index_tree"
printf '%s' "$baseline_commit"
"""
    result = _run_container_shell(container_name, command, timeout=120.0)
    baseline = (result.stdout or "").strip()
    if result.returncode != 0 or not _GIT_COMMIT_RE.fullmatch(baseline):
        detail = (result.stderr or result.stdout or "no output").strip()[-1000:]
        raise RuntimeError(
            "Could not snapshot the initial SWE-Verified workspace: "
            f"rc={result.returncode} detail={detail}"
        )
    return baseline


def _capture_sweverified_patch(
    container_name: str, baseline_commit: str
) -> str:
    if not _GIT_COMMIT_RE.fullmatch(str(baseline_commit or "")):
        raise RuntimeError("SWE-Verified workspace baseline is missing or invalid")
    git = "git -c safe.directory=/testbed -C /testbed"
    command = (
        f"{git} add -A && "
        f"{git} diff --binary --no-ext-diff {shlex.quote(baseline_commit)}"
    )
    result = _run_container_shell(container_name, command, timeout=120.0)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "no output").strip()[-1000:]
        raise RuntimeError(
            f"Could not export SWE-Verified prediction patch: {detail}"
        )
    return result.stdout or ""


def _docker_compose_down_projects(
    *,
    docker_compose_path: str | None,
    trial_name: str,
    client_container_name: str | None,
    reason: str,
    command_timeout: float,
    deadline: float | None = None,
) -> None:
    if not _env_bool("TERMINAL_ENV_COMPOSE_DOWN_CLEANUP", True):
        return
    if not docker_compose_path:
        return
    compose_path = Path(docker_compose_path)
    if not compose_path.exists():
        logger.warning(
            "Skipping docker compose down for TerminalEnv %s (%s): compose file missing: %s",
            trial_name,
            reason,
            compose_path,
        )
        return

    service_timeout = str(max(1, _env_int("TERMINAL_ENV_COMPOSE_DOWN_SERVICE_TIMEOUT", 5)))
    projects = _compose_project_candidates(trial_name, client_container_name)
    if _current_pool_namespace() != "default" and client_container_name:
        container_state, exact_project = _container_compose_project_state(
            client_container_name,
            timeout=command_timeout,
            deadline=deadline,
        )
        if container_state in {"foreign", "unknown"}:
            logger.warning(
                "Skipping docker compose down for TerminalEnv %s (%s): "
                "container ownership state=%s",
                trial_name,
                reason,
                container_state,
            )
            return
        if container_state == "owned" and exact_project:
            projects = [exact_project]
    for project in projects:
        if not _compose_project_matches_pool_namespace(
            project, timeout=command_timeout, deadline=deadline
        ):
            logger.warning(
                "Skipping docker compose down for TerminalEnv %s project=%s (%s): "
                "pool ownership could not be proven",
                trial_name,
                project,
                reason,
            )
            continue
        cmd = [
            "docker",
            "compose",
            "-p",
            project,
            "-f",
            str(compose_path),
            "down",
            "--remove-orphans",
            "-v",
            "--timeout",
            service_timeout,
        ]
        try:
            with _docker_network_lifecycle_lock():
                completed = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=_docker_cleanup_command_timeout(
                        command_timeout, deadline
                    ),
                )
            logger.warning(
                "Docker compose down finished for TerminalEnv %s project=%s "
                "reason=%s rc=%s stdout=%s stderr=%s",
                trial_name,
                project,
                reason,
                completed.returncode,
                completed.stdout.strip()[:300],
                completed.stderr.strip()[:300],
            )
        except _DockerCleanupDeadlineExceeded:
            raise
        except Exception as exc:
            logger.warning(
                "Docker compose down failed for TerminalEnv %s project=%s reason=%s: %s",
                trial_name,
                project,
                reason,
                exc,
            )


def _remove_fixed_task_services_without_running_clients(
    *,
    task_ids: set[str],
    reason: str,
    timeout: float,
    max_remove: int = 64,
    deadline: float | None = None,
) -> int:
    if not task_ids:
        return 0

    try:
        listed = subprocess.run(
            [
                "docker",
                "ps",
                "-a",
                "--format",
                "{{.ID}}\t{{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Label \"harborrl.pool-namespace\"}}",
            ],
            capture_output=True,
            text=True,
            timeout=_docker_cleanup_command_timeout(timeout, deadline),
        )
    except _DockerCleanupDeadlineExceeded:
        raise
    except Exception as exc:
        logger.warning(
            "Could not list Docker containers for fixed task service cleanup (%s): %s",
            reason,
            exc,
        )
        return 0

    running_client_task_ids: set[str] = set()
    rows: list[tuple[str, str, str, str, str]] = []
    for line in listed.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        container_id, name = parts[0], parts[1]
        image = parts[2] if len(parts) > 2 else ""
        status = parts[3] if len(parts) > 3 else ""
        pool_namespace = _clean_docker_label(parts[4] if len(parts) > 4 else "")
        rows.append((container_id, name, image, status, pool_namespace))
        if not _matches_pool_namespace(pool_namespace):
            continue
        if status.lower().startswith("up") and _is_task_container(name, image):
            task_id = _task_id_from_ref(name)
            if task_id:
                running_client_task_ids.add(task_id)

    blocked = task_ids.intersection(running_client_task_ids)
    removable_task_ids = task_ids.difference(blocked)
    if blocked:
        logger.warning(
            "Skipping fixed task service cleanup for active task id(s) %s reason=%s",
            ",".join(sorted(blocked)),
            reason,
        )
    if not removable_task_ids:
        return 0

    candidates: list[tuple[str, str, str, str, str]] = []
    for container_id, name, image, status, pool_namespace in rows:
        if not _matches_pool_namespace(pool_namespace):
            continue
        task_id = _fixed_task_service_id(name, image)
        if task_id and task_id in removable_task_ids:
            candidates.append((container_id, name, image, status, task_id))
            if max_remove > 0 and len(candidates) >= max_remove:
                break

    if not candidates:
        return 0

    logger.warning(
        "Removing %d fixed task service container(s) without running clients "
        "reason=%s task_ids=%s samples=%s",
        len(candidates),
        reason,
        ",".join(sorted(removable_task_ids)),
        "; ".join(
            f"{cid[:12]} name={name} image={image} status={status}"
            for cid, name, image, status, _task_id in candidates[:8]
        ),
    )

    removed_count = 0
    for start in range(0, len(candidates), 20):
        chunk = candidates[start : start + 20]
        ids = [item[0] for item in chunk]
        try:
            removed = subprocess.run(
                ["docker", "rm", "-f", *ids],
                capture_output=True,
                text=True,
                timeout=_docker_cleanup_command_timeout(timeout, deadline),
            )
            if removed.returncode == 0:
                removed_count += len(ids)
            logger.warning(
                "Fixed task service docker rm finished ids=%s rc=%s stdout=%s stderr=%s",
                ",".join(cid[:12] for cid in ids),
                removed.returncode,
                removed.stdout.strip()[:300],
                removed.stderr.strip()[:300],
            )
        except _DockerCleanupDeadlineExceeded:
            raise
        except Exception as exc:
            logger.warning(
                "Fixed task service docker rm failed ids=%s: %s",
                ",".join(cid[:12] for cid in ids),
                exc,
            )
    return removed_count


def _remove_inactive_compose_resources(
    *,
    resource_kind: str,
    active_project_names: set[str],
    active_task_ids: set[str],
    reason: str,
    timeout: float,
    max_remove: int,
    deadline: float | None = None,
) -> int:
    if resource_kind == "network":
        list_cmd = [
            "docker",
            "network",
            "ls",
            "--format",
            "{{.ID}}\t{{.Name}}\t{{.Label \"com.docker.compose.project\"}}\t{{.Label \"harborrl.pool-namespace\"}}",
        ]
        rm_cmd_prefix = ["docker", "network", "rm"]
        use_id = True
    elif resource_kind == "volume":
        list_cmd = [
            "docker",
            "volume",
            "ls",
            "--format",
            "{{.Name}}\t{{.Label \"com.docker.compose.project\"}}\t{{.Label \"harborrl.pool-namespace\"}}",
        ]
        rm_cmd_prefix = ["docker", "volume", "rm"]
        use_id = False
    else:
        return 0

    try:
        listed = subprocess.run(
            list_cmd,
            capture_output=True,
            text=True,
            timeout=_docker_cleanup_command_timeout(timeout, deadline),
        )
    except _DockerCleanupDeadlineExceeded:
        raise
    except Exception as exc:
        logger.warning("Could not list Docker %ss for orphan cleanup (%s): %s", resource_kind, reason, exc)
        return 0

    candidates: list[tuple[str, str, str]] = []
    for line in listed.stdout.splitlines():
        parts = line.split("\t")
        if resource_kind == "network":
            if len(parts) < 2:
                continue
            resource_id, name = parts[0], parts[1]
            compose_project = _clean_docker_label(parts[2] if len(parts) > 2 else "")
            pool_namespace = _clean_docker_label(parts[3] if len(parts) > 3 else "")
            ref = resource_id if use_id else name
        else:
            if not parts:
                continue
            name = parts[0]
            compose_project = _clean_docker_label(parts[1] if len(parts) > 1 else "")
            pool_namespace = _clean_docker_label(parts[2] if len(parts) > 2 else "")
            ref = name
        if not _matches_pool_namespace(pool_namespace):
            continue
        if compose_project and _matches_project_name(
            compose_project, active_project_names, broad=True
        ):
            continue
        task_id = _task_id_from_ref(compose_project) or _task_id_from_ref(name)
        looks_like_task_resource = (
            "slime-run" in name
            or "slime-run" in compose_project
            or (task_id is not None and task_id not in active_task_ids)
        )
        if not looks_like_task_resource:
            continue
        if task_id is not None and task_id in active_task_ids:
            continue
        if not compose_project and "slime-run" not in name:
            continue
        candidates.append((ref, name, compose_project))
        if max_remove > 0 and len(candidates) >= max_remove:
            break

    if not candidates:
        return 0

    removed_count = 0
    logger.warning(
        "Orphan Docker sweep removing %d stale compose %s(s) reason=%s samples=%s",
        len(candidates),
        resource_kind,
        reason,
        "; ".join(
            f"name={name} project={project}" for _ref, name, project in candidates[:8]
        ),
    )
    for ref, name, _project in candidates:
        try:
            removed = subprocess.run(
                [*rm_cmd_prefix, ref],
                capture_output=True,
                text=True,
                timeout=_docker_cleanup_command_timeout(timeout, deadline),
            )
            if removed.returncode == 0:
                removed_count += 1
            logger.warning(
                "Orphan Docker sweep %s rm finished name=%s rc=%s stdout=%s stderr=%s",
                resource_kind,
                name,
                removed.returncode,
                removed.stdout.strip()[:300],
                removed.stderr.strip()[:300],
            )
        except _DockerCleanupDeadlineExceeded:
            raise
        except Exception as exc:
            logger.warning(
                "Orphan Docker sweep %s rm failed name=%s: %s",
                resource_kind,
                name,
                exc,
            )
    return removed_count


def _docker_cleanup_postcondition(
    *,
    client_container_name: str | None,
    project_names: set[str],
    timeout: float,
    deadline: float | None = None,
) -> tuple[bool, list[str]]:
    """Prove that a non-default run has no remaining Compose resources."""
    remaining: list[str] = []
    direct_name = (client_container_name or "").strip()
    if direct_name:
        container_state = _docker_object_pool_namespace_state(
            "container", direct_name, timeout=timeout, deadline=deadline
        )
        if container_state != "absent":
            remaining.append(f"container:{direct_name}={container_state}")
    for project in sorted(project_names):
        project_state = _compose_project_pool_namespace_state(
            project, timeout=timeout, deadline=deadline
        )
        if project_state != "absent":
            remaining.append(f"project:{project}={project_state}")
    return not remaining, remaining


def _force_remove_docker_objects_impl(
    *,
    trial_name: str,
    client_container_name: str | None,
    docker_image_name_prefix: str | None = None,
    docker_compose_path: str | None = None,
    reason: str,
    cleanup_deadline: float,
) -> bool:
    namespace = _current_pool_namespace()
    timeout = max(
        0.001,
        _env_float("TERMINAL_ENV_FORCE_DOCKER_CLEANUP_TIMEOUT", 20.0),
    )
    broad = _env_bool("TERMINAL_ENV_FORCE_DOCKER_CLEANUP_BROAD", True)
    direct_name = (client_container_name or "").strip()
    candidate_projects = set(
        _compose_project_candidates(trial_name, client_container_name)
    )
    target_projects = set(candidate_projects)
    if namespace != "default" and direct_name:
        initial_state, exact_project = _container_compose_project_state(
            direct_name, timeout=timeout, deadline=cleanup_deadline
        )
        if initial_state in {"foreign", "unknown"}:
            logger.error(
                "Refusing force cleanup for TerminalEnv %s name=%s: "
                "container state=%s",
                trial_name,
                direct_name,
                initial_state,
            )
            return False
        if exact_project:
            target_projects = {exact_project}
    project_names = _docker_name_variants(trial_name)
    project_names.update(_docker_name_variants(client_container_name))
    project_names.update(candidate_projects)
    project_names.update(target_projects)
    image_prefixes = _docker_image_prefixes(
        docker_image_name_prefix,
        trial_name,
        client_container_name,
    )
    task_ids = {
        task_id
        for task_id in (
            _task_id_from_ref(docker_image_name_prefix),
            _task_id_from_ref(trial_name),
            _task_id_from_ref(client_container_name),
        )
        if task_id
    }
    for prefix in image_prefixes:
        task_id = _task_id_from_ref(prefix)
        if task_id:
            task_ids.add(task_id)
    if not project_names and not image_prefixes:
        return namespace == "default"

    def _run(cmd: list[str], *, check: bool = False) -> subprocess.CompletedProcess:
        return subprocess.run(
            cmd,
            check=check,
            capture_output=True,
            text=True,
            timeout=_docker_cleanup_command_timeout(timeout, cleanup_deadline),
        )

    _docker_compose_down_projects(
        docker_compose_path=docker_compose_path,
        trial_name=trial_name,
        client_container_name=client_container_name,
        reason=reason,
        command_timeout=timeout,
        deadline=cleanup_deadline,
    )

    try:
        listed = _run(
            [
                "docker",
                "ps",
                "-a",
                "--format",
                "{{.ID}}\t{{.Names}}\t{{.Image}}\t{{.Status}}\t"
                '{{.Label "harborrl.pool-namespace"}}',
            ]
        )
    except _DockerCleanupDeadlineExceeded:
        raise
    except Exception as exc:
        logger.warning(
            "Force cleanup could not list Docker containers for %s (%s): %s",
            trial_name,
            reason,
            exc,
        )
        return namespace == "default"

    container_ids: list[str] = []
    container_samples: list[str] = []
    for line in listed.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        container_id, name = parts[0], parts[1]
        image = parts[2] if len(parts) > 2 else ""
        status = parts[3] if len(parts) > 3 else ""
        pool_namespace = _clean_docker_label(parts[4] if len(parts) > 4 else "")
        if not _matches_pool_namespace(pool_namespace):
            continue
        name_match = _matches_project_name(name, project_names, broad=broad)
        # Image-prefix matching is intentionally only a fallback when we do not
        # know a project/container name; matching tb__<task>__ can otherwise
        # remove other active samples of the same task.
        image_match = not project_names and any(
            image.startswith(prefix) for prefix in image_prefixes
        )
        if name_match or image_match:
            container_ids.append(container_id)
            if len(container_samples) < 8:
                container_samples.append(
                    f"{container_id[:12]} name={name} image={image} status={status}"
                )

    if container_ids:
        logger.warning(
            "Force removing %d Docker container(s) for TerminalEnv %s (%s): %s",
            len(container_ids),
            trial_name,
            reason,
            "; ".join(container_samples),
        )
        for start in range(0, len(container_ids), 20):
            chunk = container_ids[start : start + 20]
            try:
                removed = _run(["docker", "rm", "-f", *chunk])
                logger.warning(
                    "Force docker rm finished for TerminalEnv %s ids=%s rc=%s stdout=%s stderr=%s",
                    trial_name,
                    ",".join(cid[:12] for cid in chunk),
                    removed.returncode,
                    removed.stdout.strip()[:300],
                    removed.stderr.strip()[:300],
                )
            except _DockerCleanupDeadlineExceeded:
                raise
            except Exception as exc:
                logger.warning(
                    "Force docker rm failed for TerminalEnv %s ids=%s: %s",
                    trial_name,
                    ",".join(chunk),
                    exc,
                )
    else:
        logger.warning(
            "Force cleanup matched no Docker containers for TerminalEnv %s (%s); "
            "client_container=%s image_prefixes=%s projects=%s",
            trial_name,
            reason,
            client_container_name or "",
            ",".join(sorted(image_prefixes)) or "",
            ",".join(sorted(project_names)) or "",
        )

    try:
        networks = _run(
            [
                "docker",
                "network",
                "ls",
                "--format",
                "{{.ID}}\t{{.Name}}\t"
                '{{.Label "harborrl.pool-namespace"}}',
            ]
        )
    except _DockerCleanupDeadlineExceeded:
        raise
    except Exception:
        networks = None

    network_ids: list[str] = []
    for line in (networks.stdout.splitlines() if networks is not None else []):
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        net_id, name = parts[0], parts[1]
        pool_namespace = _clean_docker_label(parts[2] if len(parts) > 2 else "")
        if _matches_pool_namespace(pool_namespace) and _matches_project_name(
            name, project_names, broad=broad
        ):
            network_ids.append(net_id)

    for net_id in network_ids:
        try:
            with _docker_network_lifecycle_lock():
                removed_net = _run(["docker", "network", "rm", net_id])
            logger.warning(
                "Force docker network rm finished for TerminalEnv %s id=%s rc=%s",
                trial_name,
                net_id[:12],
                removed_net.returncode,
            )
        except _DockerCleanupDeadlineExceeded:
            raise
        except Exception:
            pass

    try:
        volumes = _run(
            [
                "docker",
                "volume",
                "ls",
                "--format",
                "{{.Name}}\t{{.Label \"com.docker.compose.project\"}}\t"
                '{{.Label "harborrl.pool-namespace"}}',
            ]
        )
    except _DockerCleanupDeadlineExceeded:
        raise
    except Exception:
        volumes = None

    if volumes is not None:
        for line in volumes.stdout.splitlines():
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            volume_name, compose_project = parts[0], parts[1]
            pool_namespace = _clean_docker_label(
                parts[2] if len(parts) > 2 else ""
            )
            if not _matches_pool_namespace(pool_namespace):
                continue
            if not _matches_project_name(
                compose_project, project_names, broad=broad
            ):
                continue
            try:
                if not _docker_object_matches_pool_namespace(
                    "volume",
                    volume_name,
                    timeout=timeout,
                    deadline=cleanup_deadline,
                ):
                    logger.warning(
                        "Skipping Docker volume rm after ownership recheck failed: %s",
                        volume_name,
                    )
                    continue
                removed_volume = _run(["docker", "volume", "rm", volume_name])
                logger.warning(
                    "Force docker volume rm finished for TerminalEnv %s name=%s rc=%s",
                    trial_name,
                    volume_name,
                    removed_volume.returncode,
                )
            except _DockerCleanupDeadlineExceeded:
                raise
            except Exception:
                pass

    removed_fixed = _remove_fixed_task_services_without_running_clients(
        task_ids=task_ids,
        reason=f"force_cleanup:{reason}",
        timeout=timeout,
        max_remove=_env_int("TERMINAL_ENV_FIXED_SERVICE_CLEANUP_MAX_REMOVE", 64),
        deadline=cleanup_deadline,
    )
    if removed_fixed:
        logger.warning(
            "Force cleanup removed %d fixed task service container(s) for TerminalEnv %s "
            "(%s) task_ids=%s",
            removed_fixed,
            trial_name,
            reason,
            ",".join(sorted(task_ids)),
        )

    if namespace == "default":
        # Preserve legacy SETA best-effort cleanup semantics.
        return True

    postcondition_ok, remaining = _docker_cleanup_postcondition(
        client_container_name=direct_name,
        project_names=target_projects,
        timeout=timeout,
        deadline=cleanup_deadline,
    )
    if not postcondition_ok:
        logger.error(
            "Force cleanup postcondition failed for TerminalEnv %s (%s): %s",
            trial_name,
            reason,
            ", ".join(remaining),
        )
        return False
    logger.info(
        "Force cleanup postcondition verified for TerminalEnv %s (%s)",
        trial_name,
        reason,
    )
    return True


def _force_remove_docker_objects(
    *,
    trial_name: str,
    client_container_name: str | None,
    docker_image_name_prefix: str | None = None,
    docker_compose_path: str | None = None,
    reason: str,
    cleanup_deadline: float | None = None,
) -> bool:
    namespace = _current_pool_namespace()
    if not _env_bool("TERMINAL_ENV_FORCE_DOCKER_CLEANUP", True):
        return namespace == "default"
    timeout = max(
        0.001,
        _env_float("TERMINAL_ENV_FORCE_DOCKER_CLEANUP_TIMEOUT", 20.0),
    )
    deadline = cleanup_deadline or (time.monotonic() + timeout)
    try:
        return _force_remove_docker_objects_impl(
            trial_name=trial_name,
            client_container_name=client_container_name,
            docker_image_name_prefix=docker_image_name_prefix,
            docker_compose_path=docker_compose_path,
            reason=reason,
            cleanup_deadline=deadline,
        )
    except _DockerCleanupDeadlineExceeded:
        logger.error(
            "Docker cleanup deadline exceeded for TerminalEnv %s (%s)",
            trial_name,
            reason,
        )
        return namespace == "default"


async def _force_remove_docker_objects_async(
    *,
    trial_name: str,
    client_container_name: str | None,
    docker_image_name_prefix: str | None = None,
    docker_compose_path: str | None = None,
    reason: str,
) -> bool:
    loop = asyncio.get_running_loop()
    cleanup_timeout = max(
        0.001,
        _env_float("TERMINAL_ENV_FORCE_DOCKER_CLEANUP_TIMEOUT", 20.0),
    )
    cleanup_deadline = time.monotonic() + cleanup_timeout
    fut = loop.run_in_executor(
        _docker_cleanup_executor(),
        partial(
            _force_remove_docker_objects,
            trial_name=trial_name,
            client_container_name=client_container_name,
            docker_image_name_prefix=docker_image_name_prefix,
            docker_compose_path=docker_compose_path,
            reason=reason,
            cleanup_deadline=cleanup_deadline,
        ),
    )
    try:
        return bool(await asyncio.shield(fut))
    except asyncio.CancelledError:
        logger.warning(
            "Docker cleanup cancellation received for TerminalEnv %s (%s); "
            "waiting for the bounded executor cleanup before returning.",
            trial_name,
            reason,
        )
        await _join_async_task(fut)
        raise


def force_remove_orphan_docker_objects(
    *,
    active_container_names: set[str],
    active_project_names: set[str] | None = None,
    active_task_ids: set[str] | None = None,
    reason: str,
    min_age_sec: float,
    max_remove: int,
    cleanup_timeout: float | None = None,
) -> int:
    if not _env_bool("TERMINAL_ENV_FORCE_DOCKER_CLEANUP", True):
        return 0

    timeout = max(
        0.001,
        float(os.getenv("TERMINAL_ENV_FORCE_DOCKER_CLEANUP_TIMEOUT", "20")),
    )
    wall_clock_budget = max(
        0.001, timeout if cleanup_timeout is None else float(cleanup_timeout)
    )
    deadline = time.monotonic() + wall_clock_budget
    try:
        listed = subprocess.run(
            [
                "docker",
                "ps",
                "-a",
                "--format",
                "{{.ID}}\t{{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Label \"com.docker.compose.project\"}}\t{{.Label \"harborrl.pool-namespace\"}}",
            ],
            capture_output=True,
            text=True,
            timeout=_docker_cleanup_command_timeout(timeout, deadline),
        )
    except _DockerCleanupDeadlineExceeded as exc:
        raise TimeoutError(
            f"Orphan Docker sweep exceeded {wall_clock_budget:.1f}s deadline"
        ) from exc
    except Exception as exc:
        logger.warning("Orphan Docker sweep could not list containers (%s): %s", reason, exc)
        return -1
    if listed.returncode != 0:
        logger.warning(
            "Orphan Docker sweep docker ps failed (%s): rc=%s stdout=%s stderr=%s",
            reason,
            listed.returncode,
            (listed.stdout or "").strip()[:400],
            (listed.stderr or "").strip()[:400],
        )
        return -1

    active = {name for name in active_container_names if name}
    active_projects = {name for name in (active_project_names or set()) if name}
    active_tasks = {task_id for task_id in (active_task_ids or set()) if task_id}
    for name in active:
        task_id = _task_id_from_ref(name)
        if task_id:
            active_tasks.add(task_id)

    running_client_task_ids: set[str] = set()
    rows: list[tuple[str, str, str, str, str, str]] = []
    for line in listed.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        container_id, name = parts[0], parts[1]
        image = parts[2] if len(parts) > 2 else ""
        status = parts[3] if len(parts) > 3 else ""
        compose_project = _clean_docker_label(parts[4] if len(parts) > 4 else "")
        pool_namespace = _clean_docker_label(parts[5] if len(parts) > 5 else "")
        rows.append((container_id, name, image, status, compose_project, pool_namespace))
        if not _matches_pool_namespace(pool_namespace):
            continue
        if status.lower().startswith("up") and _is_task_container(name, image):
            task_id = _task_id_from_ref(name)
            if task_id:
                running_client_task_ids.add(task_id)

    candidates: list[tuple[str, str, str, str, float, str]] = []
    for container_id, name, image, status, compose_project, pool_namespace in rows:
        if not _matches_pool_namespace(pool_namespace):
            continue
        if name in active:
            continue
        if compose_project and _matches_project_name(
            compose_project, active_projects, broad=True
        ):
            continue

        fixed_task_id = _fixed_task_service_id(name, image)
        fixed_service_orphan = bool(
            fixed_task_id
            and fixed_task_id not in active_tasks
            and fixed_task_id not in running_client_task_ids
        )
        inactive_project_container = bool(
            compose_project
            and "slime-run" in compose_project
            and not _matches_project_name(compose_project, active_projects, broad=True)
            and ((image or "").startswith("tb__") or _task_id_from_ref(compose_project))
        )
        stale_client = _is_task_container(name, image)
        if not (stale_client or fixed_service_orphan or inactive_project_container):
            continue
        age_sec = _docker_status_age_seconds(status)
        if age_sec is None or age_sec < min_age_sec:
            continue
        if fixed_service_orphan:
            match_reason = f"fixed_service_task={fixed_task_id}"
        elif inactive_project_container:
            match_reason = f"inactive_project={compose_project}"
        else:
            match_reason = "stale_client"
        candidates.append((container_id, name, image, status, age_sec, match_reason))
        if max_remove > 0 and len(candidates) >= max_remove:
            break

    removed_count = 0
    if candidates:
        logger.warning(
            "Orphan Docker sweep removing %d stale task container(s) reason=%s "
            "min_age=%.1fs active=%d samples=%s",
            len(candidates),
            reason,
            min_age_sec,
            len(active),
            "; ".join(
                f"{cid[:12]} name={name} image={image} status={status} reason={why}"
                for cid, name, image, status, _age, why in candidates[:8]
            ),
        )

        for start in range(0, len(candidates), 20):
            chunk = candidates[start : start + 20]
            ids = [item[0] for item in chunk]
            try:
                removed = subprocess.run(
                    ["docker", "rm", "-f", *ids],
                    capture_output=True,
                    text=True,
                    timeout=_docker_cleanup_command_timeout(timeout, deadline),
                )
                if removed.returncode == 0:
                    removed_count += len(ids)
                logger.warning(
                    "Orphan Docker sweep rm finished ids=%s rc=%s stdout=%s stderr=%s",
                    ",".join(cid[:12] for cid in ids),
                    removed.returncode,
                    removed.stdout.strip()[:300],
                    removed.stderr.strip()[:300],
                )
            except _DockerCleanupDeadlineExceeded as exc:
                raise TimeoutError(
                    f"Orphan Docker sweep exceeded {wall_clock_budget:.1f}s deadline"
                ) from exc
            except Exception as exc:
                logger.warning(
                    "Orphan Docker sweep rm failed ids=%s: %s",
                    ",".join(cid[:12] for cid in ids),
                    exc,
                )
    if _env_bool("WORKER_ORPHAN_DOCKER_SWEEP_RESOURCES", True):
        resource_max_remove = max(0, _env_int("WORKER_ORPHAN_DOCKER_SWEEP_RESOURCE_MAX_REMOVE", 128))
        if resource_max_remove:
            try:
                with _docker_network_lifecycle_lock():
                    _remove_inactive_compose_resources(
                        resource_kind="network",
                        active_project_names=active_projects,
                        active_task_ids=active_tasks,
                        reason=reason,
                        timeout=timeout,
                        max_remove=resource_max_remove,
                        deadline=deadline,
                    )
                # A volume name is reusable, so deleting it after a list/inspect
                # ownership check still has a TOCTOU race. Non-default pools reject
                # named volumes in their Compose model and never sweep them by name.
                if _current_pool_namespace() == "default":
                    _remove_inactive_compose_resources(
                        resource_kind="volume",
                        active_project_names=active_projects,
                        active_task_ids=active_tasks,
                        reason=reason,
                        timeout=timeout,
                        max_remove=resource_max_remove,
                        deadline=deadline,
                    )
            except _DockerCleanupDeadlineExceeded as exc:
                raise TimeoutError(
                    f"Orphan Docker sweep exceeded {wall_clock_budget:.1f}s deadline"
                ) from exc
    return removed_count


def _stop_terminal_compat(terminal: Terminal, timeout: float) -> None:
    with _docker_network_lifecycle_lock():
        try:
            supports_timeout = "timeout" in inspect.signature(terminal.stop).parameters
        except (TypeError, ValueError):
            supports_timeout = False

        if supports_timeout:
            terminal.stop(timeout=timeout)
        else:
            if _env_bool("TERMINAL_ENV_FAST_CLOSE", False) or _env_bool(
                "TERMINAL_ENV_SKIP_UNBOUNDED_STOP", False
            ):
                logger.warning(
                    "Terminal.stop(timeout=...) is unsupported; skipping unbounded "
                    "Terminal.stop() under fast close."
                )
                return
            logger.warning(
                "Terminal.stop(timeout=...) is unsupported; retrying with Terminal.stop()."
            )
            terminal.stop()


def _drain_toolkit_sessions(toolkit: Any) -> None:
    sessions = getattr(toolkit, "shell_sessions", None)
    if not isinstance(sessions, dict):
        return
    lock = getattr(toolkit, "_session_lock", None)
    acquired_lock = False
    try:
        if lock is not None:
            try:
                acquired_lock = bool(lock.acquire(blocking=False))
            except TypeError:
                acquired_lock = bool(lock.acquire(False))
            if not acquired_lock:
                logger.warning(
                    "Skipping TerminalToolkit session drain because session lock "
                    "is currently held."
                )
                return
        for session in sessions.values():
            proc = session.get("process")
            if proc is not None:
                try:
                    if hasattr(proc, "terminate"):
                        proc.terminate()
                    elif hasattr(proc, "close"):
                        proc.close()
                except Exception:
                    pass
            q = session.get("output_stream")
            if q is not None:
                try:
                    while not q.empty():
                        q.get_nowait()
                except Exception:
                    pass
        sessions.clear()
    finally:
        if lock is not None and acquired_lock:
            try:
                lock.release()
            except RuntimeError:
                pass


class TerminalEnv:
    def __init__(self) -> None:
        self._lifecycle_lock = asyncio.Lock()
        self._closed = False
        self._task_spec: TaskSpec | None = None
        self._task_meta: dict[str, Any] | None = None
        self._run_ctx: RunContext | None = None
        self._timeouts: TaskTimeouts | None = None

        self._trial_handler: TrialHandler | None = None
        self._terminal: Terminal | None = None
        self._parser = None
        self._terminal_toolkit: TerminalToolkit | None = None
        self._tools: dict[str, Any] = {}
        self._agent_safetybench_env: AgentSafetyBenchEnv | None = None
        self._agentharm_env: AgentHarmEnv | None = None
        self._tau2_env: Tau2Env | None = None
        self._eval_attempt = 0
        self._last_eval: dict[str, Any] | None = None
        self._last_trial_name: str | None = None
        self._last_client_container_name: str | None = None
        self._last_docker_image_name_prefix: str | None = None
        self._last_docker_compose_path: str | None = None
        self._data_source = ""
        self._swesmith_task_commit: str | None = None
        self._swesmith_bug_commit: str | None = None
        self._sweverified_baseline_commit: str | None = None

    async def reset(
        self,
        *,
        task_meta: dict[str, Any],
        task_spec: TaskSpec,
        run_ctx: RunContext,
        timeouts: TaskTimeouts,
    ) -> tuple[str, list[dict[str, Any]]]:
        async with self._lifecycle_lock:
            return await self._reset_locked(
                task_meta=task_meta,
                task_spec=task_spec,
                run_ctx=run_ctx,
                timeouts=timeouts,
            )

    async def _reset_locked(
        self,
        *,
        task_meta: dict[str, Any],
        task_spec: TaskSpec,
        run_ctx: RunContext,
        timeouts: TaskTimeouts,
    ) -> tuple[str, list[dict[str, Any]]]:
        await self._close_locked()

        self._closed = False
        self._task_spec = task_spec
        self._task_meta = dict(task_meta)
        self._run_ctx = run_ctx
        self._timeouts = timeouts
        self._eval_attempt = 0
        self._last_eval = None
        self._data_source = str(task_meta.get("data_source") or "")
        self._swesmith_task_commit = None
        self._swesmith_bug_commit = None
        self._sweverified_baseline_commit = None

        if task_meta.get("data_source") == "agent_safetybench":
            self._agent_safetybench_env = AgentSafetyBenchEnv()
            return await self._agent_safetybench_env.reset(
                task_meta=task_meta,
                task_spec=task_spec,
                run_ctx=run_ctx,
            )
        if task_meta.get("data_source") == "agentharm":
            self._agentharm_env = AgentHarmEnv()
            return await self._agentharm_env.reset(
                task_meta=task_meta,
                task_spec=task_spec,
                run_ctx=run_ctx,
            )
        if task_meta.get("data_source") == "tau2":
            self._tau2_env = Tau2Env()
            return await self._tau2_env.reset(
                task_meta=task_meta,
                task_spec=task_spec,
                run_ctx=run_ctx,
            )

        cancel_event = threading.Event()
        reset_started = time.monotonic()

        dataset_dir = str(os.getenv("DATASET_DIR", "")).strip()
        if not dataset_dir:
            raise ValueError("DATASET_DIR is required")
        dataset_root = Path(dataset_dir).resolve()
        task_path = (dataset_root / self._task_spec.task_path).resolve()
        try:
            task_path.relative_to(dataset_root)
        except ValueError as exc:
            raise ValueError(
                f"task_path escapes DATASET_DIR: {self._task_spec.task_path!r}"
            ) from exc
        if self._data_source == "swesmith":
            from harborrl.data.convert_swesmith import (
                expected_swesmith_task_path,
                validate_task_dir_fingerprint,
            )

            expected_task_path = expected_swesmith_task_path(
                task_meta.get("task_name")
            )
            if self._task_spec.task_path != expected_task_path:
                raise RuntimeError(
                    "SWE-smith task_path does not match task identity: "
                    f"expected={expected_task_path!r} "
                    f"actual={self._task_spec.task_path!r}"
                )
            if not validate_task_dir_fingerprint(
                {"metadata": task_meta}, task_path
            ):
                raise RuntimeError(
                    "SWE-smith task directory fingerprint is missing or stale: "
                    f"{task_path}"
                )
        elif self._data_source == "sweverified":
            from harborrl.data.convert_sweverified import (
                DATASET_NAME,
                DATASET_REVISION,
                SWEBENCH_COMMIT,
                SWEBENCH_VERSION,
                TASK_FORMAT_VERSION,
                expected_task_path,
                official_image_name,
                validate_task_dir_fingerprint,
            )

            instance_id = str(task_meta.get("swe_instance_id") or "")
            expected_values = {
                "task_path": expected_task_path(instance_id),
                "source_dataset": DATASET_NAME,
                "source_revision": DATASET_REVISION,
                "swebench_harness_version": SWEBENCH_VERSION,
                "swebench_harness_commit": SWEBENCH_COMMIT,
                "task_format_version": TASK_FORMAT_VERSION,
                "image_name": official_image_name(instance_id),
            }
            for key, expected in expected_values.items():
                actual = (
                    self._task_spec.task_path
                    if key == "task_path"
                    else task_meta.get(key)
                )
                if actual != expected:
                    raise RuntimeError(
                        "SWE-Verified task metadata is stale or untrusted: "
                        f"{key} expected={expected!r} actual={actual!r}"
                    )
            if not validate_task_dir_fingerprint(
                {"metadata": task_meta}, task_path
            ):
                raise RuntimeError(
                    "SWE-Verified task directory fingerprint is missing or "
                    f"stale: {task_path}"
                )
        output_path = Path(self._run_ctx.log_dir).resolve()
        output_path.mkdir(parents=True, exist_ok=True)
        namespace = _current_pool_namespace()
        if namespace != "default":
            compose_path = task_path / "docker-compose.yaml"
            if not _compose_declares_pool_namespace(compose_path):
                raise RuntimeError(
                    "non-default Docker pool requires a single static Compose "
                    "model matching the exact generated SWE task format before "
                    "image preparation: "
                    f"{compose_path}"
                )

        image_prepare_started = time.monotonic()
        logger.info(
            "TerminalEnv reset image prepare starting task=%s uid=%s timeout=%.1fs",
            self._task_spec.task_name,
            self._run_ctx.uid,
            self._timeouts.ensure_image,
        )
        image_prepare_task = asyncio.create_task(
            asyncio.to_thread(
                prepare_task_docker_image,
                task=task_meta,
                timeout=self._timeouts.ensure_image,
                cancel_event=cancel_event,
            )
        )
        try:
            image_prep = await asyncio.shield(image_prepare_task)
        except asyncio.CancelledError:
            cancel_event.set()
            await _join_async_task(image_prepare_task)
            logger.warning(
                "TerminalEnv reset cancelled during image prepare task=%s uid=%s "
                "elapsed=%.1fs total_elapsed=%.1fs",
                self._task_spec.task_name,
                self._run_ctx.uid,
                time.monotonic() - image_prepare_started,
                time.monotonic() - reset_started,
            )
            raise
        except Exception:
            logger.exception(
                "TerminalEnv reset image prepare failed task=%s uid=%s elapsed=%.1fs "
                "total_elapsed=%.1fs",
                self._task_spec.task_name,
                self._run_ctx.uid,
                time.monotonic() - image_prepare_started,
                time.monotonic() - reset_started,
            )
            raise
        logger.info(
            "TerminalEnv reset image prepare finished task=%s uid=%s mode=%s "
            "image=%s elapsed=%.1fs total_elapsed=%.1fs",
            self._task_spec.task_name,
            self._run_ctx.uid,
            getattr(image_prep, "mode", ""),
            getattr(image_prep, "client_image_name", ""),
            time.monotonic() - image_prepare_started,
            time.monotonic() - reset_started,
        )

        def _raise_if_cancelled(stage: str) -> None:
            if cancel_event.is_set():
                raise RuntimeError(
                    f"TERMINAL_ENV_RESET_CANCELLED task={self._task_spec.task_name} "
                    f"uid={self._run_ctx.uid} stage={stage}"
                )

        def _sync_reset() -> tuple[str, list[dict[str, Any]]]:
            # Bare references do not trigger the module-level lazy __getattr__;
            # resolve through the helper (same-named locals shadow the globals).
            TrialHandler = _lazy_import("TrialHandler")  # noqa: F811
            ParserFactory = _lazy_import("ParserFactory")  # noqa: F811
            Terminal = _lazy_import("Terminal")  # noqa: F811
            TerminalToolkit = _lazy_import("TerminalToolkit")  # noqa: F811
            FunctionTool = _lazy_import("FunctionTool")  # noqa: F811
            _raise_if_cancelled("before_docker_cleanup")
            # P0 FIX: Force recreate container to avoid Docker daemon API slowdown
            # Root cause: containers.get() HTTP call hangs 360s when container runs >1h
            # Docker daemon state accumulation causes API performance degradation
            # Solution: Delete old container, create fresh one (fast API response)
            trial_name = _pool_scoped_trial_name(
                self._task_spec.task_name, self._run_ctx.uid
            )
            self._trial_handler = TrialHandler(
                trial_name=trial_name,
                input_path=task_path,
                output_path=output_path,
            )
            container_name = self._trial_handler.client_container_name
            compose_path = self._trial_handler.task_paths.docker_compose_path
            namespace = _current_pool_namespace()
            if namespace != "default":
                container_state, compose_project = _container_compose_project_state(
                    container_name, timeout=5
                )
                if container_state in {"foreign", "unknown"}:
                    raise RuntimeError(
                        "refusing reset because Docker container ownership could "
                        f"not be proven: container={container_name!r} "
                        f"state={container_state}"
                    )
                project_ref = compose_project or container_name
                project_state = _compose_project_pool_namespace_state(
                    project_ref, timeout=5
                )
                if project_state in {"foreign", "unknown"}:
                    raise RuntimeError(
                        "refusing reset because Docker project ownership could not "
                        f"be proven: project={project_ref!r} state={project_state}"
                    )
                removed_existing = _remove_owned_container_for_reset(
                    container_name, timeout=5
                )
            else:
                # Keep the existing SETA behavior: best-effort exact cleanup and
                # continue even if Docker reports that the old container is gone.
                removed_existing = False
                try:
                    removed = subprocess.run(
                        ["docker", "rm", "-f", container_name],
                        timeout=5,
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    removed_existing = removed.returncode == 0
                except Exception as exc:
                    logger.debug(
                        "Container force-remove failed (may not exist): %s", exc
                    )

            if removed_existing:
                logger.info(
                    "Forced container recreation for %s to avoid Docker API slowdown",
                    container_name,
                )

            self._last_trial_name = self._trial_handler.trial_name
            self._last_client_container_name = self._trial_handler.client_container_name
            self._last_docker_image_name_prefix = (
                self._trial_handler.docker_image_name_prefix
            )
            self._last_docker_compose_path = str(
                self._trial_handler.task_paths.docker_compose_path
            )
            _raise_if_cancelled("before_terminal_create")
            task_config = self._trial_handler.task
            self._parser = ParserFactory.get_parser(task_config.parser_name)
            client_image_name = (
                image_prep.client_image_name or self._trial_handler.client_image_name
            )

            self._terminal = Terminal(
                client_container_name=self._trial_handler.client_container_name,
                client_image_name=client_image_name,
                docker_compose_path=self._trial_handler.task_paths.docker_compose_path,
                docker_image_name_prefix=self._trial_handler.docker_image_name_prefix,
                sessions_logs_path=self._trial_handler.trial_paths.sessions_path,
                agent_logs_path=self._trial_handler.trial_paths.agent_logging_dir,
                no_rebuild=True,
                cleanup=False,
                disable_recording=task_config.disable_asciinema,
            )
            if namespace != "default":
                compose_manager = getattr(self._terminal, "_compose_manager", None)
                compose_env = getattr(compose_manager, "env", None)
                try:
                    compose_env["TERMINAL_RL_POOL_NAMESPACE"] = namespace
                except (AttributeError, TypeError) as exc:
                    raise RuntimeError(
                        "Terminal-Bench Compose manager does not expose a mutable env"
                    ) from exc
            docker_start_started = time.monotonic()
            logger.info(
                "TerminalEnv docker start starting task=%s uid=%s container=%s "
                "mode=%s timeout=%.1fs",
                self._task_spec.task_name,
                self._run_ctx.uid,
                self._trial_handler.client_container_name,
                image_prep.mode,
                self._timeouts.reset_session,
            )
            try:
                _raise_if_cancelled("before_docker_start")
                # Image preparation has already completed. Always start through
                # the bounded subprocess path; pinned Terminal-Bench start() has
                # no timeout and can strand the reset thread indefinitely.
                with _docker_network_lifecycle_lock():
                    compose_up_no_build(
                        self._terminal,
                        timeout=self._timeouts.reset_session,
                        container_name=self._trial_handler.client_container_name,
                        logger=logger,
                    )
            except Exception:
                logger.exception(
                    "TerminalEnv docker start failed task=%s uid=%s container=%s "
                    "mode=%s elapsed=%.1fs total_elapsed=%.1fs",
                    self._task_spec.task_name,
                    self._run_ctx.uid,
                    self._trial_handler.client_container_name,
                    image_prep.mode,
                    time.monotonic() - docker_start_started,
                    time.monotonic() - reset_started,
                )
                _force_remove_docker_objects(
                    trial_name=self._trial_handler.trial_name,
                    client_container_name=self._trial_handler.client_container_name,
                    docker_image_name_prefix=self._trial_handler.docker_image_name_prefix,
                    docker_compose_path=str(
                        self._trial_handler.task_paths.docker_compose_path
                    ),
                    reason="reset_start_failed",
                )
                raise
            logger.info(
                "TerminalEnv docker start finished task=%s uid=%s container=%s "
                "mode=%s elapsed=%.1fs total_elapsed=%.1fs",
                self._task_spec.task_name,
                self._run_ctx.uid,
                self._trial_handler.client_container_name,
                image_prep.mode,
                time.monotonic() - docker_start_started,
                time.monotonic() - reset_started,
            )

            if namespace != "default":
                started_state, started_project = _container_compose_project_state(
                    self._trial_handler.client_container_name, timeout=10
                )
                started_project_state = (
                    _compose_project_pool_namespace_state(
                        started_project, timeout=10
                    )
                    if started_project
                    else "unknown"
                )
                if started_state != "owned" or started_project_state != "owned":
                    _force_remove_docker_objects(
                        trial_name=self._trial_handler.trial_name,
                        client_container_name=self._trial_handler.client_container_name,
                        docker_image_name_prefix=self._trial_handler.docker_image_name_prefix,
                        docker_compose_path=str(
                            self._trial_handler.task_paths.docker_compose_path
                        ),
                        reason="namespace_post_start_failed",
                    )
                    raise RuntimeError(
                        "non-default Docker pool failed post-start ownership check: "
                        f"container={started_state} project={started_project!r} "
                        f"project_state={started_project_state}"
                    )

            if self._data_source == "swesmith":
                try:
                    (
                        self._swesmith_task_commit,
                        self._swesmith_bug_commit,
                    ) = _capture_swesmith_stage_commits(
                        self._trial_handler.client_container_name
                    )
                except Exception:
                    logger.exception(
                        "SWE-smith trusted commit capture failed task=%s uid=%s",
                        self._task_spec.task_name,
                        self._run_ctx.uid,
                    )
                    _force_remove_docker_objects(
                        trial_name=self._trial_handler.trial_name,
                        client_container_name=self._trial_handler.client_container_name,
                        docker_image_name_prefix=self._trial_handler.docker_image_name_prefix,
                        docker_compose_path=str(
                            self._trial_handler.task_paths.docker_compose_path
                        ),
                        reason="swesmith_trusted_commit_capture_failed",
                    )
                    raise
            elif self._data_source == "sweverified":
                try:
                    self._sweverified_baseline_commit = (
                        _snapshot_sweverified_workspace(
                            self._trial_handler.client_container_name,
                            task_meta,
                        )
                    )
                except Exception:
                    logger.exception(
                        "SWE-Verified workspace snapshot failed task=%s uid=%s",
                        self._task_spec.task_name,
                        self._run_ctx.uid,
                    )
                    _force_remove_docker_objects(
                        trial_name=self._trial_handler.trial_name,
                        client_container_name=self._trial_handler.client_container_name,
                        docker_image_name_prefix=self._trial_handler.docker_image_name_prefix,
                        docker_compose_path=str(
                            self._trial_handler.task_paths.docker_compose_path
                        ),
                        reason="sweverified_workspace_snapshot_failed",
                    )
                    raise

            try:
                _raise_if_cancelled("after_docker_start")
            except Exception:
                logger.warning(
                    "TerminalEnv reset cancelled after docker start task=%s uid=%s; "
                    "cleaning up container=%s",
                    self._task_spec.task_name,
                    self._run_ctx.uid,
                    self._trial_handler.client_container_name,
                )
                _force_remove_docker_objects(
                    trial_name=self._trial_handler.trial_name,
                    client_container_name=self._trial_handler.client_container_name,
                    docker_image_name_prefix=self._trial_handler.docker_image_name_prefix,
                    docker_compose_path=str(
                        self._trial_handler.task_paths.docker_compose_path
                    ),
                    reason="reset_cancelled_after_start",
                )
                raise
            session_logs_dir = (
                self._trial_handler.trial_paths.sessions_path
                / "terminal_toolkit_session_logs"
            )
            self._terminal_toolkit = TerminalToolkit(
                timeout=20.0,
                working_directory=(
                    "/testbed"
                    if is_swe_task_path(self._task_spec.task_path)
                    else None
                ),
                use_docker_backend=True,
                docker_container_name=self._trial_handler.client_container_name,
                session_logs_dir=session_logs_dir,
                safe_mode=False,
            )
            self._tools = {
                "shell_exec": self._terminal_toolkit.shell_exec,
                "shell_view": self._terminal_toolkit.shell_view,
                "shell_write_to_process": self._terminal_toolkit.shell_write_to_process,
                "shell_write_content_to_file": self._terminal_toolkit.shell_write_content_to_file,
            }

            user_msg = build_swe_user_message(
                task_name=self._task_spec.task_name,
                task_path=self._task_spec.task_path,
                instruction=self._task_spec.instruction,
            )
            function_tools = [FunctionTool(fn) for fn in self._tools.values()]
            tool_schemas = [
                func_tool.get_openai_tool_schema() for func_tool in function_tools
            ]
            return user_msg, tool_schemas

        # Keep a bounded wrapper around Docker/session startup, but leave enough
        # grace for slow compose starts after image preparation has completed.
        reset_thread_timeout = _env_float(
            "TERMINAL_ENV_RESET_THREAD_TIMEOUT",
            float(self._timeouts.reset_session) + 120.0,
        )
        reset_thread_task = asyncio.create_task(asyncio.to_thread(_sync_reset))
        try:
            return await asyncio.wait_for(
                asyncio.shield(reset_thread_task),
                timeout=reset_thread_timeout,
            )
        except asyncio.CancelledError:
            cancel_event.set()
            await _join_async_task(reset_thread_task)
            logger.warning(
                "TerminalEnv reset cancelled during docker/session startup task=%s "
                "uid=%s total_elapsed=%.1fs",
                self._task_spec.task_name,
                self._run_ctx.uid,
                time.monotonic() - reset_started,
            )
            raise
        except asyncio.TimeoutError:
            cancel_event.set()
            await _join_async_task(reset_thread_task)
            logger.error(
                "CRITICAL: reset operation hung beyond internal timeout "
                f"(timeout={reset_thread_timeout}s, reset_session={self._timeouts.reset_session}s). "
                "The reset thread was joined before returning failure."
            )
            raise TimeoutError(
                f"Reset operation exceeded timeout ({reset_thread_timeout}s). "
                "The reset was cancelled and joined."
            )

    async def exec_tool(self, name: str, arguments: dict[str, Any]) -> str:
        if self._agent_safetybench_env is not None:
            return await self._agent_safetybench_env.exec_tool(name, arguments)
        if self._agentharm_env is not None:
            return await self._agentharm_env.exec_tool(name, arguments)
        if self._tau2_env is not None:
            return await self._tau2_env.exec_tool(name, arguments)

        if not self._tools:
            raise RuntimeError("env is not initialized; call reset first")

        if name not in self._tools:
            return f"[TOOL_ERROR] unknown tool: {name}"

        fn = self._tools[name]

        try:
            if asyncio.iscoroutinefunction(fn):
                result = await fn(**arguments)
            elif hasattr(fn, "async_call") and callable(fn.async_call):
                result = await fn.async_call(**arguments)
            else:
                result = await asyncio.to_thread(partial(fn, **arguments))
        except Exception as exc:
            return f"[TOOL_ERROR] {name}: {type(exc).__name__}: {exc}"

        if isinstance(result, str):
            return result
        return json.dumps(result, ensure_ascii=False)

    async def handle_agent_reply(self, assistant_text: str) -> dict[str, Any]:
        if self._tau2_env is not None:
            return await self._tau2_env.handle_agent_reply(assistant_text)
        return {"continue": False, "user_message": ""}

    async def evaluate(self, trajectory: dict[str, Any] | None = None) -> float:
        if self._agent_safetybench_env is not None:
            return await self._agent_safetybench_env.evaluate(trajectory)
        if self._agentharm_env is not None:
            return await self._agentharm_env.evaluate(trajectory)
        if self._tau2_env is not None:
            return await self._tau2_env.evaluate(trajectory)

        if (
            self._trial_handler is None
            or self._terminal is None
            or self._parser is None
            or self._timeouts is None
        ):
            raise RuntimeError("env is not initialized; call reset first")

        is_sweverified = self._data_source == "sweverified"
        defer_sweverified = (
            is_sweverified
            and isinstance(trajectory, dict)
            and trajectory.get("swebench_defer_grading") is True
        )
        if is_sweverified and not defer_sweverified:
            raise RuntimeError(
                "SWE-Verified terminal-rl workers export predictions only. "
                "Set swebench_defer_grading=true and score predictions with "
                "the pinned official swebench.harness.run_evaluation."
            )

        def _sync_eval() -> float:
            DockerComposeManager = _lazy_import("DockerComposeManager")  # noqa: F811
            task_name = (
                self._task_spec.task_name if self._task_spec is not None else "unknown"
            )
            if is_sweverified:
                if self._task_meta is None or not self._sweverified_baseline_commit:
                    raise RuntimeError(
                        "SWE-Verified metadata/workspace baseline is unavailable"
                    )
                model_patch = _capture_sweverified_patch(
                    self._trial_handler.client_container_name,
                    self._sweverified_baseline_commit,
                )
                self._last_eval = {
                    "grader": "swebench_prediction_export",
                    "grading_deferred": True,
                    "swebench_version": self._task_meta.get(
                        "swebench_harness_version"
                    ),
                    "swebench_commit": self._task_meta.get(
                        "swebench_harness_commit"
                    ),
                    "instance_id": self._task_meta.get(
                        "swe_instance_id", task_name
                    ),
                    "repo": self._task_meta.get("repo"),
                    "version": self._task_meta.get("version"),
                    "model_patch": model_patch,
                    "patch_is_None": False,
                    "patch_exists": True,
                    "resolved": None,
                    "reward": 0.0,
                }
                logger.info(
                    "SWE-Verified prediction exported instance=%s bytes=%d",
                    self._last_eval["instance_id"],
                    len(model_patch),
                )
                return 0.0

            paths: list[Path] = [self._trial_handler.task_paths.run_tests_path]
            if self._trial_handler.task_paths.test_dir.exists():
                paths.append(self._trial_handler.task_paths.test_dir)

            self._terminal.copy_to_container(
                paths=paths,
                container_dir=str(DockerComposeManager.CONTAINER_TEST_DIR),
            )

            self._eval_attempt += 1
            run_uid = self._run_ctx.uid if self._run_ctx is not None else "unknown"
            test_session = self._terminal.create_session(
                f"tests-{run_uid}-{self._eval_attempt}",
                is_active_stream=False,
                as_configured_user=False,
            )
            test_script_path = str(
                DockerComposeManager.CONTAINER_TEST_DIR / "run-tests.sh"
            )
            test_timeout_sec = min(
                self._timeouts.eval,
                4 * self._trial_handler.task.max_test_timeout_sec,
            )
            trusted_commit_env = ""
            if self._data_source == "swesmith":
                if not self._swesmith_task_commit or not self._swesmith_bug_commit:
                    self._last_eval = {
                        "mode": "swesmith_tests",
                        "score": 0.0,
                        "reward_type": "all_tests_pass",
                        "reason": "trusted_commits_missing",
                        "task": task_name,
                    }
                    return 0.0
                trusted_commit_env = (
                    "SWESMITH_TRUSTED_TASK_COMMIT="
                    f"{shlex.quote(self._swesmith_task_commit)} "
                    "SWESMITH_TRUSTED_BUG_COMMIT="
                    f"{shlex.quote(self._swesmith_bug_commit)} "
                )
            try:
                test_session.send_keys(
                    [
                        f"{trusted_commit_env}bash {test_script_path}; "
                        "__terminal_rl_rc=$?; "
                        "echo __TERMINAL_RL_TEST_EXIT_CODE__=$__terminal_rl_rc",
                        "Enter",
                    ],
                    block=True,
                    max_timeout_sec=test_timeout_sec,
                )
            except TimeoutError as exc:
                logger.warning(
                    "Evaluation tests timed out for task=%s after %.1fs.",
                    task_name,
                    test_timeout_sec,
                )
                self._last_eval = {
                    "mode": "terminal_tests",
                    "score": 0.0,
                    "reason": "eval_timeout",
                    "task": task_name,
                    "timeout_sec": test_timeout_sec,
                    "error": str(exc),
                }
                return 0.0

            test_output = test_session.capture_pane(capture_entire=True)
            exit_matches = _TEST_EXIT_CODE_RE.findall(test_output or "")
            test_exit_code = int(exit_matches[-1]) if exit_matches else None
            if self._data_source == "swesmith" and test_exit_code != 0:
                self._last_eval = {
                    "mode": "swesmith_tests",
                    "score": 0.0,
                    "reward_type": "all_tests_pass",
                    "reason": (
                        "test_exit_missing"
                        if test_exit_code is None
                        else "test_exit_nonzero"
                    ),
                    "task": task_name,
                    "exit_code": test_exit_code,
                }
                return 0.0
            try:
                parser_results = self._parser.parse(test_output)
            except Exception as exc:
                tail = test_output[-2000:] if test_output else ""
                logger.warning(
                    "Failed to parse test output for task=%s with parser=%s: %s. Output tail:\n%s",
                    task_name,
                    type(self._parser).__name__,
                    exc,
                    tail,
                )
                # A recorded non-zero wrapper exit is a deterministic candidate
                # failure, not a grader-parser infrastructure failure. This
                # commonly happens when the candidate leaves apt/dpkg locked or
                # removes a test dependency. Preserve true parser failures for
                # exit=0 or a missing exit marker.
                if test_exit_code not in (None, 0):
                    self._last_eval = {
                        "mode": "terminal_tests",
                        "score": 0.0,
                        "reason": "test_exit_nonzero",
                        "task": task_name,
                        "parser": type(self._parser).__name__,
                        "exit_code": test_exit_code,
                        "parser_error": str(exc),
                        "output_tail": tail,
                    }
                    return 0.0
                self._last_eval = {
                    "mode": "terminal_tests",
                    "score": 0.0,
                    "reason": "eval_parse_failed",
                    "task": task_name,
                    "parser": type(self._parser).__name__,
                    "error": str(exc),
                    "exit_code": test_exit_code,
                    "output_tail": tail,
                }
                return 0.0

            if not parser_results:
                self._last_eval = {
                    "mode": "terminal_tests",
                    "score": 0.0,
                    "reason": "eval_no_results",
                    "task": task_name,
                    "parser": type(self._parser).__name__,
                    "total": 0,
                    "passed": 0,
                }
                return 0.0
            reward, passed = _terminal_test_reward(
                parser_results, self._data_source
            )
            if self._data_source == "swesmith":
                self._last_eval = {
                    "mode": "swesmith_tests",
                    "score": reward,
                    "reward_type": "all_tests_pass",
                    "task": task_name,
                    "total": len(parser_results),
                    "passed": passed,
                }
            else:
                self._last_eval = None
            return reward

        return await asyncio.wait_for(
            asyncio.to_thread(_sync_eval),
            timeout=self._timeouts.eval + 30.0,
        )

    def evaluation_depends_on_trajectory(self) -> bool:
        return (
            self._agent_safetybench_env is not None
            or self._agentharm_env is not None
            or self._data_source == "sweverified"
        )

    def last_eval_details(self) -> dict[str, Any] | None:
        if self._agent_safetybench_env is not None:
            details = getattr(self._agent_safetybench_env, "_last_eval", None)
            return details if isinstance(details, dict) else None
        if self._agentharm_env is not None:
            details = getattr(self._agentharm_env, "_last_eval", None)
            return details if isinstance(details, dict) else None
        if self._tau2_env is not None:
            details = getattr(self._tau2_env, "_last_eval", None)
            return details if isinstance(details, dict) else None
        return self._last_eval if isinstance(self._last_eval, dict) else None

    async def close(self) -> None:
        async with self._lifecycle_lock:
            await self._close_locked()

    async def _close_locked(self) -> None:
        trial_name = (
            self._trial_handler.trial_name
            if self._trial_handler is not None
            else self._last_trial_name or "unknown"
        )
        client_container_name = (
            self._trial_handler.client_container_name
            if self._trial_handler is not None
            else self._last_client_container_name
        )
        docker_image_name_prefix = (
            self._trial_handler.docker_image_name_prefix
            if self._trial_handler is not None
            else self._last_docker_image_name_prefix
        )
        docker_compose_path = (
            str(self._trial_handler.task_paths.docker_compose_path)
            if self._trial_handler is not None
            else self._last_docker_compose_path
        )
        if self._closed:
            logger.warning("TerminalEnv %s already closed", trial_name)
            return
        self._closed = True

        terminal = self._terminal
        timeouts = self._timeouts
        toolkit = self._terminal_toolkit
        agent_safetybench_env = self._agent_safetybench_env
        agentharm_env = self._agentharm_env
        tau2_env = self._tau2_env

        self._tools = {}
        self._terminal = None
        self._trial_handler = None
        self._parser = None
        self._terminal_toolkit = None
        self._task_spec = None
        self._task_meta = None
        self._run_ctx = None
        self._timeouts = None
        self._agent_safetybench_env = None
        self._agentharm_env = None
        self._tau2_env = None
        self._last_eval = None
        self._data_source = ""
        self._swesmith_task_commit = None
        self._swesmith_bug_commit = None
        self._sweverified_baseline_commit = None

        cleanup_completed = terminal is None
        cleanup_error = False
        fast_close = _env_bool("TERMINAL_ENV_FAST_CLOSE", False)
        force_cleanup_started = False
        force_cleanup_completed = False

        async def _run_force_cleanup(reason: str) -> None:
            nonlocal force_cleanup_started, force_cleanup_completed
            force_cleanup_started = True
            force_cleanup_completed = await _force_remove_docker_objects_async(
                trial_name=trial_name,
                client_container_name=client_container_name,
                docker_image_name_prefix=docker_image_name_prefix,
                docker_compose_path=docker_compose_path,
                reason=reason,
            )
            if not force_cleanup_completed:
                logger.error(
                    "Docker cleanup postcondition was not satisfied for "
                    "TerminalEnv %s (%s)",
                    trial_name,
                    reason,
                )

        try:
            if agent_safetybench_env is not None:
                try:
                    await agent_safetybench_env.close()
                except Exception:
                    logger.exception(
                        "Failed to cleanup Agent-SafetyBench env for %s", trial_name
                    )

            if agentharm_env is not None:
                try:
                    await agentharm_env.close()
                except Exception:
                    logger.exception(
                        "Failed to cleanup AgentHarm env for %s", trial_name
                    )

            if tau2_env is not None:
                try:
                    await tau2_env.close()
                except Exception:
                    logger.exception("Failed to cleanup tau2 env for %s", trial_name)

            if fast_close and terminal is not None:
                try:
                    await _run_force_cleanup("fast_close")
                except asyncio.CancelledError:
                    raise
                except Exception:
                    cleanup_error = True
                    logger.exception(
                        "Force Docker cleanup failed for TerminalEnv %s", trial_name
                    )

            if toolkit is not None:
                if fast_close:
                    logger.warning(
                        "Fast close enabled for %s; skipping TerminalToolkit.cleanup "
                        "and session drain; relying on direct Docker cleanup.",
                        trial_name,
                    )
                else:
                    try:
                        await asyncio.to_thread(toolkit.cleanup)
                    except Exception:
                        cleanup_error = True
                        logger.exception(
                            "Failed to cleanup terminal toolkit for %s", trial_name
                        )
                    try:
                        await asyncio.to_thread(_drain_toolkit_sessions, toolkit)
                    except Exception:
                        cleanup_error = True
                        logger.exception(
                            "Failed to drain toolkit sessions for %s", trial_name
                        )

            if terminal is not None and timeouts is not None:
                try:
                    close_timeout = timeouts.close_session
                    if fast_close:
                        close_timeout = min(
                            close_timeout,
                            _env_float("TERMINAL_ENV_FAST_CLOSE_STOP_TIMEOUT", 5.0),
                        )
                    namespace = _current_pool_namespace()
                    if (
                        namespace != "default"
                        and fast_close
                        and force_cleanup_completed
                    ):
                        cleanup_completed = True
                        logger.info(
                            "TerminalEnv %s fast cleanup completed; skipping "
                            "Terminal.stop",
                            trial_name,
                        )
                    else:
                        should_stop = True
                        if namespace != "default":
                            object_ref = client_container_name or trial_name
                            should_stop = await asyncio.to_thread(
                                _terminal_stop_ownership_verified,
                                object_ref,
                                timeout=min(close_timeout, 10.0),
                            )
                            if not should_stop:
                                cleanup_completed = True
                                logger.info(
                                    "TerminalEnv %s has no remaining owned "
                                    "container; skipping Terminal.stop",
                                    trial_name,
                                )
                        if should_stop:
                            await asyncio.to_thread(
                                _stop_terminal_compat, terminal, close_timeout
                            )
                            cleanup_completed = True
                            logger.info("TerminalEnv %s closed", trial_name)
                except Exception:
                    cleanup_error = True
                    logger.exception("Failed to stop terminal session during close")
        finally:
            force_always = _env_bool("TERMINAL_ENV_FORCE_DOCKER_CLEANUP_ALWAYS", False)
            compose_down_on_close = _env_bool("TERMINAL_ENV_COMPOSE_DOWN_ON_CLOSE", True)
            force_needed = (
                compose_down_on_close
                or force_always
                or fast_close
                or cleanup_error
                or not cleanup_completed
            )
            if terminal is not None and force_needed and (
                not force_cleanup_started or not force_cleanup_completed
            ):
                if fast_close:
                    reason = "fast_close"
                elif compose_down_on_close and cleanup_completed and not cleanup_error:
                    reason = "close_compose_down"
                elif force_always and cleanup_completed and not cleanup_error:
                    reason = "always"
                else:
                    reason = "close_incomplete"
                try:
                    await _run_force_cleanup(reason)
                except Exception:
                    logger.exception(
                        "Force Docker cleanup failed for TerminalEnv %s", trial_name
                    )
        if (
            terminal is not None
            and _current_pool_namespace() != "default"
            and force_needed
            and not force_cleanup_completed
        ):
            raise RuntimeError(
                f"Docker cleanup could not be verified for TerminalEnv {trial_name}"
            )

    async def force_cleanup(self, reason: str = "external") -> None:
        async with self._lifecycle_lock:
            cleaned = await _force_remove_docker_objects_async(
                trial_name=self._last_trial_name or "unknown",
                client_container_name=self._last_client_container_name,
                docker_image_name_prefix=self._last_docker_image_name_prefix,
                docker_compose_path=self._last_docker_compose_path,
                reason=reason,
            )
        if not cleaned:
            raise RuntimeError(
                "Docker force cleanup completed without satisfying its postcondition"
            )
