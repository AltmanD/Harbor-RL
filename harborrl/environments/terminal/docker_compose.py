from __future__ import annotations

import importlib
import logging
import os
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from terminal_bench.handlers.trial_handler import TrialHandler
    from terminal_bench.terminal.docker_compose_manager import DockerComposeManager
    from terminal_bench.terminal.terminal import Terminal

from harborrl.environments.terminal.validation import dockerfile_precheck_error

# Importing any terminal_bench submodule runs the package __init__, which pulls
# in the full harness stack (litellm/openai/mcp/boto3/streamlit/sqlalchemy) and
# takes tens of seconds on shared storage. These names are only used inside
# functions, so resolve them lazily to keep worker startup and preflight import
# checks fast.
_LAZY_IMPORT_MODULES = {
    "TrialHandler": "terminal_bench.handlers.trial_handler",
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

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ImagePreparationResult:
    mode: Literal["build", "pull"]
    client_image_name: str | None = None


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


_MAX_CONCURRENT_BUILDS = _env_int("WORKER_MAX_CONCURRENT_BUILDS", 4)
_BUILD_SEMAPHORE = threading.BoundedSemaphore(_MAX_CONCURRENT_BUILDS)
_BUILD_LOCKS_GUARD = threading.Lock()
_BUILD_LOCKS: dict[str, threading.Lock] = {}
_BUILD_DONE: set[str] = set()
_BUILD_FAILED: dict[str, tuple[float, str]] = {}
_BUILD_TRANSIENT_FAILED: dict[str, tuple[float, str]] = {}
_TASK_IMAGE_BLACKLISTED: dict[str, tuple[float, str]] = {}

class DockerImageBuildError(RuntimeError):
    """Deterministic task image build failure cached per task image."""


class DockerImageTransientBuildError(DockerImageBuildError):
    """Retryable registry/network build failure under a short cooldown."""


class DockerImagePreparationBacklogError(RuntimeError):
    """Transient image preparation backlog/cancellation before a build starts."""


class TaskImageBlacklistedError(DockerImageBuildError):
    """Task image is known-bad and should not hit Docker again until TTL expires."""


_BUILD_STATE_GUARD = threading.Lock()
_BUILD_ACTIVE = 0
_BUILD_WAITING = 0
_BUILD_BACKLOG_REJECTED = 0
_BUILD_CANCELLED = 0


def _safe_project_component(value: str, fallback: str = "task") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    cleaned = cleaned.strip("-.")
    return (cleaned or fallback)[:80]


def _shorten_output(text: str | None, max_chars: int = 4000) -> str:
    if not text:
        return ""
    stripped = text.strip()
    if len(stripped) <= max_chars:
        return stripped
    return f"{stripped[:max_chars]}...(truncated, total={len(stripped)} chars)"


def _docker_build_args_with_proxy(env: dict[str, str]) -> list[str]:
    """Pass configured proxy variables to BuildKit without exposing their values."""
    args = ["build"]
    for name in (
        "http_proxy",
        "https_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "no_proxy",
        "NO_PROXY",
    ):
        if env.get(name):
            args.extend(["--build-arg", name])
    return args


def _build_docker_pull_error_message(
    *,
    image: str,
    cmd: list[str],
    return_code: int | None = None,
    timeout: float | None = None,
    stdout: str | None = None,
    stderr: str | None = None,
) -> str:
    lines = [
        f"Docker image pull failed for image '{image}'.",
        f"Command: {' '.join(cmd)}",
    ]
    if return_code is not None:
        lines.append(f"Exit code: {return_code}")
    if timeout is not None:
        lines.append(f"Timeout: {timeout:.1f}s")

    out = _shorten_output(stdout)
    err = _shorten_output(stderr)
    if out:
        lines.append(f"STDOUT:\n{out}")
    if err:
        lines.append(f"STDERR:\n{err}")

    lines.append(
        "Hints: verify task_name/task image tag, run docker login for the registry, and ensure image exists."
    )
    return "\n".join(lines)


def _build_compose_up_error_message(
    *,
    cmd: list[str],
    return_code: int | None = None,
    timeout: float | None = None,
    stdout: str | None = None,
    stderr: str | None = None,
    note: str | None = None,
) -> str:
    lines = [
        "Docker compose up --no-build failed.",
        f"Command: {' '.join(cmd)}",
    ]
    if return_code is not None:
        lines.append(f"Exit code: {return_code}")
    if timeout is not None:
        lines.append(f"Timeout: {timeout:.1f}s")

    out = _shorten_output(stdout)
    err = _shorten_output(stderr)
    if out:
        lines.append(f"STDOUT:\n{out}")
    if err:
        lines.append(f"STDERR:\n{err}")
    if note:
        lines.append(f"Note: {note}")

    lines.append(
        "Hints: verify `docker compose version`; if unavailable, install Compose plugin or ensure `docker-compose` is on PATH."
    )
    return "\n".join(lines)


def _compose_plugin_maybe_missing(stderr: str | None) -> bool:
    if not stderr:
        return False
    lowered = stderr.lower()
    return (
        "unknown shorthand flag: 'p' in -p" in lowered
        or "docker: 'compose' is not a docker command" in lowered
        or 'unknown command "compose"' in lowered
    )


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value >= 0 else default


def docker_image_build_status() -> dict[str, int]:
    with _BUILD_STATE_GUARD:
        return {
            "max_concurrent_builds": _MAX_CONCURRENT_BUILDS,
            "active": _BUILD_ACTIVE,
            "waiting": _BUILD_WAITING,
            "backlog_rejected": _BUILD_BACKLOG_REJECTED,
            "cancelled_before_build": _BUILD_CANCELLED,
            "cached_done": len(_BUILD_DONE),
            "cached_failed": len(_BUILD_FAILED),
            "cached_transient_failed": len(_BUILD_TRANSIENT_FAILED),
            "blacklisted": len(_TASK_IMAGE_BLACKLISTED),
        }


def _record_build_waiting(delta: int) -> None:
    global _BUILD_WAITING
    with _BUILD_STATE_GUARD:
        _BUILD_WAITING = max(0, _BUILD_WAITING + delta)


def _record_build_active(delta: int) -> None:
    global _BUILD_ACTIVE
    with _BUILD_STATE_GUARD:
        _BUILD_ACTIVE = max(0, _BUILD_ACTIVE + delta)


def _record_build_backlog_rejected() -> None:
    global _BUILD_BACKLOG_REJECTED
    with _BUILD_STATE_GUARD:
        _BUILD_BACKLOG_REJECTED += 1


def _record_build_cancelled() -> None:
    global _BUILD_CANCELLED
    with _BUILD_STATE_GUARD:
        _BUILD_CANCELLED += 1


def _cancel_event_is_set(cancel_event: threading.Event | None) -> bool:
    return cancel_event is not None and cancel_event.is_set()


def _raise_if_cancelled(
    cancel_event: threading.Event | None, *, build_key: str, stage: str
) -> None:
    if _cancel_event_is_set(cancel_event):
        _record_build_cancelled()
        raise DockerImagePreparationBacklogError(
            f"DOCKER_IMAGE_PREP_CANCELLED image={build_key} stage={stage}"
        )


def _acquire_build_gate(
    gate: threading.Lock | threading.BoundedSemaphore,
    *,
    build_key: str,
    label: str,
    deadline: float | None,
    cancel_event: threading.Event | None,
) -> None:
    poll = max(0.05, min(1.0, _env_float("WORKER_DOCKER_BUILD_QUEUE_POLL", 0.5)))
    _record_build_waiting(1)
    try:
        while True:
            _raise_if_cancelled(cancel_event, build_key=build_key, stage=f"wait_{label}")
            timeout = poll
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    _record_build_backlog_rejected()
                    raise DockerImagePreparationBacklogError(
                        f"DOCKER_IMAGE_PREP_BACKLOG image={build_key} "
                        f"waiting_for={label} timeout={_build_queue_timeout():.1f}s"
                    )
                timeout = min(timeout, remaining)
            if gate.acquire(timeout=timeout):
                return
    finally:
        _record_build_waiting(-1)


def _build_queue_timeout() -> float:
    return _env_float("WORKER_DOCKER_BUILD_QUEUE_TIMEOUT", 90.0)


def _build_failed_ttl() -> float:
    raw = os.getenv("WORKER_DOCKER_BUILD_FAILED_TTL", "3600")
    try:
        value = float(raw)
    except ValueError:
        return 3600.0
    return max(0.0, value)


def _build_transient_failed_ttl() -> float:
    raw = os.getenv("WORKER_DOCKER_BUILD_TRANSIENT_TTL", "60")
    try:
        value = float(raw)
    except ValueError:
        return 60.0
    return max(0.0, value)


def _task_image_blacklist_ttl() -> float:
    raw = os.getenv("WORKER_DOCKER_TASK_BLACKLIST_TTL", "86400")
    try:
        value = float(raw)
    except ValueError:
        return 86400.0
    return max(0.0, value)


def _cached_build_failure(build_key: str) -> str | None:
    ttl = _build_failed_ttl()
    if ttl <= 0:
        return None
    cached = _BUILD_FAILED.get(build_key)
    if cached is None:
        return None
    ts, message = cached
    if time.time() - ts <= ttl:
        return message
    _BUILD_FAILED.pop(build_key, None)
    return None


def _cached_transient_build_failure(build_key: str) -> str | None:
    ttl = _build_transient_failed_ttl()
    if ttl <= 0:
        return None
    cached = _BUILD_TRANSIENT_FAILED.get(build_key)
    if cached is None:
        return None
    ts, message = cached
    if time.time() - ts <= ttl:
        return message
    _BUILD_TRANSIENT_FAILED.pop(build_key, None)
    return None


def _cached_task_image_blacklist(build_key: str) -> str | None:
    ttl = _task_image_blacklist_ttl()
    if ttl <= 0:
        return None
    cached = _TASK_IMAGE_BLACKLISTED.get(build_key)
    if cached is None:
        return None
    ts, message = cached
    if time.time() - ts <= ttl:
        return message
    _TASK_IMAGE_BLACKLISTED.pop(build_key, None)
    return None


def _blacklist_task_image(build_key: str, message: str) -> None:
    if _task_image_blacklist_ttl() <= 0:
        return
    _TASK_IMAGE_BLACKLISTED[build_key] = (time.time(), message)


def _is_deterministic_build_failure(message: str) -> bool:
    lowered = message.lower()
    deterministic_markers = (
        "dockerfile parse error",
        "unknown instruction:",
        "failed to read dockerfile",
        "cannot locate specified dockerfile",
        "no such file or directory",
        "yaml: line",
        "services must be a mapping",
    )
    return any(marker in lowered for marker in deterministic_markers)


def _is_transient_build_failure(message: str) -> bool:
    """Recognize registry/network failures that must never poison a task image."""
    lowered = message.lower()
    transient_markers = (
        "429 too many requests",
        "toomanyrequests",
        "rate limit",
        "tls handshake timeout",
        "connection reset by peer",
        "connection refused",
        "temporary failure in name resolution",
        "no such host",
        "context deadline exceeded",
        "i/o timeout",
        "unexpected eof",
        "service unavailable",
        "bad gateway",
        "gateway timeout",
    )
    if any(marker in lowered for marker in transient_markers):
        return True
    return bool(re.search(r"(?:head|http|https|get).*\b(?:502|503|504)\b", lowered))


def _dockerfile_precheck_error(task_path: Path) -> str | None:
    if not _env_bool("WORKER_DOCKERFILE_PRECHECK", True):
        return None
    return dockerfile_precheck_error(task_path / "Dockerfile")


def _lock_for_build_key(key: str) -> threading.Lock:
    with _BUILD_LOCKS_GUARD:
        lock = _BUILD_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _BUILD_LOCKS[key] = lock
        return lock


def build_docker_image(
    task: dict[str, Any],
    timeout: float = 1200.0,
    *,
    cancel_event: threading.Event | None = None,
) -> None:
    dataset_dir = str(os.getenv("DATASET_DIR", "")).strip()
    if not dataset_dir:
        raise ValueError("DATASET_DIR is required")

    task_path = Path(dataset_dir) / str(task.get("task_path", ""))
    task_name = _safe_project_component(str(task.get("task_name") or task_path.name))
    # Bare references do not trigger the module-level lazy __getattr__; resolve
    # through the helper (same-named locals shadow the globals).
    TrialHandler = _lazy_import("TrialHandler")  # noqa: F811
    DockerComposeManager = _lazy_import("DockerComposeManager")  # noqa: F811
    trial_handler = TrialHandler(
        trial_name=f"build_run_{task_name}",
        input_path=task_path,
        output_path=Path(
            os.getenv("TBENCH_OUTPUT_ROOT", "runs/env-worker/task_outputs")
        ),
    )

    compose_manager = DockerComposeManager(
        client_container_name=trial_handler.client_container_name,
        client_image_name=trial_handler.client_image_name,
        docker_image_name_prefix=trial_handler.docker_image_name_prefix,
        docker_compose_path=trial_handler.task_paths.docker_compose_path,
        no_rebuild=True,
        cleanup=False,
        sessions_logs_path=trial_handler.trial_paths.sessions_path,
        agent_logs_path=trial_handler.trial_paths.agent_logging_dir,
    )
    build_key = trial_handler.client_image_name
    build_dedup = _env_bool("WORKER_DOCKER_BUILD_DEDUP", True)
    skip_existing = _env_bool("WORKER_DOCKER_BUILD_SKIP_EXISTING", True)
    build_lock = _lock_for_build_key(build_key) if build_dedup else threading.Lock()
    queue_timeout = _build_queue_timeout()
    deadline = time.monotonic() + queue_timeout if queue_timeout > 0 else None
    build_lock_acquired = False
    build_semaphore_acquired = False

    _raise_if_cancelled(cancel_event, build_key=build_key, stage="before_lock")
    _acquire_build_gate(
        build_lock,
        build_key=build_key,
        label="image_lock",
        deadline=deadline,
        cancel_event=cancel_event,
    )
    build_lock_acquired = True
    try:
        _raise_if_cancelled(cancel_event, build_key=build_key, stage="after_lock")
        if build_key in _BUILD_DONE:
            return
        blacklisted_failure = _cached_task_image_blacklist(build_key)
        if blacklisted_failure is not None:
            raise TaskImageBlacklistedError(
                f"TASK_IMAGE_BLACKLISTED image={build_key}: {blacklisted_failure}"
            )
        cached_failure = _cached_build_failure(build_key)
        if cached_failure is not None:
            raise DockerImageBuildError(
                f"TASK_BUILD_FAILED cached for image={build_key}: {cached_failure}"
            )
        if skip_existing and _docker_image_exists_locally(build_key):
            logger.debug("Docker image already exists locally; skip build: %s", build_key)
            _BUILD_DONE.add(build_key)
            _BUILD_FAILED.pop(build_key, None)
            _BUILD_TRANSIENT_FAILED.pop(build_key, None)
            _TASK_IMAGE_BLACKLISTED.pop(build_key, None)
            return
        cached_transient_failure = _cached_transient_build_failure(build_key)
        if cached_transient_failure is not None:
            raise DockerImageTransientBuildError(
                "TASK_BUILD_TRANSIENT cached cooldown "
                f"for image={build_key}: {cached_transient_failure}"
            )
        precheck_error = _dockerfile_precheck_error(task_path)
        if precheck_error is not None:
            message = (
                f"TASK_DOCKERFILE_PRECHECK_FAILED task={task_name}: {precheck_error}"
            )
            _BUILD_FAILED[build_key] = (time.time(), message)
            _BUILD_TRANSIENT_FAILED.pop(build_key, None)
            _blacklist_task_image(build_key, message)
            raise TaskImageBlacklistedError(
                f"TASK_IMAGE_BLACKLISTED image={build_key}: {message}"
            )

        _acquire_build_gate(
            _BUILD_SEMAPHORE,
            build_key=build_key,
            label="global_build_semaphore",
            deadline=deadline,
            cancel_event=cancel_event,
        )
        build_semaphore_acquired = True
        _record_build_active(1)
        try:
            _raise_if_cancelled(cancel_event, build_key=build_key, stage="before_build")
            logger.info("Building Docker image for task=%s image=%s", task_name, build_key)
            try:
                for uppercase, lowercase in (
                    ("HTTP_PROXY", "http_proxy"),
                    ("HTTPS_PROXY", "https_proxy"),
                    ("NO_PROXY", "no_proxy"),
                ):
                    value = os.getenv(uppercase) or os.getenv(lowercase) or ""
                    if value:
                        compose_manager.env[uppercase] = value
                        compose_manager.env[lowercase] = value
                command = compose_manager.get_docker_compose_command(
                    _docker_build_args_with_proxy(compose_manager.env)
                )
                subprocess.run(
                    command,
                    env=compose_manager.env,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
            except Exception as exc:
                message = _shorten_output(str(exc), max_chars=1200) or type(exc).__name__
                if isinstance(exc, subprocess.CalledProcessError):
                    # str(exc) only carries the exit status; surface the build's
                    # stderr tail so the cached failure is self-diagnosing.
                    detail = _shorten_output(exc.stderr or exc.stdout, max_chars=1200)
                    if detail:
                        message = f"{message} | output: {detail}"
                if _is_deterministic_build_failure(message):
                    _BUILD_FAILED[build_key] = (time.time(), message)
                    _BUILD_TRANSIENT_FAILED.pop(build_key, None)
                    _blacklist_task_image(build_key, message)
                    raise DockerImageBuildError(
                        f"TASK_BUILD_FAILED image={build_key} task={task_name}: {message}"
                    ) from exc
                # Unknown build errors are treated as retryable. Deterministic
                # failures are explicitly recognized above; poisoning an image
                # for an hour is more harmful than a bounded retry here.
                _BUILD_TRANSIENT_FAILED[build_key] = (time.time(), message)
                marker = (
                    "TASK_BUILD_TRANSIENT"
                    if _is_transient_build_failure(message)
                    else "TASK_BUILD_RETRYABLE"
                )
                raise DockerImageTransientBuildError(
                    f"{marker} image={build_key} task={task_name}: {message}"
                ) from exc
            _BUILD_DONE.add(build_key)
            _BUILD_FAILED.pop(build_key, None)
            _BUILD_TRANSIENT_FAILED.pop(build_key, None)
            _TASK_IMAGE_BLACKLISTED.pop(build_key, None)
        finally:
            _record_build_active(-1)
            if build_semaphore_acquired:
                _BUILD_SEMAPHORE.release()
    finally:
        if build_lock_acquired:
            build_lock.release()


def _resolve_pull_image(task: dict[str, Any]) -> str:
    prefix = str(os.getenv("TBENCH_DOCKER_PULL_PREFIX", "")).strip()
    if not prefix:
        raise ValueError("TBENCH_DOCKER_PULL_PREFIX is required in pull mode")

    task_name = str(task.get("task_name", "")).strip()
    if not task_name:
        raise ValueError("task_name is required to resolve pull image")
    if "<" in task_name and ">" in task_name:
        raise ValueError(
            "task_name appears to still be a placeholder "
            f"('{task_name}'). Please provide a concrete task_name."
        )

    return f"{prefix}{task_name}"


def _docker_image_exists_locally(image: str, timeout: float = 30.0) -> bool:
    cmd = ["docker", "image", "inspect", image]
    result = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result.returncode == 0


def pull_docker_image(image: str, timeout: float = 1200.0) -> None:
    if _docker_image_exists_locally(image):
        return

    cmd = ["docker", "pull", image]
    try:
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            _build_docker_pull_error_message(
                image=image,
                cmd=cmd,
                timeout=timeout,
                stdout=exc.stdout,
                stderr=exc.stderr,
            )
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            _build_docker_pull_error_message(
                image=image,
                cmd=cmd,
                return_code=exc.returncode,
                stdout=exc.stdout,
                stderr=exc.stderr,
            )
        ) from exc


def prepare_task_docker_image(
    task: dict[str, Any],
    timeout: float = 1200.0,
    *,
    cancel_event: threading.Event | None = None,
) -> ImagePreparationResult:
    raw_mode = str(os.getenv("TBENCH_DOCKER_IMAGE_SOURCE", "")).strip().lower()
    if not raw_mode:
        raise ValueError("TBENCH_DOCKER_IMAGE_SOURCE is required")

    if raw_mode in {"build", "docker_build"}:
        build_docker_image(task=task, timeout=timeout, cancel_event=cancel_event)
        return ImagePreparationResult(mode="build", client_image_name=None)

    if raw_mode in {"pull", "docker_pull"}:
        task_name = str(task.get("task_name") or task.get("task_path") or "unknown")
        _raise_if_cancelled(cancel_event, build_key=task_name, stage="before_pull")
        image = _resolve_pull_image(task=task)
        pull_docker_image(image=image, timeout=timeout)
        _raise_if_cancelled(cancel_event, build_key=image, stage="after_pull")
        return ImagePreparationResult(mode="pull", client_image_name=image)

    raise ValueError(
        f"Unsupported docker image source '{raw_mode}'. Expected one of: build, pull"
    )


_DEFAULT_CONTAINER_MEMORY_LIMIT = os.getenv("CONTAINER_MEMORY_LIMIT", "16g")
_DEFAULT_CONTAINER_PIDS_LIMIT = os.getenv("CONTAINER_PIDS_LIMIT", "512")


def _apply_container_memory_limit(
    container_name: str,
    memory_limit: str,
    logger: logging.Logger | None = None,
) -> None:
    """Best-effort ``docker update --memory`` on a running container."""
    if not memory_limit:
        return
    cmd = [
        "docker",
        "update",
        f"--memory={memory_limit}",
        f"--memory-swap={memory_limit}",
        container_name,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=30.0)
        if logger is not None:
            logger.info(
                "Applied memory limit %s to container %s", memory_limit, container_name
            )
    except Exception as exc:
        if logger is not None:
            logger.warning(
                "Failed to apply memory limit to container %s: %s", container_name, exc
            )


def _apply_container_pids_limit(
    container_name: str,
    pids_limit: str,
    logger: logging.Logger | None = None,
) -> None:
    """Best-effort ``docker update --pids-limit`` on a running container."""
    if not pids_limit:
        return
    try:
        value = int(str(pids_limit).strip())
    except ValueError:
        if logger is not None:
            logger.warning(
                "Invalid CONTAINER_PIDS_LIMIT=%r; skipping pids limit for %s",
                pids_limit,
                container_name,
            )
        return
    if value <= 0:
        return

    cmd = ["docker", "update", f"--pids-limit={value}", container_name]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=30.0)
        if logger is not None:
            logger.info("Applied pids limit %s to container %s", value, container_name)
    except Exception as exc:
        if logger is not None:
            logger.warning(
                "Failed to apply pids limit to container %s: %s", container_name, exc
            )


def _apply_container_runtime_limits(
    container_name: str,
    logger: logging.Logger | None = None,
) -> None:
    _apply_container_pids_limit(
        container_name, _DEFAULT_CONTAINER_PIDS_LIMIT, logger=logger
    )
    _apply_container_memory_limit(
        container_name, _DEFAULT_CONTAINER_MEMORY_LIMIT, logger=logger
    )


def compose_up_no_build(
    terminal: Terminal,
    *,
    timeout: float,
    container_name: str,
    logger: logging.Logger | None = None,
) -> None:
    compose_manager = getattr(terminal, "_compose_manager")
    compose_override_path = str(os.getenv("COMPOSE_OVERRIDE_PATH", "")).strip()
    compose_command = ["up", "-d", "--no-build"]
    if compose_override_path:
        compose_command = ["-f", compose_override_path, *compose_command]

    command = compose_manager.get_docker_compose_command(compose_command)
    if logger is not None:
        logger.info("Running docker compose command: %s", " ".join(command))
        if compose_override_path:
            logger.info("Using compose override file: %s", compose_override_path)

    compose_manager.env["http_proxy"] = os.getenv("HTTP_PROXY", "")
    compose_manager.env["https_proxy"] = os.getenv("HTTPS_PROXY", "")

    try:
        subprocess.run(
            command,
            env=compose_manager.env,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        if logger is not None:
            logger.error(
                "Docker compose up --no-build timed out after %.1f sec", timeout
            )
            if exc.stdout:
                logger.error("STDOUT: %s", exc.stdout)
            if exc.stderr:
                logger.error("STDERR: %s", exc.stderr)
        raise RuntimeError(
            _build_compose_up_error_message(
                cmd=command,
                timeout=timeout,
                stdout=exc.stdout,
                stderr=exc.stderr,
            )
        ) from exc
    except subprocess.CalledProcessError as exc:
        if logger is not None:
            logger.error(
                "Docker compose up --no-build failed with code %s", exc.returncode
            )
            if exc.stdout:
                logger.error("STDOUT: %s", exc.stdout)
            if exc.stderr:
                logger.error("STDERR: %s", exc.stderr)
        if _compose_plugin_maybe_missing(exc.stderr):
            fallback_command = ["docker-compose", *command[2:]]
            if shutil.which("docker-compose"):
                if logger is not None:
                    logger.warning(
                        "docker compose plugin may be unavailable; falling back to: %s",
                        " ".join(fallback_command),
                    )
                try:
                    subprocess.run(
                        fallback_command,
                        env=compose_manager.env,
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=timeout,
                    )
                except subprocess.TimeoutExpired as fallback_exc:
                    raise RuntimeError(
                        _build_compose_up_error_message(
                            cmd=fallback_command,
                            timeout=timeout,
                            stdout=fallback_exc.stdout,
                            stderr=fallback_exc.stderr,
                            note=f"Initial command {' '.join(command)} failed with exit code {exc.returncode}.",
                        )
                    ) from fallback_exc
                except subprocess.CalledProcessError as fallback_exc:
                    raise RuntimeError(
                        _build_compose_up_error_message(
                            cmd=fallback_command,
                            return_code=fallback_exc.returncode,
                            stdout=fallback_exc.stdout,
                            stderr=fallback_exc.stderr,
                            note=f"Initial command {' '.join(command)} failed with exit code {exc.returncode}.",
                        )
                    ) from fallback_exc
                client = compose_manager._client
                if hasattr(client.api, "timeout"):
                    client.api.timeout = max(1.0, min(float(timeout), 30.0))
                container = client.containers.get(container_name)
                terminal.container = container
                compose_manager._client_container = container
                _apply_container_runtime_limits(container_name, logger=logger)
                return

            raise RuntimeError(
                _build_compose_up_error_message(
                    cmd=command,
                    return_code=exc.returncode,
                    stdout=exc.stdout,
                    stderr=exc.stderr,
                    note="Compose plugin appears unavailable and `docker-compose` binary was not found.",
                )
            ) from exc

        raise RuntimeError(
            _build_compose_up_error_message(
                cmd=command,
                return_code=exc.returncode,
                stdout=exc.stdout,
                stderr=exc.stderr,
            )
        ) from exc

    client = compose_manager._client
    if hasattr(client.api, "timeout"):
        client.api.timeout = max(1.0, min(float(timeout), 30.0))
    container = client.containers.get(container_name)
    terminal.container = container
    compose_manager._client_container = container
    _apply_container_runtime_limits(container_name, logger=logger)
