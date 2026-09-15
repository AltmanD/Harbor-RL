from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

from harborrl.types import RunContext, TaskSpec, TaskTimeouts

logger = logging.getLogger("lightrl.env.worker.admission")

_DOCKER_CLI_FAIL_STREAK = 0
_DOCKER_DEGRADED_UNTIL = 0.0
_DOCKER_DEGRADED_REASON = ""
_DISK_PRESSURE_WARN_LAST = 0.0


def _parse_timeout_overrides(
    base: TaskTimeouts, payload: dict[str, Any] | None
) -> TaskTimeouts:
    if not isinstance(payload, dict):
        return base

    def _pick(key: str, default: float, *, minimum: float | None = None) -> float:
        raw = payload.get(key, default)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return default
        if value <= 0:
            return default
        if minimum is not None and value < minimum:
            logger.debug(
                "Raising client timeout override %s=%.1fs to worker floor %.1fs",
                key,
                value,
                minimum,
            )
            return minimum
        return value

    return TaskTimeouts(
        ensure_image=_pick(
            "ensure_image",
            base.ensure_image,
            minimum=base.ensure_image,
        ),
        reset_session=_pick(
            "reset_session",
            base.reset_session,
            minimum=base.reset_session,
        ),
        close_session=_pick("close_session", base.close_session),
        eval=_pick("eval", base.eval),
    )


def _build_task_spec(task_meta: dict[str, Any]) -> TaskSpec:
    return TaskSpec(
        task_name=str(task_meta.get("task_name", "unknown")),
        task_path=str(task_meta.get("task_path", "")),
        instruction=str(task_meta.get("instruction", "")),
    )


def _build_run_ctx(
    run_ctx_payload: dict[str, Any] | None, default_log_dir: Path
) -> RunContext:
    payload = run_ctx_payload if isinstance(run_ctx_payload, dict) else {}
    uid = str(payload.get("uid") or uuid.uuid4().hex[:8])
    try:
        group_index = int(payload.get("group_index") or 0)
    except (TypeError, ValueError):
        group_index = 0
    try:
        sample_index = int(payload.get("sample_index") or 0)
    except (TypeError, ValueError):
        sample_index = 0

    log_dir_raw = payload.get("log_dir")
    if isinstance(log_dir_raw, str) and log_dir_raw:
        log_dir = Path(log_dir_raw).resolve()
    else:
        log_dir = default_log_dir.resolve()

    return RunContext(
        uid=uid,
        group_index=group_index,
        sample_index=sample_index,
        log_dir=log_dir,
    )


class CapacityError(Exception):
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


class ResourcePressureError(Exception):
    def __init__(self, code: str, message: str, details: dict[str, Any]):
        self.code = code
        self.message = message
        self.details = details
        super().__init__(message)


class ResetInProgressError(Exception):
    def __init__(self, run_lease_id: str, request_id: str | None):
        self.run_lease_id = run_lease_id
        self.request_id = request_id
        super().__init__(
            f"Run {run_lease_id} already has a different reset in progress"
        )


class ResetAdmissionBacklogError(Exception):
    def __init__(self, run_lease_id: str, timeout: float, max_concurrent: int):
        self.run_lease_id = run_lease_id
        self.timeout = timeout
        self.max_concurrent = max_concurrent
        super().__init__(
            f"WORKER_RESET_ADMISSION_BACKLOG lease_id={run_lease_id} "
            f"timeout={timeout:.1f}s max_concurrent_resets={max_concurrent}"
        )


from harborrl.env import env_float as _env_float, env_int as _env_int


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _mark_docker_degraded(reason: str) -> None:
    global _DOCKER_DEGRADED_REASON, _DOCKER_DEGRADED_UNTIL
    cooldown = max(0.0, _env_float("WORKER_DOCKER_DEGRADED_COOLDOWN", 120.0))
    if cooldown <= 0:
        return
    _DOCKER_DEGRADED_REASON = reason
    _DOCKER_DEGRADED_UNTIL = max(_DOCKER_DEGRADED_UNTIL, time.time() + cooldown)


def _record_docker_cli_probe(ok: bool, *, timeout: float) -> None:
    global _DOCKER_CLI_FAIL_STREAK, _DOCKER_DEGRADED_REASON, _DOCKER_DEGRADED_UNTIL
    if ok:
        _DOCKER_CLI_FAIL_STREAK = 0
        if time.time() >= _DOCKER_DEGRADED_UNTIL:
            _DOCKER_DEGRADED_REASON = ""
        return
    _DOCKER_CLI_FAIL_STREAK += 1
    threshold = max(1, _env_int("WORKER_DOCKER_DEGRADED_FAIL_STREAK", 2))
    if _DOCKER_CLI_FAIL_STREAK >= threshold:
        _mark_docker_degraded(
            f"docker CLI probe failed {_DOCKER_CLI_FAIL_STREAK} consecutive "
            f"time(s), timeout={timeout:.1f}s"
        )


def _docker_degraded_details() -> dict[str, Any] | None:
    remaining = _DOCKER_DEGRADED_UNTIL - time.time()
    if remaining <= 0:
        return None
    return {
        "docker_degraded_remaining_sec": round(remaining, 1),
        "docker_degraded_reason": _DOCKER_DEGRADED_REASON,
        "docker_cli_fail_streak": _DOCKER_CLI_FAIL_STREAK,
    }


def _split_env_csv(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


_TASK_ID_PREFIX_RE = re.compile(r"^([0-9]+)(?:[-_.:]|$)")
_FIXED_TASK_SERVICE_RE = re.compile(r"^tb__([0-9]+)__.*")


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


def _task_key_tokens(task_key: str) -> set[str]:
    raw = str(task_key or "").strip()
    tokens = {raw} if raw else set()
    if ":" in raw:
        task_name, task_path = raw.split(":", 1)
        if task_name:
            tokens.add(task_name)
        if task_path:
            tokens.add(task_path)
            tail = Path(task_path).name
            if tail:
                tokens.add(tail)
    task_id = _task_id_from_ref(raw)
    if task_id:
        tokens.add(task_id)
    return {token for token in tokens if token}


def _parse_task_max_runs_overrides(raw: str | None) -> dict[str, int]:
    overrides: dict[str, int] = {}
    for item in _split_env_csv(raw):
        if "=" not in item:
            logger.warning(
                "Ignoring malformed WORKER_TASK_MAX_RUNS_OVERRIDES entry %r; "
                "expected task=limit",
                item,
            )
            continue
        key, value_raw = item.split("=", 1)
        key = key.strip()
        if not key:
            continue
        try:
            value = int(value_raw.strip())
        except ValueError:
            logger.warning(
                "Ignoring invalid WORKER_TASK_MAX_RUNS_OVERRIDES entry %r", item
            )
            continue
        if value <= 0:
            logger.warning(
                "Ignoring non-positive WORKER_TASK_MAX_RUNS_OVERRIDES entry %r", item
            )
            continue
        overrides[key] = value
    return overrides


def docker_data_root_stats() -> dict[str, Any]:
    path = os.getenv("DOCKER_DATA_ROOT") or os.getenv("DOCKER_ROOT") or "/data"
    usage = shutil.disk_usage(path)
    st = os.statvfs(path)
    total_inodes = int(st.f_files)
    free_inodes = int(st.f_ffree)
    used_inodes = max(total_inodes - free_inodes, 0)
    used_pct = (usage.used * 100.0 / usage.total) if usage.total else 0.0
    inode_used_pct = (
        (used_inodes * 100.0 / total_inodes) if total_inodes else 0.0
    )
    return {
        "path": path,
        "total_gb": usage.total / 1024**3,
        "used_gb": usage.used / 1024**3,
        "free_gb": usage.free / 1024**3,
        "used_pct": used_pct,
        "total_inodes": total_inodes,
        "used_inodes": used_inodes,
        "free_inodes": free_inodes,
        "inode_used_pct": inode_used_pct,
    }


_PRESSURE_CACHE: tuple[float, dict[str, Any]] | None = None


def _read_cgroup_pids_stats() -> dict[str, Any] | None:
    try:
        lines = Path("/proc/self/cgroup").read_text().splitlines()
    except OSError:
        return None

    search_roots: list[Path] = []
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        controllers = parts[1]
        rel = parts[2].strip("/")
        if controllers == "":
            search_roots.append(Path("/sys/fs/cgroup") / rel)
        elif "pids" in controllers.split(","):
            search_roots.append(Path("/sys/fs/cgroup/pids") / rel)
            search_roots.append(Path("/sys/fs/cgroup") / rel)

    for start in search_roots:
        cur = start
        while True:
            current_file = cur / "pids.current"
            max_file = cur / "pids.max"
            if current_file.is_file() and max_file.is_file():
                try:
                    current = int(current_file.read_text().strip())
                    raw_max = max_file.read_text().strip()
                    if raw_max == "max":
                        return None
                    maximum = int(raw_max)
                except (OSError, ValueError):
                    return None
                if maximum > 0:
                    return {
                        "pids_current": current,
                        "pids_max": maximum,
                        "pids_source": str(cur),
                    }
                return None
            if cur == cur.parent:
                break
            cur = cur.parent
    return None


def _read_proc_pressure_stats() -> dict[str, Any]:
    total_procs = 0
    total_tasks = 0
    zombies = 0
    shim = 0
    runc = 0
    dockerd = 0
    containerd = 0
    docker_cli = 0

    for proc_dir in Path("/proc").glob("[0-9]*"):
        if not proc_dir.is_dir():
            continue
        total_procs += 1
        try:
            name = (proc_dir / "comm").read_text(errors="ignore").strip()
        except OSError:
            name = ""
        try:
            stat = (proc_dir / "stat").read_text(errors="ignore")
            rest = stat.split(") ", 1)[1]
            if rest.split(" ", 1)[0] == "Z":
                zombies += 1
        except (OSError, IndexError):
            pass
        try:
            total_tasks += sum(1 for p in (proc_dir / "task").iterdir() if p.is_dir())
        except OSError:
            pass

        if name == "dockerd":
            dockerd += 1
        elif name == "containerd":
            containerd += 1
        elif name.startswith("containerd-shim"):
            shim += 1
        elif name == "runc":
            runc += 1
        elif name == "docker":
            docker_cli += 1

    pids_max = 0
    try:
        pids_max = int(Path("/proc/sys/kernel/threads-max").read_text().strip())
    except (OSError, ValueError):
        pids_max = 0
    pids_current = total_tasks
    pids_source = "/proc"
    cgroup_pids = _read_cgroup_pids_stats()
    if cgroup_pids is not None:
        cgroup_max = int(cgroup_pids.get("pids_max") or 0)
        if cgroup_max > 0 and (pids_max <= 0 or cgroup_max <= pids_max):
            pids_current = int(cgroup_pids.get("pids_current") or total_tasks)
            pids_max = cgroup_max
            pids_source = str(cgroup_pids.get("pids_source") or "cgroup")

    pids_pct = (pids_current * 100.0 / pids_max) if pids_max > 0 else 0.0
    return {
        "procs": total_procs,
        "tasks": total_tasks,
        "pids_current": pids_current,
        "pids_max": pids_max,
        "pids_pct": pids_pct,
        "pids_source": pids_source,
        "zombies": zombies,
        "dockerd": dockerd,
        "containerd": containerd,
        "shim": shim,
        "runc": runc,
        "docker_cli_procs": docker_cli,
    }


def _docker_cli_ok(timeout_sec: float) -> bool:
    try:
        result = subprocess.run(
            ["docker", "ps", "-q"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout_sec,
        )
        return result.returncode == 0
    except Exception:
        return False


def worker_pressure_stats(*, force: bool = False) -> dict[str, Any]:
    global _PRESSURE_CACHE
    ttl = _env_float("WORKER_PRESSURE_CACHE_TTL", 5.0)
    now = time.time()
    if (
        not force
        and _PRESSURE_CACHE is not None
        and now - _PRESSURE_CACHE[0] <= ttl
    ):
        return dict(_PRESSURE_CACHE[1])

    stats = _read_proc_pressure_stats()
    docker_timeout = _env_float("WORKER_DOCKER_CLI_TIMEOUT", 3.0)
    docker_cli_ok = _docker_cli_ok(docker_timeout)
    _record_docker_cli_probe(docker_cli_ok, timeout=docker_timeout)
    stats["docker_cli_ok"] = docker_cli_ok
    stats["docker_cli_timeout_sec"] = docker_timeout
    degraded = _docker_degraded_details()
    if degraded is not None:
        stats.update(degraded)
    _PRESSURE_CACHE = (now, dict(stats))
    return stats


def assert_worker_has_capacity_for_docker(
    *,
    phase: str = "health",
    pending_closes: int = 0,
    pool_status: dict[str, Any] | None = None,
) -> None:
    if os.getenv("WORKER_DISK_GUARD_ENABLED", "1") == "0":
        disk_guard_enabled = False
    else:
        disk_guard_enabled = True

    if disk_guard_enabled:
        min_free_gb = _env_float("WORKER_MIN_DOCKER_FREE_GB", 50.0)
        max_used_pct = _env_float("WORKER_MAX_DOCKER_USED_PCT", 95.0)
        max_inode_pct = _env_float("WORKER_MAX_DOCKER_INODE_PCT", 80.0)

        try:
            stats = docker_data_root_stats()
        except Exception as exc:
            raise ResourcePressureError(
                "WORKER_DISK_STATS_FAILED",
                f"Failed to read Docker data-root stats: {exc}",
                {"error": str(exc), "phase": phase},
            ) from exc

        over_capacity = (
            stats["free_gb"] < min_free_gb
            or stats["used_pct"] > max_used_pct
            or stats["inode_used_pct"] > max_inode_pct
        )
        if over_capacity:
            message = (
                "Worker Docker data-root is under disk pressure: "
                f"path={stats['path']} free={stats['free_gb']:.1f}GB "
                f"used={stats['used_pct']:.1f}% inode={stats['inode_used_pct']:.1f}% "
                f"thresholds free>={min_free_gb:.1f}GB used<={max_used_pct:.1f}% "
                f"inode<={max_inode_pct:.1f}%"
            )
            details = {
                **stats,
                "phase": phase,
                "min_free_gb": min_free_gb,
                "max_used_pct": max_used_pct,
                "max_inode_pct": max_inode_pct,
            }
            if _env_bool("WORKER_DISK_GUARD_STRICT", False):
                raise ResourcePressureError(
                    "WORKER_DOCKER_DISK_PRESSURE", message, details
                )

            # The HTTP pool can still serve health/status and perform cleanup
            # while image builds are unavailable.  Keep warning mode usable on
            # small local Docker partitions, but rate-limit the warning so a
            # frequently polled /healthz endpoint does not flood the log.
            global _DISK_PRESSURE_WARN_LAST
            now = time.time()
            if now - _DISK_PRESSURE_WARN_LAST >= 60.0:
                logger.warning(
                    "%s; continuing because WORKER_DISK_GUARD_STRICT=0",
                    message,
                )
                _DISK_PRESSURE_WARN_LAST = now

    if os.getenv("WORKER_PRESSURE_GUARD_ENABLED", "1") == "0":
        return

    degraded = _docker_degraded_details()
    if degraded is not None and phase in {"allocate", "reset"}:
        raise ResourcePressureError(
            "WORKER_DOCKER_DEGRADED",
            "Worker Docker API is in short cooldown after recent CLI failures; "
            "refusing new Docker work.",
            {"phase": phase, "pending_closes": pending_closes, **degraded},
        )

    # CRITICAL FIX: Catch RuntimeError that blocks all reset operations
    # Issue: "cannot reuse already awaited coroutine" causes 100% reset failure
    # This is a defensive measure while investigating root cause
    try:
        pressure = worker_pressure_stats()
    except RuntimeError as e:
        logger.error(
            "RuntimeError in worker_pressure_stats (allowing %s to proceed): %s",
            phase,
            e,
            exc_info=True
        )
        # Degraded mode: skip pressure checks to unblock reset operations
        # This allows containers to be reset/deleted, preventing >1h uptime accumulation
        return
    except Exception as e:
        logger.exception("Unexpected error in worker_pressure_stats for phase=%s: %s", phase, e)
        return

    pids_pause_pct = _env_float("WORKER_PIDS_PAUSE_ALLOCATE_PCT", 60.0)
    pids_reject_reset_pct = _env_float("WORKER_PIDS_REJECT_RESET_PCT", 70.0)
    pids_min_free_allocate = _env_int("WORKER_PIDS_MIN_FREE_ALLOCATE", 6000)
    pids_min_free_reset = _env_int("WORKER_PIDS_MIN_FREE_RESET", 4000)
    shim_pause = _env_int("WORKER_SHIM_PAUSE_ALLOCATE", 256)
    shim_reject_reset = _env_int("WORKER_SHIM_REJECT_RESET", 384)
    pending_pause = _env_int("WORKER_PENDING_CLOSES_PAUSE_ALLOCATE", 50)
    pending_reject_reset = _env_int("WORKER_PENDING_CLOSES_REJECT_RESET", 100)

    pids_current = int(pressure.get("pids_current") or pressure.get("tasks") or 0)
    pids_max = int(pressure.get("pids_max") or 0)
    pids_free = max(pids_max - pids_current, 0) if pids_max > 0 else -1

    details = {
        **pressure,
        "phase": phase,
        "pending_closes": pending_closes,
        "pids_free": pids_free,
        "pids_min_free_allocate": pids_min_free_allocate,
        "pids_min_free_reset": pids_min_free_reset,
    }
    if pool_status is not None:
        phase_counts = pool_status.get("phase_counts", {})
        resetting = int((phase_counts or {}).get("resetting", 0) or 0)
        active_runs = int(pool_status.get("total_active_runs", 0) or 0)
        reset_age = pool_status.get("resetting_age_sec", {}) or {}
        reset_max_age = float(reset_age.get("max", 0.0) or 0.0)
        details.update(
            {
                "pool_total_active_runs": active_runs,
                "pool_resetting_runs": resetting,
                "pool_resetting_max_age_sec": reset_max_age,
            }
        )
        if (
            phase in {"allocate", "reset"}
            and _env_bool("WORKER_RESET_STORM_GUARD", True)
        ):
            block_allocate = _env_bool("WORKER_RESET_STORM_BLOCK_ALLOCATE", True)
            if phase == "reset" or block_allocate:
                min_resetting = _env_int("WORKER_RESET_STORM_MIN_RESETTING", 32)
                min_age = _env_float("WORKER_RESET_STORM_MIN_AGE", 180.0)
                ratio_threshold = _env_float("WORKER_RESET_STORM_RATIO_PCT", 50.0)
                ratio = (
                    resetting * 100.0 / max(1, active_runs)
                    if active_runs > 0
                    else 0.0
                )
                details["pool_resetting_ratio_pct"] = round(ratio, 1)
                if (
                    resetting >= min_resetting
                    and ratio >= ratio_threshold
                    and reset_max_age >= min_age
                ):
                    _mark_docker_degraded(
                        f"reset storm resetting={resetting}/{active_runs} "
                        f"ratio={ratio:.1f}% max_age={reset_max_age:.1f}s"
                    )
                    raise ResourcePressureError(
                        "WORKER_RESET_STORM",
                        "Worker has a reset storm; refusing new reset/allocation "
                        "until existing reset work drains.",
                        {
                            **details,
                            "reset_storm_min_resetting": min_resetting,
                            "reset_storm_min_age": min_age,
                            "reset_storm_ratio_pct": ratio_threshold,
                        },
                    )
    if not bool(pressure.get("docker_cli_ok", False)):
        raise ResourcePressureError(
            "WORKER_DOCKER_CLI_UNHEALTHY",
            "Worker Docker CLI probe failed; refusing new Docker work.",
            details,
        )

    if phase == "reset":
        if pressure["pids_pct"] >= pids_reject_reset_pct:
            raise ResourcePressureError(
                "WORKER_PIDS_PRESSURE",
                (
                    f"Worker pids pressure {pressure['pids_pct']:.1f}% "
                    f">= reset threshold {pids_reject_reset_pct:.1f}%"
                ),
                details,
            )
        if pids_free >= 0 and pids_free < pids_min_free_reset:
            raise ResourcePressureError(
                "WORKER_PIDS_HEADROOM_LOW",
                (
                    f"Worker pids free headroom {pids_free} "
                    f"< reset threshold {pids_min_free_reset}"
                ),
                details,
            )
        if pressure["shim"] >= shim_reject_reset:
            raise ResourcePressureError(
                "WORKER_SHIM_PRESSURE",
                f"Worker shim pressure {pressure['shim']} >= reset threshold {shim_reject_reset}",
                details,
            )
        if pending_closes >= pending_reject_reset:
            raise ResourcePressureError(
                "WORKER_PENDING_CLOSES_PRESSURE",
                f"Worker pending_closes {pending_closes} >= reset threshold {pending_reject_reset}",
                details,
            )
        return

    if phase in {"allocate", "health"}:
        if pressure["pids_pct"] >= pids_pause_pct:
            raise ResourcePressureError(
                "WORKER_PIDS_PRESSURE",
                (
                    f"Worker pids pressure {pressure['pids_pct']:.1f}% "
                    f">= allocate threshold {pids_pause_pct:.1f}%"
                ),
                details,
            )
        if pids_free >= 0 and pids_free < pids_min_free_allocate:
            raise ResourcePressureError(
                "WORKER_PIDS_HEADROOM_LOW",
                (
                    f"Worker pids free headroom {pids_free} "
                    f"< allocate threshold {pids_min_free_allocate}"
                ),
                details,
            )
        if pressure["shim"] >= shim_pause:
            raise ResourcePressureError(
                "WORKER_SHIM_PRESSURE",
                f"Worker shim pressure {pressure['shim']} >= allocate threshold {shim_pause}",
                details,
            )
        if pending_closes >= pending_pause:
            raise ResourcePressureError(
                "WORKER_PENDING_CLOSES_PRESSURE",
                f"Worker pending_closes {pending_closes} >= allocate threshold {pending_pause}",
                details,
            )
