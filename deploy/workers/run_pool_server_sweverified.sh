#!/usr/bin/env bash
# SWE-bench Verified prediction-generation Docker worker.
#
# This service only hosts model workspaces and exports patches. Authoritative
# grading is intentionally performed later by the official SWE-bench harness.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
TERMINAL_RL_DIR="${REPO_ROOT}/harborrl"

export DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/benchmarks/environments}"
export ENV_SERVER_PORT="${ENV_SERVER_PORT:-18083}"
export TERMINAL_RL_POOL_NAMESPACE="${TERMINAL_RL_POOL_NAMESPACE:-sweverified}"
export TBENCH_DOCKER_IMAGE_SOURCE="${TBENCH_DOCKER_IMAGE_SOURCE:-build}"

if [[ ! "${TERMINAL_RL_POOL_NAMESPACE}" =~ ^[a-z0-9][a-z0-9_-]{0,62}$ ]] ||
   [[ "${TERMINAL_RL_POOL_NAMESPACE}" == "default" ]]; then
  echo "[ERROR] SWE-Verified requires a dedicated non-default pool namespace." >&2
  exit 2
fi
if [[ -n "${COMPOSE_OVERRIDE_PATH:-}" ]]; then
  echo "[ERROR] COMPOSE_OVERRIDE_PATH is unsupported for SWE-Verified." >&2
  exit 2
fi

export WORKER_MAX_TASKS="${WORKER_MAX_TASKS:-4}"
export WORKER_MAX_RUNS_PER_TASK="${WORKER_MAX_RUNS_PER_TASK:-2}"
export WORKER_MAX_CONCURRENT_BUILDS="${WORKER_MAX_CONCURRENT_BUILDS:-1}"
export WORKER_DOCKER_BUILD_QUEUE_TIMEOUT="${WORKER_DOCKER_BUILD_QUEUE_TIMEOUT:-1800}"
export WORKER_MAX_CONCURRENT_RESETS="${WORKER_MAX_CONCURRENT_RESETS:-4}"
export WORKER_RESET_ADMISSION_TIMEOUT="${WORKER_RESET_ADMISSION_TIMEOUT:-300}"
export WORKER_MAX_CONCURRENT_CLOSES="${WORKER_MAX_CONCURRENT_CLOSES:-8}"
export ENSURE_IMAGE_TIMEOUT="${ENSURE_IMAGE_TIMEOUT:-3600}"
export RESET_SESSION_TIMEOUT="${RESET_SESSION_TIMEOUT:-900}"
export CLOSE_SESSION_TIMEOUT="${CLOSE_SESSION_TIMEOUT:-90}"
export EVAL_TIMEOUT="${EVAL_TIMEOUT:-300}"
export WORKER_RESET_OPERATION_TIMEOUT="${WORKER_RESET_OPERATION_TIMEOUT:-16200}"
export WORKER_RESETTING_TTL="${WORKER_RESETTING_TTL:-16800}"
export WORKER_ALLOCATED_TTL="${WORKER_ALLOCATED_TTL:-16800}"
export WORKER_RUN_IDLE_TTL="${WORKER_RUN_IDLE_TTL:-16800}"
export WORKER_REPAIR_STALE_RUNS_MIN_AGE="${WORKER_REPAIR_STALE_RUNS_MIN_AGE:-16800}"
export WORKER_MIN_DOCKER_FREE_GB="${WORKER_MIN_DOCKER_FREE_GB:-120}"
export CONTAINER_MEMORY_LIMIT="${CONTAINER_MEMORY_LIMIT:-16g}"
export CONTAINER_PIDS_LIMIT="${CONTAINER_PIDS_LIMIT:-256}"
export WORKER_DOCKER_BUILD_DEDUP="${WORKER_DOCKER_BUILD_DEDUP:-1}"
export WORKER_DOCKER_BUILD_SKIP_EXISTING="${WORKER_DOCKER_BUILD_SKIP_EXISTING:-0}"

# Coexist with SETA and SWE-smith services on the same Docker daemon. Cleanup
# may select only objects with this worker's exact pool namespace label.
export SKIP_PREFLIGHT_CLEANUP="${SKIP_PREFLIGHT_CLEANUP:-1}"
export PREFLIGHT_KILL_ORPHAN_RUNNING="${PREFLIGHT_KILL_ORPHAN_RUNNING:-0}"
export PREFLIGHT_DISK_CLEANUP="${PREFLIGHT_DISK_CLEANUP:-0}"
export PREFLIGHT_DOCKER_STORAGE_GC="${PREFLIGHT_DOCKER_STORAGE_GC:-0}"
export FINAL_DOCKER_CLEANUP="${FINAL_DOCKER_CLEANUP:-1}"
export TERMINAL_ENV_FORCE_DOCKER_CLEANUP_BROAD="${TERMINAL_ENV_FORCE_DOCKER_CLEANUP_BROAD:-0}"
export POOL_SERVER_CHILD_EXIT_CLEANUP="${POOL_SERVER_CHILD_EXIT_CLEANUP:-1}"
export WORKER_ORPHAN_DOCKER_SWEEP="${WORKER_ORPHAN_DOCKER_SWEEP:-1}"
export WORKER_SHIM_CLEANUP_ENABLED="${WORKER_SHIM_CLEANUP_ENABLED:-0}"

POOL_SERVER_PYTHON="${POOL_SERVER_PYTHON:-}"
if [[ -z "${POOL_SERVER_PYTHON}" ]]; then
  for candidate in \
    "${TERMINAL_RL_DIR}/../.venv-swesmith-worker/bin/python" \
    "${SHARED_CONDA_POOL_SERVER_VENV:-${TERMINAL_RL_DIR}/../../conda_envs/openclaw-worker-py312}/bin/python" \
    "${TERMINAL_RL_DIR}/../.venv/bin/python" \
    "$(command -v python3 || command -v python)"; do
    if [[ -x "${candidate}" ]] &&
       timeout 60 env \
         "PYTHONPATH=${TERMINAL_RL_DIR}/..${PYTHONPATH:+:${PYTHONPATH}}" \
         "${candidate}" -c \
         'import importlib; module = importlib.import_module("harborrl.platform.worker_app"); assert module.app is not None' \
         >/dev/null 2>&1; then
      POOL_SERVER_PYTHON="${candidate}"
      break
    fi
  done
fi
if [[ -z "${POOL_SERVER_PYTHON}" ]]; then
  echo "[ERROR] no compatible terminal-rl worker Python was found." >&2
  exit 2
fi
if ! timeout 60 env \
  "PYTHONPATH=${TERMINAL_RL_DIR}/..${PYTHONPATH:+:${PYTHONPATH}}" \
  "${POOL_SERVER_PYTHON}" -c \
  'import importlib; module = importlib.import_module("harborrl.platform.worker_app"); assert module.app is not None'; then
  echo "[ERROR] pool_server Python dependency preflight failed for ${POOL_SERVER_PYTHON}." >&2
  echo "        Install deploy/runtime/requirements-swesmith-worker.txt in that environment." >&2
  exit 2
fi
export POOL_SERVER_PYTHON
export POOL_SERVER_VENV="${POOL_SERVER_VENV:-$(cd "$(dirname "${POOL_SERVER_PYTHON}")/.." && pwd)}"

if ! "${POOL_SERVER_PYTHON}" -c 'import sys; assert sys.version_info >= (3, 12)'; then
  echo "[ERROR] SWE-Verified worker requires Python >= 3.12." >&2
  exit 2
fi
SWEVERIFIED_REQUIRE_PINNED_WORKER_DEPS="${SWEVERIFIED_REQUIRE_PINNED_WORKER_DEPS:-1}"
case "${SWEVERIFIED_REQUIRE_PINNED_WORKER_DEPS}" in
  0|1) ;;
  *)
    echo "[ERROR] SWEVERIFIED_REQUIRE_PINNED_WORKER_DEPS must be 0 or 1." >&2
    exit 2
    ;;
esac
"${POOL_SERVER_PYTHON}" - "${PROMPT_DATA}" "${ENV_DIR}" "${REPO_ROOT}" <<'PY'
import importlib.metadata as metadata
import json
import sys

strict = sys.argv[1] == "1"
expected = {
    "anyio": "4.12.1",
    "camel-ai": "0.2.90",
    "docker": "7.1.0",
    "fastapi": "0.128.0",
    "pydantic": "2.12.0",
    "PyYAML": "6.0.3",
    "terminal-bench": "0.2.18",
    "uvicorn": "0.40.0",
}
actual = {}
for name in expected:
    try:
        actual[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        actual[name] = "<missing>"
try:
    distribution = metadata.distribution("terminal-bench")
    direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
    commit = str((direct_url.get("vcs_info") or {}).get("commit_id") or "")
except metadata.PackageNotFoundError:
    commit = ""
expected_commit = "d28711d0da2675d0bb1d56de45ae5df6082438a3"
mismatches = [
    f"{name}={actual[name]} (expected {version})"
    for name, version in expected.items()
    if actual[name] != version
]
if commit != expected_commit:
    mismatches.append(
        f"terminal-bench commit={commit or '<unknown>'} "
        f"(expected {expected_commit})"
    )
print(
    "[sweverified-worker] dependency_versions="
    + ",".join(f"{name}={actual[name]}" for name in sorted(actual))
    + f" terminal_bench_commit={commit or '<unknown>'}"
)
if strict and mismatches:
    raise SystemExit(
        "[ERROR] SWE-Verified worker dependencies do not match "
        "deploy/runtime/requirements-swesmith-worker.txt: "
        + "; ".join(mismatches)
    )
PY

PROMPT_DATA="${SWEVERIFIED_PROMPT_DATA:-${DATASET_DIR}/sweverified_convert/test.jsonl}"
ENV_DIR="${SWEVERIFIED_ENV_DIR:-${DATASET_DIR}/sweverified_env}"
if [[ ! -s "${PROMPT_DATA}" || ! -d "${ENV_DIR}" ]]; then
  echo "[ERROR] missing SWE-Verified data or task directory." >&2
  echo "        data=${PROMPT_DATA}" >&2
  echo "        env=${ENV_DIR}" >&2
  exit 2
fi

LOCK_PATH="${SWEVERIFIED_ARTIFACT_LOCK:-${DATASET_DIR}/.sweverified_artifact.lock}"
command -v flock >/dev/null 2>&1 || {
  echo "[ERROR] flock is required." >&2
  exit 2
}
exec 8>"${LOCK_PATH}"
flock -s -n 8 || {
  echo "[ERROR] SWE-Verified dataset publication is active: ${LOCK_PATH}" >&2
  exit 2
}

"${POOL_SERVER_PYTHON}" - "${PROMPT_DATA}" "${ENV_DIR}" "${REPO_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

prompt_path = Path(sys.argv[1])
env_dir = Path(sys.argv[2])
sys.path.insert(0, sys.argv[3])
from harborrl.data.convert_sweverified import (
    DATASET_NAME,
    DATASET_REVISION,
    OFFICIAL_INSTANCE_COUNT,
    SWEBENCH_COMMIT,
    SWEBENCH_VERSION,
    TASK_FORMAT_VERSION,
    expected_task_path,
    validate_task_dir_fingerprint,
)

ids = set()
task_paths = []
for line_no, line in enumerate(prompt_path.read_text(encoding="utf-8").splitlines(), 1):
    if not line.strip():
        continue
    row = json.loads(line)
    meta = row.get("metadata") or {}
    instance_id = str(meta.get("swe_instance_id") or "")
    if not instance_id or instance_id in ids:
        raise SystemExit(f"[ERROR] invalid/duplicate instance ID at row {line_no}")
    ids.add(instance_id)
    expected = {
        "data_source": "sweverified",
        "source_dataset": DATASET_NAME,
        "source_revision": DATASET_REVISION,
        "swebench_harness_version": SWEBENCH_VERSION,
        "swebench_harness_commit": SWEBENCH_COMMIT,
        "task_format_version": TASK_FORMAT_VERSION,
        "task_path": expected_task_path(instance_id),
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            raise SystemExit(
                f"[ERROR] row {line_no} has invalid {key}: "
                f"expected={value!r} actual={meta.get(key)!r}"
            )
    task_paths.append((line_no, instance_id, meta["task_path"], row))
if len(ids) != OFFICIAL_INSTANCE_COUNT:
    raise SystemExit(
        f"[ERROR] formal SWE-Verified worker requires "
        f"{OFFICIAL_INSTANCE_COUNT} rows; found {len(ids)}"
    )
for line_no, instance_id, task_path, row in task_paths:
    task_dir = env_dir / task_path.split("/", 1)[1]
    if not validate_task_dir_fingerprint(row, task_dir):
        raise SystemExit(
            f"[ERROR] missing/stale task artifact at row {line_no} "
            f"instance={instance_id} path={task_dir}"
        )
print(
    f"[sweverified-worker] data_preflight=ok rows={len(ids)} "
    f"task_dirs={len(task_paths)} "
    f"dataset_revision={DATASET_REVISION} harness={SWEBENCH_VERSION}@{SWEBENCH_COMMIT}"
)
PY

echo "[sweverified-worker] port=${ENV_SERVER_PORT} namespace=${TERMINAL_RL_POOL_NAMESPACE}"
echo "[sweverified-worker] data=${PROMPT_DATA}"
echo "[sweverified-worker] max_tasks=${WORKER_MAX_TASKS} runs_per_task=${WORKER_MAX_RUNS_PER_TASK} builds=${WORKER_MAX_CONCURRENT_BUILDS}"
echo "[sweverified-worker] python=${POOL_SERVER_PYTHON}"

if [[ "${WORKER_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "[sweverified-worker] preflight-only complete"
  exit 0
fi

exec bash "${SCRIPT_DIR}/run_pool_server.sh" "$@"
