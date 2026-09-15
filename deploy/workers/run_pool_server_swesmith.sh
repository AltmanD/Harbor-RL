#!/usr/bin/env bash
# SWE-smith Docker worker launcher.
#
# Run on a CPU/docker worker that shares the HarborRL filesystem with the GPU
# trainer. It delegates to run_pool_server.sh but pins conservative
# defaults for SWE-smith image build/start latency.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
TERMINAL_RL="${REPO_ROOT}/harborrl"

export DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/benchmarks/environments}"
export TBENCH_DOCKER_IMAGE_SOURCE="${TBENCH_DOCKER_IMAGE_SOURCE:-build}"
export ENV_SERVER_PORT="${ENV_SERVER_PORT:-18082}"
export TERMINAL_RL_POOL_NAMESPACE="${TERMINAL_RL_POOL_NAMESPACE:-swesmith}"
if [[ ! "${TERMINAL_RL_POOL_NAMESPACE}" =~ ^[a-z0-9][a-z0-9_-]{0,62}$ ]]; then
  echo "[ERROR] TERMINAL_RL_POOL_NAMESPACE must match ^[a-z0-9][a-z0-9_-]{0,62}$." >&2
  exit 1
fi
if [[ "${TERMINAL_RL_POOL_NAMESPACE}" == "default" ]]; then
  echo "[ERROR] the SWE-smith worker requires a non-default Docker pool namespace." >&2
  echo "        Unset TERMINAL_RL_POOL_NAMESPACE or choose a dedicated value such as swesmith." >&2
  exit 1
fi
SWESMITH_WORKER_REQUIRE_FULL_DATA="${SWESMITH_WORKER_REQUIRE_FULL_DATA:-1}"
case "${SWESMITH_WORKER_REQUIRE_FULL_DATA}" in
  0|1) ;;
  *)
    echo "[ERROR] SWESMITH_WORKER_REQUIRE_FULL_DATA must be 0 or 1." >&2
    exit 1
    ;;
esac

if [[ -n "${COMPOSE_OVERRIDE_PATH:-}" ]]; then
  echo "[ERROR] COMPOSE_OVERRIDE_PATH is unsupported by the namespaced SWE-smith worker." >&2
  echo "        Regenerate the task Compose file instead of merging an unchecked override." >&2
  exit 1
fi

# SWE-smith images are heavier than SETA tasks. Keep build/reset concurrency
# bounded unless the worker has been explicitly provisioned for more.
export WORKER_MAX_TASKS="${WORKER_MAX_TASKS:-8}"
export WORKER_MAX_RUNS_PER_TASK="${WORKER_MAX_RUNS_PER_TASK:-4}"
WORKER_TOTAL_CAPACITY=$((WORKER_MAX_TASKS * WORKER_MAX_RUNS_PER_TASK))
export WORKER_MAX_CONCURRENT_BUILDS="${WORKER_MAX_CONCURRENT_BUILDS:-1}"
export WORKER_DOCKER_BUILD_QUEUE_TIMEOUT="${WORKER_DOCKER_BUILD_QUEUE_TIMEOUT:-1800}"
export WORKER_MAX_CONCURRENT_CLOSES="${WORKER_MAX_CONCURRENT_CLOSES:-8}"
export WORKER_MAX_CONCURRENT_RESETS="${WORKER_MAX_CONCURRENT_RESETS:-${WORKER_TOTAL_CAPACITY}}"
export WORKER_RESET_ADMISSION_TIMEOUT="${WORKER_RESET_ADMISSION_TIMEOUT:-300}"
export ENSURE_IMAGE_TIMEOUT="${ENSURE_IMAGE_TIMEOUT:-1800}"
export RESET_SESSION_TIMEOUT="${RESET_SESSION_TIMEOUT:-900}"
export CLOSE_SESSION_TIMEOUT="${CLOSE_SESSION_TIMEOUT:-90}"
export EVAL_TIMEOUT="${EVAL_TIMEOUT:-1200}"
export WORKER_CLOSE_SESSION_TIMEOUT="${WORKER_CLOSE_SESSION_TIMEOUT:-90}"
export WORKER_RESET_OPERATION_TIMEOUT="${WORKER_RESET_OPERATION_TIMEOUT:-4620}"
export WORKER_RESETTING_TTL="${WORKER_RESETTING_TTL:-5400}"
export WORKER_ALLOCATED_TTL="${WORKER_ALLOCATED_TTL:-8400}"
export WORKER_RUN_IDLE_TTL="${WORKER_RUN_IDLE_TTL:-8400}"
export WORKER_REPAIR_STALE_RUNS_MIN_AGE="${WORKER_REPAIR_STALE_RUNS_MIN_AGE:-8400}"
export WORKER_MIN_DOCKER_FREE_GB="${WORKER_MIN_DOCKER_FREE_GB:-80}"
export CONTAINER_MEMORY_LIMIT="${CONTAINER_MEMORY_LIMIT:-16g}"
export CONTAINER_PIDS_LIMIT="${CONTAINER_PIDS_LIMIT:-256}"

# Rebuild task images by default for SWE-smith so fixes in generated Dockerfiles
# (tmux/asciinema runtime, runner scripts) are not hidden by stale cached images.
export WORKER_DOCKER_BUILD_SKIP_EXISTING="${WORKER_DOCKER_BUILD_SKIP_EXISTING:-0}"
export WORKER_DOCKER_BUILD_DEDUP="${WORKER_DOCKER_BUILD_DEDUP:-1}"

# Coexistence mode: this script is commonly launched on the same Docker host as
# an existing SETA pool server. Host-wide cleanup stays disabled. Final and
# child-exit cleanup are safe because the shared launcher selects only Docker
# objects whose harborrl.pool-namespace label exactly matches `swesmith`.
export SKIP_PREFLIGHT_CLEANUP="${SKIP_PREFLIGHT_CLEANUP:-1}"
export PREFLIGHT_KILL_ORPHAN_RUNNING="${PREFLIGHT_KILL_ORPHAN_RUNNING:-0}"
export PREFLIGHT_DISK_CLEANUP="${PREFLIGHT_DISK_CLEANUP:-0}"
export PREFLIGHT_DOCKER_STORAGE_GC="${PREFLIGHT_DOCKER_STORAGE_GC:-0}"
export FINAL_DOCKER_CLEANUP="${FINAL_DOCKER_CLEANUP:-1}"
export TERMINAL_ENV_FORCE_DOCKER_CLEANUP_BROAD="${TERMINAL_ENV_FORCE_DOCKER_CLEANUP_BROAD:-0}"
export POOL_SERVER_CHILD_EXIT_CLEANUP="${POOL_SERVER_CHILD_EXIT_CLEANUP:-1}"
# Periodically recover stale SWE-smith objects after worker crashes. The sweep
# is restricted by TERMINAL_RL_POOL_NAMESPACE and cannot select SETA objects.
export WORKER_ORPHAN_DOCKER_SWEEP="${WORKER_ORPHAN_DOCKER_SWEEP:-1}"
export WORKER_SHIM_CLEANUP_ENABLED="${WORKER_SHIM_CLEANUP_ENABLED:-0}"

# Default to SWE-bench semantics: FAIL_TO_PASS must pass and PASS_TO_PASS must
# not regress. Skipping PASS_TO_PASS is allowed only when full-data enforcement
# is explicitly disabled for a smoke/custom throughput experiment.
export SWESMITH_RUN_PASS_TO_PASS="${SWESMITH_RUN_PASS_TO_PASS:-1}"
if [[ "${SWESMITH_WORKER_REQUIRE_FULL_DATA}" == "1" && "${SWESMITH_RUN_PASS_TO_PASS}" != "1" ]]; then
  echo "[ERROR] formal SWE-smith worker requires SWESMITH_RUN_PASS_TO_PASS=1." >&2
  echo "        Disable SWESMITH_WORKER_REQUIRE_FULL_DATA only for smoke/custom throughput tests." >&2
  exit 1
fi

SWESMITH_ARTIFACT_LOCK="${SWESMITH_ARTIFACT_LOCK:-${DATASET_DIR}/.swesmith_artifact.lock}"
if ! command -v flock >/dev/null 2>&1; then
  echo "[ERROR] flock is required for SWE-smith artifact/task consistency." >&2
  exit 1
fi
mkdir -p "$(dirname -- "${SWESMITH_ARTIFACT_LOCK}")"
exec 8>"${SWESMITH_ARTIFACT_LOCK}"
if ! flock -s -n 8; then
  echo "[ERROR] SWE-smith conversion/publication is active: ${SWESMITH_ARTIFACT_LOCK}" >&2
  echo "        Wait for conversion to finish, then restart the worker." >&2
  exit 1
fi

echo "[swesmith-worker] DATASET_DIR=${DATASET_DIR}"
echo "[swesmith-worker] TBENCH_DOCKER_IMAGE_SOURCE=${TBENCH_DOCKER_IMAGE_SOURCE}"
echo "[swesmith-worker] ENV_SERVER_PORT=${ENV_SERVER_PORT}"
echo "[swesmith-worker] pool_namespace=${TERMINAL_RL_POOL_NAMESPACE}"
echo "[swesmith-worker] max_tasks=${WORKER_MAX_TASKS} runs_per_task=${WORKER_MAX_RUNS_PER_TASK} builds=${WORKER_MAX_CONCURRENT_BUILDS}"
echo "[swesmith-worker] reset_admission concurrency=${WORKER_MAX_CONCURRENT_RESETS} timeout=${WORKER_RESET_ADMISSION_TIMEOUT}s"
echo "[swesmith-worker] ttl allocated=${WORKER_ALLOCATED_TTL}s resetting=${WORKER_RESETTING_TTL}s idle=${WORKER_RUN_IDLE_TTL}s"
echo "[swesmith-worker] timeout ensure_image=${ENSURE_IMAGE_TIMEOUT}s build_queue=${WORKER_DOCKER_BUILD_QUEUE_TIMEOUT}s reset=${RESET_SESSION_TIMEOUT}s reset_op=${WORKER_RESET_OPERATION_TIMEOUT}s eval=${EVAL_TIMEOUT}s"
echo "[swesmith-worker] cleanup preflight=${SKIP_PREFLIGHT_CLEANUP} kill_running=${PREFLIGHT_KILL_ORPHAN_RUNNING} disk=${PREFLIGHT_DISK_CLEANUP} storage_gc=${PREFLIGHT_DOCKER_STORAGE_GC} orphan_sweep=${WORKER_ORPHAN_DOCKER_SWEEP} shim=${WORKER_SHIM_CLEANUP_ENABLED} final=${FINAL_DOCKER_CLEANUP} broad=${TERMINAL_ENV_FORCE_DOCKER_CLEANUP_BROAD} child_exit=${POOL_SERVER_CHILD_EXIT_CLEANUP}"
echo "[swesmith-worker] docker_build_skip_existing=${WORKER_DOCKER_BUILD_SKIP_EXISTING} dedup=${WORKER_DOCKER_BUILD_DEDUP}"

pool_python_ready() {
  local python_bin="$1"
  [[ -x "${python_bin}" ]] || return 1
  timeout "${POOL_SERVER_PYTHON_PREFLIGHT_TIMEOUT:-60}" "${python_bin}" - <<'PY' >/dev/null 2>&1
import anyio._core  # noqa: F401
from camel.toolkits import FunctionTool, TerminalToolkit  # noqa: F401
import fastapi  # noqa: F401
from terminal_bench.terminal.terminal import Terminal  # noqa: F401
import uvicorn  # noqa: F401
import yaml  # noqa: F401
PY
}

POOL_SERVER_PYTHON="${POOL_SERVER_PYTHON:-}"
if [[ -z "${POOL_SERVER_PYTHON}" ]]; then
  SHARED_WORKER_VENV="${SHARED_CONDA_POOL_SERVER_VENV:-${TERMINAL_RL}/../../conda_envs/openclaw-worker-py312}"
  python_candidates=(
    "${POOL_SERVER_VENV:+${POOL_SERVER_VENV}/bin/python}"
    "${SHARED_WORKER_VENV}/bin/python"
    "${TERMINAL_RL}/../.venv/bin/python"
    "$(command -v python3 || command -v python)"
  )
  for candidate in "${python_candidates[@]}"; do
    if pool_python_ready "${candidate}"; then
      POOL_SERVER_PYTHON="${candidate}"
      break
    fi
  done
fi

if [[ -z "${POOL_SERVER_PYTHON}" ]] || ! pool_python_ready "${POOL_SERVER_PYTHON}"; then
  shown_python="${POOL_SERVER_PYTHON:-<auto-detect found no compatible Python>}"
  echo "[ERROR] pool_server Python dependency preflight failed for ${shown_python}." >&2
  echo "        Required imports: anyio, fastapi, uvicorn, yaml, camel, terminal_bench." >&2
  echo "        Set POOL_SERVER_PYTHON to the prepared terminal-rl worker Python," >&2
  echo "        or repair that environment before restarting the service." >&2
  exit 1
fi
export POOL_SERVER_PYTHON
if [[ -z "${POOL_SERVER_VENV:-}" ]]; then
  POOL_SERVER_VENV="$(cd -- "$(dirname -- "${POOL_SERVER_PYTHON}")/.." &>/dev/null && pwd)"
fi
export POOL_SERVER_VENV

if ! "${POOL_SERVER_PYTHON}" -c 'import sys; assert sys.version_info >= (3, 12)'; then
  echo "[ERROR] pool_server Python dependency preflight failed for ${POOL_SERVER_PYTHON}." >&2
  echo "        Python >= 3.12 is required." >&2
  exit 1
fi

SWESMITH_REQUIRE_PINNED_WORKER_DEPS="${SWESMITH_REQUIRE_PINNED_WORKER_DEPS:-1}"
case "${SWESMITH_REQUIRE_PINNED_WORKER_DEPS}" in
  0|1) ;;
  *)
    echo "[ERROR] SWESMITH_REQUIRE_PINNED_WORKER_DEPS must be 0 or 1." >&2
    exit 1
    ;;
esac
"${POOL_SERVER_PYTHON}" - "${SWESMITH_REQUIRE_PINNED_WORKER_DEPS}" <<'PY'
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
actual = {name: metadata.version(name) for name in expected}
terminal_bench = metadata.distribution("terminal-bench")
direct_url = json.loads(terminal_bench.read_text("direct_url.json") or "{}")
commit = str((direct_url.get("vcs_info") or {}).get("commit_id") or "")
expected_commit = "d28711d0da2675d0bb1d56de45ae5df6082438a3"
mismatches = [
    f"{name}={actual[name]} (expected {version})"
    for name, version in expected.items()
    if actual[name] != version
]
if commit != expected_commit:
    mismatches.append(
        f"terminal-bench commit={commit or '<unknown>'} (expected {expected_commit})"
    )
print(
    "[swesmith-worker] dependency_versions="
    + ",".join(f"{name}={actual[name]}" for name in sorted(actual))
    + f" terminal_bench_commit={commit or '<unknown>'}"
)
if strict and mismatches:
    raise SystemExit(
        "[ERROR] SWE-smith worker dependencies do not match "
        "deploy/runtime/requirements-swesmith-worker.txt: "
        + "; ".join(mismatches)
    )
PY

if [[ "${SWESMITH_WORKER_DATA_PREFLIGHT:-1}" == "1" ]]; then
  EXPECTED_SWESMITH_ENV_DIR="${DATASET_DIR}/swesmith_env"
  SWESMITH_ENV_DIR="${SWESMITH_ENV_DIR:-${EXPECTED_SWESMITH_ENV_DIR}}"
  if [[ "$(realpath -m -- "${SWESMITH_ENV_DIR}")" != "$(realpath -m -- "${EXPECTED_SWESMITH_ENV_DIR}")" ]]; then
    echo "[ERROR] SWESMITH_ENV_DIR must be DATASET_DIR/swesmith_env at runtime." >&2
    echo "        expected=${EXPECTED_SWESMITH_ENV_DIR} actual=${SWESMITH_ENV_DIR}" >&2
    exit 1
  fi
  if [[ -z "${SWESMITH_PROMPT_DATA:-}" ]]; then
    if [[ -s "${DATASET_DIR}/swesmith_convert/train.jsonl" ]]; then
      SWESMITH_PROMPT_DATA="${DATASET_DIR}/swesmith_convert/train.jsonl"
    elif [[ "${SWESMITH_WORKER_REQUIRE_FULL_DATA}" == "0" ]]; then
      SWESMITH_PROMPT_DATA="${DATASET_DIR}/swesmith_convert/smoke.jsonl"
    else
      echo "[ERROR] formal SWE-smith worker requires swesmith_convert/train.jsonl." >&2
      echo "        Set SWESMITH_WORKER_REQUIRE_FULL_DATA=0 only for smoke/custom validation." >&2
      exit 1
    fi
  fi
  if [[ ! -s "${SWESMITH_PROMPT_DATA}" ]]; then
    echo "[ERROR] SWE-smith worker data preflight found no JSONL: ${SWESMITH_PROMPT_DATA}" >&2
    echo "        Generate data first, or set SWESMITH_PROMPT_DATA explicitly." >&2
    exit 1
  fi
  "${POOL_SERVER_PYTHON}" - "${SWESMITH_PROMPT_DATA}" "${SWESMITH_ENV_DIR}" "${REPO_ROOT}" "${SWESMITH_WORKER_REQUIRE_FULL_DATA}" "${SWESMITH_STATS_PATH:-}" "${SWESMITH_EXPECTED_SAMPLES:-}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

prompt_path = Path(sys.argv[1])
env_dir = Path(sys.argv[2])
sys.path.insert(0, sys.argv[3])
require_full = sys.argv[4] == "1"
stats_path = Path(sys.argv[5]) if sys.argv[5] else prompt_path.with_name("convert_stats.json")
expected_samples = int(sys.argv[6]) if sys.argv[6] else None
from harborrl.data.convert_swesmith import (
    OFFICIAL_TEST_COMMANDS,
    TASK_FORMAT_MARKER,
    TASK_FORMAT_VERSION,
    expected_swesmith_task_path,
    infer_test_runner,
    validate_swesmith_artifact_manifest,
    validate_task_dir_fingerprint,
)

digest = hashlib.sha256()
row = None
rows = 0
with prompt_path.open("rb") as handle:
    for raw_line in handle:
        digest.update(raw_line)
        if not raw_line.strip():
            continue
        rows += 1
        if row is None:
            row = json.loads(raw_line)
if row is None:
    raise SystemExit(f"[ERROR] SWE-smith JSONL is empty: {prompt_path}")
if require_full:
    try:
        validate_swesmith_artifact_manifest(
            prompt_path,
            stats_path=stats_path,
            require_full=True,
            expected_samples=expected_samples,
            artifact_rows=rows,
            artifact_sha256=digest.hexdigest(),
        )
    except ValueError as exc:
        raise SystemExit(f"[ERROR] {exc}") from exc
metadata = row.get("metadata") or {}
expected_runner = infer_test_runner(metadata)
expected_command = OFFICIAL_TEST_COMMANDS.get(
    str(metadata.get("repo") or "").lower(), ""
)
try:
    expected_task_path = expected_swesmith_task_path(metadata.get("task_name"))
except ValueError as exc:
    raise SystemExit(f"[ERROR] invalid SWE-smith task identity: {exc}") from exc
if (
    metadata.get("data_source") != "swesmith"
    or str(metadata.get("task_format_version") or "") != TASK_FORMAT_VERSION
    or str(metadata.get("test_runner") or "") != expected_runner
    or str(metadata.get("test_command") or "") != expected_command
    or str(metadata.get("swesmith_instance_id") or "")
    != str(metadata.get("task_name") or "")
    or str(metadata.get("task_path") or "") != expected_task_path
    or expected_runner == "unsupported"
):
    raise SystemExit(
        "[ERROR] SWE-smith worker JSONL uses a stale or unsupported task profile; "
        "re-run conversion"
    )
task_path = str(metadata.get("task_path") or "")
if not task_path.startswith("swesmith_env/"):
    raise SystemExit(f"[ERROR] invalid SWE-smith task_path: {task_path!r}")
task_dir = env_dir / task_path.split("/", 1)[1]
required = [
    "task.yaml",
    "docker-compose.yaml",
    "Dockerfile",
    "run-tests.sh",
    "tests/fail_to_pass.txt",
    "tests/pass_to_pass.txt",
    TASK_FORMAT_MARKER,
]
missing = [name for name in required if not (task_dir / name).is_file()]
if missing:
    raise SystemExit(
        f"[ERROR] SWE-smith representative task is incomplete: {task_dir}; "
        f"missing={missing}"
    )
if not validate_task_dir_fingerprint(row, task_dir):
    raise SystemExit(
        f"[ERROR] SWE-smith representative task fingerprint is stale: {task_dir}"
    )
print(
    f"[swesmith-worker] data_preflight=ok format=v{TASK_FORMAT_VERSION} "
    f"runner={expected_runner} task={metadata.get('task_name')} "
    f"rows={rows} require_full={int(require_full)}"
)
PY
fi

exec bash "${SCRIPT_DIR}/run_pool_server.sh" "$@"
