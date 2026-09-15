#!/usr/bin/env bash
# One-click 4-GPU evaluation: Qwen3-8B + SETA fixed12 + camel-agent.
#
# This is an evaluation-only invocation of the existing Slime launcher.  The
# launcher starts a local Ray runtime and therefore may stop local Ray/SGLang
# processes; require an explicit acknowledgement before doing that.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." >/dev/null 2>&1 && pwd)"
cd "${REPO_ROOT}"

usage() {
  cat <<'EOF'
Usage:
  CONFIRM_LOCAL_CLEANUP=1 bash examples/evaluation/run_qwen3_8b_seta_fixed12_camel_4gpu.sh
  bash examples/evaluation/run_qwen3_8b_seta_fixed12_camel_4gpu.sh --dry-run

The real run expects a reachable SETA worker at WORKER_URLS (default:
http://100.98.250.90:18082).  Override model/checkpoint, worker, or run settings
through environment variables; see examples/evaluation/README.md.

For hot worker switching, edit deploy/workers/worker_urls.txt while the run is
active.  The local router reloads that file every few seconds; existing leases
finish on their original worker and new leases use the new URL.
EOF
}

DRY_RUN="${DRY_RUN:-0}"
# Foreground is the safe one-click default: startup and evaluation progress are
# visible and the shell receives the real exit code.  Set BACKGROUND=1 when a
# detached run is explicitly desired.
BACKGROUND="${BACKGROUND:-0}"
case "${1:-}" in
  --dry-run)
    DRY_RUN=1
    BACKGROUND=0
    shift
    ;;
  --help|-h)
    usage
    exit 0
    ;;
  "") ;;
  *)
    printf '[seta-fixed12] ERROR: unknown argument: %s\n' "$1" >&2
    usage >&2
    exit 2
    ;;
esac
[[ "$#" -eq 0 ]] || { printf '[seta-fixed12] ERROR: unexpected arguments\n' >&2; exit 2; }

die() { printf '[seta-fixed12] ERROR: %s\n' "$*" >&2; exit 1; }
warn() { printf '[seta-fixed12] WARN: %s\n' "$*" >&2; }

# Fixed protocol and 4-GPU topology.  Every value remains overridable so the
# recipe can also be reused on a private worker or a different model mount.
export REPO_ROOT
export WORKER_URLS_FILE="${WORKER_URLS_FILE:-${REPO_ROOT}/deploy/workers/worker_urls.txt}"
resolve_worker_urls_file() {
  python3 - "$1" <<'PY'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(0)
urls = []
for raw in path.read_text(encoding="utf-8").splitlines():
    line = raw.split("#", 1)[0].strip()
    if not line:
        continue
    if line.startswith("export "):
        line = line[7:].strip()
    if line.startswith("WORKER_URLS="):
        line = line.split("=", 1)[1].strip()
    line = line.strip("\"'")
    urls.extend(part.rstrip("/") for part in re.split(r"[,\s]+", line) if part)
print(",".join(urls))
PY
}
file_worker_urls="$(resolve_worker_urls_file "${WORKER_URLS_FILE}")"
if [[ -n "${file_worker_urls}" && "${WORKER_URLS_PREFER_ENV:-0}" != "1" ]]; then
  # The file is authoritative by default, preventing a stale WORKER_URLS left
  # in an activated shell from silently defeating hot reload.
  WORKER_URLS="${file_worker_urls}"
fi
export WORKER_URLS="${WORKER_URLS:-http://100.98.250.90:18082}"
if [[ ! -s "${WORKER_URLS_FILE}" ]]; then
  mkdir -p "$(dirname -- "${WORKER_URLS_FILE}")"
  printf '%s\n' "${WORKER_URLS}" > "${WORKER_URLS_FILE}"
fi


export DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/benchmarks/environments}"
export NUM_GPUS="${NUM_GPUS:-4}"
export ACTOR_GPUS="${ACTOR_GPUS:-2}"
export ROLLOUT_GPUS="${ROLLOUT_GPUS:-2}"
export TP_SIZE="${TP_SIZE:-2}"
export ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-2}"

export MODEL_TAG="${MODEL_TAG:-qwen3-8b}"
export MODEL_ARGS_FILE="${MODEL_ARGS_FILE:-qwen3-8B}"
export HF_CKPT="${HF_CKPT:-/mnt/shared-storage-user/puyuan/code/slime/Qwen3-8B}"
export REF_LOAD="${REF_LOAD:-/mnt/shared-storage-user/puyuan/code/slime/Qwen3-8B_torch_dist}"
export SLIME_DIR="${SLIME_DIR:-${REPO_ROOT}/backends/slime}"
export SLIME_ENTRYPOINT="${SLIME_ENTRYPOINT:-${REPO_ROOT}/backends/slime/eval_only.py}"
export CUSTOM_CONFIG_PATH="${CUSTOM_CONFIG_PATH:-${REPO_ROOT}/configs/rollout/rollout_qwen3_think.yaml}"
export EVAL_CONFIG="${EVAL_CONFIG:-${REPO_ROOT}/configs/evaluation/seta_fixed12_score_v1.yaml}"

export DATASET="seta"
export ALGO="dapo"
export HARNESS_OPTION="camel-agent"
# Run a local B-layer router so worker_urls.txt can be hot-reloaded without
# restarting Ray/Slime.  The router itself listens only on localhost.
export START_ENV_POOL_SERVER="${START_ENV_POOL_SERVER:-1}"
export ROUTER_PORT="${ROUTER_PORT:-18080}"
export ENV_SERVER_URL="${ENV_SERVER_URL:-http://127.0.0.1:${ROUTER_PORT}}"
export WORKER_URLS_RELOAD_INTERVAL="${WORKER_URLS_RELOAD_INTERVAL:-5}"
export DAPO_DYNAMIC_SAMPLING="${DAPO_DYNAMIC_SAMPLING:-0}"
export EXPLORATION_PROFILE="${EXPLORATION_PROFILE:-off}"
export MAX_TURN="${MAX_TURN:-10}"
export NUM_ROLLOUT="${NUM_ROLLOUT:-0}"
# Slime still validates training batch arithmetic in eval_only.py:
# global_batch_size = rollout_batch_size * n_samples / num_steps_per_rollout
# and this recipe fixes num_steps_per_rollout=2.  Keep the smallest legal value
# (2*1/2=1); setting this to 1 produces global_batch_size=0 and an assertion.
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-2}"
export N_SAMPLES="${N_SAMPLES:-1}"
export ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-8192}"
export ROLLOUT_MAX_CONTEXT_LEN="${ROLLOUT_MAX_CONTEXT_LEN:-16384}"
export ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1}"
export EVAL_N_SAMPLES="${EVAL_N_SAMPLES:-1}"
export EVAL_MAX_RESPONSE_LEN="${EVAL_MAX_RESPONSE_LEN:-8192}"
export EVAL_MAX_CONTEXT_LEN="${EVAL_MAX_CONTEXT_LEN:-16384}"
export EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0}"
export EVAL_TOP_P="${EVAL_TOP_P:-1}"
export EVAL_TOP_K="${EVAL_TOP_K:--1}"
export EVAL_SEED="${EVAL_SEED:-20260809}"
export EVAL_STEPS="${EVAL_STEPS:-0}"

# Evaluation does not need checkpoints or online W&B.  Trajectories remain
# enabled so every one of the 12 fixed tasks can be inspected afterwards.
export MAX_CKPT_KEEP="${MAX_CKPT_KEEP:-0}"
export TRAJECTORY_SAVE_INTERVAL_SETA="${TRAJECTORY_SAVE_INTERVAL_SETA:-1}"
export TRAJECTORY_SAVE_POLICY="${TRAJECTORY_SAVE_POLICY:-step_interval}"
# The worker may need to build a different Docker image for each SETA task.
# One-at-a-time admission avoids the build semaphore backlog (503) seen when
# three fresh leases are reset concurrently on a nearly-full Docker root.
export EVAL_ROLLOUT_MAX_CONCURRENCY="${EVAL_ROLLOUT_MAX_CONCURRENCY:-1}"
export ENV_REMOTE_MAX_ACTIVE_TASKS="${ENV_REMOTE_MAX_ACTIVE_TASKS:-1}"
export ENV_RESET_MAX_RETRIES="${ENV_RESET_MAX_RETRIES:-3}"
export ENV_RESET_FRESH_LEASE_RETRIES="${ENV_RESET_FRESH_LEASE_RETRIES:-3}"
export ENV_RESET_HTTP_TIMEOUT="${ENV_RESET_HTTP_TIMEOUT:-180}"
export WANDB_ENABLE="${WANDB_ENABLE:-0}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"
export RUN_ID="${RUN_ID:-qwen3-8b-seta-fixed12-camel-4gpu-${RUN_TIMESTAMP}}"
export RUN_NAME="${RUN_NAME:-${RUN_ID}}"
export RUNS_ROOT="${RUNS_ROOT:-${REPO_ROOT}/runs}"
export RUN_CATEGORY="${RUN_CATEGORY:-evaluation}"
if [[ "${DEBUG_MODE:-0}" == "1" ]]; then
  export RUN_CATEGORY="testing/debug"
fi
case "${RUN_CATEGORY}" in
  training|evaluation|testing|testing/debug) ;;
  train) export RUN_CATEGORY="training" ;;
  eval|evaluate) export RUN_CATEGORY="evaluation" ;;
  test) export RUN_CATEGORY="testing" ;;
  debug) export RUN_CATEGORY="testing/debug" ;;
  *) printf '[seta-fixed12] ERROR: unknown RUN_CATEGORY=%s\n' "${RUN_CATEGORY}" >&2; exit 2 ;;
esac
export RUN_DIR="${RUN_DIR:-${RUNS_ROOT}/${RUN_CATEGORY}/${RUN_ID}}"
export LOG_FILE="${LOG_FILE:-${RUN_DIR}/launcher.log}"
export BACKGROUND
export DRY_RUN

[[ -f "${SLIME_ENTRYPOINT}" ]] || die "Slime eval entrypoint not found: ${SLIME_ENTRYPOINT}"
[[ -f "${CUSTOM_CONFIG_PATH}" ]] || die "rollout config not found: ${CUSTOM_CONFIG_PATH}"
[[ -f "${EVAL_CONFIG}" ]] || die "SETA eval config not found: ${EVAL_CONFIG}"
[[ -f "${REPO_ROOT}/benchmarks/datasets/seta_env_convert/eval_fixed12.jsonl" ]] \
  || die "SETA fixed12 dataset not found under benchmarks/datasets/seta_env_convert"
[[ -d "${DATASET_DIR}/seta_env" ]] || die "SETA environment directory not found: ${DATASET_DIR}/seta_env"
command -v python3 >/dev/null 2>&1 || die "python3 is required"
launcher_python="python3"
if [[ -n "${LIGHTRFT_PY312_BIN:-}" && -x "${LIGHTRFT_PY312_BIN}/python3" ]]; then
  launcher_python="${LIGHTRFT_PY312_BIN}/python3"
fi
"${launcher_python}" -c 'import yaml' >/dev/null 2>&1 || die \
  "PyYAML is required by the Slime launcher; activate the HarborRL runtime or install PyYAML before running"

if [[ "${DRY_RUN}" != "1" ]]; then
  [[ "${NUM_GPUS}" -eq 4 && "${ACTOR_GPUS}" -eq 2 && "${ROLLOUT_GPUS}" -eq 2 && "${TP_SIZE}" -eq 2 ]] \
    || die "the fixed recipe requires NUM_GPUS=4, ACTOR_GPUS=2, ROLLOUT_GPUS=2, TP_SIZE=2"
  command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is required on the GPU machine"
  gpu_count="$(nvidia-smi -L | sed -n '/^GPU /p' | wc -l)"
  [[ "${gpu_count}" -ge 4 ]] || die "requires 4 visible GPUs, found ${gpu_count}"
  command -v curl >/dev/null 2>&1 || die "curl is required for the SETA worker health check"
  [[ "${CONFIRM_LOCAL_CLEANUP:-0}" == "1" ]] || die \
    "the launcher stops local Ray/SGLang processes; re-run with CONFIRM_LOCAL_CLEANUP=1 on a dedicated machine"
  if [[ "${SKIP_WORKER_HEALTHCHECK:-0}" != "1" ]]; then
    first_worker="${WORKER_URLS%%,*}"
    first_worker="${first_worker%/}"
    curl --noproxy '*' --fail --silent --show-error --max-time 10 \
      "${first_worker}/healthz" >/dev/null \
      || die "SETA worker is unavailable at ${first_worker}/healthz; start a private/co-located worker or override WORKER_URLS. This recipe does not mutate shared workers."
  else
    warn "SKIP_WORKER_HEALTHCHECK=1: worker connectivity will be checked by the launcher"
  fi
fi

printf '[seta-fixed12] run_id=%s\n' "${RUN_ID}"
printf '[seta-fixed12] topology=4 GPU (2 actor + 2 rollout, TP=2)\n'
printf '[seta-fixed12] worker_urls=%s\n' "${WORKER_URLS}"
printf '[seta-fixed12] worker_urls_file=%s reload=%ss router=%s\n' \
  "${WORKER_URLS_FILE}" "${WORKER_URLS_RELOAD_INTERVAL}" "${ENV_SERVER_URL}"
printf '[seta-fixed12] eval_config=%s\n' "${EVAL_CONFIG}"
printf '[seta-fixed12] output=%s\n' "${RUN_DIR}"
printf '[seta-fixed12] remote_env_concurrency=%s reset_retries=%s fresh_lease_retries=%s\n' \
  "${EVAL_ROLLOUT_MAX_CONCURRENCY}" "${ENV_RESET_MAX_RETRIES}" "${ENV_RESET_FRESH_LEASE_RETRIES}"

if [[ "${DRY_RUN}" == "1" ]]; then
  bash "${REPO_ROOT}/examples/training/train_qwen3_8b_seta_dapo.sh" --dry-run
else
  mkdir -p "${RUN_DIR}"
  # slime_train enables shell tracing for diagnostics.  Keep that noisy trace
  # in a separate file so the terminal shows only meaningful progress lines.
  exec 9>>"${RUN_DIR}/launcher.trace.log"
  export BASH_XTRACEFD=9
  if [[ "${BACKGROUND}" == "1" ]]; then
    bash "${REPO_ROOT}/examples/training/train_qwen3_8b_seta_dapo.sh"
    printf '[seta-fixed12] detached evaluator started; log=%s\n' "${LOG_FILE}"
    printf '[seta-fixed12] monitor: tail -f %s\n' "${LOG_FILE}"
    printf '[seta-fixed12] trajectories=%s/trajectories/\n' "${RUN_DIR}"
    exit 0
  fi
  printf '[seta-fixed12] starting foreground evaluator (log=%s)\n' "${LOG_FILE}"
  # Keep a complete local log while preserving the launcher's live progress
  # bars.  PIPESTATUS ensures failures are not hidden by tee.
  set +e
  bash "${REPO_ROOT}/examples/training/train_qwen3_8b_seta_dapo.sh" 2>&1 | tee "${LOG_FILE}"
  rc=${PIPESTATUS[0]}
  set -e
  if [[ "${rc}" -eq 0 ]]; then
    printf '[seta-fixed12] completed successfully; log=%s\n' "${LOG_FILE}"
    printf '[seta-fixed12] trajectories=%s/trajectories/\n' "${RUN_DIR}"
  else
    printf '[seta-fixed12] FAILED rc=%s; inspect %s\n' "${rc}" "${LOG_FILE}" >&2
  fi
  exit "${rc}"
fi
