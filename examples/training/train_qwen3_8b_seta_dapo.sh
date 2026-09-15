#!/usr/bin/env bash
# Complete 4-GPU recipe: Qwen3-8B + SETA + Slime DAPO.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)"
cd "${REPO_ROOT}"

# Infrastructure and topology. Override any value through the environment.
: "${WORKER_URLS:?set WORKER_URLS to the comma-separated SETA worker endpoint(s)}"
export WORKER_URLS
export NUM_GPUS="${NUM_GPUS:-4}"
export ACTOR_GPUS="${ACTOR_GPUS:-2}"
export ROLLOUT_GPUS="${ROLLOUT_GPUS:-2}"
export TP_SIZE="${TP_SIZE:-2}"
export ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-2}"

# Model, environment and algorithm recipe.
export MODEL_TAG="${MODEL_TAG:-qwen3-8b}"
export MODEL_ARGS_FILE="${MODEL_ARGS_FILE:-qwen3-8B}"
export HF_CKPT="${HF_CKPT:-${REPO_ROOT}/models/Qwen3-8B}"
export REF_LOAD="${REF_LOAD:-${REPO_ROOT}/models/Qwen3-8B_torch_dist}"
export CUSTOM_CONFIG_PATH="${CUSTOM_CONFIG_PATH:-${REPO_ROOT}/configs/rollout/rollout_qwen3_think.yaml}"
export DATASET="${DATASET:-seta}"
export ALGO="${ALGO:-dapo}"
export HARNESS_OPTION="${HARNESS_OPTION:-camel-agent}"
export MAX_TURN="${MAX_TURN:-10}"
export DAPO_DYNAMIC_SAMPLING="${DAPO_DYNAMIC_SAMPLING:-0}"
export EXPLORATION_PROFILE="${EXPLORATION_PROFILE:-off}"

export RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"
export RUN_ID="${RUN_ID:-seta-dapo-4g-${RUN_TIMESTAMP}}"
export RUN_NAME="${RUN_NAME:-${RUN_ID}}"
export RUNS_ROOT="${RUNS_ROOT:-${REPO_ROOT}/runs}"
export RUN_CATEGORY="${RUN_CATEGORY:-training}"
if [[ "${DEBUG_MODE:-0}" == "1" ]]; then
  export RUN_CATEGORY="testing/debug"
fi
case "${RUN_CATEGORY}" in
  training|evaluation|testing|testing/debug) ;;
  train) export RUN_CATEGORY="training" ;;
  eval|evaluate) export RUN_CATEGORY="evaluation" ;;
  test) export RUN_CATEGORY="testing" ;;
  debug) export RUN_CATEGORY="testing/debug" ;;
  *) printf '[seta-dapo] ERROR: unknown RUN_CATEGORY=%s\n' "${RUN_CATEGORY}" >&2; exit 2 ;;
esac
RUN_DIR="${RUN_DIR:-${RUNS_ROOT}/${RUN_CATEGORY}/${RUN_ID}}"
export RUN_DIR
LOG_FILE="${LOG_FILE:-${RUN_DIR}/launcher.log}"
BACKGROUND="${BACKGROUND:-0}"
DRY_RUN="${DRY_RUN:-0}"

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  export DRY_RUN
  shift
fi

die() { printf '[seta-dapo] ERROR: %s\n' "$*" >&2; exit 1; }

if [[ "${DRY_RUN}" != "1" ]]; then
  command -v nvidia-smi >/dev/null || die "nvidia-smi is required inside the GPU job"
  gpu_count="$(nvidia-smi -L | sed -n '/^GPU /p' | wc -l)"
  # NUM_GPUS is the Ray resource budget, not a claim that the container must
  # expose exactly that many physical devices. Dedicated 8-GPU jobs may run
  # this 4-GPU recipe; reject only true under-allocation.  Ray is started with
  # --num-gpus=${NUM_GPUS} and therefore cannot schedule onto the extra GPUs.
  [[ "${gpu_count}" -ge "${NUM_GPUS}" || "${ALLOW_GPU_COUNT_MISMATCH:-0}" == "1" ]] \
    || die "requires at least ${NUM_GPUS} visible GPUs, found ${gpu_count}"
  command -v curl >/dev/null || die "curl is required for the worker health check"
  curl --noproxy '*' --fail --silent --show-error --max-time 10 \
    "${WORKER_URLS%%,*}/healthz" >/dev/null \
    || die "Docker worker is unreachable: ${WORKER_URLS%%,*}/healthz. If this came from a stale shell export, run: unset WORKER_URLS WORKER_URLS_FILE"
fi

printf '[seta-dapo] recipe=%s\n' "${BASH_SOURCE[0]}"
printf '[seta-dapo] runtime=%s\n' "harborrl/platform/slime_train.sh"
printf '[seta-dapo] final_entry=%s\n' "slime/train_async.py"
printf '[seta-dapo] run_id=%s worker=%s gpus=%s (%s actor + %s rollout)\n' \
  "${RUN_ID}" "${WORKER_URLS}" "${NUM_GPUS}" "${ACTOR_GPUS}" "${ROLLOUT_GPUS}"

launcher=(bash harborrl/platform/slime_train.sh "$@")
if [[ "${BACKGROUND}" != "1" ]]; then
  exec "${launcher[@]}"
fi

mkdir -p "${RUN_DIR}"
if ! mkdir "${RUN_DIR}/.launch-once" 2>/dev/null; then
  die "${RUN_ID} was already launched; inspect ${LOG_FILE} or choose another RUN_ID"
fi
nohup "${launcher[@]}" >"${LOG_FILE}" 2>&1 </dev/null &
pid="$!"
printf '%s\n' "${pid}" >"${RUN_DIR}/launcher.pid"
printf '[seta-dapo] background pid=%s log=%s\n' "${pid}" "${LOG_FILE}"
