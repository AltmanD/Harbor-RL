#!/usr/bin/env bash
# GLM-5.1 + SETA + Slime DAPO. Supply the three model paths below.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)"
cd "${REPO_ROOT}"
[[ "${1:-}" == "--dry-run" ]] && export DRY_RUN=1 && shift
if [[ "${DRY_RUN:-0}" != "1" ]]; then
  : "${HF_CKPT:?HF_CKPT is required for GLM-5.1 training}"
  : "${REF_LOAD:?REF_LOAD is required for GLM-5.1 training}"
  : "${MODEL_ARGS_FILE:?MODEL_ARGS_FILE is required for GLM-5.1 training}"
fi
export MODEL_TAG="${MODEL_TAG:-glm-5.1}"
export CUSTOM_CONFIG_PATH="${CUSTOM_CONFIG_PATH:-${REPO_ROOT}/configs/rollout/rollout_glm51_think.yaml}"
export DATASET="${DATASET:-seta}"
export ALGO="${ALGO:-dapo}"
export HARNESS_OPTION="${HARNESS_OPTION:-camel-agent}"
export DAPO_DYNAMIC_SAMPLING="${DAPO_DYNAMIC_SAMPLING:-0}"
export EXPLORATION_PROFILE="${EXPLORATION_PROFILE:-off}"
exec bash harborrl/platform/slime_train.sh "$@"
