#!/usr/bin/env bash
# Qwen3-8B + mixed SETA/Agent-SafetyBench/AgentHarm + Slime DAPO.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)"
cd "${REPO_ROOT}"
export MODEL_TAG="${MODEL_TAG:-qwen3-8b}"
export MODEL_ARGS_FILE="${MODEL_ARGS_FILE:-qwen3-8B}"
export CUSTOM_CONFIG_PATH="${CUSTOM_CONFIG_PATH:-${REPO_ROOT}/configs/rollout/rollout_qwen3_think.yaml}"
export DATASET="${DATASET:-mixed}"
export ALGO="${ALGO:-dapo}"
export HARNESS_OPTION="${HARNESS_OPTION:-camel-agent}"
export DAPO_DYNAMIC_SAMPLING="${DAPO_DYNAMIC_SAMPLING:-0}"
export MIX_SETA_RATIO="${MIX_SETA_RATIO:-6}"
export MIX_SAFETY_RATIO="${MIX_SAFETY_RATIO:-2}"
export MIX_AGENTHARM_RATIO="${MIX_AGENTHARM_RATIO:-2}"
export EXPLORATION_PROFILE="${EXPLORATION_PROFILE:-off}"
[[ "${1:-}" == "--dry-run" ]] && export DRY_RUN=1 && shift
exec bash harborrl/platform/slime_train.sh "$@"
