#!/usr/bin/env bash
# HarborRL Slime model- and dataset-selectable GRPO/DAPO baseline.
#
# Defaults:
#   * DATASET=mixed with seta:safety:agentharm = 6:2:2; supports swe-smith
#   * ALGO=dapo, DAPO_DYNAMIC_SAMPLING=0 for the no-dynamic DAPO baseline
#   * CUSTOM_CONFIG_PATH=configs/rollout_qwen3_think.yaml
#   * HARNESS_OPTION=camel-agent, MAX_TURN=10
#
# Prerequisites (remote worker):
#   1. Pool server(s) running on reachable host(s), default port 18081 for SETA:
#        bash deploy/workers/run_pool_server.sh
#      SWE-smith uses a separate worker, default port 18082:
#        bash deploy/workers/run_pool_server_swesmith.sh
#   2. WORKER_URLS exported for the selected dataset, e.g.
#        export WORKER_URLS="http://<worker-ip>:18081"
#        export WORKER_URLS="http://<worker-ip>:18082"
#   3. Converted dataset files available.
#
# Usage:
#   DATASET=mixed ALGO=dapo bash harborrl/platform/slime_train.sh
#   DATASET=swe-smith ALGO=dapo bash harborrl/platform/slime_train.sh
#   DATASET=seta ALGO=grpo bash harborrl/platform/slime_train.sh
#
# Structured reward observability:
#   TERMINAL_STRUCTURED_METRICS=1 writes per-rollout dataset reward breakdowns
#   to logs and to ${RUN_DIR}/logs/metrics.jsonl.

set -euo pipefail
set -x

log() { echo "[$(date +'%F %T')] $*"; }
require_cmd() { command -v "$1" >/dev/null 2>&1 || { echo "[ERROR] missing cmd: $1"; exit 1; }; }
normalize_dataset() {
  case "$1" in
    swemith|swe-smith|swe_smith|SWE-Smith|SWESMITH)
      echo "[WARN] DATASET=$1 normalized to DATASET=swesmith" >&2
      echo "swesmith"
      ;;
    *)
      echo "$1"
      ;;
  esac
}


# ── Paths ────────────────────────────────────────────────────────────
PLATFORM_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SCRIPT_DIR="$(cd -- "${PLATFORM_DIR}/.." &>/dev/null && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
export REPO_ROOT
export SLIME_DIR="${SLIME_DIR:-${REPO_ROOT}/backends/slime}"
export MEGATRON_DIR="${MEGATRON_DIR:-${REPO_ROOT}/backends/Megatron-LM}"

CUSTOM_CONFIG_PATH="${CUSTOM_CONFIG_PATH:-${REPO_ROOT}/configs/rollout/rollout_qwen3_think.yaml}"

# ── Sourced stages (execution order preserved from the pre-split monolith) ──
# slime_train.sh was split on 2026-08-07; each lib holds a contiguous slice of
# the original flow, so sourcing in this order is byte-for-byte equivalent to
# running the old single file.
_SLIME_TRAIN_LIB_DIR="${PLATFORM_DIR}/slime_train"
source "${_SLIME_TRAIN_LIB_DIR}/lib_bootstrap.sh"    # conda env, process cleanup, GPU split, remaining run paths
source "${_SLIME_TRAIN_LIB_DIR}/lib_json.sh"         # safe runtime-env JSON helpers
source "${_SLIME_TRAIN_LIB_DIR}/lib_run_dir.sh"      # dataset/algo tags, unified run dir, claude-code preflight
source "${_SLIME_TRAIN_LIB_DIR}/lib_rollout_cfg.sh"  # rollout knobs, log mirroring, model args
source "${_SLIME_TRAIN_LIB_DIR}/lib_dataset.sh"      # dataset & reward config, agentharm/swesmith provisioning
source "${_SLIME_TRAIN_LIB_DIR}/lib_worker.sh"       # worker/router URLs, readiness probes, stale-lease repair, and trajectory knobs
source "${_SLIME_TRAIN_LIB_DIR}/lib_args.sh"         # backend command assembly
source "${_SLIME_TRAIN_LIB_DIR}/lib_launch.sh"       # router start, run-config dump, Ray/runtime env, job monitor, ckpt GC, failure capture, case study
