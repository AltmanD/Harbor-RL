#!/usr/bin/env bash
# Qwen3-8B + Terminal-Bench 2.1 converted tasks + HarborRL SPEAR.
#
# TB2.1 is represented by the converted JSONL/task-compose tree used by the
# existing terminal worker.  Keep DATASET=seta so the shared Slime launcher
# selects the terminal-env adapter, while allowing the data and env roots to be
# overridden for smoke subsets or a site-specific mount.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." >/dev/null 2>&1 && pwd)"
TB21_DATASET="${TB21_DATASET:-/mnt/shared-storage-gpfs2/trustcyberdata/private/docker-infra/tmp/puyuan/lightrl/datasets/tb21_full89.jsonl}"
TB21_DATASET_DIR="${TB21_DATASET_DIR:-/mnt/shared-storage-gpfs2/trustcyberdata/private/docker-infra/tmp/puyuan/lightrl/envs}"

[[ -f "${TB21_DATASET}" ]] || {
  echo "[tb21-spear] missing TB21_DATASET=${TB21_DATASET}" >&2
  exit 1
}
[[ -d "${TB21_DATASET_DIR}" ]] || {
  echo "[tb21-spear] missing TB21_DATASET_DIR=${TB21_DATASET_DIR}" >&2
  exit 1
}

export DATASET="seta"
export ROLLOUT_PROMPT_DATA="${TB21_DATASET}"
export DATASET_DIR="${TB21_DATASET_DIR}"
# The shared launcher applies its SETA blacklist by default.  Converted TB2.1
# smoke artifacts may already be filtered; avoid creating a second, ambiguous
# dataset copy unless the caller explicitly chooses another policy.
if [[ "${TB21_DATASET}" == *.filtered.jsonl && -z "${USE_BLACKLIST+x}" ]]; then
  export USE_BLACKLIST=0
fi

exec bash "${REPO_ROOT}/examples/training/train_qwen3_8b_seta_spear.sh" "$@"
