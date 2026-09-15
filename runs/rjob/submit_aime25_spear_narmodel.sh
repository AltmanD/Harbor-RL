#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
NAME="${RJOB_NAME:-aime25-spear-$(date +%Y%m%d-%H%M%S)}"
MODEL="${HF_CKPT:?set HF_CKPT to the Qwen3-32B checkpoint}"
CMD="cd ${ROOT} && MATH_DATA_ROOT=/mnt/shared-storage-user/puyuan/data/math_rlvr TRAIN_DATASET=aime-2025 EVAL_DATASETS=aime-2025 EVAL_N_SAMPLES=1 HF_CKPT=${MODEL} REF_LOAD=${REF_LOAD:-${MODEL}} bash examples/training/train_qwen3_8b_dapo_math.sh"
exec rjob submit --namespace=ailab-narmodel --name="${NAME}" --gpu="${RJOB_GPU:-8}" --cpu="${RJOB_CPU:-64}" --memory="${RJOB_MEMORY:-512000}" --charged-group="${RJOB_CHARGED_GROUP:-narmodel_gpu}" --private-machine=group --preemptible=no --mount=gpfs://gpfs1/puyuan:/mnt/shared-storage-user/puyuan -- bash -lc "${CMD}"
