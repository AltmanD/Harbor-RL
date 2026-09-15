#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
RJOB_NAMESPACE="${RJOB_NAMESPACE:-ailab-narmodel}"
RJOB_IMAGE="${RJOB_IMAGE:-registry.h.pjlab.org.cn/ailab-rlinfra-rlinfra_gpu/rft:20260408}"
RJOB_GPU="${RJOB_GPU:-4}"
RJOB_CPU="${RJOB_CPU:-64}"
RJOB_MEMORY_MB="${RJOB_MEMORY_MB:-512000}"
RJOB_NAME="${RJOB_NAME:-alfworld-spear-qwen3-8b-$(date +%Y%m%d-%H%M%S)}"
RJOB_MOUNT_PUYUAN="${RJOB_MOUNT_PUYUAN:-gpfs://gpfs1/puyuan:/mnt/shared-storage-user/puyuan}"
RJOB_MOUNT_NARMODEL="${RJOB_MOUNT_NARMODEL:-gpfs://gpfs2/narmodel:/mnt/shared-storage-user/narmodel}"
RJOB_MOUNT_TRUSTCYBER="${RJOB_MOUNT_TRUSTCYBER:-gpfs://gpfs2/trustcyberdata:/mnt/shared-storage-gpfs2/trustcyberdata}"

command -v rjob >/dev/null || { echo "rjob CLI not found" >&2; exit 1; }
CMD="cd ${ROOT} && MODEL_PATH=${MODEL_PATH:-/mnt/shared-storage-user/puyuan/code/slime/Qwen3-8B} ALFWORLD_DATA_DIR=${ALFWORLD_DATA_DIR:-/mnt/shared-storage-user/puyuan/data/alfworld} SPEAR_VERL_AGENT_ROOT=${SPEAR_VERL_AGENT_ROOT:-/mnt/shared-storage-user/puyuan/code/archive/SPEAR/verl-agent} N_GPUS=${N_GPUS:-${RJOB_GPU}} bash examples/training/train_alfworld_spear.sh"

echo "[alfworld-rjob] submitting ${RJOB_NAME} (${RJOB_GPU} GPU, ${RJOB_CPU} CPU, ${RJOB_MEMORY_MB} MiB)"
exec rjob submit \
  --namespace="${RJOB_NAMESPACE}" \
  --name="${RJOB_NAME}" \
  --gpu="${RJOB_GPU}" --cpu="${RJOB_CPU}" --memory="${RJOB_MEMORY_MB}" \
  --charged-group="${RJOB_CHARGED_GROUP:-narmodel_gpu}" \
  --priority="${RJOB_PRIORITY:-5}" --image="${RJOB_IMAGE}" \
  --gpu-affinity-type="${RJOB_GPU_AFFINITY_TYPE:-guaranteed}" \
  --private-machine="${RJOB_PRIVATE_MACHINE:-group}" \
  --preemptible="${RJOB_PREEMPTIBLE:-no}" \
  --mount="${RJOB_MOUNT_PUYUAN}" --mount="${RJOB_MOUNT_NARMODEL}" \
  --mount="${RJOB_MOUNT_TRUSTCYBER}" --privileged=false --host-network=false \
  --share-host-shm=true --termination-grace-period-seconds=30 \
  -- bash -lc "${CMD}"
