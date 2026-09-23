#!/usr/bin/env bash
# Native-only HarborRL training launcher for the official Slime v0.3.2 backend.
#
# Required environment (normally provided by `harborrl train --config ...`):
#   SLIME_DIR, ROLLOUT_PROMPT_DATA, HF_CKPT, REF_LOAD, MODEL_ARGS_FILE,
#   RUN_DIR, CKPT_ROOT, NUM_GPUS, ACTOR_GPUS, ROLLOUT_GPUS, TP_SIZE,
#   ROLLOUT_NUM_GPUS_PER_ENGINE, HARBORRL_GPU_LAYOUT
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

die() { echo "[ERROR] $*" >&2; exit 1; }

: "${SLIME_DIR:?SLIME_DIR must point to the official Slime v0.3.2 checkout}"
SLIME_DIR="$(cd "${SLIME_DIR}" && pwd)"
[ -f "${SLIME_DIR}/train.py" ] || die "Slime entrypoint not found: ${SLIME_DIR}/train.py"

MODEL_ARGS_FILE="${MODEL_ARGS_FILE:-qwen3-8B}"
MODEL_ARGS_PATH="${SLIME_DIR}/scripts/models/${MODEL_ARGS_FILE}.sh"
[ -f "${MODEL_ARGS_PATH}" ] || die "MODEL_ARGS_FILE=${MODEL_ARGS_FILE} not found at ${MODEL_ARGS_PATH}"
# shellcheck disable=SC1090
source "${MODEL_ARGS_PATH}"

NUM_GPUS="${NUM_GPUS:?NUM_GPUS required}"
ACTOR_GPUS="${ACTOR_GPUS:?ACTOR_GPUS required}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:?ROLLOUT_GPUS required}"
TP_SIZE="${TP_SIZE:?TP_SIZE required}"
ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:?ROLLOUT_NUM_GPUS_PER_ENGINE required}"
for value in "${NUM_GPUS}" "${ACTOR_GPUS}" "${ROLLOUT_GPUS}" "${TP_SIZE}" "${ROLLOUT_NUM_GPUS_PER_ENGINE}"; do
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || die "GPU counts must be positive integers: ${value}"
done
(( ACTOR_GPUS % TP_SIZE == 0 )) || die "ACTOR_GPUS must be divisible by TP_SIZE"
(( ROLLOUT_GPUS % ROLLOUT_NUM_GPUS_PER_ENGINE == 0 )) || die "ROLLOUT_GPUS must be divisible by ROLLOUT_NUM_GPUS_PER_ENGINE"
if [[ "${HARBORRL_GPU_LAYOUT:-split}" == "split" ]]; then
  (( ACTOR_GPUS + ROLLOUT_GPUS <= NUM_GPUS )) || die "split layout exceeds the GPU budget"
elif [[ "${HARBORRL_GPU_LAYOUT:-split}" == "colocate" ]]; then
  (( ACTOR_GPUS == ROLLOUT_GPUS && ROLLOUT_GPUS == NUM_GPUS )) || die "colocate requires equal actor/rollout/total GPUs"
else
  die "unknown HARBORRL_GPU_LAYOUT: ${HARBORRL_GPU_LAYOUT}"
fi

RUN_DIR="${RUN_DIR:?RUN_DIR required}"
CKPT_ROOT="${CKPT_ROOT:-${RUN_DIR}/checkpoints}"
ROLLOUT_PROMPT_DATA="${ROLLOUT_PROMPT_DATA:?ROLLOUT_PROMPT_DATA required}"
HF_CKPT="${HF_CKPT:?HF_CKPT required}"
REF_LOAD="${REF_LOAD:?REF_LOAD required}"
mkdir -p "${RUN_DIR}" "${CKPT_ROOT}" "${RUN_DIR}/metrics/wandb"

NUM_ROLLOUT="${NUM_ROLLOUT:-1}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-1}"
N_SAMPLES="${N_SAMPLES:-2}"
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-8192}"
ROLLOUT_MAX_CONTEXT_LEN="${ROLLOUT_MAX_CONTEXT_LEN:-16384}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-16384}"
LR="${LR:-1e-6}"

CKPT_ARGS=(
  --hf-checkpoint "${HF_CKPT}"
  --ref-load "${REF_LOAD}"
  --rotary-base 1000000
  --megatron-to-hf-mode bridge
  --save "${CKPT_ROOT}/${RUN_ID:-run}"
  --save-interval "${SAVE_INTERVAL}"
  --max-ckpt-keep "${MAX_CKPT_KEEP:-2}"
  --checkpoint-min-free-gb "${CHECKPOINT_MIN_FREE_GB:-128}"
  --checkpoint-expected-gb "${CHECKPOINT_EXPECTED_GB:-0}"
  --checkpoint-space-margin-ratio "${CHECKPOINT_SPACE_MARGIN_RATIO:-1.15}"
)
if [[ -n "${RESUME_LOAD:-}" ]]; then
  CKPT_ARGS+=(--load "${RESUME_LOAD}")
else
  CKPT_ARGS+=(--load "${HF_CKPT}")
fi

DATA_ARGS=(
  --prompt-data "${ROLLOUT_PROMPT_DATA}"
  --input-key task
  --rollout-shuffle
  --reward-key score
  --num-rollout "${NUM_ROLLOUT}"
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
  --n-samples-per-prompt "${N_SAMPLES}"
  --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
  --rollout-max-context-len "${ROLLOUT_MAX_CONTEXT_LEN}"
  --rollout-temperature 1
  --num-steps-per-rollout 1
  --balance-data
  --rollout-generation-max-retries 0
  --rollout-generation-retry-initial-backoff 60
  --rollout-generation-retry-max-backoff 300
  --rollout-generation-retry-backoff-multiplier 2.0
)

OPT_ARGS=(
  --optimizer adam
  --lr "${LR}"
  --lr-decay-style constant
  --weight-decay 0.0
  --adam-beta1 0.9
  --adam-beta2 0.98
  --clip-grad 1.0
  --optimizer-cpu-offload
  --overlap-cpu-optimizer-d2h-h2d
  --use-precision-aware-optimizer
  --advantage-estimator grpo
  --use-rollout-logprobs
  --disable-grpo-std-normalization
  --use-wandb
  --wandb-mode offline
  --wandb-project harborrl
  --wandb-group "${MODEL_ARGS_FILE}_${ACTOR_GPUS}gpu"
  --wandb-dir "${RUN_DIR}/metrics/wandb"
)

PARALLEL_ARGS=(
  --actor-num-nodes 1
  --num-gpus-per-node "${NUM_GPUS}"
  --actor-num-gpus-per-node "${ACTOR_GPUS}"
  --rollout-num-gpus "${ROLLOUT_GPUS}"
  --tensor-model-parallel-size "${TP_SIZE}"
  --sequence-parallel
  --pipeline-model-parallel-size 1
  --context-parallel-size 1
  --expert-model-parallel-size 1
  --expert-tensor-parallel-size 1
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
  --log-probs-chunk-size 1024
  --n-samples-per-eval-prompt 16
  --eval-max-response-len "${ROLLOUT_MAX_CONTEXT_LEN}"
  --eval-top-p 1
  --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}"
  --sglang-mem-fraction-static 0.6
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
  --no-gradient-accumulation-fusion
)

CUSTOM_ARGS=(
  --rollout-function-path harborrl.backends.slime_v032.rollout.generate_rollout
  --custom-convert-samples-to-train-data-path harborrl.backends.slime_v032.converter.convert_samples_to_train_data
  --custom-advantage-function-path harborrl.backends.slime_v032.advantage.compute_advantages_and_returns
  --loss-type custom_loss
  --custom-loss-function-path harborrl.backends.slime_v032.loss.loss_function
  --rollout-data-postprocess-path harborrl.backends.slime_v032.postprocess.rollout_data_postprocess
)
if [[ -f "${RUN_DIR}/config/rollout_config.yaml" ]]; then
  CUSTOM_ARGS+=(--custom-config-path "${RUN_DIR}/config/rollout_config.yaml")
fi

TRAIN_ARGS=(
  "${MODEL_ARGS[@]}"
  "${CKPT_ARGS[@]}"
  "${DATA_ARGS[@]}"
  "${OPT_ARGS[@]}"
  "${PARALLEL_ARGS[@]}"
  "${CUSTOM_ARGS[@]}"
)

TRAIN_PYTHON="${TRAIN_PYTHON:-python3}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '[dry-run] '
  printf '%q ' "${TRAIN_PYTHON}" -u "${SLIME_DIR}/train.py" "${TRAIN_ARGS[@]}"
  printf '\n'
  exit 0
fi

exec "${TRAIN_PYTHON}" -u "${SLIME_DIR}/train.py" "${TRAIN_ARGS[@]}"
