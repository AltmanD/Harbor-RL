#!/usr/bin/env bash
# Qwen3-8B Math RLVR/DAPO recipe.
#
# This entrypoint is intentionally independent of the terminal-environment
# recipe.  It can run inside an rjob worker and accepts the same environment
# variables when called directly.  Use DRY_RUN=1 to inspect the exact Slime
# argv without starting Ray or allocating a GPU.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# Silent checkpoint fallbacks are dangerous for a capability experiment.
: "${HF_CKPT:?set HF_CKPT to the checkpoint being trained}"
: "${REF_LOAD:?set REF_LOAD to the reference checkpoint}"
if [[ -z "${MATH_DATA_ROOT:-}" ]]; then
  MATH_DATA_ROOT="$(PYTHONPATH="${REPO_ROOT}" "${TRAIN_PYTHON:-python3}" -c \
    'from tools.evaluation.math_rlvr.data import resolve_data_root; print(resolve_data_root())')"
fi
TRAIN_DATASET="${TRAIN_DATASET:-aime-2025}"
REWARD_TYPE="${REWARD_TYPE:-math}"
# DAPO paper: 16,384 expected tokens plus a 4,096-token soft-punish cache,
# hence a 20,480-token generation cap.
RESPONSE_CAP="${RESPONSE_CAP:-20480}"
# Use the rollout engine's logprobs as the PPO old policy (PPO-bypass).  The
# Megatron old-policy recomputation disagreed with SGLang by ~8 nats/token in
# v8, which poisons the IS ratio; bypassing it also skips one full forward.
USE_ROLLOUT_LOGPROBS="${USE_ROLLOUT_LOGPROBS:-0}"
if [[ -z "${ROLLOUT_BATCH_SIZE:-}" ]]; then
  if [[ "${TRAIN_DATASET}" == "dapo" || "${TRAIN_DATASET}" == "dapo-math-17k" ]]; then
    # DAPO paper: 512 prompts per rollout.  AIME's 30-row dataset uses the
    # small-dataset fallback below to avoid an empty floor-divided epoch.
    ROLLOUT_BATCH_SIZE=512
  else
    # AIME has only 30 prompts. Four prompts x 16 samples keeps the paper's
    # group size while still allowing several updates per epoch; callers with
    # more memory can override this explicitly.
    ROLLOUT_BATCH_SIZE=4
  fi
fi
N_SAMPLES="${N_SAMPLES:-16}"
NUM_EPOCHS="${NUM_EPOCHS:-10}"
NUM_ROLLOUT="${NUM_ROLLOUT:-}"
if [[ -z "${GLOBAL_BATCH_SIZE:-}" ]]; then
  if [[ "${TRAIN_DATASET}" == "dapo" || "${TRAIN_DATASET}" == "dapo-math-17k" ]]; then
    # 8,192 sampled responses / 512-token minibatch = 16 updates per rollout.
    GLOBAL_BATCH_SIZE=512
  else
    # AIME fallback keeps a valid small-dataset epoch (4 prompts x 16 samples).
    GLOBAL_BATCH_SIZE=$((ROLLOUT_BATCH_SIZE * N_SAMPLES))
  fi
fi
EVAL_DATASETS="${EVAL_DATASETS:-aime-2025,aime-2024}"
EVAL_N_SAMPLES="${EVAL_N_SAMPLES:-1}"
# Evaluation is deliberately sparse; the training loop no longer adds an
# implicit eval at every short epoch boundary.
EVAL_INTERVAL="${EVAL_INTERVAL:-20}"
EVAL_TOP_P="${EVAL_TOP_P:-0.7}"
ENABLE_TRAJECTORY_REPLAY="${ENABLE_TRAJECTORY_REPLAY:-1}"
TRAJECTORY_BUFFER_SIZE="${TRAJECTORY_BUFFER_SIZE:-2048}"
TRAJECTORY_SCORE_THRESHOLD="${TRAJECTORY_SCORE_THRESHOLD:-1}"
TRAJECTORY_TOLERATE_STEPS="${TRAJECTORY_TOLERATE_STEPS:-5}"
REPLAY_LOSS_COEF="${REPLAY_LOSS_COEF:-1}"
MAX_REPLAY_LOSS_STEPS="${MAX_REPLAY_LOSS_STEPS:-300}"
BASELINE_BUFFER_SIZE="${BASELINE_BUFFER_SIZE:-10240}"
EVAL_ROLLOUT_MAX_CONCURRENCY="${EVAL_ROLLOUT_MAX_CONCURRENCY:-8}"
SGLANG_SERVER_CONCURRENCY="${SGLANG_SERVER_CONCURRENCY:-64}"
USE_FAULT_TOLERANCE="${USE_FAULT_TOLERANCE:-1}"
ROLLOUT_GENERATION_MAX_RETRIES="${ROLLOUT_GENERATION_MAX_RETRIES:-3}"
ROLLOUT_GENERATION_RETRY_INITIAL_BACKOFF="${ROLLOUT_GENERATION_RETRY_INITIAL_BACKOFF:-30}"
ROLLOUT_GENERATION_RETRY_MAX_BACKOFF="${ROLLOUT_GENERATION_RETRY_MAX_BACKOFF:-120}"
ROLLOUT_GENERATION_RETRY_BACKOFF_MULTIPLIER="${ROLLOUT_GENERATION_RETRY_BACKOFF_MULTIPLIER:-2}"
ROLLOUT_HEALTH_CHECK_INTERVAL="${ROLLOUT_HEALTH_CHECK_INTERVAL:-30}"
ROLLOUT_HEALTH_CHECK_TIMEOUT="${ROLLOUT_HEALTH_CHECK_TIMEOUT:-30}"
ROLLOUT_HEALTH_CHECK_FIRST_WAIT="${ROLLOUT_HEALTH_CHECK_FIRST_WAIT:-60}"
export EVAL_ROLLOUT_MAX_CONCURRENCY
SEED="${SEED:-1}"
NUM_GPUS="${NUM_GPUS:-1}"
ACTOR_GPUS="${ACTOR_GPUS:-${NUM_GPUS}}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-0}"
ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}"
COLOCATE="${COLOCATE:-0}"
TRAIN_BACKEND="${TRAIN_BACKEND:-megatron}"
MODEL_TRANSFORMER_IMPL="${MODEL_TRANSFORMER_IMPL:-transformer_engine}"
MODEL_ATTENTION_BACKEND="${MODEL_ATTENTION_BACKEND:-flash}"
MODEL_NUM_LAYERS="${MODEL_NUM_LAYERS:-36}"
MODEL_VOCAB_SIZE="${MODEL_VOCAB_SIZE:-151936}"
MODEL_HIDDEN_SIZE="${MODEL_HIDDEN_SIZE:-4096}"
MODEL_NUM_ATTENTION_HEADS="${MODEL_NUM_ATTENTION_HEADS:-32}"
MODEL_FFN_HIDDEN_SIZE="${MODEL_FFN_HIDDEN_SIZE:-12288}"
MODEL_MAX_POSITION_EMBEDDINGS="${MODEL_MAX_POSITION_EMBEDDINGS:-40960}"
MODEL_NUM_QUERY_GROUPS="${MODEL_NUM_QUERY_GROUPS:-8}"
# Qwen3 attention applies QK layernorm; Megatron defaults it off and silently
# skips the checkpoint's q/k norm weights, corrupting the train-side forward.
MODEL_KV_CHANNELS="${MODEL_KV_CHANNELS:-128}"
MODEL_NORM_EPSILON="${MODEL_NORM_EPSILON:-1e-6}"
MODEL_ROTARY_BASE="${MODEL_ROTARY_BASE:-1000000}"
TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-2}"
SEQUENCE_PARALLEL="${SEQUENCE_PARALLEL:-1}"
RECOMPUTE_GRANULARITY="${RECOMPUTE_GRANULARITY:-full}"
RECOMPUTE_METHOD="${RECOMPUTE_METHOD:-uniform}"
RECOMPUTE_NUM_LAYERS="${RECOMPUTE_NUM_LAYERS:-1}"
LR="${LR:-1e-6}"
LR_DECAY_STYLE="${LR_DECAY_STYLE:-constant}"
LR_WARMUP_ITERS="${LR_WARMUP_ITERS:-20}"
CLIP_GRAD="${CLIP_GRAD:-1.0}"
NUM_STEPS_PER_ROLLOUT="${NUM_STEPS_PER_ROLLOUT:-1}"
OVER_SAMPLING_BATCH_SIZE="${OVER_SAMPLING_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE * 2))}"
DYNAMIC_SAMPLING_MAX_GROUPS="${DYNAMIC_SAMPLING_MAX_GROUPS:-256}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-4096}"
ROLLOUT_SHUFFLE="${ROLLOUT_SHUFFLE:-1}"
BALANCE_DATA="${BALANCE_DATA:-1}"
USE_DYNAMIC_BATCH_SIZE="${USE_DYNAMIC_BATCH_SIZE:-1}"
APPLY_CHAT_TEMPLATE="${APPLY_CHAT_TEMPLATE:-1}"
# Debug rollout dumps are opt-in.  When enabled, save evaluation samples by
# default; set DEBUG_ROLLOUT_DATA_SCOPE=train or both when training samples are
# also needed for a forensic run.
DUMP_DETAILS="${DUMP_DETAILS:-}"
DEBUG_ROLLOUT_DATA_SCOPE="${DEBUG_ROLLOUT_DATA_SCOPE:-eval}"
case "${DEBUG_ROLLOUT_DATA_SCOPE}" in
  eval|train|both) ;;
  *) echo "[math-dapo] invalid DEBUG_ROLLOUT_DATA_SCOPE=${DEBUG_ROLLOUT_DATA_SCOPE} (expected eval, train, or both)" >&2; exit 2 ;;
esac
SLIME_DIR="${SLIME_DIR:-${REPO_ROOT}/backends/slime}"
TRAIN_PYTHON="${TRAIN_PYTHON:-python3}"
RUN_ID="${RUN_ID:-math-dapo-${TRAIN_DATASET}-seed${SEED}-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="${RUN_DIR:-${REPO_ROOT}/runs/training/${RUN_ID}}"
# Persist evaluation trajectories by default.  The scope remains eval-only so
# training rollouts do not multiply disk usage; callers can set DUMP_DETAILS=
# explicitly to choose another location.
if [[ -z "${DUMP_DETAILS}" ]]; then
  DUMP_DETAILS="${RUN_DIR}/eval_trajectories"
fi

# DAPO overlong reward shaping: 20,480 generation tokens with a 4,096-token
# soft-punish cache implies an expected length of 16,384.
ALGO="${ALGO:-dapo}"
DAPO_OVERLONG_BUFFER_ENABLE="${DAPO_OVERLONG_BUFFER_ENABLE:-1}"
DAPO_MAX_RESPONSE_LEN="${DAPO_MAX_RESPONSE_LEN:-${RESPONSE_CAP}}"
DAPO_OVERLONG_BUFFER_LEN="${DAPO_OVERLONG_BUFFER_LEN:-4096}"
DAPO_OVERLONG_PENALTY_FACTOR="${DAPO_OVERLONG_PENALTY_FACTOR:-1.0}"
(( DAPO_MAX_RESPONSE_LEN > 0 && DAPO_OVERLONG_BUFFER_LEN > 0 && DAPO_OVERLONG_BUFFER_LEN <= DAPO_MAX_RESPONSE_LEN )) || {
  echo "[math-dapo] invalid DAPO overlong shaping lengths" >&2
  exit 2
}
export ALGO DAPO_OVERLONG_BUFFER_ENABLE DAPO_MAX_RESPONSE_LEN DAPO_OVERLONG_BUFFER_LEN DAPO_OVERLONG_PENALTY_FACTOR

dataset_path() {
  case "$1" in
    aime-2025|aime2025) echo "${MATH_DATA_ROOT}/aime-2025.jsonl" ;;
    aime-2024|aime2024) echo "${MATH_DATA_ROOT}/aime-2024.jsonl" ;;
    amc23|amc-23) echo "${MATH_DATA_ROOT}/amc23.jsonl" ;;
    math-500|math500) echo "${MATH_DATA_ROOT}/math-500.jsonl" ;;
    dapo|dapo-math-17k) echo "${MATH_DATA_ROOT}/dapo-math-17k.jsonl" ;;
    *) echo "$1" ;;
  esac
}
if [[ -z "${TRAIN_DATA:-}" ]]; then
  TRAIN_DATA="$(dataset_path "${TRAIN_DATASET}")"
fi
export TRAIN_DATA TRAIN_DATASET
[[ -f "${TRAIN_DATA}" ]] || { echo "[math-dapo] TRAIN_DATA does not exist: ${TRAIN_DATA}" >&2; exit 2; }

case "${REWARD_TYPE}" in math|dapo|boxed) ;; *) echo "[math-dapo] invalid REWARD_TYPE=${REWARD_TYPE}" >&2; exit 2 ;; esac
(( RESPONSE_CAP > 0 && ROLLOUT_BATCH_SIZE > 0 && N_SAMPLES > 0 )) || { echo "[math-dapo] cap/batch/n must be positive" >&2; exit 2; }
if [[ -z "${NUM_ROLLOUT}" ]]; then
  (( NUM_EPOCHS > 0 )) || { echo "[math-dapo] NUM_EPOCHS must be positive" >&2; exit 2; }
else
  (( NUM_ROLLOUT > 0 )) || { echo "[math-dapo] NUM_ROLLOUT must be positive" >&2; exit 2; }
fi
[[ -d "${HF_CKPT}" || -f "${HF_CKPT}" ]] || { echo "[math-dapo] HF_CKPT does not exist: ${HF_CKPT}" >&2; exit 2; }
[[ -d "${REF_LOAD}" || -f "${REF_LOAD}" ]] || { echo "[math-dapo] REF_LOAD does not exist: ${REF_LOAD}" >&2; exit 2; }

ROW_COUNT="$(PYTHONPATH="${REPO_ROOT}" "${TRAIN_PYTHON}" -c 'import sys; from tools.evaluation.math_rlvr.data import load_dataset; print(len(load_dataset(sys.argv[1], deduplicate=True)))' "${TRAIN_DATA}")"
[[ "${ROW_COUNT}" =~ ^[0-9]+$ && "${ROW_COUNT}" -gt 0 ]] || { echo "[math-dapo] empty training data" >&2; exit 2; }
if [[ "${TRAIN_DATASET}" == "dapo-math-17k" || "${TRAIN_DATASET}" == "dapo" ]]; then
  echo "[math-dapo] DAPO-Math-17k unique rows=${ROW_COUNT}; batch=${ROLLOUT_BATCH_SIZE}; eval=${EVAL_DATASETS}"
fi

EVAL_ARGS=()
IFS=',' read -r -a _EVAL_NAMES <<< "${EVAL_DATASETS}"
for eval_name in "${_EVAL_NAMES[@]}"; do
  eval_path="$(dataset_path "${eval_name}")"
  [[ -f "${eval_path}" ]] || { echo "[math-dapo] eval dataset does not exist: ${eval_path}" >&2; exit 2; }
  EVAL_ARGS+=(--eval-prompt-data "${eval_name}" "${eval_path}")
done

CMD=("${TRAIN_PYTHON}" -u "${SLIME_DIR}/train_async.py"
  --hf-checkpoint "${HF_CKPT}" --ref-load "${REF_LOAD}"
  --prompt-data "${TRAIN_DATA}" --input-key prompt --label-key label
  --reward-key score --rm-type "${REWARD_TYPE}"
  --custom-rm-path tools.evaluation.math_rlvr.reward.reward_func
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
  --n-samples-per-prompt "${N_SAMPLES}" --global-batch-size "${GLOBAL_BATCH_SIZE}"
  --num-steps-per-rollout "${NUM_STEPS_PER_ROLLOUT}"
  --rollout-max-response-len "${RESPONSE_CAP}" --rollout-max-context-len "$((RESPONSE_CAP + 4096))"
  --rollout-temperature 1.0 --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}"
  --sglang-server-concurrency "${SGLANG_SERVER_CONCURRENCY}"
  --advantage-estimator grpo --eps-clip 0.2 --eps-clip-high 0.28
  --calculate-per-token-loss --eval-interval "${EVAL_INTERVAL}"
  --n-samples-per-eval-prompt "${EVAL_N_SAMPLES}" --eval-max-response-len "${RESPONSE_CAP}"
  --eval-input-key prompt --eval-label-key label --eval-reward-key score
  --eval-top-p "${EVAL_TOP_P}" --log-passrate --train-backend "${TRAIN_BACKEND}"
  --attention-backend "${MODEL_ATTENTION_BACKEND}"
  --transformer-impl "${MODEL_TRANSFORMER_IMPL}"
  --num-layers "${MODEL_NUM_LAYERS}" --vocab-size "${MODEL_VOCAB_SIZE}" --hidden-size "${MODEL_HIDDEN_SIZE}"
  --num-attention-heads "${MODEL_NUM_ATTENTION_HEADS}" --ffn-hidden-size "${MODEL_FFN_HIDDEN_SIZE}"
  --max-position-embeddings "${MODEL_MAX_POSITION_EMBEDDINGS}"
  --normalization RMSNorm --norm-epsilon "${MODEL_NORM_EPSILON}"
  --position-embedding-type rope --rotary-base "${MODEL_ROTARY_BASE}"
  --group-query-attention --num-query-groups "${MODEL_NUM_QUERY_GROUPS}"
  --qk-layernorm --kv-channels "${MODEL_KV_CHANNELS}"
  --swiglu --disable-bias-linear --untie-embeddings-and-output-weights
  --attention-dropout 0.0 --hidden-dropout 0.0
  --tensor-model-parallel-size "${TENSOR_MODEL_PARALLEL_SIZE}"
  --recompute-granularity "${RECOMPUTE_GRANULARITY}" --recompute-method "${RECOMPUTE_METHOD}"
  --recompute-num-layers "${RECOMPUTE_NUM_LAYERS}"
  --optimizer adam --lr "${LR}" --lr-decay-style "${LR_DECAY_STYLE}"
  --lr-warmup-iters "${LR_WARMUP_ITERS}" --clip-grad "${CLIP_GRAD}"
  --attention-softmax-in-fp32 --accumulate-allreduce-grads-in-fp32
  --actor-num-nodes 1 --actor-num-gpus-per-node "${ACTOR_GPUS}"
  --seed "${SEED}" --save "${RUN_DIR}/checkpoints" --save-interval "${SAVE_INTERVAL:-20}"
  "${EVAL_ARGS[@]}")

if [[ "${USE_FAULT_TOLERANCE}" == "1" ]]; then
  CMD+=(--use-fault-tolerance
    --rollout-generation-max-retries "${ROLLOUT_GENERATION_MAX_RETRIES}"
    --rollout-generation-retry-initial-backoff "${ROLLOUT_GENERATION_RETRY_INITIAL_BACKOFF}"
    --rollout-generation-retry-max-backoff "${ROLLOUT_GENERATION_RETRY_MAX_BACKOFF}"
    --rollout-generation-retry-backoff-multiplier "${ROLLOUT_GENERATION_RETRY_BACKOFF_MULTIPLIER}"
    --rollout-health-check-interval "${ROLLOUT_HEALTH_CHECK_INTERVAL}"
    --rollout-health-check-timeout "${ROLLOUT_HEALTH_CHECK_TIMEOUT}"
    --rollout-health-check-first-wait "${ROLLOUT_HEALTH_CHECK_FIRST_WAIT}")
fi

# Normalise both plain string prompts and OpenAI-style message lists before
# tokenisation.  Without this flag list prompts reach tokenizer.encode during
# in-training evaluation and abort after otherwise valid updates.
if [[ "${APPLY_CHAT_TEMPLATE}" == "1" ]]; then
  CMD+=(--apply-chat-template)
fi
if [[ "${ROLLOUT_SHUFFLE}" == "1" ]]; then
  CMD+=(--rollout-shuffle --rollout-seed "${SEED}")
fi
if [[ "${BALANCE_DATA}" == "1" ]]; then
  CMD+=(--balance-data)
fi
if [[ "${USE_DYNAMIC_BATCH_SIZE}" == "1" ]]; then
  CMD+=(--use-dynamic-batch-size --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}")
fi
if [[ "${USE_ROLLOUT_LOGPROBS}" == "1" ]]; then
  CMD+=(--use-rollout-logprobs)
fi
if [[ -n "${DUMP_DETAILS}" ]]; then
  CMD+=(--dump-details "${DUMP_DETAILS}" --debug-rollout-data-scope "${DEBUG_ROLLOUT_DATA_SCOPE}")
fi
if (( OVER_SAMPLING_BATCH_SIZE > ROLLOUT_BATCH_SIZE )); then
  CMD+=(--over-sampling-batch-size "${OVER_SAMPLING_BATCH_SIZE}"
    --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
    --dynamic-sampling-max-groups "${DYNAMIC_SAMPLING_MAX_GROUPS}")
fi

if [[ -n "${NUM_ROLLOUT}" ]]; then
  CMD+=(--num-rollout "${NUM_ROLLOUT}")
else
  CMD+=(--num-epoch "${NUM_EPOCHS}")
fi
if [[ "${ENABLE_TRAJECTORY_REPLAY}" == "1" ]]; then
  CMD+=(--enable-trajectory-replay --enable-trajectory-posadv
    --trajectory-buffer-size "${TRAJECTORY_BUFFER_SIZE}"
    --trajectory-score-threshold "${TRAJECTORY_SCORE_THRESHOLD}"
    --trajectory-tolerate-steps "${TRAJECTORY_TOLERATE_STEPS}"
    --replay-loss-coef "${REPLAY_LOSS_COEF}"
    --max-replay-loss-steps "${MAX_REPLAY_LOSS_STEPS}"
    --weight-decay-trajectory-replay -1
    --baseline-buffer-size "${BASELINE_BUFFER_SIZE}")
fi
if [[ "${SEQUENCE_PARALLEL}" == "1" ]]; then
  CMD+=(--sequence-parallel)
fi

if [[ "${COLOCATE}" == "1" ]]; then
  CMD+=(--colocate)
else
  (( ROLLOUT_GPUS > 0 )) || { echo "[math-dapo] ROLLOUT_GPUS must be positive when COLOCATE=0" >&2; exit 2; }
  CMD+=(--rollout-num-gpus "${ROLLOUT_GPUS}")
fi

mkdir -p "${RUN_DIR}/config" "${RUN_DIR}/logs"
PYTHONPATH="${REPO_ROOT}" "${TRAIN_PYTHON}" -c '
import json, sys
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "train_data": sys.argv[2], "train_rows": int(sys.argv[3]),
    "rollout_batch_size": int(sys.argv[4]), "n_samples": int(sys.argv[5]),
    "global_batch_size": int(sys.argv[6]), "num_epochs": int(sys.argv[7]),
    "num_rollout": None if sys.argv[8] == "" else int(sys.argv[8]),
    "eval_datasets": sys.argv[9].split(","), "reward_type": sys.argv[10],
    "response_cap": int(sys.argv[11]), "seed": int(sys.argv[12]),
    "tensor_model_parallel_size": int(sys.argv[13]), "sequence_parallel": sys.argv[14] == "1",
    "recompute_granularity": sys.argv[15], "lr": float(sys.argv[16]),
    "num_steps_per_rollout": int(sys.argv[17]), "over_sampling_batch_size": int(sys.argv[18]),
    "dynamic_sampling_filter": sys.argv[19], "dynamic_sampling_max_groups": int(sys.argv[20]),
    "use_dynamic_batch_size": sys.argv[21] == "1", "max_tokens_per_gpu": int(sys.argv[22]),
    "apply_chat_template": sys.argv[23] == "1", "rollout_shuffle": sys.argv[24] == "1",
    "balance_data": sys.argv[25] == "1", "use_rollout_logprobs": sys.argv[26] == "1",
    "dump_details": sys.argv[27] or None, "debug_rollout_data_scope": sys.argv[28],
    "eval_interval": int(sys.argv[29]), "eval_n_samples": int(sys.argv[30]),
    "eval_top_p": float(sys.argv[31]), "eval_pass_k": 1,
    # Megatron Adam uses decoupled weight decay by default (AdamW mode).
    "optimizer": "adam", "decoupled_weight_decay": True, "lr_warmup_rollouts": int(sys.argv[32]),
    "dapo_overlong": {"max_response_len": int(sys.argv[33]), "expected_len": int(sys.argv[34]),
                       "buffer_len": int(sys.argv[35]), "penalty_factor": float(sys.argv[36])},
}
path.write_text(json.dumps(payload, indent=2) + "\n")
' "${RUN_DIR}/config/math_rlvr.json" "${TRAIN_DATA}" "${ROW_COUNT}" "${ROLLOUT_BATCH_SIZE}" "${N_SAMPLES}" "${GLOBAL_BATCH_SIZE}" "${NUM_EPOCHS}" "${NUM_ROLLOUT}" "${EVAL_DATASETS}" "${REWARD_TYPE}" "${RESPONSE_CAP}" "${SEED}" "${TENSOR_MODEL_PARALLEL_SIZE}" "${SEQUENCE_PARALLEL}" "${RECOMPUTE_GRANULARITY}" "${LR}" "${NUM_STEPS_PER_ROLLOUT}" "${OVER_SAMPLING_BATCH_SIZE}" "slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std" "${DYNAMIC_SAMPLING_MAX_GROUPS}" "${USE_DYNAMIC_BATCH_SIZE}" "${MAX_TOKENS_PER_GPU}" "${APPLY_CHAT_TEMPLATE}" "${ROLLOUT_SHUFFLE}" "${BALANCE_DATA}" "${USE_ROLLOUT_LOGPROBS}" "${DUMP_DETAILS}" "${DEBUG_ROLLOUT_DATA_SCOPE}" "${EVAL_INTERVAL}" "${EVAL_N_SAMPLES}" "${EVAL_TOP_P}" "${LR_WARMUP_ITERS}" "${DAPO_MAX_RESPONSE_LEN}" "$((DAPO_MAX_RESPONSE_LEN - DAPO_OVERLONG_BUFFER_LEN))" "${DAPO_OVERLONG_BUFFER_LEN}" "${DAPO_OVERLONG_PENALTY_FACTOR}"
export MATH_RLVR_REWARD_TYPE="${REWARD_TYPE}" MATH_RLVR_RESPONSE_CAP="${RESPONSE_CAP}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '[dry-run] '
  printf '%q ' "${CMD[@]}"
  printf '\n'
  exit 0
fi
exec "${CMD[@]}" 2>&1 | tee "${RUN_DIR}/logs/train.log"
