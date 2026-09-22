# ── Args ─────────────────────────────────────────────────────────────
# The launch stage uses this entrypoint for the Ray job.  Define it here too
# so `--dry-run` prints eval_only.py when an evaluation recipe overrides it.
SLIME_ENTRYPOINT="${SLIME_ENTRYPOINT:-${SLIME_DIR}/train_async.py}"

CKPT_ARGS=(
  --hf-checkpoint "${HF_CKPT}"
  --ref-load "${REF_LOAD}"
  --rotary-base 1000000
)
if [[ "${HARBORRL_NATIVE_ROLLOUT:-0}" == "1" ]]; then
  CKPT_ARGS+=(--megatron-to-hf-mode bridge)
fi
# Only add --save / --load / --save-interval when checkpointing is enabled
if [[ -n "${SAVE_CKPT}" ]]; then
  CKPT_ARGS+=(
    --save "${SAVE_CKPT}"
    --save-interval "${SAVE_INTERVAL}"
    --max-ckpt-keep "${MAX_CKPT_KEEP}"
    --checkpoint-min-free-gb "${CHECKPOINT_MIN_FREE_GB:-128}"
    --checkpoint-expected-gb "${CHECKPOINT_EXPECTED_GB:-0}"
    --checkpoint-space-margin-ratio "${CHECKPOINT_SPACE_MARGIN_RATIO:-1.15}"
  )
  if [[ "${SAVE_FIRST_ROLLOUT}" == "1" ]]; then
    CKPT_ARGS+=(--save-first-rollout)
  fi
  if [[ "${CHECKPOINT_SAVE_FATAL:-0}" == "1" ]]; then
    CKPT_ARGS+=(--checkpoint-save-fatal)
  fi
fi
if [[ "${HARBORRL_NATIVE_ROLLOUT:-0}" == "1" && -z "${RESUME_LOAD:-}" ]]; then
  # Bridge mode must load the locked HF checkpoint on the first run.  The save
  # directory is intentionally not a valid resume source until iteration 1.
  CKPT_ARGS+=(--load "${HF_CKPT}")
elif [[ -n "${RESUME_LOAD}" ]]; then
  CKPT_ARGS+=(--load "${RESUME_LOAD}")
fi

if [[ "${DEBUG_MODE}" == "1" ]]; then
  # Preserve DEBUG_MODE safeguards while allowing launchers to request a smaller smoke shape.
  NUM_ROLLOUT="${NUM_ROLLOUT:-4}"
  ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-4}"
  N_SAMPLES="${N_SAMPLES:-2}"
  MAX_TOKENS_PER_GPU=8192
else
  NUM_ROLLOUT="${NUM_ROLLOUT:-2000}"
  # each rollout = ROLLOUT_BATCH_SIZE * N_SAMPLES concurrent lease requests.
  # Keep this baseline explicit and predictable without dynamic sampling.
  ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
  if [[ "${DATASET}" == "swesmith" ]]; then
    N_SAMPLES="${N_SAMPLES:-4}"
  else
    N_SAMPLES="${N_SAMPLES:-8}"
  fi
  MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-16384}"
fi
if [[ "${HARBORRL_STRUCTURED_LAUNCH:-0}" == "1" && "${HARBORRL_NATIVE_ROLLOUT:-0}" != "1" ]]; then
  actor_dp=$((ACTOR_GPUS / TP_SIZE))
  initial_batch=$((ROLLOUT_BATCH_SIZE * N_SAMPLES / 2))
  if (( initial_batch < actor_dp || initial_batch % actor_dp != 0 )); then
    echo "[ERROR] initial global batch (${ROLLOUT_BATCH_SIZE} * ${N_SAMPLES} / 2 = ${initial_batch}) must be divisible by actor DP=${actor_dp}; adjust ROLLOUT_BATCH_SIZE or N_SAMPLES" >&2
    exit 1
  fi
fi
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-8192}"
ROLLOUT_MAX_CONTEXT_LEN="${ROLLOUT_MAX_CONTEXT_LEN:-16384}"
ROLLOUT_GENERATION_MAX_RETRIES="${ROLLOUT_GENERATION_MAX_RETRIES:-$([[ "${HARBORRL_NATIVE_ROLLOUT:-0}" == "1" ]] && echo 0 || echo 3)}"
ROLLOUT_GENERATION_RETRY_INITIAL_BACKOFF="${ROLLOUT_GENERATION_RETRY_INITIAL_BACKOFF:-60}"
ROLLOUT_GENERATION_RETRY_MAX_BACKOFF="${ROLLOUT_GENERATION_RETRY_MAX_BACKOFF:-300}"
ROLLOUT_GENERATION_RETRY_BACKOFF_MULTIPLIER="${ROLLOUT_GENERATION_RETRY_BACKOFF_MULTIPLIER:-2.0}"
ROLLOUT_GENERATION_ENV_STORM_MAX_RETRIES="${ROLLOUT_GENERATION_ENV_STORM_MAX_RETRIES:-3}"
ROLLOUT_GENERATION_SKIP_ON_FAILURE="${ROLLOUT_GENERATION_SKIP_ON_FAILURE:-0}"

ROLLOUT_ARGS=(
  --prompt-data "${ROLLOUT_PROMPT_DATA}"
  --input-key task
  --rollout-shuffle
  --reward-key score
  --num-rollout "${NUM_ROLLOUT}"
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
  --n-samples-per-prompt "${N_SAMPLES}"
  --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
  --rollout-max-context-len "${ROLLOUT_MAX_CONTEXT_LEN}"
  --rollout-temperature "${ROLLOUT_TEMPERATURE:-1}"
  --num-steps-per-rollout "${NUM_STEPS_PER_ROLLOUT:-$([[ "${HARBORRL_NATIVE_ROLLOUT:-0}" == "1" ]] && echo 1 || echo 2)}"
  --balance-data
  --rollout-generation-max-retries "${ROLLOUT_GENERATION_MAX_RETRIES}"
  --rollout-generation-retry-initial-backoff "${ROLLOUT_GENERATION_RETRY_INITIAL_BACKOFF}"
  --rollout-generation-retry-max-backoff "${ROLLOUT_GENERATION_RETRY_MAX_BACKOFF}"
  --rollout-generation-retry-backoff-multiplier "${ROLLOUT_GENERATION_RETRY_BACKOFF_MULTIPLIER}"
  --rollout-generation-env-storm-max-retries "${ROLLOUT_GENERATION_ENV_STORM_MAX_RETRIES}"
)
if [[ "${ROLLOUT_GENERATION_SKIP_ON_FAILURE}" == "1" ]]; then
  ROLLOUT_ARGS+=(--rollout-generation-skip-on-failure)
fi

EVAL_N_SAMPLES="${EVAL_N_SAMPLES:-16}"
EVAL_MAX_RESPONSE_LEN="${EVAL_MAX_RESPONSE_LEN:-16384}"
EVAL_TOP_P="${EVAL_TOP_P:-1}"
EVAL_ARGS=(
  --n-samples-per-eval-prompt "${EVAL_N_SAMPLES}"
  --eval-max-response-len "${EVAL_MAX_RESPONSE_LEN}"
  --eval-top-p "${EVAL_TOP_P}"
)
if [[ -n "${EVAL_CONFIG:-}" ]]; then
  EVAL_ARGS+=(--eval-config "${EVAL_CONFIG}")
elif [[ -n "${EVAL_PROMPT_DATA:-}" ]]; then
  EVAL_DATASET_NAME="${EVAL_DATASET_NAME:-seta}"
  EVAL_ARGS+=(--eval-prompt-data "${EVAL_DATASET_NAME}" "${EVAL_PROMPT_DATA}")
fi
if [[ -n "${EVAL_INTERVAL:-}" ]]; then
  EVAL_ARGS+=(--eval-interval "${EVAL_INTERVAL}")
fi
if [[ -n "${EVAL_FUNCTION_PATH:-}" ]]; then
  EVAL_ARGS+=(--eval-function-path "${EVAL_FUNCTION_PATH}")
fi
if [[ -n "${EVAL_TEMPERATURE:-}" ]]; then
  EVAL_ARGS+=(--eval-temperature "${EVAL_TEMPERATURE}")
fi
if [[ -n "${EVAL_SEED:-}" ]]; then
  EVAL_ARGS+=(--eval-seed "${EVAL_SEED}")
fi
if [[ -n "${EVAL_STEPS:-}" ]]; then
  read -r -a _EVAL_STEPS_ARRAY <<<"${EVAL_STEPS//,/ }"
  EVAL_ARGS+=(--eval-steps "${_EVAL_STEPS_ARRAY[@]}")
fi
if [[ -n "${EVAL_TOP_K:-}" ]]; then
  EVAL_ARGS+=(--eval-top-k "${EVAL_TOP_K}")
fi
if [[ -n "${EVAL_MAX_CONTEXT_LEN:-}" ]]; then
  EVAL_ARGS+=(--eval-max-context-len "${EVAL_MAX_CONTEXT_LEN}")
fi

PERF_ARGS=(
  --tensor-model-parallel-size "${TP_SIZE}"
  --sequence-parallel
  --pipeline-model-parallel-size "${PP_SIZE:-1}"
  --context-parallel-size "${CP_SIZE:-1}"
  --expert-model-parallel-size "${EP_SIZE:-1}"
  --expert-tensor-parallel-size "${ETP_SIZE:-1}"
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
  --log-probs-chunk-size 1024
)

GRPO_ARGS=(
  --advantage-estimator grpo
)
if [[ "${HARBORRL_NATIVE_ROLLOUT:-0}" != "1" ]]; then
  GRPO_ARGS+=(--dynamic_history --use-kl-loss --kl-loss-coef 0.01 --kl-loss-type k3)
else
  GRPO_ARGS+=(--use-rollout-logprobs --disable-grpo-std-normalization)
fi

DAPO_EPS_CLIP_LOW="${DAPO_EPS_CLIP_LOW:-0.2}"
DAPO_EPS_CLIP_HIGH="${DAPO_EPS_CLIP_HIGH:-0.28}"
DAPO_USE_KL_LOSS="${DAPO_USE_KL_LOSS:-0}"
DAPO_KL_LOSS_COEF="${DAPO_KL_LOSS_COEF:-0.0}"
DAPO_KL_LOSS_TYPE="${DAPO_KL_LOSS_TYPE:-k3}"
DAPO_CALCULATE_PER_TOKEN_LOSS="${DAPO_CALCULATE_PER_TOKEN_LOSS:-1}"
DAPO_DYNAMIC_SAMPLING="${DAPO_DYNAMIC_SAMPLING:-0}"
DAPO_DYNAMIC_FILTER_PATH="${DAPO_DYNAMIC_FILTER_PATH:-slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std}"
DAPO_OVER_SAMPLING_BATCH_SIZE="${DAPO_OVER_SAMPLING_BATCH_SIZE:-${ROLLOUT_BATCH_SIZE}}"
DAPO_FAILED_GROUP_ABORT_MIN_GROUPS="${DAPO_FAILED_GROUP_ABORT_MIN_GROUPS:-${ROLLOUT_BATCH_SIZE}}"
DAPO_FAILED_GROUP_ABORT_RATIO="${DAPO_FAILED_GROUP_ABORT_RATIO:-1.0}"
DAPO_GRPO_STD_NORMALIZATION="${DAPO_GRPO_STD_NORMALIZATION:-1}"
DAPO_OVERLONG_BUFFER_ENABLE="${DAPO_OVERLONG_BUFFER_ENABLE:-1}"
DAPO_OVERLONG_BUFFER_LEN="${DAPO_OVERLONG_BUFFER_LEN:-4096}"
DAPO_OVERLONG_PENALTY_FACTOR="${DAPO_OVERLONG_PENALTY_FACTOR:-1.0}"

DAPO_ARGS=(
  --advantage-estimator grpo
  --dynamic_history
  --eps-clip "${DAPO_EPS_CLIP_LOW}"
  --eps-clip-high "${DAPO_EPS_CLIP_HIGH}"
)

if [[ "${DAPO_CALCULATE_PER_TOKEN_LOSS}" == "1" ]]; then
  DAPO_ARGS+=(--calculate-per-token-loss)
fi
if [[ "${DAPO_USE_KL_LOSS}" == "1" ]]; then
  DAPO_ARGS+=(--use-kl-loss --kl-loss-coef "${DAPO_KL_LOSS_COEF}" --kl-loss-type "${DAPO_KL_LOSS_TYPE}")
fi
if [[ "${DAPO_DYNAMIC_SAMPLING}" == "1" ]]; then
  if (( DAPO_OVER_SAMPLING_BATCH_SIZE < ROLLOUT_BATCH_SIZE )); then
    echo "[ERROR] DAPO_OVER_SAMPLING_BATCH_SIZE(${DAPO_OVER_SAMPLING_BATCH_SIZE}) must be >= ROLLOUT_BATCH_SIZE(${ROLLOUT_BATCH_SIZE})"
    exit 1
  fi
  DAPO_ARGS+=(
    --dynamic-sampling-filter-path "${DAPO_DYNAMIC_FILTER_PATH}"
    --over-sampling-batch-size "${DAPO_OVER_SAMPLING_BATCH_SIZE}"
  )
  if (( DAPO_FAILED_GROUP_ABORT_MIN_GROUPS > 0 )); then
    DAPO_ARGS+=(
      --dynamic-sampling-failed-group-abort-min-groups "${DAPO_FAILED_GROUP_ABORT_MIN_GROUPS}"
      --dynamic-sampling-failed-group-abort-ratio "${DAPO_FAILED_GROUP_ABORT_RATIO}"
    )
  fi
fi
if [[ "${DAPO_GRPO_STD_NORMALIZATION}" == "0" ]]; then
  DAPO_ARGS+=(--disable-grpo-std-normalization)
fi

case "${ALGO}" in
  grpo)
    ALGO_ARGS=("${GRPO_ARGS[@]}")
    ALGO_EXTRA_ARGS="${EXTRA_GRPO_ARGS:-} ${EXTRA_ALGO_ARGS:-}"
    ;;
  dapo)
    ALGO_ARGS=("${DAPO_ARGS[@]}")
    ALGO_EXTRA_ARGS="${EXTRA_DAPO_ARGS:-} ${EXTRA_ALGO_ARGS:-}"
    ;;
esac
ALGO_EXTRA_ARGS_ARRAY=()
if [[ -n "${ALGO_EXTRA_ARGS// }" ]]; then
  # Preserve the existing unquoted EXTRA_GRPO_ARGS behavior for compatibility.
  ALGO_EXTRA_ARGS_ARRAY=(${ALGO_EXTRA_ARGS})
fi
log "Algorithm config: ALGO=${ALGO} args=${ALGO_ARGS[*]} extra=${ALGO_EXTRA_ARGS_ARRAY[*]:-<none>}"
log "Rollout retry config: max_retries=${ROLLOUT_GENERATION_MAX_RETRIES} initial_backoff=${ROLLOUT_GENERATION_RETRY_INITIAL_BACKOFF}s max_backoff=${ROLLOUT_GENERATION_RETRY_MAX_BACKOFF}s multiplier=${ROLLOUT_GENERATION_RETRY_BACKOFF_MULTIPLIER} env_storm_max_retries=${ROLLOUT_GENERATION_ENV_STORM_MAX_RETRIES} skip_on_failure=${ROLLOUT_GENERATION_SKIP_ON_FAILURE}"
log "Exploration config: profile=${EXPLORATION_PROFILE} entropy=${EXPLORE_ENTROPY_COEF} intrinsic=${EXPLORE_INTRINSIC_ENABLED}/${EXPLORE_INTRINSIC} coef=${EXPLORE_INTRINSIC_COEF} schedule=${EXPLORE_INTRINSIC_SCHEDULE}/${EXPLORE_INTRINSIC_DECAY_STEPS} reducer=${EXPLORE_INTRINSIC_REDUCER} granularity=${EXPLORE_INTRINSIC_GRANULARITY} scope=${EXPLORE_INTRINSIC_SCOPE} score_components=${EXPLORE_SCORE_BONUS_COMPONENTS} safety_filter=${EXPLORE_SAFETY_FILTER_ENABLED}/${EXPLORE_SAFETY_FILTER} lprnd=${EXPLORE_LPRND_ENABLED}/${EXPLORE_LPRND} coef=${EXPLORE_LPRND_COEF} schedule=${EXPLORE_LPRND_SCHEDULE}/${EXPLORE_LPRND_DECAY_STEPS} agent57=${EXPLORE_AGENT57_LITE_ENABLED}/${EXPLORE_AGENT57_LITE} k=${EXPLORE_AGENT57_K} controller=${EXPLORE_AGENT57_CONTROLLER} ucb_eps=${EXPLORE_AGENT57_UCB_EPSILON} ucb_min=${EXPLORE_AGENT57_UCB_MIN_PER_ARM} ucb_value=${EXPLORE_AGENT57_UCB_VALUE} dataset_aware=${EXPLORE_AGENT57_UCB_DATASET_AWARE} ucb_seed=${EXPLORE_AGENT57_UCB_RANDOM_SEED:-<legacy>} episodic=${EXPLORE_AGENT57_EPISODIC_BACKEND} episodic_obs=${EXPLORE_AGENT57_EPISODIC_OBS_MODE} episodic_turn=${EXPLORE_AGENT57_EPISODIC_INCLUDE_TURN} episodic_probe=${EXPLORE_AGENT57_EPISODIC_MULTI_PROBE_RADIUS} episodic_floor=${EXPLORE_AGENT57_EPISODIC_NOVELTY_FLOOR} combine=${EXPLORE_AGENT57_COMBINE_MODE} ngu_clip=${EXPLORE_AGENT57_NGU_MOD_CLIP} ngu_reducer=${EXPLORE_AGENT57_NGU_EPISODIC_REDUCER} life_mod=${EXPLORE_AGENT57_NGU_LIFE_MOD_MODE}/${EXPLORE_AGENT57_NGU_LIFE_MOD_STD_CLIP} max_bonus=${EXPLORE_AGENT57_MAX_BONUS} betas=${EXPLORE_AGENT57_ARM_BETAS} temps=${EXPLORE_AGENT57_ARM_TEMPERATURES:-<inherit>} temp_warmup=${EXPLORE_AGENT57_ARM_TEMPERATURE_WARMUP_ROLLOUTS} lifelong=${EXPLORE_AGENT57_LIFELONG_ENABLED}/${EXPLORE_AGENT57_LIFELONG} life_coef=${EXPLORE_AGENT57_LIFELONG_COEF} life_backend=${EXPLORE_AGENT57_LIFELONG_BACKEND} life_key=${EXPLORE_AGENT57_LIFELONG_KEY_VERSION} life_dataset=${EXPLORE_AGENT57_LIFELONG_INCLUDE_DATASET} life_task=${EXPLORE_AGENT57_LIFELONG_INCLUDE_TASK} life_turn=${EXPLORE_AGENT57_LIFELONG_INCLUDE_TURN} life_obs=${EXPLORE_AGENT57_LIFELONG_OBS_MODE} life_hier=${EXPLORE_AGENT57_LIFELONG_HIERARCHICAL} life_weights=${EXPLORE_AGENT57_LIFELONG_TASK_WEIGHT}/${EXPLORE_AGENT57_LIFELONG_SKILL_WEIGHT}/${EXPLORE_AGENT57_LIFELONG_GLOBAL_WEIGHT} sqlite_timeout_ms=${EXPLORE_AGENT57_SQLITE_BUSY_TIMEOUT_MS} sqlite_wal=${EXPLORE_AGENT57_SQLITE_WAL} life_decay=${EXPLORE_AGENT57_LIFELONG_COUNT_DECAY} life_capacity=${EXPLORE_AGENT57_LIFELONG_CAPACITY} trust=${EXPLORE_AGENT57_TRUST_GATE} cde_actor=${EXPLORE_CDE_ACTOR_ENABLED}/${EXPLORE_CDE_ACTOR} omega=${EXPLORE_CDE_ACTOR_OMEGA} alpha=${EXPLORE_CDE_ACTOR_ALPHA} kappa=${EXPLORE_CDE_ACTOR_KAPPA} gate=${EXPLORE_CDE_ACTOR_REWARD_GATE} decay_steps=${EXPLORE_CDE_ACTOR_DECAY_STEPS} post_norm_bonus=${EXPLORE_ADVANTAGE_BONUS_ENABLED}/${EXPLORE_ADVANTAGE_BONUS} mode=${EXPLORE_ADVANTAGE_BONUS_MODE} components=${EXPLORE_ADVANTAGE_BONUS_COMPONENTS} coef=${EXPLORE_ADVANTAGE_BONUS_COEF} lambda=${EXPLORE_ADVANTAGE_LAMBDA} lambda_schedule=${EXPLORE_ADVANTAGE_LAMBDA_SCHEDULE}/${EXPLORE_ADVANTAGE_LAMBDA_DECAY_STEPS} intrinsic_key=${EXPLORE_ADVANTAGE_INTRINSIC_KEY} arm_weight=${EXPLORE_ADVANTAGE_ARM_WEIGHT_MODE} clip=${EXPLORE_ADVANTAGE_BONUS_CLIP} trunc_intr_scale=${EXPLORE_ADVANTAGE_TRUNCATED_INTRINSIC_SCALE} fail_intr_scale=${EXPLORE_ADVANTAGE_FAILED_INTRINSIC_SCALE} trunc_penalty=${EXPLORE_TRUNCATION_PENALTY} skip_zero_trainable=${SLIME_SKIP_ZERO_TRAINABLE_ROLLOUT}/${SLIME_SKIP_ZERO_TRAINABLE_TRAIN}"
if [[ "${ALGO}" == "dapo" ]]; then
  log "DAPO knobs: clip_low=${DAPO_EPS_CLIP_LOW} clip_high=${DAPO_EPS_CLIP_HIGH} token_loss=${DAPO_CALCULATE_PER_TOKEN_LOSS} dynamic_sampling=${DAPO_DYNAMIC_SAMPLING} failed_group_abort=${DAPO_FAILED_GROUP_ABORT_MIN_GROUPS}/${DAPO_FAILED_GROUP_ABORT_RATIO} overlong=${DAPO_OVERLONG_BUFFER_ENABLE}/${DAPO_OVERLONG_BUFFER_LEN}/${DAPO_OVERLONG_PENALTY_FACTOR}"
fi

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr 1e-6
  --lr-decay-style constant
  --weight-decay $([[ "${HARBORRL_NATIVE_ROLLOUT:-0}" == "1" ]] && echo 0.0 || echo 0.1)
  --adam-beta1 0.9
  --adam-beta2 0.98
  --clip-grad 1.0
)
if [[ "${OPTIMIZER_CPU_OFFLOAD:-1}" == "1" ]]; then
  # The cpu-offload optimizer package (hybrid optimizer).  Skip it for LoRA
  # smokes whose optimizer state is tiny: the hybrid path is heavier and has
  # more collective surface during initialization.
  OPTIMIZER_ARGS+=(
    --optimizer-cpu-offload
    --overlap-cpu-optimizer-d2h-h2d
    --use-precision-aware-optimizer
  )
fi

WANDB_MODE="${WANDB_MODE:-offline}"
WANDB_ENABLE="${WANDB_ENABLE:-1}"
case "${WANDB_ENABLE,,}" in
  1|true|yes|on)
    WANDB_ENABLE_RESOLVED=1
    ;;
  *)
    WANDB_ENABLE_RESOLVED=0
    ;;
esac

if [[ "${WANDB_MODE,,}" == "offline" ]]; then
  # Offline logging neither needs nor should propagate credentials into Ray
  # runtime metadata, process listings, or generated run configs.
  unset WANDB_API_KEY WANDB_KEY
fi

if [[ "${WANDB_MODE,,}" != "disabled" ]] && (( WANDB_ENABLE_RESOLVED )); then
  WANDB_ARGS=(
    --use-wandb
    --wandb-mode    "${WANDB_MODE}"
    --wandb-project "${WANDB_PROJECT:-terminal_rl}"
    --wandb-group   "${WANDB_GROUP:-${MODEL_TAG}_4gpu}"
    --wandb-dir     "${WANDB_DIR}"
  )
else
  WANDB_ARGS=()
fi

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}"
  # Large colocated models (for example GLM-5.1 TP16) need a higher static
  # pool than the historical 0.6 default.  Keep the old default while
  # allowing the model runner to select a validated value explicitly.
  --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.6}"
)
if [[ "${SGLANG_DISABLE_CUDA_GRAPH:-0}" == "1" ]]; then
  SGLANG_ARGS+=(--sglang-disable-cuda-graph)
fi
if [[ "${SGLANG_ENABLE_WEIGHTS_CPU_BACKUP:-0}" == "1" ]]; then
  # Keep a CPU copy of the SGLang weights so torch-memory-saver can unmap the
  # GPU weight pool during colocated Megatron initialization and restore it
  # before the next rollout.  This is essential for very large TP16 models.
  SGLANG_ARGS+=(--sglang-enable-weights-cpu-backup)
fi
if [[ -n "${SGLANG_ATTENTION_BACKEND:-}" ]]; then
  # ``torch_native`` is a safe fallback for memory-saver resume tests: the
  # installed SGLang 0.5.10 Triton token-pool writer can receive a CPU index
  # tensor after a full KV pause.  Keep the production default untouched and
  # let the smoke opt into the fallback explicitly.
  SGLANG_ARGS+=(--sglang-attention-backend "${SGLANG_ATTENTION_BACKEND}")
fi
if [[ -n "${SGLANG_SERVER_CONCURRENCY:-}" ]]; then
  SGLANG_ARGS+=(--sglang-server-concurrency "${SGLANG_SERVER_CONCURRENCY}")
fi
if [[ "${SLIME_USE_FAULT_TOLERANCE:-0}" == "1" ]]; then
  SGLANG_ARGS+=(--use-fault-tolerance)
fi
if [[ "${SAVE_EVAL_TRAJECTORIES:-0}" == "1" && -z "${SLIME_SAVE_DEBUG_ROLLOUT_DATA:-}" ]]; then
  # Keep evaluation traces under the current run directory.  The rollout
  # manager applies the eval-only scope, so training rollouts are not dumped.
  SLIME_SAVE_DEBUG_ROLLOUT_DATA="${RUN_DIR}/metrics/traces/eval_{rollout_id}.pt"
  export SLIME_SAVE_DEBUG_ROLLOUT_DATA
fi
if [[ -n "${SLIME_SAVE_DEBUG_ROLLOUT_DATA:-}" ]]; then
  SGLANG_ARGS+=(--save-debug-rollout-data "${SLIME_SAVE_DEBUG_ROLLOUT_DATA}")
fi

MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
  --no-gradient-accumulation-fusion
)

CUSTOM_ARGS=(
  --custom-rollout-log-function-path harborrl.misc.rollout_log.rollout_log
  --custom-eval-rollout-log-function-path harborrl.misc.rollout_log.eval_rollout_log
)
if [[ "${HARBORRL_NATIVE_ROLLOUT:-0}" != "1" ]]; then
  CUSTOM_ARGS+=(--custom-generate-function-path harborrl.rollout.entrypoint.generate)
fi
if [[ "${HARBORRL_NATIVE_ROLLOUT:-0}" == "1" ]]; then
  CUSTOM_ARGS+=(
    --loss-type custom_loss
    --custom-loss-function-path harborrl.rollout.exporters.native_slime.loss_function
    --custom-convert-samples-to-train-data-path harborrl.rollout.exporters.native_slime.convert_samples
  )
fi
if [[ "${EXPLORE_ADVANTAGE_BONUS_ENABLED}" == "1" ]]; then
  # Keep the historical hook as the default, while allowing a cluster-job variant to
  # test a drop-in post-process fix without forking this large launcher.
  CUSTOM_REWARD_POST_PROCESS_PATH="${CUSTOM_REWARD_POST_PROCESS_PATH:-harborrl.algorithms.dive_po.rewards.dual_stream.post_process_rewards}"
  CUSTOM_ARGS+=(--custom-reward-post-process-path "${CUSTOM_REWARD_POST_PROCESS_PATH}")
fi
# --custom-config-path is optional in slime; only attach it if the yaml exists.
if [[ -f "${CUSTOM_CONFIG_PATH}" ]]; then
  CUSTOM_ARGS+=(--custom-config-path "${CUSTOM_CONFIG_PATH}")
else
  echo "WARN: custom config not found at ${CUSTOM_CONFIG_PATH}; skipping --custom-config-path"
fi

LAYOUT_ARGS=()
if [[ "${HARBORRL_GPU_LAYOUT}" == colocate ]]; then
  LAYOUT_ARGS+=(--colocate --offload-train --offload-rollout)
fi

TRAIN_ARGS=(
  "${LAYOUT_ARGS[@]}"
  --actor-num-nodes "${ACTOR_NUM_NODES:-1}"
  --num-gpus-per-node "${NUM_GPUS}"
  --actor-num-gpus-per-node "${ACTOR_GPUS}"
  --num-gpus-per-node "${NUM_GPUS}"
  --rollout-num-gpus "${ROLLOUT_GPUS}"
  "${MODEL_ARGS[@]}"
  "${CKPT_ARGS[@]}"
  "${ROLLOUT_ARGS[@]}"
  "${OPTIMIZER_ARGS[@]}"
  "${ALGO_ARGS[@]}"
  "${ALGO_EXTRA_ARGS_ARRAY[@]}"
  "${WANDB_ARGS[@]}"
  "${PERF_ARGS[@]}"
  "${EVAL_ARGS[@]}"
  "${SGLANG_ARGS[@]}"
  "${MISC_ARGS[@]}"
  "${CUSTOM_ARGS[@]}"
)

if [[ "${DRY_RUN}" == "1" ]]; then
  log "DRY_RUN=1: final Slime command only; router/Ray/training will not start"
  printf '[dry-run] '
  # Keep dry-run output aligned with lib_launch.sh, which honors
  # SLIME_ENTRYPOINT for eval-only recipes as well as training.
  printf '%q ' "${TRAIN_PYTHON}" -u "${SLIME_ENTRYPOINT:-${SLIME_DIR}/train_async.py}" "${TRAIN_ARGS[@]}"
  printf '\n'
  exit 0
fi

# NOTE: safety reward params are passed via env vars (RUNTIME_ENV_JSON below),
# not CLI flags, because slime's argparse rejects unknown flags.
