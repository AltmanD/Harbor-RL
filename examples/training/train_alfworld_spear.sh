#!/usr/bin/env bash
set -euo pipefail

# HarborRL wrapper for the official SPEAR ALFWorld recipe.  The command below
# mirrors archive/SPEAR and parameterizes paths/resources needed by RJob.
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
OFFICIAL_ROOT="${SPEAR_VERL_AGENT_ROOT:-/mnt/shared-storage-user/puyuan/code/archive/SPEAR/verl-agent}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_PATH="${MODEL_PATH:-/mnt/shared-storage-user/puyuan/code/slime/Qwen3-8B}"
ALFWORLD_DATA_DIR="${ALFWORLD_DATA_DIR:-/mnt/shared-storage-user/puyuan/data/alfworld}"
ALFWORLD_HOME="${ALFWORLD_HOME:-/mnt/shared-storage-user/puyuan/runtime/alfworld-home}"
ENV_ROOT="${ALFWORLD_ENV_ROOT:-/mnt/shared-storage-user/puyuan/runtime/alfworld-environments}"
ENV_TAR="${ALFWORLD_ENV_TAR:-${OFFICIAL_ROOT}/agent_system/environments.tar}"
TRAIN_DATA_SIZE="${TRAIN_DATA_SIZE:-32}"
VAL_DATA_SIZE="${VAL_DATA_SIZE:-128}"
GROUP_SIZE="${GROUP_SIZE:-8}"
N_GPUS="${N_GPUS:-${RJOB_GPU:-2}}"
N_NODES="${N_NODES:-1}"
MAX_STEPS="${MAX_STEPS:-50}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-200}"
SAVE_FREQ="${SAVE_FREQ:-10}"
TEST_FREQ="${TEST_FREQ:-5}"
PROJECT_NAME="${PROJECT_NAME:-verl_agent_alfworld}"
EXP_NAME="${EXP_NAME:-grpo_qwen3_8b_spear}"
LOCAL_DIR="${LOCAL_DIR:-${OFFICIAL_ROOT}/checkpoint/${PROJECT_NAME}/${EXP_NAME}}"

[[ -d "${OFFICIAL_ROOT}/verl" ]] || { echo "missing SPEAR verl-agent runtime: ${OFFICIAL_ROOT}" >&2; exit 2; }
[[ -d "${MODEL_PATH}" ]] || { echo "missing model checkpoint: ${MODEL_PATH}" >&2; exit 2; }
mkdir -p "${ALFWORLD_DATA_DIR}" "${ALFWORLD_HOME}" "${ENV_ROOT}" \
  "${OFFICIAL_ROOT}/data/verl-agent/text" "${LOCAL_DIR}/rollout" "${LOCAL_DIR}/validation"

if [[ ! -f "${ENV_ROOT}/environments/__init__.py" ]]; then
  [[ -f "${ENV_TAR}" ]] || { echo "missing ALFWorld environment archive: ${ENV_TAR}" >&2; exit 2; }
  tar -xf "${ENV_TAR}" -C "${ENV_ROOT}"
fi

export HOME="${ALFWORLD_HOME}"
export ALFWORLD_DATA="${ALFWORLD_DATA_DIR}"
export ALFWORLD_DATA_DIR
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
export PYTHONPATH="${ENV_ROOT}:${OFFICIAL_ROOT}:${ROOT}:${PYTHONPATH:-}"

if [[ ! -d "${ALFWORLD_DATA_DIR}/json_2.1.1" && ! -d "${ALFWORLD_DATA_DIR}/generated" ]]; then
  ALFWORLD_DATA_DIR="${ALFWORLD_DATA_DIR}" ALFWORLD_HOME="${ALFWORLD_HOME}" \
    ALFWORLD_VENV="${ALFWORLD_VENV:-/mnt/shared-storage-user/puyuan/runtime/alfworld-venv}" \
    bash "${ROOT}/examples/training/download_alfworld.sh"
fi

cd "${OFFICIAL_ROOT}"
"${PYTHON_BIN}" -m examples.data_preprocess.prepare \
  --mode text --train_data_size "${TRAIN_DATA_SIZE}" --val_data_size "${VAL_DATA_SIZE}"

exec "${PYTHON_BIN}" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="${OFFICIAL_ROOT}/data/verl-agent/text/train.parquet" \
  data.val_files="${OFFICIAL_ROOT}/data/verl-agent/text/test.parquet" \
  data.train_batch_size="${TRAIN_DATA_SIZE}" data.val_batch_size="${VAL_DATA_SIZE}" \
  data.max_prompt_length=2048 data.max_response_length=512 \
  data.filter_overlong_prompts=True data.truncation=error data.return_raw_chat=True \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.actor.policy_loss.loss_mode=clip_cov \
  actor_rollout_ref.actor.policy_loss.clip_cov_ratio_replay=0.02 \
  actor_rollout_ref.actor.policy_loss.clip_cov_lb_replay=2 \
  actor_rollout_ref.actor.policy_loss.clip_cov_ub_replay=60.0 \
  actor_rollout_ref.actor.policy_loss.kl_cov_ratio_replay=0.02 \
  actor_rollout_ref.actor.clip_ratio_c=10.0 actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=1024 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.actor.use_kl_loss=False actor_rollout_ref.actor.kl_loss_coef=0.0 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.clip_ratio_low=0.2 actor_rollout_ref.actor.clip_ratio_high=0.28 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.enforce_eager=False actor_rollout_ref.rollout.free_cache_engine=False \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.use_invalid_action_penalty=True \
  actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
  actor_rollout_ref.actor.enable_trajectory_replay=True \
  actor_rollout_ref.actor.replay_loss_coef=1 \
  actor_rollout_ref.actor.weight_decay_trajectory_replay=-1 \
  actor_rollout_ref.actor.trajectory_buffer_size=2048 \
  actor_rollout_ref.actor.baseline_buffer_size=10240 \
  actor_rollout_ref.actor.trajectory_score_threshold=1 \
  actor_rollout_ref.actor.trajectory_tolerate_steps=5 \
  actor_rollout_ref.rollout.rollout_filter_type=std \
  actor_rollout_ref.rollout.rollout_filter_ratio=0.75 \
  actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum-norm \
  algorithm.norm_adv_by_std_in_grpo=False \
  algorithm.filter_overlong_responses=True algorithm.filter_incomplete_responses=True \
  algorithm.filter_repetitive_responses=True algorithm.filter_unreadable_responses=True \
  algorithm.use_kl_in_reward=False algorithm.use_toolcall_reward=cosine \
  algorithm.max_toolcall_steps=200 \
  env.env_name=alfworld/AlfredTWEnv env.resources_per_worker.num_cpus=0.1 \
  env.seed=0 env.max_steps="${MAX_STEPS}" env.rollout.n="${GROUP_SIZE}" \
  trainer.critic_warmup=0 trainer.logger="['console','wandb']" \
  trainer.project_name="${PROJECT_NAME}" trainer.experiment_name="${EXP_NAME}" \
  trainer.default_local_dir="${LOCAL_DIR}" \
  trainer.rollout_data_dir="${LOCAL_DIR}/rollout" \
  trainer.validation_data_dir="${LOCAL_DIR}/validation" \
  trainer.n_gpus_per_node="${N_GPUS}" trainer.nnodes="${N_NODES}" \
  trainer.save_freq="${SAVE_FREQ}" trainer.test_freq="${TEST_FREQ}" \
  trainer.total_epochs="${TOTAL_EPOCHS}" trainer.val_before_train=False "$@"
