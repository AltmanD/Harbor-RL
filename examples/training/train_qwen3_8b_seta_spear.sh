#!/usr/bin/env bash
# Qwen3-8B + SETA + DAPO with the SPEAR self-imitation replay baseline.
#
# SPEAR is enabled through the integrated slime replay path. Successful
# trajectories (score >= TRAJECTORY_SCORE_THRESHOLD and, by default, positive
# group-relative advantage) enter a bounded SIL FIFO. The FIFO also keeps the
# historical p50 reward baseline and removes stale trajectories, matching the
# reference SPEAR replay discipline.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)"
cd "${REPO_ROOT}"

: "${WORKER_URLS:?set WORKER_URLS to the comma-separated SETA worker endpoint(s)}"
export MODEL_TAG="${MODEL_TAG:-qwen3-8b}"
export MODEL_ARGS_FILE="${MODEL_ARGS_FILE:-qwen3-8B}"
export DATASET="${DATASET:-seta}"
export ALGO="${ALGO:-dapo}"
export HARNESS_OPTION="${HARNESS_OPTION:-camel-agent}"
export DAPO_DYNAMIC_SAMPLING="${DAPO_DYNAMIC_SAMPLING:-0}"
export NUM_GPUS="${NUM_GPUS:-4}"
export ACTOR_GPUS="${ACTOR_GPUS:-2}"
export ROLLOUT_GPUS="${ROLLOUT_GPUS:-2}"
export TP_SIZE="${TP_SIZE:-2}"
export ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-2}"
# Keep evaluation inexpensive for the SPEAR baseline: evaluate pass@1 only,
# use the paper's sampling settings, and checkpoint/evaluate less frequently.
export EVAL_N_SAMPLES="${EVAL_N_SAMPLES:-1}"
export EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-1.0}"
export EVAL_TOP_P="${EVAL_TOP_P:-0.7}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-200}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-200}"
# Ask the platform launcher to place eval-only debug trajectories below this
# run's metrics directory.  An explicit SLIME_SAVE_DEBUG_ROLLOUT_DATA still
# takes precedence when a different path is needed.
export SAVE_EVAL_TRAJECTORIES="${SAVE_EVAL_TRAJECTORIES:-1}"

_spear_args=(
  --loss-type decoupled_policy_loss
  --use-rollout-logprobs
  --trajectory-buffer-size "${TRAJECTORY_BUFFER_SIZE:-2048}"
  --trajectory-score-threshold "${TRAJECTORY_SCORE_THRESHOLD:-1.0}"
  --baseline-buffer-size "${BASELINE_BUFFER_SIZE:-10240}"
  --trajectory-tolerate-steps "${TRAJECTORY_TOLERATE_STEPS:-10}"
  --replay-loss-coef "${REPLAY_LOSS_COEF:-0.001}"
  --max-replay-loss-steps "${MAX_REPLAY_LOSS_STEPS:-200}"
  --train-iters-per-rollout "${TRAIN_ITERS_PER_ROLLOUT:-2}"
  --buffer-sampling-strategy "${BUFFER_SAMPLING_STRATEGY:-fifo_staleness}"
  --buffer-max-size "${BUFFER_MAX_SIZE:-1000}"
  --max-staleness "${MAX_STALENESS:-2}"
  --prox-logp-method "${PROX_LOGP_METHOD:-loglinear}"
)
if [[ "${ENABLE_TRAJECTORY_REPLAY:-1}" == "1" ]]; then
  _spear_args+=(--enable-trajectory-replay)
fi
if [[ "${ENABLE_TRAJECTORY_POSADV:-1}" == "1" ]]; then
  _spear_args+=(--enable-trajectory-posadv)
fi
if [[ "${BUFFER_REMOVE_ON_SAMPLE:-0}" == "0" ]]; then
  _spear_args+=(--buffer-remove-on-sample false --buffer-reuse-samples "${BUFFER_REUSE_SAMPLES:-4}")
fi

_joined="${EXTRA_ALGO_ARGS:-}"
for _arg in "${_spear_args[@]}"; do _joined+=" ${_arg}"; done
export EXTRA_ALGO_ARGS="${_joined}"

exec bash harborrl/platform/slime_train.sh "$@"
