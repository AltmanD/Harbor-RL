#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
RUNS_ROOT="${RUNS_ROOT:-${REPO_ROOT}/runs}"

WM_TRAJECTORIES="${WM_TRAJECTORIES:-${RUNS_ROOT}/training/terminal-rl_qwen3-8b_8gpu_seta_dapo_nodynamic_exploration_simhash_life_fp_ucb_v0710_k6_quality_balanced1000_epturn0_none_dualadv_think_2026-07-08_185609/trajectories}"
WM_OUTPUT_DIR="${WM_OUTPUT_DIR:-${RUNS_ROOT}/training/world_model_seta_latent/$(date +%Y%m%d_%H%M%S)}"
WM_ENCODER="${WM_ENCODER:-hash}"
WM_HF_MODEL="${WM_HF_MODEL:-${REPO_ROOT}/models/Qwen3-8B}"
WM_MAX_TRAJECTORIES="${WM_MAX_TRAJECTORIES:-4}"
WM_MAX_TRANSITIONS="${WM_MAX_TRANSITIONS:-32}"
WM_EPOCHS="${WM_EPOCHS:-1}"
WM_BATCH_SIZE="${WM_BATCH_SIZE:-8}"
WM_LATENT_DIM="${WM_LATENT_DIM:-64}"
WM_BACKPROP_TO_LLM="${WM_BACKPROP_TO_LLM:-0}"
WM_SAVE_UPDATED_LLM="${WM_SAVE_UPDATED_LLM:-0}"
WM_LLM_LR="${WM_LLM_LR:-1e-6}"
WM_USE_DAPO_REPLAY_BUFFER="${WM_USE_DAPO_REPLAY_BUFFER:-0}"
WM_REPLAY_BUFFER_SIZE="${WM_REPLAY_BUFFER_SIZE:-2048}"
PYTHON_BIN="${PYTHON_BIN:-python}"

args=(
  --input "${WM_TRAJECTORIES}"
  --output-dir "${WM_OUTPUT_DIR}"
  --encoder "${WM_ENCODER}"
  --max-trajectories "${WM_MAX_TRAJECTORIES}"
  --max-transitions "${WM_MAX_TRANSITIONS}"
  --epochs "${WM_EPOCHS}"
  --batch-size "${WM_BATCH_SIZE}"
  --latent-dim "${WM_LATENT_DIM}"
  --llm-lr "${WM_LLM_LR}"
  --replay-buffer-size "${WM_REPLAY_BUFFER_SIZE}"
  --predictor-type adaln
  --predictor-num-heads 4
  --val-ratio 0.2
)

if [[ "${WM_ENCODER}" == "hf-policy" ]]; then
  args+=(--hf-model "${WM_HF_MODEL}" --hf-local-files-only)
fi
if [[ "${WM_BACKPROP_TO_LLM}" == "1" ]]; then
  args+=(--backprop-to-llm)
fi
if [[ "${WM_SAVE_UPDATED_LLM}" == "1" ]]; then
  args+=(--save-updated-llm)
fi
if [[ "${WM_USE_DAPO_REPLAY_BUFFER}" == "1" ]]; then
  args+=(--use-dapo-replay-buffer)
fi

PYTHONPATH="${REPO_ROOT}/backends/slime:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON_BIN}" -m slime.world_model.train_latent "${args[@]}"

echo "[wm-seta-latent] outputs: ${WM_OUTPUT_DIR}"
