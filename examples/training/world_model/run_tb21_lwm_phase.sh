#!/usr/bin/env bash
# Run one reproducible offline LWM phase.  The same script is suitable as the
# command passed to rjob; it has no cluster-specific assumptions.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." >/dev/null 2>&1 && pwd)"
cd "${REPO_ROOT}"

PHASE="${WM_PHASE:-baseline}"
case "${PHASE}" in baseline|replay|value_mpc) ;; *) echo "[lwm] WM_PHASE must be baseline, replay, or value_mpc" >&2; exit 2 ;; esac

RUNS_ROOT="${RUNS_ROOT:-${REPO_ROOT}/runs}"
WM_INPUT="${WM_INPUT:-${RUNS_ROOT}/evaluation}"
WM_SUPPLEMENT_INPUT="${WM_SUPPLEMENT_INPUT:-${RUNS_ROOT}/training}"
WM_ENCODER="${WM_ENCODER:-hash}"
WM_HF_MODEL="${WM_HF_MODEL:-${REPO_ROOT}/models/Qwen3-8B}"
WM_STAMP="${WM_STAMP:-$(date +%Y%m%d-%H%M%S)}"
WM_SEED="${WM_SEED:-42}"
WM_OUTPUT_DIR="${WM_OUTPUT_DIR:-${RUNS_ROOT}/training/lwm_offline_verify/${PHASE}_seed${WM_SEED}_${WM_STAMP}}"
WM_MAX_TRAJECTORIES="${WM_MAX_TRAJECTORIES:-256}"
WM_MAX_TRANSITIONS="${WM_MAX_TRANSITIONS:-0}"
WM_EPOCHS="${WM_EPOCHS:-5}"
WM_BATCH_SIZE="${WM_BATCH_SIZE:-32}"
WM_LATENT_DIM="${WM_LATENT_DIM:-128}"
WM_LR="${WM_LR:-1e-4}"
WM_VALUE_COEF="${WM_VALUE_COEF:-0.0}"
WM_BACKPROP_TO_LLM="${WM_BACKPROP_TO_LLM:-0}"
WM_REPLAY_BUFFER_SIZE="${WM_REPLAY_BUFFER_SIZE:-4096}"
WM_REPLAY_RATIO="${WM_REPLAY_RATIO:-1.0}"
WM_REPLAY_SAMPLES_PER_EPOCH="${WM_REPLAY_SAMPLES_PER_EPOCH:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ "${WM_MAX_TRANSITIONS}" == "0" ]]; then
  WM_MAX_TRANSITIONS_ARG=()
else
  WM_MAX_TRANSITIONS_ARG=(--max-transitions "${WM_MAX_TRANSITIONS}")
fi

args=(
  --input "${WM_INPUT}"
  --data-source auto
  --supplement-input "${WM_SUPPLEMENT_INPUT}"
  --output-dir "${WM_OUTPUT_DIR}"
  --encoder "${WM_ENCODER}"
  --max-trajectories "${WM_MAX_TRAJECTORIES}"
  "${WM_MAX_TRANSITIONS_ARG[@]}"
  --phase "${PHASE}"
  --epochs "${WM_EPOCHS}"
  --batch-size "${WM_BATCH_SIZE}"
  --latent-dim "${WM_LATENT_DIM}"
  --lr "${WM_LR}"
  --value-coef "${WM_VALUE_COEF}"
  --gamma "${WM_GAMMA:-0.99}"
  --gradient-clip "${WM_GRADIENT_CLIP:-1.0}"
  --seed "${WM_SEED}"
  --stop-grad-target
  --predictor-type adaln
  --predictor-num-heads "${WM_PREDICTOR_HEADS:-4}"
  --val-ratio "${WM_VAL_RATIO:-0.2}"
)
if [[ "${WM_ENCODER}" == "hf-policy" ]]; then
  args+=(--hf-model "${WM_HF_MODEL}" --hf-local-files-only)
fi
if [[ "${WM_BACKPROP_TO_LLM}" == "1" ]]; then args+=(--backprop-to-llm); fi
if [[ "${PHASE}" == "replay" || "${WM_USE_DAPO_REPLAY_BUFFER:-0}" == "1" ]]; then
  args+=(--replay --replay-buffer-size "${WM_REPLAY_BUFFER_SIZE}" --replay-ratio "${WM_REPLAY_RATIO}" --replay-samples-per-epoch "${WM_REPLAY_SAMPLES_PER_EPOCH}")
fi
if [[ "${WM_REQUIRE_TOOL_FEEDBACK:-0}" == "1" ]]; then args+=(--require-tool-feedback); fi
if [[ "${WM_DRY_RUN:-0}" == "1" ]]; then
  printf '[lwm] phase=%s input=%s supplement=%s output=%s encoder=%s\n' "${PHASE}" "${WM_INPUT}" "${WM_SUPPLEMENT_INPUT}" "${WM_OUTPUT_DIR}" "${WM_ENCODER}"
  printf '[lwm] command:'; printf ' %q' "${PYTHON_BIN}" -m slime.world_model.train_latent "${args[@]}"; printf '\n'
  exit 0
fi

mkdir -p "${WM_OUTPUT_DIR}/logs"
printf '[lwm] phase=%s input=%s supplement=%s output=%s encoder=%s seed=%s\n' \
  "${PHASE}" "${WM_INPUT}" "${WM_SUPPLEMENT_INPUT}" "${WM_OUTPUT_DIR}" "${WM_ENCODER}" "${WM_SEED}" \
  | tee "${WM_OUTPUT_DIR}/logs/phase.log"
PYTHONPATH="${REPO_ROOT}/backends/slime:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON_BIN}" -m slime.world_model.train_latent "${args[@]}" \
  2>&1 | tee -a "${WM_OUTPUT_DIR}/logs/phase.log"

if [[ "${PHASE}" == "value_mpc" && -f "${WM_OUTPUT_DIR}/latent_world_model.pt" && -f "${WM_OUTPUT_DIR}/hidden_cache.pt" ]]; then
  PYTHONPATH="${REPO_ROOT}/backends/slime:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON_BIN}" -m slime.world_model.plan_mpc \
      --checkpoint "${WM_OUTPUT_DIR}/latent_world_model.pt" \
      --input "${WM_OUTPUT_DIR}/hidden_cache.pt" \
      --state-index "${WM_MPC_STATE_INDEX:-0}" \
      --output "${WM_OUTPUT_DIR}/mpc_plan.json" \
      2>&1 | tee -a "${WM_OUTPUT_DIR}/logs/phase.log"
fi
echo "[lwm] completed: ${WM_OUTPUT_DIR}"
