#!/usr/bin/env bash
# Run one reproducible streaming LWM A/B (noreplay vs replay) on tb2.1
# trajectories.  The same script is suitable as the command passed to rjob;
# it has no cluster-specific assumptions.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." >/dev/null 2>&1 && pwd)"
cd "${REPO_ROOT}"

RUNS_ROOT="${RUNS_ROOT:-${REPO_ROOT}/runs}"
WM_INPUT="${WM_INPUT:-${RUNS_ROOT}/evaluation}"
WM_SUPPLEMENT_INPUT="${WM_SUPPLEMENT_INPUT:-${RUNS_ROOT}/training}"
WM_ENCODER="${WM_ENCODER:-hash}"
WM_HF_MODEL="${WM_HF_MODEL:-${REPO_ROOT}/models/Qwen3-8B}"
WM_STAMP="${WM_STAMP:-$(date +%Y%m%d-%H%M%S)}"
WM_SEED="${WM_SEED:-42}"
WM_OUTPUT_DIR="${WM_OUTPUT_DIR:-${RUNS_ROOT}/training/lwm_stream_verify/stream_seed${WM_SEED}_${WM_STAMP}}"
WM_MAX_TRAJECTORIES="${WM_MAX_TRAJECTORIES:-256}"
WM_MAX_TRANSITIONS="${WM_MAX_TRANSITIONS:-0}"
WM_BATCH_SIZE="${WM_BATCH_SIZE:-32}"
WM_LATENT_DIM="${WM_LATENT_DIM:-128}"
WM_LR="${WM_LR:-1e-4}"
WM_VALUE_COEF="${WM_VALUE_COEF:-0.0}"
WM_BACKPROP_TO_LLM="${WM_BACKPROP_TO_LLM:-0}"
WM_STREAM_CHUNKS="${WM_STREAM_CHUNKS:-8}"
WM_STREAM_ARMS="${WM_STREAM_ARMS:-noreplay,replay}"
WM_STREAM_EPOCHS_PER_CHUNK="${WM_STREAM_EPOCHS_PER_CHUNK:-1}"
WM_REPLAY_BUFFER_SIZE="${WM_REPLAY_BUFFER_SIZE:-4096}"
WM_REPLAY_RATIO="${WM_REPLAY_RATIO:-0.5}"
WM_REPLAY_WARMUP_CHUNKS="${WM_REPLAY_WARMUP_CHUNKS:-0}"
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
  --batch-size "${WM_BATCH_SIZE}"
  --encode-batch-size "${WM_ENCODE_BATCH_SIZE:-8}"
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
  --stream-chunks "${WM_STREAM_CHUNKS}"
  --stream-arms "${WM_STREAM_ARMS}"
  --stream-epochs-per-chunk "${WM_STREAM_EPOCHS_PER_CHUNK}"
  --replay-buffer-size "${WM_REPLAY_BUFFER_SIZE}"
  --replay-ratio "${WM_REPLAY_RATIO}"
  --replay-warmup-chunks "${WM_REPLAY_WARMUP_CHUNKS}"
)
if [[ "${WM_ENCODER}" == "hf-policy" ]]; then
  args+=(--hf-model "${WM_HF_MODEL}" --hf-local-files-only)
fi
if [[ "${WM_BACKPROP_TO_LLM}" == "1" ]]; then args+=(--backprop-to-llm); fi
if [[ "${WM_REQUIRE_TOOL_FEEDBACK:-0}" == "1" ]]; then args+=(--require-tool-feedback); fi
if [[ "${WM_STREAM_KEEP_SOURCE_ORDER:-0}" == "1" ]]; then args+=(--stream-keep-source-order); fi
if [[ "${WM_DRY_RUN:-0}" == "1" ]]; then
  printf '[lwm-stream] arms=%s chunks=%s input=%s output=%s encoder=%s\n' "${WM_STREAM_ARMS}" "${WM_STREAM_CHUNKS}" "${WM_INPUT}" "${WM_OUTPUT_DIR}" "${WM_ENCODER}"
  printf '[lwm-stream] command:'; printf ' %q' "${PYTHON_BIN}" -m slime.world_model.stream_latent "${args[@]}"; printf '\n'
  exit 0
fi

mkdir -p "${WM_OUTPUT_DIR}/logs"
printf '[lwm-stream] arms=%s chunks=%s input=%s supplement=%s output=%s encoder=%s seed=%s\n' \
  "${WM_STREAM_ARMS}" "${WM_STREAM_CHUNKS}" "${WM_INPUT}" "${WM_SUPPLEMENT_INPUT}" "${WM_OUTPUT_DIR}" "${WM_ENCODER}" "${WM_SEED}" \
  | tee "${WM_OUTPUT_DIR}/logs/phase.log"
PYTHONPATH="${REPO_ROOT}/backends/slime:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON_BIN}" -m slime.world_model.stream_latent "${args[@]}" \
  2>&1 | tee -a "${WM_OUTPUT_DIR}/logs/phase.log"
echo "[lwm-stream] completed: ${WM_OUTPUT_DIR}"
