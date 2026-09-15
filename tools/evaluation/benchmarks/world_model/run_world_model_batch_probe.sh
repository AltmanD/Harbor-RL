#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"
RUNS_ROOT="${RUNS_ROOT:-${REPO_ROOT}/runs}"

PYTHON_BIN="${PYTHON_BIN:-python3}"

WM_OUT_DIR="${WM_OUT_DIR:-${RUNS_ROOT}/evaluation/world_model_probe_batch/$(date +%Y%m%d_%H%M%S)}"
WM_INPUT_GLOB="${WM_INPUT_GLOB:-${RUNS_ROOT}/testing/world_model_smoke/*/metadata/rollout_*.pt}"
WM_INPUTS="${WM_INPUTS:-}"
WM_RECORDS="${WM_RECORDS:-}"
WM_CONTEXT_MAX_CHARS="${WM_CONTEXT_MAX_CHARS:-4096}"

WM_ENCODER="${WM_ENCODER:-hash}"
WM_ALLOW_HF="${WM_ALLOW_HF:-0}"
WM_HASH_HIDDEN_DIM="${WM_HASH_HIDDEN_DIM:-128}"
WM_HF_MODEL="${WM_HF_MODEL:-}"
WM_HF_MAX_LENGTH="${WM_HF_MAX_LENGTH:-2048}"
WM_HF_POOLING="${WM_HF_POOLING:-mean}"
WM_CACHE_BATCH_SIZE="${WM_CACHE_BATCH_SIZE:-4}"
WM_DEVICE="${WM_DEVICE:-auto}"

if [[ "${WM_ENCODER}" == "hf" ]]; then
  WM_LATENT_DIM="${WM_LATENT_DIM:-1024}"
else
  WM_LATENT_DIM="${WM_LATENT_DIM:-128}"
fi
WM_EPOCHS="${WM_EPOCHS:-3}"
WM_TRAIN_BATCH_SIZE="${WM_TRAIN_BATCH_SIZE:-16}"
WM_LR="${WM_LR:-1e-4}"
WM_SIGREG_COEF="${WM_SIGREG_COEF:-0.1}"
WM_ACTION_CONTRAST_COEF="${WM_ACTION_CONTRAST_COEF:-0.1}"
WM_VALUE_COEF="${WM_VALUE_COEF:-0.0}"
WM_VAL_RATIO="${WM_VAL_RATIO:-0.1}"
WM_SEED="${WM_SEED:-42}"

mkdir -p "${WM_OUT_DIR}/records_shards"

collect_inputs() {
  local specs=()
  local spec
  local matches=()
  shopt -s nullglob
  if [[ -n "${WM_INPUTS}" ]]; then
    # Space-separated paths or globs. Paths with spaces are intentionally unsupported for simple cluster usage.
    read -r -a specs <<< "${WM_INPUTS}"
    for spec in "${specs[@]}"; do
      matches=( ${spec} )
      if (( ${#matches[@]} == 0 )); then
        echo "[wm-batch-probe] warning: no matches for ${spec}" >&2
      else
        printf '%s\n' "${matches[@]}"
      fi
    done
  else
    matches=( ${WM_INPUT_GLOB} )
    printf '%s\n' "${matches[@]}"
  fi
  shopt -u nullglob
}

if [[ -z "${WM_RECORDS}" ]]; then
  mapfile -t INPUT_FILES < <(collect_inputs | sort -u)
  if (( ${#INPUT_FILES[@]} == 0 )); then
    echo "[wm-batch-probe] no rollout inputs found. Set WM_INPUTS or WM_INPUT_GLOB." >&2
    exit 1
  fi

  COMBINED_RECORDS="${WM_OUT_DIR}/records.jsonl"
  : > "${COMBINED_RECORDS}"
  shard_idx=0
  for input in "${INPUT_FILES[@]}"; do
    shard="${WM_OUT_DIR}/records_shards/$(printf '%05d' "${shard_idx}").jsonl"
    echo "[wm-batch-probe] extract records: ${input}"
    PYTHONPATH="${REPO_ROOT}/backends/slime:${REPO_ROOT}/harborrl" "${PYTHON_BIN}" -m slime.world_model.build_dataset \
      --input "${input}" \
      --output "${shard}" \
      --context-max-chars "${WM_CONTEXT_MAX_CHARS}"
    if [[ -s "${shard}" ]]; then
      cat "${shard}" >> "${COMBINED_RECORDS}"
    fi
    shard_idx=$((shard_idx + 1))
  done
  WM_RECORDS="${COMBINED_RECORDS}"
fi

if [[ ! -s "${WM_RECORDS}" ]]; then
  echo "[wm-batch-probe] records file not found or empty: ${WM_RECORDS}" >&2
  exit 1
fi

RECORD_COUNT="$(wc -l < "${WM_RECORDS}" | tr -d ' ')"
echo "[wm-batch-probe] records: ${WM_RECORDS} (${RECORD_COUNT})"
echo "[wm-batch-probe] output:  ${WM_OUT_DIR}"

CACHE_ARGS=(
  -m slime.world_model.cache_text_hidden
  --input "${WM_RECORDS}"
  --output "${WM_OUT_DIR}/cached_hidden.pt"
  --encoder "${WM_ENCODER}"
  --batch-size "${WM_CACHE_BATCH_SIZE}"
  --device "${WM_DEVICE}"
)

if [[ "${WM_ENCODER}" == "hash" ]]; then
  CACHE_ARGS+=(--hidden-dim "${WM_HASH_HIDDEN_DIM}")
else
  if [[ "${WM_ALLOW_HF}" != "1" ]]; then
    echo "[wm-batch-probe] HF encoder is fail-closed. Set WM_ALLOW_HF=1 and WM_HF_MODEL=/local/path." >&2
    exit 1
  fi
  if [[ -z "${WM_HF_MODEL}" || ! -d "${WM_HF_MODEL}" ]]; then
    echo "[wm-batch-probe] missing local WM_HF_MODEL: ${WM_HF_MODEL}" >&2
    exit 1
  fi
  CACHE_ARGS+=(--hf-model "${WM_HF_MODEL}" --max-length "${WM_HF_MAX_LENGTH}" --pooling "${WM_HF_POOLING}")
fi

PYTHONPATH="${REPO_ROOT}/backends/slime:${REPO_ROOT}/harborrl" "${PYTHON_BIN}" "${CACHE_ARGS[@]}"

PYTHONPATH="${REPO_ROOT}/backends/slime:${REPO_ROOT}/harborrl" "${PYTHON_BIN}" -m slime.world_model.train_probe \
  --input "${WM_OUT_DIR}/cached_hidden.pt" \
  --output "${WM_OUT_DIR}/probe.pt" \
  --latent-dim "${WM_LATENT_DIM}" \
  --batch-size "${WM_TRAIN_BATCH_SIZE}" \
  --epochs "${WM_EPOCHS}" \
  --lr "${WM_LR}" \
  --sigreg-coef "${WM_SIGREG_COEF}" \
  --action-contrast-coef "${WM_ACTION_CONTRAST_COEF}" \
  --value-coef "${WM_VALUE_COEF}" \
  --val-ratio "${WM_VAL_RATIO}" \
  --seed "${WM_SEED}"

PYTHONPATH="${REPO_ROOT}/backends/slime:${REPO_ROOT}/harborrl" "${PYTHON_BIN}" -m slime.world_model.rank_candidates \
  --checkpoint "${WM_OUT_DIR}/probe.pt" \
  --input "${WM_OUT_DIR}/cached_hidden.pt" \
  --output "${WM_OUT_DIR}/rankings.jsonl"

echo "[wm-batch-probe] done. Outputs: ${WM_OUT_DIR}"
