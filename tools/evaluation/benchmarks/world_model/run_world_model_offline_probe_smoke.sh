#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"
RUNS_ROOT="${RUNS_ROOT:-${REPO_ROOT}/runs}"

PYTHON_BIN="${PYTHON_BIN:-python3}"

WM_RECORDS="${WM_RECORDS:-}"
WM_OUT_DIR="${WM_OUT_DIR:-${RUNS_ROOT}/evaluation/world_model_probe_smoke/$(date +%Y%m%d_%H%M%S)}"
WM_ENCODER="${WM_ENCODER:-hash}"
WM_ALLOW_HF="${WM_ALLOW_HF:-0}"
WM_HASH_HIDDEN_DIM="${WM_HASH_HIDDEN_DIM:-128}"
WM_LATENT_DIM="${WM_LATENT_DIM:-32}"
WM_EPOCHS="${WM_EPOCHS:-3}"
WM_BATCH_SIZE="${WM_BATCH_SIZE:-4}"
WM_LR="${WM_LR:-1e-3}"
WM_SIGREG_COEF="${WM_SIGREG_COEF:-0.0}"
WM_ACTION_CONTRAST_COEF="${WM_ACTION_CONTRAST_COEF:-0.0}"
WM_VALUE_COEF="${WM_VALUE_COEF:-0.0}"
WM_HF_MODEL="${WM_HF_MODEL:-}"
WM_HF_MAX_LENGTH="${WM_HF_MAX_LENGTH:-2048}"
WM_HF_POOLING="${WM_HF_POOLING:-mean}"

mkdir -p "${WM_OUT_DIR}"

if [[ -z "${WM_RECORDS}" ]]; then
  WM_RECORDS="$(find "${RUNS_ROOT}/testing/world_model_smoke" -path "*/metadata/records.jsonl" -type f 2>/dev/null | sort | tail -1 || true)"
fi

if [[ ! -s "${WM_RECORDS}" ]]; then
  echo "[wm-offline-probe] records file not found or empty. Set WM_RECORDS=/path/to/records.jsonl." >&2
  exit 1
fi

echo "[wm-offline-probe] records: ${WM_RECORDS}"
echo "[wm-offline-probe] output:  ${WM_OUT_DIR}"

CACHE_ARGS=(
  -m slime.world_model.cache_text_hidden
  --input "${WM_RECORDS}"
  --output "${WM_OUT_DIR}/cached_hidden.pt"
  --encoder "${WM_ENCODER}"
)

if [[ "${WM_ENCODER}" == "hash" ]]; then
  CACHE_ARGS+=(--hidden-dim "${WM_HASH_HIDDEN_DIM}")
else
  if [[ "${WM_ALLOW_HF}" != "1" ]]; then
    echo "[wm-offline-probe] HF encoder is fail-closed. Set WM_ALLOW_HF=1 and WM_HF_MODEL=/local/path." >&2
    exit 1
  fi
  if [[ -z "${WM_HF_MODEL}" || ! -d "${WM_HF_MODEL}" ]]; then
    echo "[wm-offline-probe] missing local WM_HF_MODEL: ${WM_HF_MODEL}" >&2
    exit 1
  fi
  CACHE_ARGS+=(--hf-model "${WM_HF_MODEL}" --max-length "${WM_HF_MAX_LENGTH}" --pooling "${WM_HF_POOLING}")
fi

PYTHONPATH="${REPO_ROOT}/backends/slime:${REPO_ROOT}/harborrl" "${PYTHON_BIN}" "${CACHE_ARGS[@]}"

PYTHONPATH="${REPO_ROOT}/backends/slime:${REPO_ROOT}/harborrl" "${PYTHON_BIN}" -m slime.world_model.train_probe \
  --input "${WM_OUT_DIR}/cached_hidden.pt" \
  --output "${WM_OUT_DIR}/probe.pt" \
  --latent-dim "${WM_LATENT_DIM}" \
  --batch-size "${WM_BATCH_SIZE}" \
  --epochs "${WM_EPOCHS}" \
  --lr "${WM_LR}" \
  --sigreg-coef "${WM_SIGREG_COEF}" \
  --action-contrast-coef "${WM_ACTION_CONTRAST_COEF}" \
  --value-coef "${WM_VALUE_COEF}"

PYTHONPATH="${REPO_ROOT}/backends/slime:${REPO_ROOT}/harborrl" "${PYTHON_BIN}" -m slime.world_model.rank_candidates \
  --checkpoint "${WM_OUT_DIR}/probe.pt" \
  --input "${WM_OUT_DIR}/cached_hidden.pt" \
  --output "${WM_OUT_DIR}/rankings.jsonl"

echo "[wm-offline-probe] done. Outputs: ${WM_OUT_DIR}"
