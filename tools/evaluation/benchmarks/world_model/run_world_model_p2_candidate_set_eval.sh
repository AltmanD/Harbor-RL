#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"
RUNS_ROOT="${RUNS_ROOT:-${REPO_ROOT}/runs}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="${REPO_ROOT}/backends/slime:${PYTHONPATH:-}"

STAMP="$(date +%Y%m%d_%H%M%S)"
WM_P2_BASE_EXP="${WM_P2_BASE_EXP:-}"
WM_P2_BUCKET="${WM_P2_BUCKET:-clean}"
WM_P2_OUT_DIR="${WM_P2_OUT_DIR:-${RUNS_ROOT}/evaluation/world_model_p2_candidate_set_eval/candidate_set_${STAMP}}"
WM_P2_GROUP_KEY="${WM_P2_GROUP_KEY:-context_hash}"
WM_P2_MIN_CANDIDATES="${WM_P2_MIN_CANDIDATES:-2}"
WM_P2_MAX_CANDIDATES="${WM_P2_MAX_CANDIDATES:-8}"
WM_P2_DEVICE="${WM_P2_DEVICE:-auto}"
WM_P2_UNCERTAINTY_COEF="${WM_P2_UNCERTAINTY_COEF:-0.0}"

mkdir -p "${WM_P2_OUT_DIR}/logs"

if [[ -n "${WM_P2_BASE_EXP}" ]]; then
  RECORDS="${WM_P2_RECORDS:-${WM_P2_BASE_EXP}/${WM_P2_BUCKET}/records.jsonl}"
  CACHE="${WM_P2_CACHE:-${WM_P2_BASE_EXP}/${WM_P2_BUCKET}/cached_hidden.pt}"
  CHECKPOINT="${WM_P2_CHECKPOINT:-${WM_P2_BASE_EXP}/${WM_P2_BUCKET}/probe.pt}"
else
  RECORDS="${WM_P2_RECORDS:-}"
  CACHE="${WM_P2_CACHE:-}"
  CHECKPOINT="${WM_P2_CHECKPOINT:-}"
fi
SUMMARY="${WM_P2_OUT_DIR}/candidate_set_summary.json"
GROUPS_OUT="${WM_P2_OUT_DIR}/candidate_groups.jsonl"

cat <<EOF | tee "${WM_P2_OUT_DIR}/logs/config.txt"
[wm-p2] repo:            ${REPO_ROOT}
[wm-p2] base_exp:        ${WM_P2_BASE_EXP}
[wm-p2] bucket:          ${WM_P2_BUCKET}
[wm-p2] records:         ${RECORDS}
[wm-p2] cache:           ${CACHE}
[wm-p2] checkpoint:      ${CHECKPOINT}
[wm-p2] out_dir:         ${WM_P2_OUT_DIR}
[wm-p2] groups_output:   ${GROUPS_OUT}
[wm-p2] group_key:       ${WM_P2_GROUP_KEY}
[wm-p2] min_candidates:  ${WM_P2_MIN_CANDIDATES}
[wm-p2] max_candidates:  ${WM_P2_MAX_CANDIDATES}
[wm-p2] device:          ${WM_P2_DEVICE}
EOF

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[wm-p2] DRY_RUN=1; command wiring only."
  exit 0
fi

if [[ ! -s "${RECORDS}" || ! -s "${CACHE}" || ! -s "${CHECKPOINT}" ]]; then
  echo "[wm-p2] missing inputs. Set WM_P2_BASE_EXP or WM_P2_RECORDS/WM_P2_CACHE/WM_P2_CHECKPOINT explicitly." >&2
  exit 1
fi

"${PYTHON_BIN}" -m slime.world_model.candidate_set_eval \
  --checkpoint "${CHECKPOINT}" \
  --cache "${CACHE}" \
  --records "${RECORDS}" \
  --output "${SUMMARY}" \
  --groups-output "${GROUPS_OUT}" \
  --group-key "${WM_P2_GROUP_KEY}" \
  --min-candidates "${WM_P2_MIN_CANDIDATES}" \
  --max-candidates "${WM_P2_MAX_CANDIDATES}" \
  --device "${WM_P2_DEVICE}" \
  --uncertainty-coef "${WM_P2_UNCERTAINTY_COEF}" \
  2>&1 | tee "${WM_P2_OUT_DIR}/logs/candidate_set_eval.log"

echo "[wm-p2] done. Outputs: ${WM_P2_OUT_DIR}"
