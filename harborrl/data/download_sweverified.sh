#!/usr/bin/env bash
# Prepare pinned SWE-bench Verified metadata and Terminal-Bench task directories.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
TERMINAL_RL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODE="${MODE:-smoke}" # smoke | full
OUTPUT_DIR="${OUTPUT_DIR:-${TERMINAL_RL_DIR}/dataset/sweverified_convert}"
ENV_DIR="${ENV_DIR:-${TERMINAL_RL_DIR}/dataset/sweverified_env}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/tmp/hf_sweverified_cache}"
HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
INPUT_JSONL="${INPUT_JSONL:-}"
INSTALL_DEPS="${INSTALL_DEPS:-1}"
OVERWRITE="${OVERWRITE:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-8}"

case "${MODE}" in
  smoke)
    [[ "${MAX_SAMPLES}" =~ ^[1-9][0-9]*$ ]] || {
      echo "[ERROR] smoke MAX_SAMPLES must be a positive integer." >&2
      exit 2
    }
    MODE_ARGS=(--max-samples "${MAX_SAMPLES}")
    ;;
  full)
    MODE_ARGS=(--formal)
    if [[ -n "${INPUT_JSONL}" ]]; then
      echo "[ERROR] full conversion requires the pinned Hugging Face source." >&2
      exit 2
    fi
    ;;
  *)
    echo "[ERROR] MODE must be smoke or full." >&2
    exit 2
    ;;
esac

if [[ "${INSTALL_DEPS}" == "1" ]]; then
  if ! "${PYTHON_BIN}" -c 'import huggingface_hub, pyarrow' >/dev/null 2>&1; then
    "${PYTHON_BIN}" -m pip install --user \
      'huggingface_hub>=0.25,<2' 'pyarrow>=15,<23'
  fi
fi

mkdir -p "$(dirname -- "${OUTPUT_DIR}")" "$(dirname -- "${ENV_DIR}")"
LOCK_PATH="${SWEVERIFIED_ARTIFACT_LOCK:-${TERMINAL_RL_DIR}/dataset/.sweverified_artifact.lock}"
command -v flock >/dev/null 2>&1 || {
  echo "[ERROR] flock is required." >&2
  exit 2
}
exec 9>"${LOCK_PATH}"
flock -n 9 || {
  echo "[ERROR] another SWE-Verified conversion/worker holds ${LOCK_PATH}" >&2
  exit 2
}

if [[ "${OVERWRITE}" != "1" ]] &&
   { [[ -e "${OUTPUT_DIR}/test.jsonl" ]] || [[ -d "${ENV_DIR}" ]]; }; then
  echo "[ERROR] target artifacts already exist; set OVERWRITE=1 to regenerate." >&2
  exit 2
fi

OVERWRITE_ARGS=()
if [[ "${OVERWRITE}" == "1" ]]; then
  OVERWRITE_ARGS=(--overwrite-output --overwrite-env-dirs)
fi
INPUT_ARGS=()
if [[ -n "${INPUT_JSONL}" ]]; then
  INPUT_ARGS=(--input-jsonl "${INPUT_JSONL}")
fi

echo "[sweverified-data] mode=${MODE}"
echo "[sweverified-data] output=${OUTPUT_DIR}/test.jsonl"
echo "[sweverified-data] env_dir=${ENV_DIR}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/convert_sweverified.py" \
  --output-dir "${OUTPUT_DIR}" \
  --env-dir "${ENV_DIR}" \
  --create-env-dirs \
  --hf-cache-dir "${HF_CACHE_DIR}" \
  --hf-endpoint "${HF_ENDPOINT}" \
  "${INPUT_ARGS[@]}" \
  "${MODE_ARGS[@]}" \
  "${OVERWRITE_ARGS[@]}"
