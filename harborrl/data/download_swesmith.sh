#!/usr/bin/env bash
# Low-memory SWE-smith downloader + converter for harborrl.
#
# Default mode creates a small smoke subset for path validation. Formal full
# conversion must opt in with MODE=full ALLOW_FULL=1.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
DATASET_NAME="${DATASET_NAME:-SWE-bench/SWE-smith}"
SPLIT="${SPLIT:-train}"
CANONICAL_DATASET_REVISION="ea6d7173829c7ec8fa16c22055699ff2e9188091"
DATASET_REVISION="${DATASET_REVISION:-${CANONICAL_DATASET_REVISION}}"
MODE="${MODE:-smoke}"                  # smoke | full
OUTPUT_NAME="${OUTPUT_NAME:-}"
STATS_NAME="${STATS_NAME:-}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
MAX_SOURCE_SAMPLES="${MAX_SOURCE_SAMPLES:-}"
MIN_TEST_COUNT="${MIN_TEST_COUNT:-1}"
MAX_FAIL_TO_PASS_COUNT="${MAX_FAIL_TO_PASS_COUNT-}"
MAX_PASS_TO_PASS_COUNT="${MAX_PASS_TO_PASS_COUNT-}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/tmp/hf_swesmith_cache}"
CREATE_ENV_DIRS="${CREATE_ENV_DIRS:-1}"
OVERWRITE_ENV_DIRS="${OVERWRITE_ENV_DIRS:-0}"
BACKUP_EXISTING="${BACKUP_EXISTING:-0}"
INSTALL_DATASETS="${INSTALL_DATASETS:-1}"
USE_STREAMING="${USE_STREAMING:-1}"
SOURCE_BACKEND="${SOURCE_BACKEND:-parquet}"
AUTO_PROXY="${AUTO_PROXY:-0}"
PROXY_ENV_FILE="${PROXY_ENV_FILE:-}"
HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
ALLOW_FULL="${ALLOW_FULL:-0}"
ALLOW_SMOKE_TRAIN_NAME="${ALLOW_SMOKE_TRAIN_NAME:-0}"

case "${BACKUP_EXISTING}" in
  0|1) ;;
  *)
    echo "[ERROR] BACKUP_EXISTING must be 0 or 1." >&2
    exit 1
    ;;
esac

if [[ "${AUTO_PROXY}" == "1" ]]; then
  if [[ -z "${PROXY_ENV_FILE}" || ! -f "${PROXY_ENV_FILE}" ]]; then
    echo "[ERROR] AUTO_PROXY=1 requires PROXY_ENV_FILE to point to a local env file." >&2
    echo "        Allowed keys: http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY HF_ENDPOINT" >&2
    exit 1
  fi
  while IFS= read -r line; do
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    [[ -z "${line}" || "${line}" == \#* ]] && continue
    line="${line#export }"
    key="${line%%=*}"
    value="${line#*=}"
    case "${key}" in
      http_proxy|https_proxy|HTTP_PROXY|HTTPS_PROXY|no_proxy|NO_PROXY|HF_ENDPOINT)
        value="${value%\"}"
        value="${value#\"}"
        value="${value%\'}"
        value="${value#\'}"
        export "${key}=${value}"
        ;;
    esac
  done < "${PROXY_ENV_FILE}"
  echo "[DL] Loaded proxy env from ${PROXY_ENV_FILE}"
fi

if [[ -n "${HF_TOKEN:-}" ]]; then
  export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
  export HF_TOKEN
  echo "[DL] HF token set"
fi
if [[ -n "${HF_ENDPOINT:-}" ]]; then
  export HF_ENDPOINT
  echo "[DL] HF endpoint: ${HF_ENDPOINT}"
fi

case "${MODE}" in
  smoke)
    MAX_SAMPLES="${MAX_SAMPLES:-64}"
    MAX_FAIL_TO_PASS_COUNT="${MAX_FAIL_TO_PASS_COUNT:-50}"
    MAX_PASS_TO_PASS_COUNT="${MAX_PASS_TO_PASS_COUNT:-200}"
    ;;
  full)
    if [[ "${ALLOW_FULL}" != "1" ]]; then
      echo "[ERROR] Full SWE-smith conversion is disabled by default."
      echo "        Re-run with MODE=full ALLOW_FULL=1 after confirming disk/network budget."
      exit 1
    fi
    if [[ "${DATASET_NAME}" != "SWE-bench/SWE-smith" || "${SPLIT}" != "train" ]]; then
      echo "[ERROR] Formal full conversion requires SWE-bench/SWE-smith split=train." >&2
      exit 1
    fi
    if [[ "${DATASET_REVISION}" != "${CANONICAL_DATASET_REVISION}" ]]; then
      echo "[ERROR] Formal full conversion requires audited revision ${CANONICAL_DATASET_REVISION}." >&2
      exit 1
    fi
    if [[ "${SOURCE_BACKEND}" != "parquet" || "${HF_ENDPOINT%/}" != "https://huggingface.co" ]]; then
      echo "[ERROR] Formal full conversion requires the canonical Hugging Face parquet source." >&2
      exit 1
    fi
    if [[ -n "${INPUT_JSONL:-}" || -n "${MAX_SAMPLES}" || -n "${MAX_SOURCE_SAMPLES}" ||
          -n "${REPO_INCLUDE:-}" || -n "${REPO_EXCLUDE:-}" || -n "${SAMPLE_PER_REPO:-}" ]]; then
      echo "[ERROR] Formal full conversion rejects local input, row caps, and repo filters." >&2
      exit 1
    fi
    if [[ "${MIN_TEST_COUNT}" != "1" ]]; then
      echo "[ERROR] Formal full conversion requires MIN_TEST_COUNT=1." >&2
      exit 1
    fi
    if [[ -n "${MAX_FAIL_TO_PASS_COUNT}" && "${MAX_FAIL_TO_PASS_COUNT}" != "50" ]] ||
       [[ -n "${MAX_PASS_TO_PASS_COUNT}" && "${MAX_PASS_TO_PASS_COUNT}" != "200" ]]; then
      echo "[ERROR] Formal full conversion requires audited training caps F2P=50/P2P=200." >&2
      exit 1
    fi
    MAX_FAIL_TO_PASS_COUNT=50
    MAX_PASS_TO_PASS_COUNT=200
    CREATE_ENV_DIRS=1
    FAIL_ON_SOURCE_GAPS=1
    ;;
  *)
    echo "[ERROR] Unknown MODE=${MODE}. Use: smoke|full"
    exit 1
    ;;
esac

if [[ -z "${OUTPUT_DIR}" ]]; then
  if [[ "${MODE}" == "smoke" ]]; then
    OUTPUT_DIR="${SCRIPT_DIR}/../dataset/swesmith_smoke/swesmith_convert"
  else
    OUTPUT_DIR="${SCRIPT_DIR}/../dataset/swesmith_convert"
  fi
fi
if [[ -z "${OUTPUT_NAME}" ]]; then
  if [[ "${MODE}" == "smoke" ]]; then
    OUTPUT_NAME="smoke.jsonl"
  else
    OUTPUT_NAME="train.jsonl"
  fi
fi
if [[ -z "${STATS_NAME}" ]]; then
  if [[ "${MODE}" == "smoke" ]]; then
    STATS_NAME="smoke_stats.json"
  else
    STATS_NAME="convert_stats.json"
  fi
fi
if [[ "${MODE}" == "full" && ( "${OUTPUT_NAME}" != "train.jsonl" || "${STATS_NAME}" != "convert_stats.json" ) ]]; then
  echo "[ERROR] Formal full conversion must publish train.jsonl + convert_stats.json." >&2
  exit 1
fi
if [[ "${MODE}" == "smoke" && "${OUTPUT_NAME}" == "train.jsonl" && "${ALLOW_SMOKE_TRAIN_NAME}" != "1" ]]; then
  echo "[ERROR] smoke mode cannot write train.jsonl." >&2
  echo "        Use the default smoke.jsonl, or set ALLOW_SMOKE_TRAIN_NAME=1 for an intentional custom experiment." >&2
  exit 1
fi

if [[ "${INSTALL_DATASETS}" == "1" ]]; then
  if ! "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import datasets  # noqa: F401
import huggingface_hub  # noqa: F401
import pyarrow  # noqa: F401
PY
  then
    echo "[DL] Installing Python dependencies for ${PYTHON_BIN} (datasets, huggingface_hub, pyarrow)..."
    "${PYTHON_BIN}" -m pip install --user -q 'datasets>=2.18.0' huggingface_hub pyarrow
  fi
fi

mkdir -p "${OUTPUT_DIR}"
DATASET_ROOT="$(cd -- "${OUTPUT_DIR}/.." && pwd)"
SWESMITH_ARTIFACT_LOCK="${SWESMITH_ARTIFACT_LOCK:-${DATASET_ROOT}/.swesmith_artifact.lock}"
if ! command -v flock >/dev/null 2>&1; then
  echo "[ERROR] flock is required for atomic SWE-smith conversion/publication." >&2
  exit 1
fi
exec 9>"${SWESMITH_ARTIFACT_LOCK}"
if ! flock -n 9; then
  echo "[ERROR] SWE-smith artifact is in use by a worker/trainer or another converter:" >&2
  echo "        ${SWESMITH_ARTIFACT_LOCK}" >&2
  exit 1
fi
OUT_PATH="${OUTPUT_DIR}/${OUTPUT_NAME}"
STAMP="$(date +%Y%m%d_%H%M%S)"
TMP_OUTPUT_NAME=".${OUTPUT_NAME}.tmp_${STAMP}"
TMP_STATS_NAME=".${STATS_NAME}.tmp_${STAMP}"
TMP_OUT_PATH="${OUTPUT_DIR}/${TMP_OUTPUT_NAME}"
TMP_STATS_PATH="${OUTPUT_DIR}/${TMP_STATS_NAME}"
ENV_PATH="${DATASET_ROOT}/swesmith_env"
TMP_ENV_PARENT="${DATASET_ROOT}/.swesmith_generation_${STAMP}"
TMP_ENV_DIR="${TMP_ENV_PARENT}/swesmith_env"
PUBLISH_STARTED=0
PUBLISH_COMPLETE=0
BACKUP_PATH=""
STATS_BACKUP_PATH=""
ENV_BACKUP_PATH=""
cleanup_tmp() {
  rm -f "${TMP_OUT_PATH}" "${TMP_STATS_PATH}" 2>/dev/null || true
  rm -rf "${TMP_ENV_PARENT}" 2>/dev/null || true
  if [[ "${PUBLISH_COMPLETE}" != "1" ]]; then
    if [[ "${PUBLISH_STARTED}" == "1" ]]; then
      rm -f "${OUT_PATH}" "${STATS_PATH:-}" 2>/dev/null || true
      if [[ "${CREATE_ENV_DIRS}" == "1" ]]; then
        rm -rf "${ENV_PATH}" 2>/dev/null || true
      fi
    fi
    if [[ -n "${BACKUP_PATH}" && -e "${BACKUP_PATH}" ]]; then
      mv "${BACKUP_PATH}" "${OUT_PATH}" 2>/dev/null || true
    fi
    if [[ -n "${STATS_BACKUP_PATH}" && -e "${STATS_BACKUP_PATH}" ]]; then
      mv "${STATS_BACKUP_PATH}" "${STATS_PATH}" 2>/dev/null || true
    fi
    if [[ -n "${ENV_BACKUP_PATH}" && -e "${ENV_BACKUP_PATH}" ]]; then
      mv "${ENV_BACKUP_PATH}" "${ENV_PATH}" 2>/dev/null || true
    fi
  fi
}
trap cleanup_tmp EXIT

EXTRA_ARGS=()
if [[ -n "${INPUT_JSONL:-}" ]]; then
  EXTRA_ARGS+=(--input-jsonl "${INPUT_JSONL}")
fi
if [[ -n "${MAX_SAMPLES}" ]]; then
  EXTRA_ARGS+=(--max-samples "${MAX_SAMPLES}")
fi
if [[ -n "${MAX_SOURCE_SAMPLES}" ]]; then
  EXTRA_ARGS+=(--max-source-samples "${MAX_SOURCE_SAMPLES}")
fi
if [[ -n "${MAX_FAIL_TO_PASS_COUNT}" ]]; then
  EXTRA_ARGS+=(--max-fail-to-pass-count "${MAX_FAIL_TO_PASS_COUNT}")
fi
if [[ -n "${MAX_PASS_TO_PASS_COUNT}" ]]; then
  EXTRA_ARGS+=(--max-pass-to-pass-count "${MAX_PASS_TO_PASS_COUNT}")
fi
if [[ "${CREATE_ENV_DIRS}" == "1" ]]; then
  rm -rf "${TMP_ENV_PARENT}"
  EXTRA_ARGS+=(--create-env-dirs --env-dir "${TMP_ENV_DIR}")
fi
if [[ "${OVERWRITE_ENV_DIRS}" == "1" ]]; then
  EXTRA_ARGS+=(--overwrite-env-dirs)
fi
if [[ "${USE_STREAMING}" == "1" ]]; then
  EXTRA_ARGS+=(--streaming)
else
  EXTRA_ARGS+=(--no-streaming)
fi
if [[ -n "${REPO_INCLUDE:-}" ]]; then
  EXTRA_ARGS+=(--repo-include "${REPO_INCLUDE}")
fi
if [[ -n "${REPO_EXCLUDE:-}" ]]; then
  EXTRA_ARGS+=(--repo-exclude "${REPO_EXCLUDE}")
fi
if [[ -n "${SAMPLE_PER_REPO:-}" ]]; then
  EXTRA_ARGS+=(--sample-per-repo "${SAMPLE_PER_REPO}")
fi
if [[ "${FAIL_ON_SOURCE_GAPS:-0}" == "1" ]]; then
  EXTRA_ARGS+=(
    --fail-on-missing-instance
    --fail-on-missing-image
    --fail-on-too-few-tests
    --fail-on-duplicate-instance
    --fail-on-unsupported-runner
  )
fi

echo "[DL] Converting SWE-smith dataset"
echo "[DL] repo=${REPO_ROOT}"
echo "[DL] source=${INPUT_JSONL:-${DATASET_NAME}/${SPLIT}} revision=${DATASET_REVISION} mode=${MODE}"
echo "[DL] output=${OUT_PATH}"
echo "[DL] backend=${SOURCE_BACKEND} max_samples=${MAX_SAMPLES:-all} min_test_count=${MIN_TEST_COUNT}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/convert_swesmith.py" \
  --dataset-name "${DATASET_NAME}" \
  --split "${SPLIT}" \
  --revision "${DATASET_REVISION}" \
  --output-dir "${OUTPUT_DIR}" \
  --output-name "${TMP_OUTPUT_NAME}" \
  --stats-name "${TMP_STATS_NAME}" \
  --min-test-count "${MIN_TEST_COUNT}" \
  --hf-cache-dir "${HF_CACHE_DIR}" \
  --source-backend "${SOURCE_BACKEND}" \
  --hf-endpoint "${HF_ENDPOINT}" \
  --overwrite-output \
  "${EXTRA_ARGS[@]}"

if [[ ! -f "${TMP_OUT_PATH}" || ! -f "${TMP_STATS_PATH}" ]] ||
   [[ "${CREATE_ENV_DIRS}" == "1" && ! -d "${TMP_ENV_DIR}" ]]; then
  echo "[ERROR] Converter did not produce the expected temporary artifact pair." >&2
  exit 1
fi

"${PYTHON_BIN}" - "${TMP_STATS_PATH}" "${TMP_OUT_PATH}" "${OUT_PATH}" "${MODE}" "${DATASET_REVISION}" "${SCRIPT_DIR}/convert_swesmith.py" "${ENV_PATH}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

stats_path = Path(sys.argv[1])
artifact_path = Path(sys.argv[2])
published_path = Path(sys.argv[3])
mode = sys.argv[4]
revision = sys.argv[5]
converter_path = Path(sys.argv[6])
published_env_path = Path(sys.argv[7])
stats = json.loads(stats_path.read_text(encoding="utf-8"))
digest = hashlib.sha256()
rows = 0
with artifact_path.open("rb") as handle:
    for line in handle:
        digest.update(line)
        if line.strip():
            rows += 1
stats["manifest_schema_version"] = 2
stats["output_path"] = str(published_path)
stats["env_dir"] = str(published_env_path)
stats["conversion_mode"] = mode
stats["dataset_revision"] = revision
stats["artifact_sha256"] = digest.hexdigest()
stats["artifact_bytes"] = artifact_path.stat().st_size
stats["artifact_rows"] = rows
stats["converter_sha256"] = hashlib.sha256(converter_path.read_bytes()).hexdigest()
if int(stats.get("converted", -1)) != rows:
    raise SystemExit(
        f"[ERROR] stats converted={stats.get('converted')} does not match rows={rows}"
    )
stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
if mode == "full":
    sys.path.insert(0, str(converter_path.parent))
    from convert_swesmith import validate_swesmith_artifact_manifest

    validate_swesmith_artifact_manifest(
        artifact_path,
        stats_path=stats_path,
        require_full=True,
        artifact_rows=rows,
        artifact_sha256=digest.hexdigest(),
    )
PY

STATS_PATH="${OUTPUT_DIR}/${STATS_NAME}"
if [[ -f "${OUT_PATH}" ]]; then
  if [[ "${BACKUP_EXISTING}" == "1" ]]; then
    BACKUP_PATH="${OUT_PATH}.bak_${STAMP}"
    echo "[DL] Existing output moved to ${BACKUP_PATH}"
  else
    BACKUP_PATH="${OUT_PATH}.rollback_${STAMP}"
  fi
  mv "${OUT_PATH}" "${BACKUP_PATH}"
fi
if [[ -f "${STATS_PATH}" ]]; then
  if [[ "${BACKUP_EXISTING}" == "1" ]]; then
    STATS_BACKUP_PATH="${STATS_PATH}.bak_${STAMP}"
    echo "[DL] Existing manifest moved to ${STATS_BACKUP_PATH}"
  else
    STATS_BACKUP_PATH="${STATS_PATH}.rollback_${STAMP}"
  fi
  mv "${STATS_PATH}" "${STATS_BACKUP_PATH}"
fi
if [[ "${CREATE_ENV_DIRS}" == "1" && -d "${ENV_PATH}" ]]; then
  if [[ "${BACKUP_EXISTING}" == "1" ]]; then
    ENV_BACKUP_PATH="${ENV_PATH}.bak_${STAMP}"
    echo "[DL] Existing task dirs moved to ${ENV_BACKUP_PATH}"
  else
    ENV_BACKUP_PATH="${ENV_PATH}.rollback_${STAMP}"
  fi
  mv "${ENV_PATH}" "${ENV_BACKUP_PATH}"
fi
PUBLISH_STARTED=1
if [[ "${CREATE_ENV_DIRS}" == "1" ]]; then
  mv "${TMP_ENV_DIR}" "${ENV_PATH}"
fi
mv "${TMP_OUT_PATH}" "${OUT_PATH}"
mv "${TMP_STATS_PATH}" "${STATS_PATH}"
PUBLISH_COMPLETE=1
if [[ "${BACKUP_EXISTING}" != "1" ]]; then
  rm -f "${BACKUP_PATH}" "${STATS_BACKUP_PATH}"
  rm -rf "${ENV_BACKUP_PATH}"
fi
rm -rf "${TMP_ENV_PARENT}"
trap - EXIT

echo "[DL] Done: ${OUT_PATH}"
echo "[DL] Verify: wc -l ${OUT_PATH}"
