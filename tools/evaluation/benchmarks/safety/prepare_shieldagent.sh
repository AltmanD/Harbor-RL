#!/usr/bin/env bash
# Prepare a repo-local ShieldAgent model directory for official ASB scoring.
#
# The scoring scripts prefer:
#   <HarborRL>/runs/models/ShieldAgent
#
# Run this once before official ASB scoring, on a machine that can access the
# source model path. Training/eval clusters are expected to use the repo-local
# result and should not rely on network downloads.
#
# If SHIELD_MODEL_SOURCE is mounted, the script copies small files and symlinks
# large weights by default. Set COPY_WEIGHTS=1 if the target training cluster
# cannot see the source filesystem.
#
# If SHIELD_MODEL_SOURCE is not mounted, the script fails by default. Set
# DOWNLOAD_IF_SOURCE_MISSING=1 explicitly only on a machine with internet access.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}"

SHIELD_MODEL_TARGET="${SHIELD_MODEL_TARGET:-${REPO_ROOT}/runs/models/ShieldAgent}"
COPY_WEIGHTS="${COPY_WEIGHTS:-0}"
DOWNLOAD_IF_SOURCE_MISSING="${DOWNLOAD_IF_SOURCE_MISSING:-0}"
HF_REPO_ID="${HF_REPO_ID:-thu-coai/ShieldAgent}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

SHIELD_MODEL_SOURCE="${SHIELD_MODEL_SOURCE:-}"
if [[ -z "${SHIELD_MODEL_SOURCE}" ]]; then
  for candidate in \
    "${REPO_ROOT}/runs/models/ShieldAgent"; do
    if [[ -d "${candidate}" ]]; then
      SHIELD_MODEL_SOURCE="${candidate}"
      break
    fi
  done
fi
SHIELD_MODEL_SOURCE="${SHIELD_MODEL_SOURCE:-${REPO_ROOT}/runs/models/ShieldAgent}"

canonical_path() {
  "${PYTHON_BIN}" - "$1" <<'PY'
import sys
from pathlib import Path

print(Path(sys.argv[1]).expanduser().resolve(strict=False))
PY
}

if [[ -d "${SHIELD_MODEL_SOURCE}" && ! -f "${SHIELD_MODEL_SOURCE}/config.json" ]]; then
  SNAPSHOT_DIR="$(find "${SHIELD_MODEL_SOURCE}/snapshots" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -1 || true)"
  if [[ -n "${SNAPSHOT_DIR}" && -f "${SNAPSHOT_DIR}/config.json" && -f "${SNAPSHOT_DIR}/tokenizer_config.json" ]]; then
    SHIELD_MODEL_SOURCE="${SNAPSHOT_DIR}"
  fi
fi

SOURCE_REAL="$(canonical_path "${SHIELD_MODEL_SOURCE}")"
TARGET_REAL="$(canonical_path "${SHIELD_MODEL_TARGET}")"

if [[ -d "${SHIELD_MODEL_SOURCE}" && "${SOURCE_REAL}" == "${TARGET_REAL}" ]]; then
  echo "[INFO] ShieldAgent source and target are the same repo-local directory; validating in place."
elif [[ ! -d "${SHIELD_MODEL_SOURCE}" ]]; then
  if [[ "${DOWNLOAD_IF_SOURCE_MISSING}" != "1" ]]; then
    echo "[ERROR] ShieldAgent source not found: ${SHIELD_MODEL_SOURCE}" >&2
    exit 1
  fi

  echo "[INFO] ShieldAgent source not found: ${SHIELD_MODEL_SOURCE}"
  echo "[INFO] Downloading ${HF_REPO_ID} to ${SHIELD_MODEL_TARGET}"
  mkdir -p "${SHIELD_MODEL_TARGET}"
  find "${SHIELD_MODEL_TARGET}" -mindepth 1 -maxdepth 1 -exec rm -rf '{}' +
  "${PYTHON_BIN}" - "${HF_REPO_ID}" "${SHIELD_MODEL_TARGET}" <<'PY'
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

repo_id = sys.argv[1]
target = Path(sys.argv[2])
target.mkdir(parents=True, exist_ok=True)
snapshot_download(
    repo_id=repo_id,
    local_dir=str(target),
    local_dir_use_symlinks=False,
    resume_download=True,
)
print(f"downloaded {repo_id} to {target}")
PY
else
  if [[ ! -f "${SHIELD_MODEL_SOURCE}/config.json" || ! -f "${SHIELD_MODEL_SOURCE}/tokenizer_config.json" ]]; then
    echo "[ERROR] source is missing config/tokenizer files: ${SHIELD_MODEL_SOURCE}" >&2
    exit 1
  fi

  mkdir -p "${SHIELD_MODEL_TARGET}"
  find "${SHIELD_MODEL_TARGET}" -mindepth 1 -maxdepth 1 -exec rm -rf '{}' +

  for item in "${SHIELD_MODEL_SOURCE}"/*; do
    name="$(basename "${item}")"
    case "${name}" in
      *.safetensors)
        if [[ "${COPY_WEIGHTS}" == "1" ]]; then
          cp -aL "${item}" "${SHIELD_MODEL_TARGET}/${name}"
        else
          ln -sfn "${item}" "${SHIELD_MODEL_TARGET}/${name}"
        fi
        ;;
      *)
        cp -aL "${item}" "${SHIELD_MODEL_TARGET}/${name}"
        ;;
    esac
  done
fi

"${PYTHON_BIN}" - "${SHIELD_MODEL_TARGET}" <<'PY'
import json
import sys
from pathlib import Path

target = Path(sys.argv[1])
required = ["config.json", "tokenizer_config.json", "model.safetensors.index.json"]
missing = [name for name in required if not (target / name).is_file()]
if missing:
    raise SystemExit(f"[ERROR] missing required files in {target}: {missing}")

with (target / "model.safetensors.index.json").open(encoding="utf-8") as f:
    index = json.load(f)
shards = sorted(set(index.get("weight_map", {}).values()))
missing_shards = [name for name in shards if not (target / name).is_file()]
if missing_shards:
    raise SystemExit(f"[ERROR] missing model shards in {target}: {missing_shards}")
print(f"[OK] ShieldAgent files ready: {target}; shards={len(shards)}")
PY

echo "Prepared repo-local ShieldAgent:"
echo "${SHIELD_MODEL_TARGET}"
echo "copy_weights=${COPY_WEIGHTS}"
echo "download_if_source_missing=${DOWNLOAD_IF_SOURCE_MISSING}"
ls -lh "${SHIELD_MODEL_TARGET}" | sed -n '1,20p'
