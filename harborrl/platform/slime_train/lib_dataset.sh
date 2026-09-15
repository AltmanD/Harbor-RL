# ── Dataset & Reward Configuration ───────────────────────────────────
# DATASET: which data to train on
#   seta    = seta_env only (capability tasks, Docker-based evaluation)
#   safety  = Agent-SafetyBench only (safety tasks, no Docker needed)
#   agentharm = inspect_evals/agentharm only (safety tool tasks, no Docker needed)
#   mixed   = configurable mix of seta / safety / agentharm
#
# SETA_SAFETY: safety reward mode for seta_env data
#   none       = pure outcome reward (2*accuracy - 1), no safety signal
#
# SAFETY_BENCH_REWARD: reward mode for Agent-SafetyBench data
#   rule       = fulfillable-based: refuse harmful=+1, comply harmful=-1
#   dense_rule = rule-compatible dense shaping for GRPO
#
# AGENTHARM_REWARD: reward mode for AgentHarm data
#   rule       = target-tool/refusal based safety reward
#   dense_rule = rule-compatible dense shaping for GRPO
#
# ALGO:
#   grpo = existing baseline path
#   dapo = verl DAPO recipe knobs on top of GRPO estimator:
#          clip-higher, token-level loss, optional dynamic sampling, overlong shaping
ALGO="${ALGO:-dapo}"
case "${ALGO}" in
  grpo|dapo) ;;
  *)
    echo "[ERROR] Unknown ALGO=${ALGO}. Use: grpo|dapo"
    exit 1
    ;;
esac
export ALGO

DATASET="$(normalize_dataset "${DATASET:-mixed}")"
SETA_SAFETY="${SETA_SAFETY:-none}"
SAFETY_BENCH_REWARD="${SAFETY_BENCH_REWARD:-dense_rule}"
AGENT_SAFETYBENCH_REMOTE_ENV="${AGENT_SAFETYBENCH_REMOTE_ENV:-0}"
AGENT_SAFETYBENCH_ROOT="${AGENT_SAFETYBENCH_ROOT:-${REPO_ROOT}/../Agent-SafetyBench}"
AGENTHARM_REWARD="${AGENTHARM_REWARD:-dense_rule}"
AGENTHARM_REMOTE_ENV="${AGENTHARM_REMOTE_ENV:-0}"
AGENTHARM_ROOT="${AGENTHARM_ROOT:-${REPO_ROOT}/../inspect_evals/src/inspect_evals/agentharm}"

SETA_DATA="${REPO_ROOT}/benchmarks/datasets/seta_env_convert/train.jsonl"
SAFETY_DATA="${REPO_ROOT}/benchmarks/datasets/agent_safetybench_convert/train.jsonl"
AGENTHARM_RAW_DIR="${REPO_ROOT}/benchmarks/datasets/agentharm"
AGENTHARM_OUTPUT_DIR="${RUN_DIR}/config/agentharm"
AGENTHARM_DATA="${AGENTHARM_OUTPUT_DIR}/train.jsonl"
SWESMITH_DATA="${REPO_ROOT}/benchmarks/datasets/swesmith_convert/train.jsonl"
SWESMITH_ENV_DIR="${SWESMITH_ENV_DIR:-${REPO_ROOT}/benchmarks/environments/swesmith_env}"
SWESMITH_REQUIRE_FULL_DATA="${SWESMITH_REQUIRE_FULL_DATA:-1}"
SWESMITH_ARTIFACT_SHA256=""
SWESMITH_CONVERSION_MODE="not_applicable"
SWESMITH_DATASET_REVISION=""
SWESMITH_CONVERTER_SHA256=""
SWESMITH_SOURCE_PROMPT_DATA=""
SWESMITH_SOURCE_DEVICE=""
SWESMITH_SOURCE_INODE=""
case "${SWESMITH_REQUIRE_FULL_DATA}" in
  0|1) ;;
  *)
    echo "[ERROR] SWESMITH_REQUIRE_FULL_DATA must be 0 or 1."
    exit 1
    ;;
esac

ensure_agentharm_dataset() {
  if [[ ! -d "${AGENTHARM_RAW_DIR}" ]]; then
    echo "[ERROR] AgentHarm raw data dir not found: ${AGENTHARM_RAW_DIR}"
    exit 1
  fi
  python3 "${REPO_ROOT}/harborrl/data/convert_agentharm_to_dataset.py" \
    --input-dir "${AGENTHARM_RAW_DIR}" \
    --output-dir "${AGENTHARM_OUTPUT_DIR}"
}

ensure_swesmith_env_dirs() {
  local prompt_data="$1"
  local env_status total_count missing_count image_missing_count duplicate_count
  local exact_missing_count expected_mismatch_count unsupported_runner_count
  local profile_mismatch_count foreign_row_count
  local artifact_sha256 conversion_mode dataset_revision converter_sha256
  local source_device source_inode
  local swesmith_dataset_root
  swesmith_dataset_root="$(dirname -- "${SWESMITH_ENV_DIR}")"
  env_status="$(python3 - "$prompt_data" "${swesmith_dataset_root}" "${SWESMITH_ENV_DIR}" "${SWESMITH_EXPECTED_SAMPLES:-}" "${REPO_ROOT}" "${SWESMITH_REQUIRE_FULL_DATA}" "${SWESMITH_STATS_PATH:-}" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

prompt = Path(sys.argv[1])
dataset_root = Path(sys.argv[2])
env_dir = Path(sys.argv[3])
expected_samples = int(sys.argv[4]) if sys.argv[4] else None
sys.path.insert(0, sys.argv[5])
require_full = sys.argv[6] == "1"
stats_path = Path(sys.argv[7]) if sys.argv[7] else prompt.with_name("convert_stats.json")
from harborrl.data.convert_swesmith import (
    OFFICIAL_TEST_COMMANDS,
    TASK_FORMAT_VERSION,
    expected_swesmith_task_path,
    infer_test_runner,
    validate_swesmith_artifact_manifest,
    validate_task_dir_fingerprint,
)

missing = 0
image_missing = 0
duplicates = 0
unsupported_runners = 0
profile_mismatches = 0
foreign_rows = 0
total = 0
artifact_rows = 0
exact_missing = 1
stat_limit = 2000
seen_task_names = set()
required_names = [
    "task.yaml",
    "docker-compose.yaml",
    "Dockerfile",
    "run-tests.sh",
    "tests/fail_to_pass.txt",
    "tests/pass_to_pass.txt",
    ".terminal-rl-swesmith-format",
]
artifact_digest = hashlib.sha256()
with prompt.open("rb") as f:
    artifact_stat = os.fstat(f.fileno())
    for raw_line in f:
        artifact_digest.update(raw_line)
        if not raw_line.strip():
            continue
        artifact_rows += 1
        obj = json.loads(raw_line)
        meta = obj.get("metadata") or {}
        if meta.get("data_source") != "swesmith":
            foreign_rows += 1
            continue
        total += 1
        if str(meta.get("test_runner") or "unsupported") == "unsupported":
            unsupported_runners += 1
        expected_runner = infer_test_runner(meta)
        expected_command = OFFICIAL_TEST_COMMANDS.get(
            str(meta.get("repo") or "").lower(), ""
        )
        try:
            expected_task_path = expected_swesmith_task_path(meta.get("task_name"))
        except ValueError:
            expected_task_path = ""
        if (
            str(meta.get("task_format_version") or "") != TASK_FORMAT_VERSION
            or str(meta.get("test_runner") or "") != expected_runner
            or str(meta.get("test_command") or "") != expected_command
            or str(meta.get("swesmith_instance_id") or "")
            != str(meta.get("task_name") or "")
            or str(meta.get("task_path") or "") != expected_task_path
        ):
            profile_mismatches += 1
        task_name = str(meta.get("task_name") or "")
        if not task_name or task_name in seen_task_names:
            duplicates += 1
        else:
            seen_task_names.add(task_name)
        task_path = str(meta.get("task_path") or "")
        missing_image = not str(meta.get("image_name") or "").strip()
        if missing_image:
            image_missing += 1
        if not task_path:
            missing += 1
            continue
        task_dir = dataset_root / task_path
        if task_path.startswith("swesmith_env/"):
            task_dir = env_dir / task_path.split("/", 1)[1]
        if missing < stat_limit:
            complete = all((task_dir / rel).is_file() for rel in required_names)
            if complete:
                complete = validate_task_dir_fingerprint(obj, task_dir)
            if not complete:
                missing += 1
        else:
            exact_missing = 0
manifest = {}
if require_full:
    try:
        manifest = validate_swesmith_artifact_manifest(
            prompt,
            stats_path=stats_path,
            require_full=True,
            expected_samples=expected_samples,
            artifact_rows=artifact_rows,
            artifact_sha256=artifact_digest.hexdigest(),
        )
    except ValueError as exc:
        raise SystemExit(f"[ERROR] {exc}") from exc
expected_mismatch = int(expected_samples is not None and total != expected_samples)
print(
    f"{total} {missing} {image_missing} {duplicates} "
    f"{exact_missing} {expected_mismatch} {unsupported_runners} "
    f"{profile_mismatches} {foreign_rows} {artifact_digest.hexdigest()} "
    f"{manifest.get('conversion_mode', 'custom')} "
    f"{manifest.get('dataset_revision', '-')} "
    f"{manifest.get('converter_sha256', '-')} "
    f"{artifact_stat.st_dev} {artifact_stat.st_ino}"
)
PY
)"
  total_count="$(awk '{print $1}' <<<"${env_status}")"
  missing_count="$(awk '{print $2}' <<<"${env_status}")"
  image_missing_count="$(awk '{print $3}' <<<"${env_status}")"
  duplicate_count="$(awk '{print $4}' <<<"${env_status}")"
  exact_missing_count="$(awk '{print $5}' <<<"${env_status}")"
  expected_mismatch_count="$(awk '{print $6}' <<<"${env_status}")"
  unsupported_runner_count="$(awk '{print $7}' <<<"${env_status}")"
  profile_mismatch_count="$(awk '{print $8}' <<<"${env_status}")"
  foreign_row_count="$(awk '{print $9}' <<<"${env_status}")"
  artifact_sha256="$(awk '{print $10}' <<<"${env_status}")"
  conversion_mode="$(awk '{print $11}' <<<"${env_status}")"
  dataset_revision="$(awk '{print $12}' <<<"${env_status}")"
  converter_sha256="$(awk '{print $13}' <<<"${env_status}")"
  source_device="$(awk '{print $14}' <<<"${env_status}")"
  source_inode="$(awk '{print $15}' <<<"${env_status}")"
  SWESMITH_ARTIFACT_SHA256="${artifact_sha256}"
  SWESMITH_CONVERSION_MODE="${conversion_mode}"
  [[ "${dataset_revision}" != "-" ]] && SWESMITH_DATASET_REVISION="${dataset_revision}"
  [[ "${converter_sha256}" != "-" ]] && SWESMITH_CONVERTER_SHA256="${converter_sha256}"
  SWESMITH_SOURCE_DEVICE="${source_device}"
  SWESMITH_SOURCE_INODE="${source_inode}"

  if (( total_count == 0 )); then
    echo "[ERROR] DATASET=swesmith selected, but ${prompt_data} contains no SWE-smith rows."
    exit 1
  fi
  if (( foreign_row_count > 0 )); then
    echo "[ERROR] ${prompt_data} contains ${foreign_row_count} non-SWE-smith row(s)."
    echo "        DATASET=swesmith requires a homogeneous converted JSONL."
    exit 1
  fi
  if (( image_missing_count > 0 )); then
    echo "[ERROR] ${image_missing_count} SWE-smith JSONL row(s) are missing metadata.image_name."
    echo "        Re-run: CREATE_ENV_DIRS=1 bash ${REPO_ROOT}/harborrl/data/download_swesmith.sh"
    exit 1
  fi
  if (( duplicate_count > 0 )); then
    echo "[ERROR] ${duplicate_count} SWE-smith row(s) have missing or duplicate metadata.task_name."
    exit 1
  fi
  if (( unsupported_runner_count > 0 )); then
    echo "[ERROR] ${unsupported_runner_count} SWE-smith row(s) have no supported test runner/profile."
    echo "        Re-convert with a pinned dataset revision and the current converter."
    exit 1
  fi
  if (( profile_mismatch_count > 0 )); then
    echo "[ERROR] ${profile_mismatch_count} SWE-smith row(s) use stale or untrusted task format/test profiles."
    echo "        Re-run full dataset conversion with the current converter; env-only regeneration is insufficient."
    exit 1
  fi
  if (( expected_mismatch_count > 0 )); then
    echo "[ERROR] SWE-smith row count mismatch: expected=${SWESMITH_EXPECTED_SAMPLES} actual=${total_count}."
    exit 1
  fi
  if (( missing_count == 0 )); then
    return
  fi

  if (( exact_missing_count == 1 )); then
    echo "[WARN] SWE-smith env task dirs incomplete or outdated: ${missing_count} task(s)"
  else
    echo "[WARN] SWE-smith env task dirs incomplete or outdated: at least ${missing_count} task(s)"
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[ERROR] DRY_RUN=1 found incomplete/outdated SWE-smith task directories."
    echo "        Regenerate task format v6 before launching training."
    exit 1
  fi
  echo "[ERROR] SWE-smith env dirs are incomplete; training preflight is read-only."
  echo "        Regenerate task format v6 with harborrl/data/download_swesmith.sh, then retry."
  exit 1
}

INCLUDES_SETA="0"
INCLUDES_SAFETY="0"
INCLUDES_AGENTHARM="0"
INCLUDES_SWESMITH="0"
MIX_SETA_RATIO="${MIX_SETA_RATIO:-6}"
MIX_SAFETY_RATIO="${MIX_SAFETY_RATIO:-2}"
MIX_AGENTHARM_RATIO="${MIX_AGENTHARM_RATIO:-2}"
MIX_MODE="${MIX_MODE:-all_visible}"
export MIX_MODE

case "${DATASET}" in
  harbor_terminal)
    : "${ROLLOUT_PROMPT_DATA:?Set ROLLOUT_PROMPT_DATA to a verified Harbor catalog}"
    ;;
  seta)
    INCLUDES_SETA="1"
    ROLLOUT_PROMPT_DATA="${ROLLOUT_PROMPT_DATA:-${SETA_DATA}}"
    ;;
  safety)
    INCLUDES_SAFETY="1"
    ROLLOUT_PROMPT_DATA="${ROLLOUT_PROMPT_DATA:-${SAFETY_DATA}}"
    ;;
  agentharm)
    INCLUDES_AGENTHARM="1"
    ensure_agentharm_dataset
    ROLLOUT_PROMPT_DATA="${ROLLOUT_PROMPT_DATA:-${AGENTHARM_DATA}}"
    ;;
  swesmith)
    INCLUDES_SWESMITH="1"
    ROLLOUT_PROMPT_DATA="${ROLLOUT_PROMPT_DATA:-${SWESMITH_DATA}}"
    ;;
  mixed)
    if [[ -n "${MIX_AGENTHARM_RATIO:-}" ]]; then
      ensure_agentharm_dataset
      MIXED_DATA="${RUN_DIR}/config/mixed_sources.jsonl"
      MIX_ARGS=(
        --output "${MIXED_DATA}"
        --seed "${MIX_SEED:-42}"
        --mode "${MIX_MODE}"
      )
      MIX_LABELS=()
      add_mix_source() {
        local path="$1"
        local ratio="$2"
        local label="$3"
        if [[ -z "${ratio}" || "${ratio}" == "0" ]]; then
          return
        fi
        if [[ ! -f "${path}" ]]; then
          echo "[ERROR] mixed source not found: ${path}"
          exit 1
        fi
        MIX_ARGS+=(--source "${path}:${ratio}")
        MIX_LABELS+=("${label}(${ratio})")
      }
      add_mix_source "${SETA_DATA}" "${MIX_SETA_RATIO:-}" "seta"
      add_mix_source "${SAFETY_DATA}" "${MIX_SAFETY_RATIO:-}" "safety"
      add_mix_source "${AGENTHARM_DATA}" "${MIX_AGENTHARM_RATIO:-}" "agentharm"
      if [[ "${#MIX_LABELS[@]}" -eq 0 ]]; then
        echo "[ERROR] No mixed sources selected. Set MIX_SETA_RATIO, MIX_SAFETY_RATIO, or MIX_AGENTHARM_RATIO to a positive value."
        exit 1
      fi
      [[ -n "${MIX_SETA_RATIO:-}" && "${MIX_SETA_RATIO}" != "0" ]] && INCLUDES_SETA="1"
      [[ -n "${MIX_SAFETY_RATIO:-}" && "${MIX_SAFETY_RATIO}" != "0" ]] && INCLUDES_SAFETY="1"
      [[ -n "${MIX_AGENTHARM_RATIO:-}" && "${MIX_AGENTHARM_RATIO}" != "0" ]] && INCLUDES_AGENTHARM="1"
      if [[ -n "${MIX_TOTAL:-}" ]]; then
        MIX_ARGS+=(--total "${MIX_TOTAL}")
      fi
      if [[ -n "${MIX_OVERSAMPLE:-}" ]]; then
        MIX_ARGS+=(--oversample)
      fi
      python "${REPO_ROOT}/harborrl/data/mix_jsonl_datasets.py" "${MIX_ARGS[@]}"
      echo "[dataset] mixed sources: ${MIX_LABELS[*]} -> ${MIXED_DATA}"
    else
      INCLUDES_SETA="1"
      INCLUDES_SAFETY="1"
      MIXED_DATA="${RUN_DIR}/config/mixed_seta_safety.jsonl"
      if [[ ! -f "${MIXED_DATA}" ]] || [[ "${SETA_DATA}" -nt "${MIXED_DATA}" ]] || [[ "${SAFETY_DATA}" -nt "${MIXED_DATA}" ]]; then
        if [[ -n "${MIX_SETA_RATIO:-}" ]] || [[ -n "${MIX_SAFETY_RATIO:-}" ]]; then
          MIX_ARGS=(
            --source "${SETA_DATA}:${MIX_SETA_RATIO:-1}"
            --source "${SAFETY_DATA}:${MIX_SAFETY_RATIO:-1}"
            --output "${MIXED_DATA}"
            --seed "${MIX_SEED:-42}"
            --mode "${MIX_MODE}"
          )
          if [[ -n "${MIX_TOTAL:-}" ]]; then
            MIX_ARGS+=(--total "${MIX_TOTAL}")
          fi
          if [[ -n "${MIX_OVERSAMPLE:-}" ]]; then
            MIX_ARGS+=(--oversample)
          fi
          python "${REPO_ROOT}/harborrl/data/mix_jsonl_datasets.py" "${MIX_ARGS[@]}"
        else
          cat "${SETA_DATA}" "${SAFETY_DATA}" > "${MIXED_DATA}"
          echo "[dataset] merged seta($(wc -l < "${SETA_DATA}")) + safety($(wc -l < "${SAFETY_DATA}")) -> ${MIXED_DATA}"
        fi
      fi
    fi
    ROLLOUT_PROMPT_DATA="${ROLLOUT_PROMPT_DATA:-${MIXED_DATA}}"
    ;;
  *)
    echo "[ERROR] Unknown DATASET=${DATASET}. Use: seta|safety|agentharm|mixed|swesmith"
    exit 1
    ;;
esac

if [[ -z "${ROLLOUT_PROMPT_DATA}" ]]; then
  echo "[ERROR] ROLLOUT_PROMPT_DATA is unset."
  exit 1
fi
if [[ ! -f "${ROLLOUT_PROMPT_DATA}" ]]; then
  if [[ "${INCLUDES_SWESMITH}" == "1" && "${ROLLOUT_PROMPT_DATA}" == "${SWESMITH_DATA}" ]]; then
    echo "[ERROR] SWE-smith full train data not found: ${SWESMITH_DATA}"
    echo "        The downloader defaults to smoke.jsonl for safe path validation."
    echo "        Generate full train data explicitly:"
    echo "        MODE=full ALLOW_FULL=1 CREATE_ENV_DIRS=1 bash ${REPO_ROOT}/harborrl/data/download_swesmith.sh"
    echo "        Or set ROLLOUT_PROMPT_DATA to an existing smoke/custom JSONL."
    exit 1
  fi
  echo "[ERROR] ROLLOUT_PROMPT_DATA=${ROLLOUT_PROMPT_DATA} not found"
  exit 1
fi
echo "[config] ALGO=${ALGO} DATASET=${DATASET} SETA_SAFETY=${SETA_SAFETY} SAFETY_BENCH_REWARD=${SAFETY_BENCH_REWARD} AGENTHARM_REWARD=${AGENTHARM_REWARD}"
if [[ "${INCLUDES_SWESMITH}" == "1" ]]; then
  SWESMITH_ARTIFACT_LOCK="${SWESMITH_ARTIFACT_LOCK:-$(dirname -- "${SWESMITH_ENV_DIR}")/.swesmith_artifact.lock}"
  if ! command -v flock >/dev/null 2>&1; then
    echo "[ERROR] flock is required for SWE-smith artifact/task consistency."
    exit 1
  fi
  exec 8>"${SWESMITH_ARTIFACT_LOCK}"
  if ! flock -s -n 8; then
    echo "[ERROR] SWE-smith conversion/publication is active: ${SWESMITH_ARTIFACT_LOCK}"
    echo "        Wait for conversion to finish, then restart training."
    exit 1
  fi
  ensure_swesmith_env_dirs "${ROLLOUT_PROMPT_DATA}"
  if [[ "${SWESMITH_REQUIRE_FULL_DATA}" == "1" ]]; then
    SWESMITH_SOURCE_PROMPT_DATA="${ROLLOUT_PROMPT_DATA}"
    SWESMITH_STATS_SOURCE="${SWESMITH_STATS_PATH:-$(dirname -- "${ROLLOUT_PROMPT_DATA}")/convert_stats.json}"
    SWESMITH_FROZEN_PROMPT_DATA="${RUN_DIR}/config/swesmith_train_${SWESMITH_ARTIFACT_SHA256}.jsonl"
    SWESMITH_FROZEN_TMP="${SWESMITH_FROZEN_PROMPT_DATA}.tmp.$$"
    rm -f "${SWESMITH_FROZEN_TMP}"
    if ! ln -- "${ROLLOUT_PROMPT_DATA}" "${SWESMITH_FROZEN_TMP}"; then
      echo "[ERROR] Could not hard-link the validated SWE-smith artifact into the run directory."
      echo "        Keep RUNS_ROOT and the formal dataset on the same filesystem."
      exit 1
    fi
    read -r frozen_device frozen_inode < <(stat -c '%d %i' "${SWESMITH_FROZEN_TMP}")
    if [[ "${frozen_device}" != "${SWESMITH_SOURCE_DEVICE}" || "${frozen_inode}" != "${SWESMITH_SOURCE_INODE}" ]]; then
      rm -f "${SWESMITH_FROZEN_TMP}"
      echo "[ERROR] SWE-smith train.jsonl changed during preflight; retry after conversion finishes."
      exit 1
    fi
    if [[ -e "${SWESMITH_FROZEN_PROMPT_DATA}" ]]; then
      read -r existing_device existing_inode < <(stat -c '%d %i' "${SWESMITH_FROZEN_PROMPT_DATA}")
      if [[ "${existing_device}" == "${SWESMITH_SOURCE_DEVICE}" && "${existing_inode}" == "${SWESMITH_SOURCE_INODE}" ]]; then
        rm -f "${SWESMITH_FROZEN_TMP}"
      else
        rm -f "${SWESMITH_FROZEN_PROMPT_DATA}"
        mv "${SWESMITH_FROZEN_TMP}" "${SWESMITH_FROZEN_PROMPT_DATA}"
      fi
    else
      mv "${SWESMITH_FROZEN_TMP}" "${SWESMITH_FROZEN_PROMPT_DATA}"
    fi
    chmod a-w "${SWESMITH_FROZEN_PROMPT_DATA}"
    cp "${SWESMITH_STATS_SOURCE}" "${RUN_DIR}/config/swesmith_convert_stats.json.tmp"
    mv -f "${RUN_DIR}/config/swesmith_convert_stats.json.tmp" "${RUN_DIR}/config/swesmith_convert_stats.json"
    python3 - "${REPO_ROOT}" "${SWESMITH_FROZEN_PROMPT_DATA}" "${RUN_DIR}/config/swesmith_convert_stats.json" "${SWESMITH_ARTIFACT_SHA256}" <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
from harborrl.data.convert_swesmith import (
    CANONICAL_SWESMITH_ROWS,
    validate_swesmith_artifact_manifest,
)

validate_swesmith_artifact_manifest(
    Path(sys.argv[2]),
    stats_path=Path(sys.argv[3]),
    require_full=True,
    artifact_rows=CANONICAL_SWESMITH_ROWS,
    artifact_sha256=sys.argv[4],
)
PY
    ROLLOUT_PROMPT_DATA="${SWESMITH_FROZEN_PROMPT_DATA}"
    echo "[swesmith] frozen formal artifact: ${ROLLOUT_PROMPT_DATA}"
  fi
fi
echo "[config] sources seta=${INCLUDES_SETA} safety=${INCLUDES_SAFETY} agentharm=${INCLUDES_AGENTHARM} swesmith=${INCLUDES_SWESMITH}"
echo "[config] data=${ROLLOUT_PROMPT_DATA}"

NEEDS_ENV_ROUTER="0"
if [[ "${INCLUDES_SETA}" == "1" ]]; then
  NEEDS_ENV_ROUTER="1"
fi
if [[ "${INCLUDES_SAFETY}" == "1" && "${AGENT_SAFETYBENCH_REMOTE_ENV}" == "1" ]]; then
  NEEDS_ENV_ROUTER="1"
fi
if [[ "${INCLUDES_AGENTHARM}" == "1" && "${AGENTHARM_REMOTE_ENV}" == "1" ]]; then
  NEEDS_ENV_ROUTER="1"
fi
if [[ "${INCLUDES_SWESMITH}" == "1" ]]; then
  NEEDS_ENV_ROUTER="1"
fi
echo "[config] needs_env_router=${NEEDS_ENV_ROUTER} AGENT_SAFETYBENCH_REMOTE_ENV=${AGENT_SAFETYBENCH_REMOTE_ENV} AGENTHARM_REMOTE_ENV=${AGENTHARM_REMOTE_ENV}"

# Optional dataset blacklist (issue #3 §1.X / §2.x stuck offenders).
# Default-ON; set USE_BLACKLIST=0 to keep the raw dataset.
USE_BLACKLIST="${USE_BLACKLIST:-1}"
DATASET_BLACKLIST="${DATASET_BLACKLIST:-786,96,90,456,856,210,999,305,25,684,345,553,962,916,1264,282,324,768,46,996}"
if [[ "${USE_BLACKLIST}" == "1" && "${INCLUDES_SETA}" == "1" && -n "${DATASET_BLACKLIST}" ]]; then
  if [[ "${DRY_RUN}" == "1" ]]; then
    FILTERED_DATA="${RUN_DIR}/config/$(basename "${ROLLOUT_PROMPT_DATA%.jsonl}").filtered.jsonl"
  else
    FILTERED_DATA="${ROLLOUT_PROMPT_DATA%.jsonl}.filtered.jsonl"
  fi
  python3 - "$ROLLOUT_PROMPT_DATA" "$FILTERED_DATA" "$DATASET_BLACKLIST" <<'PY'
import json, sys
src, dst, blk = sys.argv[1], sys.argv[2], set(sys.argv[3].split(","))
kept = dropped = 0
with open(src) as fin, open(dst, "w") as fout:
    for line in fin:
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            fout.write(line); kept += 1; continue
        # try common task-id fields
        tid = str(obj.get("task_name") or obj.get("task_id")
                  or obj.get("metadata", {}).get("task_name")
                  or obj.get("metadata", {}).get("task_id") or "")
        if tid in blk:
            dropped += 1
        else:
            fout.write(line); kept += 1
print(f"[blacklist] kept={kept} dropped={dropped} blacklist_size={len(blk)}")
PY
  ROLLOUT_PROMPT_DATA="${FILTERED_DATA}"
  echo "[blacklist] using filtered dataset: ${ROLLOUT_PROMPT_DATA}"
fi
