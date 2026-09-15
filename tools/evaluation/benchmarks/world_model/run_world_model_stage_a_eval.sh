#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"
RUNS_ROOT="${RUNS_ROOT:-${REPO_ROOT}/runs}"

PYTHON_BIN="${PYTHON_BIN:-python3}"

export PYTHONPATH="${REPO_ROOT}/backends/slime:${PYTHONPATH:-}"

STAMP="$(date +%Y%m%d_%H%M%S)"
WM_OUT_DIR="${WM_OUT_DIR:-${RUNS_ROOT}/evaluation/world_model_stage_a_eval/${STAMP}}"
WM_DEFAULT_RECORDS="${WM_DEFAULT_RECORDS:-}"
WM_USE_DEFAULT_RECORDS="${WM_USE_DEFAULT_RECORDS:-0}"
WM_SOURCE_RECORDS="${WM_SOURCE_RECORDS:-}"
WM_INPUT_GLOB="${WM_INPUT_GLOB:-${RUNS_ROOT}/testing/world_model_smoke/*/metadata/rollout_*.pt}"
WM_FILTERS="${WM_FILTERS:-full,clean,tool_only}"
WM_REQUIRED_FILTERS="${WM_REQUIRED_FILTERS:-full,clean}"
WM_BAD_EVAL_REASONS="${WM_BAD_EVAL_REASONS:-eval_timeout,eval_parse_failed}"
WM_MAX_RECORDS="${WM_MAX_RECORDS:-0}"
WM_MIN_RECORDS="${WM_MIN_RECORDS:-1}"
WM_REUSE_CACHE="${WM_REUSE_CACHE:-1}"
WM_RUN_RANK="${WM_RUN_RANK:-1}"

WM_ENCODER="${WM_ENCODER:-hash}"
WM_ALLOW_HF="${WM_ALLOW_HF:-0}"
WM_HF_MODEL="${WM_HF_MODEL:-}"
WM_HF_LOCAL_FILES_ONLY="${WM_HF_LOCAL_FILES_ONLY:-1}"
WM_HASH_HIDDEN_DIM="${WM_HASH_HIDDEN_DIM:-256}"
WM_CONTEXT_MAX_CHARS="${WM_CONTEXT_MAX_CHARS:-4096}"
WM_CONTEXT_SOURCE="${WM_CONTEXT_SOURCE:-world_model}"
WM_CONTEXT_TRUNCATION="${WM_CONTEXT_TRUNCATION:-head_tail}"
WM_HF_MAX_LENGTH="${WM_HF_MAX_LENGTH:-2048}"
WM_HF_POOLING="${WM_HF_POOLING:-mean}"
WM_CACHE_BATCH_SIZE="${WM_CACHE_BATCH_SIZE:-2}"
WM_TRAIN_BATCH_SIZE="${WM_TRAIN_BATCH_SIZE:-8}"
WM_SEED="${WM_SEED:-42}"
WM_BOOTSTRAP_SAMPLES="${WM_BOOTSTRAP_SAMPLES:-500}"

if [[ "${WM_ENCODER}" == "hf" ]]; then
  WM_LATENT_DIM="${WM_LATENT_DIM:-1024}"
  WM_EPOCHS="${WM_EPOCHS:-5}"
else
  WM_LATENT_DIM="${WM_LATENT_DIM:-128}"
  WM_EPOCHS="${WM_EPOCHS:-3}"
fi
WM_VAL_RATIO="${WM_VAL_RATIO:-0.25}"
WM_LR="${WM_LR:-1e-4}"
WM_SIGREG_COEF="${WM_SIGREG_COEF:-0.1}"
WM_ACTION_CONTRAST_COEF="${WM_ACTION_CONTRAST_COEF:-0.1}"
WM_VALUE_COEF="${WM_VALUE_COEF:-0.0}"

mkdir -p "${WM_OUT_DIR}/logs" "${WM_OUT_DIR}/shards"

if [[ -z "${WM_SOURCE_RECORDS}" && "${WM_USE_DEFAULT_RECORDS}" == "1" && -n "${WM_DEFAULT_RECORDS}" && -s "${WM_DEFAULT_RECORDS}" ]]; then
  WM_SOURCE_RECORDS="${WM_DEFAULT_RECORDS}"
fi

ALL_RECORDS="${WM_OUT_DIR}/records_all.jsonl"
if [[ -n "${WM_SOURCE_RECORDS}" ]]; then
  if [[ ! -s "${WM_SOURCE_RECORDS}" ]]; then
    echo "[wm-stage-a] WM_SOURCE_RECORDS missing or empty: ${WM_SOURCE_RECORDS}" >&2
    exit 1
  fi
  cp "${WM_SOURCE_RECORDS}" "${ALL_RECORDS}"
  echo "[wm-stage-a] copied source records: ${WM_SOURCE_RECORDS}"
else
  mapfile -t ROLLOUT_FILES < <(compgen -G "${WM_INPUT_GLOB}" | sort)
  if (( ${#ROLLOUT_FILES[@]} == 0 )); then
    echo "[wm-stage-a] no source records and no rollout files matched: ${WM_INPUT_GLOB}" >&2
    exit 1
  fi
  : > "${ALL_RECORDS}"
  idx=0
  for rollout in "${ROLLOUT_FILES[@]}"; do
    shard="${WM_OUT_DIR}/shards/records_${idx}.jsonl"
    "${PYTHON_BIN}" -m slime.world_model.build_dataset \
      --input "${rollout}" \
      --output "${shard}" \
      --context-max-chars "${WM_CONTEXT_MAX_CHARS}" \
      --context-source "${WM_CONTEXT_SOURCE}" \
      --context-truncation "${WM_CONTEXT_TRUNCATION}" \
      --summary-output "${WM_OUT_DIR}/shards/summary_${idx}.json" \
      > "${WM_OUT_DIR}/logs/build_${idx}.log" 2>&1
    cat "${shard}" >> "${ALL_RECORDS}"
    idx=$((idx + 1))
  done
  echo "[wm-stage-a] built records from ${#ROLLOUT_FILES[@]} rollout files"
fi

if [[ ! -s "${ALL_RECORDS}" ]]; then
  echo "[wm-stage-a] no world-model records found after collection" >&2
  exit 1
fi

COUNTS_TSV="${WM_OUT_DIR}/bucket_counts.tsv"
"${PYTHON_BIN}" - "${ALL_RECORDS}" "${WM_OUT_DIR}" "${WM_FILTERS}" "${WM_BAD_EVAL_REASONS}" "${WM_MAX_RECORDS}" "${WM_CONTEXT_MAX_CHARS}" <<'PY'
import json
import os
import sys
from pathlib import Path

from slime.world_model.build_dataset import _eval_reason, _observation_source, summarize_world_model_records

src = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
filters = [item.strip() for item in sys.argv[3].split(",") if item.strip()]
bad_reasons = {item.strip().lower() for item in sys.argv[4].split(",") if item.strip()}
max_records = int(sys.argv[5])
context_max_chars = int(sys.argv[6])

rows = [json.loads(line) for line in src.open("r", encoding="utf-8") if line.strip()]

def match(name, row):
    reason = _eval_reason(row)
    reason = None if reason is None else str(reason).lower()
    status = str(row.get("status", "")).lower()
    if name == "full":
        return True
    if name == "clean":
        return status == "completed" and reason not in bad_reasons
    if name == "tool_only":
        return bool(row.get("has_tool_result"))
    if name == "completed":
        return status == "completed"
    if name == "eval_summary_only":
        return _observation_source(row) == "eval_summary"
    if name == "failed_or_truncated":
        return status in {"failed", "truncated", "aborted"}
    if name == "timeout_parse_failed":
        return reason in bad_reasons
    raise ValueError(f"unknown WM_FILTERS bucket: {name}")

counts = []
for name in filters:
    selected = [row for row in rows if match(name, row)]
    if max_records > 0:
        selected = selected[:max_records]
    bucket_dir = out_dir / name
    bucket_dir.mkdir(parents=True, exist_ok=True)
    records_path = bucket_dir / "records.jsonl"
    with records_path.open("w", encoding="utf-8") as fh:
        for row in selected:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary = summarize_world_model_records(
        selected,
        context_max_chars=context_max_chars,
        input_record_count=len(rows),
        filter_args={
            "bucket": name,
            "bad_eval_reasons": sorted(bad_reasons),
            "max_records": max_records,
        },
    )
    (bucket_dir / "records_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    counts.append((name, len(selected), str(records_path)))

with (out_dir / "bucket_counts.tsv").open("w", encoding="utf-8") as fh:
    for name, count, records_path in counts:
        fh.write(f"{name}\t{count}\t{records_path}\n")

print("[wm-stage-a] buckets:", json.dumps(
    [{"bucket": name, "records": count, "path": records_path} for name, count, records_path in counts],
    ensure_ascii=False,
    sort_keys=True,
))
PY

select_gpus() {
  if [[ -n "${WM_DUAL_GPU_IDS:-}" ]]; then
    echo "${WM_DUAL_GPU_IDS}"
    return
  fi
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "${CUDA_VISIBLE_DEVICES}"
    return
  fi
  echo "0,1"
}

IFS=',' read -r -a GPU_IDS <<< "$(select_gpus)"
if (( ${#GPU_IDS[@]} == 0 )); then
  GPU_IDS=("0")
fi

if [[ "${WM_ENCODER}" == "hf" ]]; then
  if [[ "${WM_ALLOW_HF}" != "1" ]]; then
    echo "[wm-stage-a] HF encoder is fail-closed. Set WM_ALLOW_HF=1 and WM_HF_MODEL=/local/path explicitly." >&2
    exit 1
  fi
  if [[ "${WM_HF_LOCAL_FILES_ONLY}" == "1" && ! -d "${WM_HF_MODEL}" ]]; then
    echo "[wm-stage-a] missing local WM_HF_MODEL: ${WM_HF_MODEL}" >&2
    exit 1
  fi
fi

is_required_bucket() {
  local bucket="$1"
  [[ ",${WM_REQUIRED_FILTERS}," == *",${bucket},"* ]]
}

run_bucket() {
  local bucket="$1"
  local count="$2"
  local records="$3"
  local gpu_id="$4"
  local bucket_dir="${WM_OUT_DIR}/${bucket}"
  local log_file="${bucket_dir}/logs/stage_a.log"
  local config_file="${bucket_dir}/stage_a_config.json"
  local next_config="${bucket_dir}/stage_a_config.next.json"
  mkdir -p "${bucket_dir}/logs"

  if (( count < WM_MIN_RECORDS )); then
    echo "[wm-stage-a:${bucket}] skip count=${count} < WM_MIN_RECORDS=${WM_MIN_RECORDS}" | tee "${log_file}"
    if is_required_bucket "${bucket}"; then
      echo "[wm-stage-a:${bucket}] required bucket is missing enough records" | tee -a "${log_file}" >&2
      return 2
    fi
    return 0
  fi

  echo "[wm-stage-a:${bucket}] gpu=${gpu_id} records=${count} encoder=${WM_ENCODER} out=${bucket_dir}" | tee "${log_file}"

  "${PYTHON_BIN}" - "${records}" "${next_config}" <<PY
import hashlib
import json
import sys
from pathlib import Path

records_path = Path(sys.argv[1])
digest = hashlib.sha256(records_path.read_bytes()).hexdigest()
config = {
    "script_version": "run_world_model_stage_a_eval_v1",
    "bucket": ${bucket@Q},
    "records_path": str(records_path),
    "records_sha256": digest,
    "record_count": int(${count@Q}),
    "encoder": ${WM_ENCODER@Q},
    "allow_hf": ${WM_ALLOW_HF@Q},
    "hf_model": ${WM_HF_MODEL@Q} if ${WM_ENCODER@Q} == "hf" else None,
    "hf_local_files_only": ${WM_HF_LOCAL_FILES_ONLY@Q},
    "hash_hidden_dim": int(${WM_HASH_HIDDEN_DIM@Q}),
    "hf_max_length": int(${WM_HF_MAX_LENGTH@Q}),
    "hf_pooling": ${WM_HF_POOLING@Q},
    "context_source": ${WM_CONTEXT_SOURCE@Q},
    "context_truncation": ${WM_CONTEXT_TRUNCATION@Q},
    "cache_batch_size": int(${WM_CACHE_BATCH_SIZE@Q}),
    "latent_dim": int(${WM_LATENT_DIM@Q}),
    "train_batch_size": int(${WM_TRAIN_BATCH_SIZE@Q}),
    "epochs": int(${WM_EPOCHS@Q}),
    "lr": float(${WM_LR@Q}),
    "sigreg_coef": float(${WM_SIGREG_COEF@Q}),
    "action_contrast_coef": float(${WM_ACTION_CONTRAST_COEF@Q}),
    "value_coef": float(${WM_VALUE_COEF@Q}),
    "val_ratio": float(${WM_VAL_RATIO@Q}),
    "seed": int(${WM_SEED@Q}),
    "bootstrap_samples": int(${WM_BOOTSTRAP_SAMPLES@Q}),
}
Path(sys.argv[2]).write_text(json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

  local reuse_artifacts=0
  if [[ "${WM_REUSE_CACHE}" == "1" && -s "${bucket_dir}/cached_hidden.pt" && -s "${bucket_dir}/probe.pt" ]] \
    && cmp -s "${next_config}" "${config_file}"; then
    reuse_artifacts=1
  fi

  if (( reuse_artifacts == 0 )); then
    local cache_tmp="${bucket_dir}/cached_hidden.pt.tmp.$$"
    cache_cmd=(
      "${PYTHON_BIN}" -m slime.world_model.cache_text_hidden
      --input "${records}"
      --output "${cache_tmp}"
      --encoder "${WM_ENCODER}"
      --batch-size "${WM_CACHE_BATCH_SIZE}"
    )
    if [[ "${WM_ENCODER}" == "hash" ]]; then
      cache_cmd+=(--hidden-dim "${WM_HASH_HIDDEN_DIM}")
    else
      cache_cmd+=(
        --hf-model "${WM_HF_MODEL}"
        --max-length "${WM_HF_MAX_LENGTH}"
        --pooling "${WM_HF_POOLING}"
        --device auto
      )
      if [[ "${WM_HF_LOCAL_FILES_ONLY}" == "1" ]]; then
        cache_cmd+=(--hf-local-files-only)
      fi
    fi
    CUDA_VISIBLE_DEVICES="${gpu_id}" "${cache_cmd[@]}" >> "${log_file}" 2>&1
    mv "${cache_tmp}" "${bucket_dir}/cached_hidden.pt"
  else
    echo "[wm-stage-a:${bucket}] reuse cache ${bucket_dir}/cached_hidden.pt" >> "${log_file}"
  fi

  if (( reuse_artifacts == 0 )); then
    local probe_tmp="${bucket_dir}/probe.pt.tmp.$$"
    CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON_BIN}" -m slime.world_model.train_probe \
      --input "${bucket_dir}/cached_hidden.pt" \
      --output "${probe_tmp}" \
      --latent-dim "${WM_LATENT_DIM}" \
      --batch-size "${WM_TRAIN_BATCH_SIZE}" \
      --epochs "${WM_EPOCHS}" \
      --lr "${WM_LR}" \
      --sigreg-coef "${WM_SIGREG_COEF}" \
      --action-contrast-coef "${WM_ACTION_CONTRAST_COEF}" \
      --value-coef "${WM_VALUE_COEF}" \
      --val-ratio "${WM_VAL_RATIO}" \
      --seed "${WM_SEED}" >> "${log_file}" 2>&1
    mv "${probe_tmp}" "${bucket_dir}/probe.pt"
  else
    echo "[wm-stage-a:${bucket}] reuse probe ${bucket_dir}/probe.pt" >> "${log_file}"
  fi

  local eval_tmp="${bucket_dir}/eval_summary.json.tmp.$$"
  CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON_BIN}" -m slime.world_model.evaluate_probe \
    --checkpoint "${bucket_dir}/probe.pt" \
    --input "${bucket_dir}/cached_hidden.pt" \
    --output "${eval_tmp}" \
    --device auto \
    --seed "${WM_SEED}" \
    --bootstrap-samples "${WM_BOOTSTRAP_SAMPLES}" >> "${log_file}" 2>&1
  mv "${eval_tmp}" "${bucket_dir}/eval_summary.json"

  if [[ "${WM_RUN_RANK}" == "1" ]]; then
    local rankings_tmp="${bucket_dir}/rankings.jsonl.tmp.$$"
    CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON_BIN}" -m slime.world_model.rank_candidates \
      --checkpoint "${bucket_dir}/probe.pt" \
      --input "${bucket_dir}/cached_hidden.pt" \
      --output "${rankings_tmp}" >> "${log_file}" 2>&1
    mv "${rankings_tmp}" "${bucket_dir}/rankings.jsonl"
  fi
  mv "${next_config}" "${config_file}"
}

FAILED=0
ACTIVE_PIDS=()

wait_oldest() {
  local pid="${ACTIVE_PIDS[0]}"
  if ! wait "${pid}"; then
    FAILED=1
  fi
  ACTIVE_PIDS=("${ACTIVE_PIDS[@]:1}")
}

while IFS=$'\t' read -r bucket count records; do
  [[ -z "${bucket}" ]] && continue
  while (( ${#ACTIVE_PIDS[@]} >= ${#GPU_IDS[@]} )); do
    wait_oldest
  done
  gpu_id="${GPU_IDS[${#ACTIVE_PIDS[@]}]}"
  run_bucket "${bucket}" "${count}" "${records}" "${gpu_id}" > "${WM_OUT_DIR}/logs/${bucket}.driver.log" 2>&1 &
  ACTIVE_PIDS+=("$!")
done < "${COUNTS_TSV}"

while (( ${#ACTIVE_PIDS[@]} > 0 )); do
  wait_oldest
done

if (( FAILED != 0 )); then
  echo "[wm-stage-a] one or more bucket jobs failed. Logs: ${WM_OUT_DIR}/logs" >&2
  exit 1
fi

"${PYTHON_BIN}" - "${WM_OUT_DIR}" <<'PY'
import json
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
rows = []
counts = []
counts_path = out_dir / "bucket_counts.tsv"
if counts_path.exists():
    for line in counts_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        name, count, records = line.split("\t")
        counts.append({"bucket": name, "records": int(count), "records_path": records})

for item in counts:
    bucket_dir = out_dir / item["bucket"]
    eval_path = bucket_dir / "eval_summary.json"
    record_summary_path = bucket_dir / "records_summary.json"
    row = dict(item)
    row["out_dir"] = str(bucket_dir)
    row["eval_summary"] = str(eval_path) if eval_path.exists() else None
    if eval_path.exists():
        payload = json.loads(eval_path.read_text(encoding="utf-8"))
        metrics = payload.get("metrics", {})
        row["pred_mse_real"] = metrics.get("pred_mse_real")
        row["shuffle_gap_mse_shuffled_minus_real"] = metrics.get("shuffle_gap_mse_shuffled_minus_real")
        row["action_delta"] = metrics.get("action_delta")
        row["value_reward_spearman"] = (metrics.get("value_reward") or {}).get("spearman")
    if record_summary_path.exists():
        row["records_summary"] = str(record_summary_path)
    rows.append(row)

summary = {
    "schema_version": "lightrl_text_jepa_stage_a_eval_summary_v1",
    "out_dir": str(out_dir),
    "buckets": rows,
}
(out_dir / "summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print("[wm-stage-a] verify:", json.dumps(rows, ensure_ascii=False, sort_keys=True))
PY

echo "[wm-stage-a] done. Outputs: ${WM_OUT_DIR}"
