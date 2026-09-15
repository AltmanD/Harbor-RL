# ── Rollout knobs (env-configurable, baked into per-run yaml below) ──────
# MAX_TURN: max model turns per rollout (terminal_max_iterations in generate.py).
#   Empirical guidance based on 05-21 trajectory analysis (1743 trajectories):
#     - 30.0% trajectories hit max_iteration=15 (TRUNCATED) → most tasks need fewer turns
#     - Pass cases averaged 5-9 turns; tasks taking 10+ turns rarely passed
#     - Lowering to 10 trims tail-latency rollouts ≈ 33%, saving ~3 hours / 78 rollouts at 14h
#     - For exploratory runs needing more turns, override with MAX_TURN=15 or higher.
MAX_TURN="${MAX_TURN:-10}"
# TRAJECTORY_SAVE_INTERVAL controls full trajectory artifact storage globally.
# Per-dataset knobs override the global value and keep eval/training metrics
# unchanged; they only throttle full traj.json/meta.json artifact writes.
#   unset / config value 1: save every rollout step (backward compatible)
#   N>1: save only when train_step % N == 0
#   0: disable trajectory artifact writes even when TERMINAL_SAVE_TRAJ_DIR is set
TRAJECTORY_SAVE_INTERVAL="${TRAJECTORY_SAVE_INTERVAL:-}"
if [[ -n "${TRAJECTORY_SAVE_INTERVAL}" ]]; then
  DEFAULT_TRAJECTORY_SAVE_INTERVAL_SETA=""
  DEFAULT_TRAJECTORY_SAVE_INTERVAL_AGENT_SAFETYBENCH=""
  DEFAULT_TRAJECTORY_SAVE_INTERVAL_AGENTHARM=""
elif [[ "${DEBUG_MODE}" == "1" ]]; then
  DEFAULT_TRAJECTORY_SAVE_INTERVAL_SETA="1"
  DEFAULT_TRAJECTORY_SAVE_INTERVAL_AGENT_SAFETYBENCH="1"
  DEFAULT_TRAJECTORY_SAVE_INTERVAL_AGENTHARM="1"
else
  DEFAULT_TRAJECTORY_SAVE_INTERVAL_SETA="5"
  DEFAULT_TRAJECTORY_SAVE_INTERVAL_AGENT_SAFETYBENCH="5"
  DEFAULT_TRAJECTORY_SAVE_INTERVAL_AGENTHARM="10"
fi
TRAJECTORY_SAVE_INTERVAL_SETA="${TRAJECTORY_SAVE_INTERVAL_SETA:-${SAVE_INTERVAL_SETA:-${DEFAULT_TRAJECTORY_SAVE_INTERVAL_SETA}}}"
TRAJECTORY_SAVE_INTERVAL_AGENT_SAFETYBENCH="${TRAJECTORY_SAVE_INTERVAL_AGENT_SAFETYBENCH:-${TRAJECTORY_SAVE_INTERVAL_ASB:-${SAVE_INTERVAL_AGENT_SAFETYBENCH:-${SAVE_INTERVAL_ASB:-${DEFAULT_TRAJECTORY_SAVE_INTERVAL_AGENT_SAFETYBENCH}}}}}"
TRAJECTORY_SAVE_INTERVAL_AGENTHARM="${TRAJECTORY_SAVE_INTERVAL_AGENTHARM:-${SAVE_INTERVAL_AGENTHARM:-${DEFAULT_TRAJECTORY_SAVE_INTERVAL_AGENTHARM}}}"
TRAJECTORY_SAVE_POLICY="${TRAJECTORY_SAVE_POLICY:-step_interval}"
TRAJECTORY_TASK_SAVE_INTERVAL="${TRAJECTORY_TASK_SAVE_INTERVAL:-}"
TRAJECTORY_TASK_MAX_PER_STEP="${TRAJECTORY_TASK_MAX_PER_STEP:-2}"
TRAJECTORY_TASK_MAX_PER_TASK="${TRAJECTORY_TASK_MAX_PER_TASK:-24}"
TRAJECTORY_MAX_TOTAL="${TRAJECTORY_MAX_TOTAL:-5000}"
TRAJECTORY_SAVE_REWARD_STRATA="${TRAJECTORY_SAVE_REWARD_STRATA:-best,worst}"
TRAJECTORY_SAVE_LOG_DECISIONS="${TRAJECTORY_SAVE_LOG_DECISIONS:-0}"

# Generate a per-run yaml that overlays MAX_TURN onto the base CUSTOM_CONFIG_PATH.
# This is cleaner than mutating the base yaml — different concurrent runs can pick
# different MAX_TURN without stepping on each other.
BASE_CUSTOM_CONFIG_PATH="${CUSTOM_CONFIG_PATH}"
RUN_CUSTOM_CONFIG_PATH="${RUN_DIR}/config/rollout_config.yaml"
mkdir -p "$(dirname "${RUN_CUSTOM_CONFIG_PATH}")"
if [[ -f "${BASE_CUSTOM_CONFIG_PATH}" ]]; then
  python3 - "$BASE_CUSTOM_CONFIG_PATH" "$RUN_CUSTOM_CONFIG_PATH" "$MAX_TURN" "$TRAJECTORY_SAVE_INTERVAL" "$HARNESS_OPTION" "$TRAJECTORY_SAVE_INTERVAL_SETA" "$TRAJECTORY_SAVE_INTERVAL_AGENT_SAFETYBENCH" "$TRAJECTORY_SAVE_INTERVAL_AGENTHARM" <<'PY'
import sys, yaml
(
    src,
    dst,
    max_turn,
    traj_interval,
    harness_option,
    traj_seta,
    traj_asb,
    traj_agentharm,
) = (
    sys.argv[1],
    sys.argv[2],
    int(sys.argv[3]),
    sys.argv[4].strip(),
    sys.argv[5].strip(),
    sys.argv[6].strip(),
    sys.argv[7].strip(),
    sys.argv[8].strip(),
)
with open(src) as f:
    cfg = yaml.safe_load(f) or {}
cfg["max_iteration"] = max_turn
cfg["harness_option"] = harness_option
if traj_interval:
    cfg["trajectory_save_interval"] = int(traj_interval)
if traj_seta:
    cfg["trajectory_save_interval_seta"] = int(traj_seta)
if traj_asb:
    cfg["trajectory_save_interval_agent_safetybench"] = int(traj_asb)
if traj_agentharm:
    cfg["trajectory_save_interval_agentharm"] = int(traj_agentharm)
with open(dst, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=True)
PY
  CUSTOM_CONFIG_PATH="${RUN_CUSTOM_CONFIG_PATH}"
  if [[ -n "${TRAJECTORY_SAVE_INTERVAL}" ]]; then
    echo "[config] rollout yaml -> ${RUN_CUSTOM_CONFIG_PATH} (max_iteration=${MAX_TURN}, harness_option=${HARNESS_OPTION}, trajectory_save_interval=${TRAJECTORY_SAVE_INTERVAL}, per_dataset=seta:${TRAJECTORY_SAVE_INTERVAL_SETA}/asb:${TRAJECTORY_SAVE_INTERVAL_AGENT_SAFETYBENCH}/agentharm:${TRAJECTORY_SAVE_INTERVAL_AGENTHARM}, traj_policy=${TRAJECTORY_SAVE_POLICY}, task_interval=${TRAJECTORY_TASK_SAVE_INTERVAL:-<per-dataset>}, per_step=${TRAJECTORY_TASK_MAX_PER_STEP}, per_task=${TRAJECTORY_TASK_MAX_PER_TASK}, max_total=${TRAJECTORY_MAX_TOTAL})"
  else
    echo "[config] rollout yaml -> ${RUN_CUSTOM_CONFIG_PATH} (max_iteration=${MAX_TURN}, harness_option=${HARNESS_OPTION}, per_dataset=seta:${TRAJECTORY_SAVE_INTERVAL_SETA}/asb:${TRAJECTORY_SAVE_INTERVAL_AGENT_SAFETYBENCH}/agentharm:${TRAJECTORY_SAVE_INTERVAL_AGENTHARM}, traj_policy=${TRAJECTORY_SAVE_POLICY}, task_interval=${TRAJECTORY_TASK_SAVE_INTERVAL:-<per-dataset>}, per_step=${TRAJECTORY_TASK_MAX_PER_STEP}, per_task=${TRAJECTORY_TASK_MAX_PER_TASK}, max_total=${TRAJECTORY_MAX_TOTAL})"
  fi
else
  echo "[config] base yaml ${BASE_CUSTOM_CONFIG_PATH} not found; MAX_TURN=${MAX_TURN} will not take effect"
fi


if [[ "${HARNESS_OPTION}" == "claude-code" && "${DRY_RUN}" != "1" ]]; then
  claude_code_preflight
fi

# Keep one current-run pointer under the repository-level runs directory.
if [[ "${DRY_RUN}" != "1" ]]; then
  ln -sfn "${RUN_DIR}" "${RUNS_ROOT}/latest" 2>/dev/null || true
fi

# Only create ckpt dir and set SAVE_CKPT when saving is enabled
if (( MAX_CKPT_KEEP > 0 )); then
  SAVE_CKPT="${SAVE_CKPT:-${CKPT_ROOT}/${RUN_ID}}"
else
  SAVE_CKPT=""
fi
RESUME_LOAD="${RESUME_LOAD:-${SAVE_CKPT}}"

# Pre-flight: check the filesystem that will actually receive checkpoints.
# CKPT_ROOT may intentionally override the legacy EXPORT_ROOT default; probing
# EXPORT_ROOT in that case can either inspect the wrong disk or abort under
# set -e when the legacy mount is absent.
if (( MAX_CKPT_KEEP > 0 )) && [[ "${DRY_RUN}" != "1" ]]; then
  CKPT_SPACE_PATH="${CKPT_ROOT}"
  if [[ ! -d "${CKPT_SPACE_PATH}" ]]; then
    echo "[WARN] CHECKPOINT_STORAGE_UNAVAILABLE_NONFATAL path=${CKPT_SPACE_PATH}; checkpointing disabled, training continues" >&2
    MAX_CKPT_KEEP=0
    SAVE_CKPT=""
  elif ! AVAIL_GB_RAW=$(df -BG --output=avail "${CKPT_SPACE_PATH}" 2>/dev/null); then
    echo "[WARN] CHECKPOINT_STORAGE_UNAVAILABLE_NONFATAL cannot inspect ${CKPT_SPACE_PATH}; checkpointing disabled, training continues" >&2
    MAX_CKPT_KEEP=0
    SAVE_CKPT=""
  else
    AVAIL_GB=$(printf '%s\n' "${AVAIL_GB_RAW}" | tail -1 | tr -dc '0-9')
    CHECKPOINT_MIN_FREE_GB="${CHECKPOINT_MIN_FREE_GB:-128}"
    if [[ -z "${AVAIL_GB}" ]] || [[ "${AVAIL_GB}" -lt "${CHECKPOINT_MIN_FREE_GB}" ]]; then
      echo "[WARN] CHECKPOINT_STORAGE_LOW_NONFATAL path=${CKPT_SPACE_PATH} free=${AVAIL_GB:-unknown}G required=${CHECKPOINT_MIN_FREE_GB}G; checkpointing disabled, training continues" >&2
      MAX_CKPT_KEEP=0
      SAVE_CKPT=""
    fi
  fi
fi

RUN_LOG="${RUN_LOG_DIR}/train.log"

# ── Auto-mirror logs inside the unified run directory ──
# Canonical logs and compatibility shortcuts live under the categorized
# RUN_DIR (runs/training, runs/evaluation, or runs/testing).
# runs/latest is the only repository-level pointer to the current run.
TMP_DOC_ROOT="${RUN_LOG_DIR}/mirror"
TMP_DOC_LATEST="${TMP_DOC_ROOT}"
mkdir -p "${TMP_DOC_ROOT}"

GPU_RUN_LOG="${TMP_DOC_ROOT}/gpu_run.log"      # full stdout/stderr
GPU_ERR_LOG="${TMP_DOC_ROOT}/gpu_err.log"      # filtered errors (populated on failure)
GPU_TAIL_LOG="${TMP_DOC_ROOT}/gpu_tail.log"    # last ~300 lines (populated on failure)
if [[ "${DRY_RUN}" != "1" ]]; then
  ln -sfnT "${GPU_RUN_LOG}" "${RUN_DIR}/gpu_run.log" 2>/dev/null || true
  ln -sfnT "${GPU_ERR_LOG}" "${RUN_DIR}/gpu_err.log" 2>/dev/null || true
  ln -sfnT "${GPU_TAIL_LOG}" "${RUN_DIR}/gpu_tail.log" 2>/dev/null || true
fi

# Tee everything to both the run-specific file and tmp_doc copy
exec > >(tee -a "${RUN_LOG}" "${GPU_RUN_LOG}") 2>&1
echo "========================================"
echo "  Terminal-RL Run: ${RUN_NAME}"
echo "  Log dir:  ${RUN_LOG_DIR}"
echo "  Metrics:  ${TERMINAL_METRICS_JSONL} (structured=${TERMINAL_STRUCTURED_METRICS}, wandb=${TERMINAL_WANDB_METRIC_PROFILE})"
echo "  Harness:  ${HARNESS_OPTION}"
echo "  Ckpt:     ${SAVE_CKPT:-<disabled>}"
echo "  HF_CKPT:  ${HF_CKPT}"
echo "  REF_LOAD: ${REF_LOAD}"
echo "  MAX_CKPT_KEEP: ${MAX_CKPT_KEEP}"
echo "========================================"

# ── Model args selected by MODEL_ARGS_FILE ──────────────────────────────────
MODEL_ARGS_PATH="${SLIME_DIR}/scripts/models/${MODEL_ARGS_FILE}.sh"
if [[ ! -f "${MODEL_ARGS_PATH}" ]]; then
  echo "[ERROR] MODEL_ARGS_FILE=${MODEL_ARGS_FILE} not found at ${MODEL_ARGS_PATH}" >&2
  exit 1
fi
source "${MODEL_ARGS_PATH}"
