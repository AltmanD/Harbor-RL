short_mode() {
  case "$1" in
    dense_rule) echo "dense" ;;
    *) echo "$1" ;;
  esac
}

sanitize_run_part() {
  printf '%s' "$1" | tr -c 'A-Za-z0-9_.-' '-' | sed 's/-\\{1,\\}/-/g; s/^-//; s/-$//'
}

build_dataset_tag() {
  case "${DATASET}" in
    seta)
      echo "seta-$(short_mode "${SETA_SAFETY}")"
      ;;
    safety)
      echo "asb-$(short_mode "${SAFETY_BENCH_REWARD}")"
      ;;
    agentharm)
      echo "agentharm-$(short_mode "${AGENTHARM_REWARD}")"
      ;;
    swesmith)
      echo "swesmith"
      ;;
    mixed)
      local seta_ratio safety_ratio agentharm_ratio
      if [[ -n "${MIX_AGENTHARM_RATIO:-}" ]]; then
        seta_ratio="${MIX_SETA_RATIO:-0}"
        safety_ratio="${MIX_SAFETY_RATIO:-0}"
        agentharm_ratio="${MIX_AGENTHARM_RATIO:-0}"
      else
        seta_ratio="${MIX_SETA_RATIO:-1}"
        safety_ratio="${MIX_SAFETY_RATIO:-1}"
        agentharm_ratio="0"
      fi
      echo "mixed-s${seta_ratio}_asb${safety_ratio}_ah${agentharm_ratio}-rw$(short_mode "${SETA_SAFETY}")_$(short_mode "${SAFETY_BENCH_REWARD}")_$(short_mode "${AGENTHARM_REWARD}")"
      ;;
    *)
      echo "${DATASET}"
      ;;
  esac
}

build_algo_tag() {
  case "${ALGO}" in
    dapo)
      echo "dapo-ch${DAPO_EPS_CLIP_HIGH}-tok${DAPO_CALCULATE_PER_TOKEN_LOSS}-dyn${DAPO_DYNAMIC_SAMPLING}"
      ;;
    *)
      echo "${ALGO}"
      ;;
  esac
}

RUN_DATASET_TAG="$(sanitize_run_part "$(build_dataset_tag)")"
RUN_ALGO_TAG="$(sanitize_run_part "$(build_algo_tag)")"
RUN_HARNESS_TAG="$(sanitize_run_part "${HARNESS_OPTION}")"
RUN_ALGO_NAME_TAG="${ALGO}"
if [[ "${ALGO}" == "dapo" && "${DAPO_DYNAMIC_SAMPLING}" == "0" ]]; then
  RUN_ALGO_NAME_TAG="dapo_nodynamic"
fi
# Checkpoint saving keeps only the latest N checkpoints by default.
# When enabled, only the latest N checkpoints are kept; older ones are auto-deleted.
MAX_CKPT_KEEP="${MAX_CKPT_KEEP:-2}"
SAVE_INTERVAL="${SAVE_INTERVAL:-8}"
SAVE_FIRST_ROLLOUT="${SAVE_FIRST_ROLLOUT:-0}"
if [[ "${DEBUG_MODE}" == "1" ]]; then
  if [[ "${DATASET}" == "swesmith" ]]; then
    RUN_NAME="${RUN_NAME:-terminal-rl_${MODEL_TAG}_${NUM_GPUS}gpu_debug_swesmith_${RUN_ALGO_NAME_TAG}_think_harness-${RUN_HARNESS_TAG}_mt${MAX_TURN}_${RUN_TIMESTAMP}}"
  else
    RUN_NAME="${RUN_NAME:-terminal-rl_${MODEL_TAG}_${NUM_GPUS}gpu_debug_mixed_${RUN_ALGO_NAME_TAG}_think_s${MIX_SETA_RATIO}_asb${MIX_SAFETY_RATIO}_ah${MIX_AGENTHARM_RATIO}_harness-${RUN_HARNESS_TAG}_mt${MAX_TURN}_${RUN_TIMESTAMP}}"
  fi
  # Debug mode: never save checkpoints regardless of MAX_CKPT_KEEP
  MAX_CKPT_KEEP=0
else
  if [[ "${DATASET}" == "swesmith" ]]; then
    RUN_NAME="${RUN_NAME:-terminal-rl_${MODEL_TAG}_${NUM_GPUS}gpu_swesmith_${RUN_ALGO_NAME_TAG}_think_harness-${RUN_HARNESS_TAG}_mt${MAX_TURN}_${RUN_TIMESTAMP}}"
  else
    RUN_NAME="${RUN_NAME:-terminal-rl_${MODEL_TAG}_${NUM_GPUS}gpu_mixed_${RUN_ALGO_NAME_TAG}_think_s${MIX_SETA_RATIO}_asb${MIX_SAFETY_RATIO}_ah${MIX_AGENTHARM_RATIO}_harness-${RUN_HARNESS_TAG}_mt${MAX_TURN}_${RUN_TIMESTAMP}}"
  fi
fi

# ── Unified run directory (see STORAGE.md) ───────────────────────────────
# New runs are grouped by lifecycle: training, evaluation, or testing. Debug
# mode is a testing variant and therefore lives under testing/debug. An
# explicitly supplied RUN_DIR remains authoritative for resume/compatibility.
RUNS_ROOT="${RUNS_ROOT:-${REPO_ROOT}/runs}"
LIGHTRL_PERSIST_ROOT="${LIGHTRL_PERSIST_ROOT:-${RUNS_ROOT}/.persistent}"
CKPT_ROOT="${CKPT_ROOT:-${LIGHTRL_PERSIST_ROOT}/checkpoints}"
RUN_ID="${RUN_ID:-${RUN_NAME}}"
RUN_CATEGORY="${RUN_CATEGORY:-training}"
if [[ "${DEBUG_MODE}" == "1" ]]; then
  RUN_CATEGORY="testing/debug"
fi
case "${RUN_CATEGORY}" in
  training|evaluation|testing|testing/debug) ;;
  train) RUN_CATEGORY="training" ;;
  eval|evaluate) RUN_CATEGORY="evaluation" ;;
  test) RUN_CATEGORY="testing" ;;
  debug) RUN_CATEGORY="testing/debug" ;;
  *)
    echo "[ERROR] Unknown RUN_CATEGORY=${RUN_CATEGORY}. Use: training|evaluation|testing|testing/debug" >&2
    exit 1
    ;;
esac
RUN_DIR="${RUN_DIR:-${RUNS_ROOT}/${RUN_CATEGORY}/${RUN_ID}}"
TBENCH_OUTPUT_ROOT="${TBENCH_OUTPUT_ROOT:-${RUN_DIR}/environment_outputs}"
WANDB_DIR="${WANDB_DIR:-${LIGHTRL_PERSIST_ROOT}/wandb/${RUN_ID}}"
export RUNS_ROOT RUN_ID RUN_DIR RUN_CATEGORY TBENCH_OUTPUT_ROOT CKPT_ROOT WANDB_DIR

# Create directory structure via run_paths.py
# A dry-run must be executable on a login/debug node where the production
# checkpoint mount is intentionally unavailable or read-only.  Keep the real
# CKPT_ROOT untouched for normal training, but direct path initialization to a
# disposable directory under the run directory while validating the launcher.
RUN_PATHS_CKPT_ROOT="${CKPT_ROOT}"
if [[ "${DRY_RUN}" == "1" ]]; then
  RUN_PATHS_CKPT_ROOT="${RUN_DIR}/checkpoints"
  WANDB_DIR="${RUN_DIR}/metrics/wandb"
  export WANDB_DIR
else
  # Persistent artifacts must never prevent the metric/train logs from being
  # produced.  Disable checkpointing or fall back to run-local W&B storage if
  # either external path is unavailable at startup.
  if (( MAX_CKPT_KEEP > 0 )) && { ! mkdir -p "${CKPT_ROOT}/${RUN_ID}" || [[ ! -w "${CKPT_ROOT}/${RUN_ID}" ]]; }; then
    echo "[WARN] CHECKPOINT_STORAGE_UNAVAILABLE_NONFATAL path=${CKPT_ROOT}/${RUN_ID}; checkpointing disabled, training continues" >&2
    MAX_CKPT_KEEP=0
  fi
  if ! mkdir -p "${WANDB_DIR}" || [[ ! -w "${WANDB_DIR}" ]]; then
    echo "[WARN] WANDB_STORAGE_UNAVAILABLE path=${WANDB_DIR}; falling back to ${RUN_DIR}/metrics/wandb" >&2
    WANDB_DIR="${RUN_DIR}/metrics/wandb"
    export WANDB_DIR
  fi
fi
if [[ "${RESUME_EXISTING_RUN:-0}" == "1" ]]; then
  [[ -f "${RUN_DIR}/meta.json" ]] || {
    echo "[ERROR] Resume requested but original run metadata is missing: ${RUN_DIR}/meta.json" >&2
    exit 1
  }
else
  MAX_CKPT_KEEP="${MAX_CKPT_KEEP}" python3 -m harborrl.platform.paths init \
    --runs-root "${RUNS_ROOT}" \
    --ckpt-root "${RUN_PATHS_CKPT_ROOT}" \
    --run-id "${RUN_ID}" \
    --category "${RUN_CATEGORY}" \
    --run-dir "${RUN_DIR}" > /dev/null
fi

# Derive all paths from RUN_DIR
RUN_LOG_DIR="${RUN_DIR}/logs"
if [[ "${TERMINAL_SAVE_TRAJ_DIR+x}" ]]; then
  TERMINAL_SAVE_TRAJ_DIR="${TERMINAL_SAVE_TRAJ_DIR}"
else
  TERMINAL_SAVE_TRAJ_DIR="${RUN_DIR}/trajectories"
fi
TERMINAL_STRUCTURED_METRICS="${TERMINAL_STRUCTURED_METRICS:-1}"
TERMINAL_METRICS_JSONL="${TERMINAL_METRICS_JSONL:-${RUN_LOG_DIR}/metrics.jsonl}"
TERMINAL_WANDB_METRIC_PROFILE="${TERMINAL_WANDB_METRIC_PROFILE:-full}"
export TERMINAL_STRUCTURED_METRICS TERMINAL_METRICS_JSONL TERMINAL_WANDB_METRIC_PROFILE
TRAIN_PYTHON="${TRAIN_PYTHON:-python3}"
CLAUDE_CODE_XTRACE_WAS_ON=0
if [[ "${HARNESS_OPTION}" == "claude-code" && "$-" == *x* ]]; then
  CLAUDE_CODE_XTRACE_WAS_ON=1
  set +x
fi
CLAUDE_CODE_CLI="${CLAUDE_CODE_CLI:-claude}"
CLAUDE_CODE_LLM_BACKEND="${CLAUDE_CODE_LLM_BACKEND:-sglang}"
case "${CLAUDE_CODE_LLM_BACKEND}" in
  sglang|qwen|qwen-sglang|local|local-sglang)
    CLAUDE_CODE_LLM_BACKEND="sglang"
    ;;
  anthropic|claude|claude-api|external)
    CLAUDE_CODE_LLM_BACKEND="anthropic"
    ;;
  *)
    echo "[ERROR] Unknown CLAUDE_CODE_LLM_BACKEND=${CLAUDE_CODE_LLM_BACKEND}. Use: sglang|anthropic" >&2
    exit 1
    ;;
esac
CLAUDE_CODE_MODEL="${CLAUDE_CODE_MODEL:-}"
CLAUDE_CODE_QWEN_GATEWAY_MODEL="${CLAUDE_CODE_QWEN_GATEWAY_MODEL:-qwen-8b-sglang}"
CLAUDE_CODE_LOCAL_RUN_ROOT="${CLAUDE_CODE_LOCAL_RUN_ROOT:-${CLAUDE_CODE_WORKSPACE_ROOT:-${RUN_LOG_DIR}/claude_code_cli}}"
CLAUDE_CODE_WORKSPACE_ROOT="${CLAUDE_CODE_WORKSPACE_ROOT:-${CLAUDE_CODE_LOCAL_RUN_ROOT}}"
CLAUDE_CODE_TURN_TIMEOUT_SEC="${CLAUDE_CODE_TURN_TIMEOUT_SEC:-900}"
CLAUDE_CODE_TOOL_TIMEOUT_MS="${CLAUDE_CODE_TOOL_TIMEOUT_MS:-300000}"
CLAUDE_CODE_MAX_TOOL_ROUNDS="${CLAUDE_CODE_MAX_TOOL_ROUNDS:-10}"
CLAUDE_CODE_OUTPUT_FORMAT="${CLAUDE_CODE_OUTPUT_FORMAT:-json}"
if [[ -z "${CLAUDE_CODE_PERMISSION_MODE+x}" ]]; then
  # Claude Code rejects bypassPermissions when the CLI runs as root, which is
  # the normal execution user in privileged GPU RJobs. dontAsk prevents a
  # headless permission prompt while explicit allowedTools remains the only
  # tool allowlist.
  if [[ "$(id -u)" -eq 0 ]]; then
    CLAUDE_CODE_PERMISSION_MODE="dontAsk"
  else
    CLAUDE_CODE_PERMISSION_MODE="bypassPermissions"
  fi
fi
CLAUDE_CODE_ALLOWED_TOOLS="${CLAUDE_CODE_ALLOWED_TOOLS:-mcp__terminal_rl__shell_exec,mcp__terminal_rl__shell_view,mcp__terminal_rl__shell_write_to_process,mcp__terminal_rl__shell_write_content_to_file,mcp__terminal_rl__read_file,mcp__terminal_rl__write_file,mcp__terminal_rl__list_dir}"
CLAUDE_CODE_DISALLOWED_TOOLS="${CLAUDE_CODE_DISALLOWED_TOOLS:-}"
CLAUDE_CODE_EXTRA_ARGS="${CLAUDE_CODE_EXTRA_ARGS:-}"
CLAUDE_CODE_SYSTEM_PROMPT="${CLAUDE_CODE_SYSTEM_PROMPT:-}"
CLAUDE_CODE_MCP_PYTHON="${CLAUDE_CODE_MCP_PYTHON:-${TRAIN_PYTHON}}"
CLAUDE_CODE_HTTP_MAX_RETRIES="${CLAUDE_CODE_HTTP_MAX_RETRIES:-3}"
CLAUDE_CODE_HTTP_RETRY_DELAY="${CLAUDE_CODE_HTTP_RETRY_DELAY:-1.0}"
if [[ -z "${CLAUDE_CODE_MARK_NON_TRAINABLE+x}" ]]; then
  if [[ "${CLAUDE_CODE_LLM_BACKEND}" == "sglang" ]]; then
    CLAUDE_CODE_MARK_NON_TRAINABLE="0"
  else
    CLAUDE_CODE_MARK_NON_TRAINABLE="1"
  fi
fi
ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}"
ANTHROPIC_AUTH_TOKEN="${ANTHROPIC_AUTH_TOKEN:-}"
ANTHROPIC_BASE_URL="${ANTHROPIC_BASE_URL:-}"
ANTHROPIC_API_URL="${ANTHROPIC_API_URL:-}"
if [[ "${CLAUDE_CODE_XTRACE_WAS_ON}" == "1" ]]; then
  set -x
fi
ENV_HTTP_MAX_RETRIES="${ENV_HTTP_MAX_RETRIES:-10}"
ENV_ALLOCATE_MAX_RETRIES="${ENV_ALLOCATE_MAX_RETRIES:-20}"
ENV_ALLOCATE_RETRY_BASE_DELAY="${ENV_ALLOCATE_RETRY_BASE_DELAY:-2.0}"
ENV_ALLOCATE_RETRY_MAX_DELAY="${ENV_ALLOCATE_RETRY_MAX_DELAY:-30.0}"
ENV_ALLOCATE_RETRY_BACKOFF="${ENV_ALLOCATE_RETRY_BACKOFF:-2.0}"
ENV_ALLOCATE_RETRY_JITTER="${ENV_ALLOCATE_RETRY_JITTER:-0.25}"
HTTP_RETRY_LOG_EVERY_N="${HTTP_RETRY_LOG_EVERY_N:-25}"
HTTP_RETRY_LOG_RESPONSE_CHARS="${HTTP_RETRY_LOG_RESPONSE_CHARS:-512}"
TERMINAL_RL_GENERATE_FAILURE_TRACEBACK="${TERMINAL_RL_GENERATE_FAILURE_TRACEBACK:-0}"
ENV_EVALUATE_MAX_RETRIES="${ENV_EVALUATE_MAX_RETRIES:-1}"
ENV_CLOSE_MAX_RETRIES="${ENV_CLOSE_MAX_RETRIES:-3}"
ENV_EXEC_TOOL_MAX_RETRIES="${ENV_EXEC_TOOL_MAX_RETRIES:-3}"
ENV_ALLOCATE_HTTP_TIMEOUT="${ENV_ALLOCATE_HTTP_TIMEOUT:-300}"
ENV_RESET_MAX_RETRIES="${ENV_RESET_MAX_RETRIES:-1}"
ENV_RESET_LEASE_MAX_ATTEMPTS="${ENV_RESET_LEASE_MAX_ATTEMPTS:-30}"
ENV_RESET_LEASE_RETRY_BASE_SLEEP="${ENV_RESET_LEASE_RETRY_BASE_SLEEP:-15}"
ENV_RESET_LEASE_RETRY_MAX_SLEEP="${ENV_RESET_LEASE_RETRY_MAX_SLEEP:-60}"

if [[ "${DATASET}" == "swesmith" ]]; then
  ENV_RESET_HTTP_TIMEOUT="${ENV_RESET_HTTP_TIMEOUT:-5400}"
else
  ENV_RESET_HTTP_TIMEOUT="${ENV_RESET_HTTP_TIMEOUT:-2100}"
fi

ENV_CLOSE_HTTP_TIMEOUT="${ENV_CLOSE_HTTP_TIMEOUT:-90}"
ENV_REMOTE_MAX_ACTIVE_TASKS="${ENV_REMOTE_MAX_ACTIVE_TASKS:-12}"
ENV_REMOTE_MAX_ACTIVE_RUNS="${ENV_REMOTE_MAX_ACTIVE_RUNS:-0}"
if [[ "${DATASET}" == "swesmith" ]]; then
  ENV_REMOTE_MAX_RUNS_PER_TASK="${ENV_REMOTE_MAX_RUNS_PER_TASK:-4}"
else
  ENV_REMOTE_MAX_RUNS_PER_TASK="${ENV_REMOTE_MAX_RUNS_PER_TASK:-8}"
fi
ENV_REMOTE_ADMISSION_TIMEOUT="${ENV_REMOTE_ADMISSION_TIMEOUT:-900}"
ENV_REMOTE_ADMISSION_LOG_INTERVAL="${ENV_REMOTE_ADMISSION_LOG_INTERVAL:-30}"
ENV_REMOTE_MAX_CONCURRENT_CLOSES="${ENV_REMOTE_MAX_CONCURRENT_CLOSES:-8}"
EVAL_ROLLOUT_MAX_CONCURRENCY="${EVAL_ROLLOUT_MAX_CONCURRENCY:-0}"


claude_code_preflight() {
  mkdir -p "${RUN_LOG_DIR}"
  if ! command -v "${CLAUDE_CODE_CLI}" >/dev/null 2>&1; then
    echo "[ERROR] HARNESS_OPTION=claude-code but CLAUDE_CODE_CLI=${CLAUDE_CODE_CLI} is not on PATH."
    echo "[ERROR] Install Claude Code CLI or set CLAUDE_CODE_CLI=/absolute/path/to/claude."
    return 1
  fi
  if ! "${CLAUDE_CODE_MCP_PYTHON}" -c "import mcp.server.fastmcp" > "${RUN_LOG_DIR}/claude_code_mcp_import_check.log" 2>&1; then
    echo "[ERROR] HARNESS_OPTION=claude-code but Python cannot import mcp.server.fastmcp."
    echo "[ERROR] Set CLAUDE_CODE_MCP_PYTHON to a Python with the mcp package installed."
    echo "[ERROR] Import log: ${RUN_LOG_DIR}/claude_code_mcp_import_check.log"
    return 1
  fi
  if [[ "${CLAUDE_CODE_LLM_BACKEND}" != "sglang" && -z "${ANTHROPIC_API_KEY}${ANTHROPIC_AUTH_TOKEN}" ]]; then
    echo "[WARN] claude-code auth env vars are empty. This is OK only if the Claude Code CLI is already authenticated via its own config."
  fi
}
