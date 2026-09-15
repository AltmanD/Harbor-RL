#!/usr/bin/env bash
# run_pool_server.sh — Hardened pool_server launcher for CPU/docker worker
#
# Incorporates all lessons from issue #3:
#   坑1: pool capacity must be >= rollout-batch-size × n-samples-per-prompt
#   坑2: max-concurrent-closes must be ~1.5× peak-close-rate
#   坑3: docker address-pool must be expanded for high-concurrency (check only, fix separately)
#   坑4: nofile ulimit must be ≥65k (checks and raises via prlimit if needed)
#   坑5: pre-flight cleanup of orphaned containers/networks from previous runs
#   坑6: connectivity probe before starting training
#   Extra: docker daemon health check before start
#
# Usage (on CPU/docker worker):
#   bash deploy/workers/run_pool_server.sh
#   bash deploy/workers/run_pool_server.sh --restart   # stop the listener on
#       ENV_SERVER_PORT (default 18082 in this mode), then start; replaces the
#       old restart_seta_docker_server.sh wrapper
#
# Key env vars:
#   WORKER_MAX_TASKS            (default 16)   — pool_server --max-tasks
#   WORKER_MAX_RUNS_PER_TASK    (default 8)    — pool_server --max-runs-per-task
#   WORKER_SERIAL_TASK_IDS      (default 892,1133) — per-task serialization
#   WORKER_MAX_CONCURRENT_CLOSES (default 16)  — pool_server --max-concurrent-closes
#   ENV_SERVER_PORT             (default 18081)
#   STOP_EXISTING_SERVER        (default 0) — stop an older pool server bound
#                                           to ENV_SERVER_PORT before starting
#   SKIP_PREFLIGHT_CLEANUP      (default 0)    — set 1 to skip orphan cleanup
#   PROXY_ENV_FILE              (default /etc/seta_build_proxy.env)
#   SKIP_PROXY_ENV              (default 0)    — set 1 to avoid sourcing proxy env
#   DOCKER_DATA_ROOT            (default /data) — Docker data root to guard
#   WORKER_MIN_DOCKER_FREE_GB   (default 50) — startup warning threshold
#   WORKER_MAX_DOCKER_USED_PCT  (default 95) — startup warning threshold
#   WORKER_MAX_DOCKER_INODE_PCT (default 80) — startup warning threshold
#   WORKER_DISK_GUARD_STRICT    (default 0) — set 1 to abort startup when the
#                                             disk thresholds are still exceeded
#   WORKER_MAX_CONCURRENT_BUILDS (default 2) — cap concurrent docker compose builds
#   WORKER_MAX_CONCURRENT_RESETS (default 16) — cap reset admission before image prep
#   WORKER_DOCKER_BUILD_QUEUE_TIMEOUT (default 90) — fail queued image prep before reset storm
#   WORKER_PRESSURE_GUARD_ENABLED (default 1) — pids/shim/docker-cli admission guard
#   CONTAINER_PIDS_LIMIT        (default 512) — docker update --pids-limit per task container
#
# Logs written by default:
#   deploy/workers/logs/<worker>/<server-run>/cpu_pool.log
#   deploy/workers/logs/<worker>/<server-run>/cpu_err.log
# If RUN_DIR/RUN_ID is not provided, they fall back to deploy/workers/logs/.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"

# Usage: run_pool_server.sh [--restart]
#   --restart  Replace the local SETA pool server in one command: stop the
#              listener on ENV_SERVER_PORT first (default port 18082 in this
#              mode), then start normally. Host-wide docker prune stays out of
#              scope; use deploy/ops/docker_storage_gc.py separately when
#              storage reclamation is needed.
RESTART_MODE=0
for arg in "$@"; do
    case "${arg}" in
        --restart) RESTART_MODE=1 ;;
        *)
            echo "[ERROR] unknown argument: ${arg} (supported: --restart)" >&2
            exit 2
            ;;
    esac
done

if [[ "${RESTART_MODE}" == "1" ]]; then
    ENV_SERVER_PORT="${ENV_SERVER_PORT:-18082}"
    STOP_EXISTING_SERVER=1
    export ENV_SERVER_PORT STOP_EXISTING_SERVER
fi

log() { echo "[$(date +'%F %T')] $*"; }

docker_network_rm_safe() {
    local timeout_s="$1"
    if ! command -v flock >/dev/null 2>&1; then
        log "  WARN: flock is unavailable; skipping unsafe Docker network removal"
        return 0
    fi
    flock -w "${timeout_s}" "${DOCKER_NETWORK_LIFECYCLE_LOCK}" \
        xargs -r -n 20 timeout "${timeout_s}" docker network rm \
        >/dev/null 2>&1 || true
}

detect_docker_data_root() {
    local detected=""
    if [[ -n "${DOCKER_DATA_ROOT:-}" ]]; then
        printf '%s\n' "${DOCKER_DATA_ROOT}"
        return 0
    fi
    if [[ -n "${DOCKER_ROOT:-}" ]]; then
        printf '%s\n' "${DOCKER_ROOT}"
        return 0
    fi
    if command -v docker >/dev/null 2>&1; then
        detected="$(timeout 10 docker info --format '{{.DockerRootDir}}' 2>/dev/null || true)"
        if [[ -n "${detected}" ]]; then
            printf '%s\n' "${detected}"
            return 0
        fi
    fi
    if [[ -f /etc/docker/daemon.json ]] && command -v python3 >/dev/null 2>&1; then
        detected="$(python3 - <<'PY' 2>/dev/null || true
import json
with open("/etc/docker/daemon.json") as f:
    print(json.load(f).get("data-root", ""))
PY
)"
        if [[ -n "${detected}" ]]; then
            printf '%s\n' "${detected}"
            return 0
        fi
    fi
    printf '%s\n' "/var/lib/docker"
}

# ── Configuration ─────────────────────────────────────────────────────────────
# 坑1: capacity must balance rollout demand and Docker isolation limits.
# Most tasks can run in parallel; known compose-unsafe tasks are serialized
# through WORKER_SERIAL_TASK_IDS or explicit WORKER_TASK_MAX_RUNS_OVERRIDES.
WORKER_MAX_TASKS="${WORKER_MAX_TASKS:-16}"
WORKER_MAX_RUNS_PER_TASK="${WORKER_MAX_RUNS_PER_TASK:-8}"
export WORKER_MAX_TOTAL_RUNS="${WORKER_MAX_TOTAL_RUNS:-$((WORKER_MAX_TASKS * WORKER_MAX_RUNS_PER_TASK))}"
export TERMINAL_RL_POOL_NAMESPACE="${TERMINAL_RL_POOL_NAMESPACE:-default}"
if [[ ! "${TERMINAL_RL_POOL_NAMESPACE}" =~ ^[a-z0-9][a-z0-9_-]{0,62}$ ]]; then
    echo "[ERROR] TERMINAL_RL_POOL_NAMESPACE must match ^[a-z0-9][a-z0-9_-]{0,62}$." >&2
    exit 1
fi
# Host-wide `docker system prune` cannot distinguish concurrent terminal-rl
# pools. Keep it opt-in and rely on lease/namespace-owned cleanup by default.
export WORKER_SHIM_CLEANUP_ENABLED="${WORKER_SHIM_CLEANUP_ENABLED:-0}"
WORKER_SERIAL_TASK_IDS="${WORKER_SERIAL_TASK_IDS:-892,1133}"
WORKER_TASK_MAX_RUNS_OVERRIDES="${WORKER_TASK_MAX_RUNS_OVERRIDES:-}"
WORKER_AUTO_SERIALIZE_UNSAFE_COMPOSE="${WORKER_AUTO_SERIALIZE_UNSAFE_COMPOSE:-0}"
# Close/build fan-out also consumes host PIDs under pressure.
WORKER_MAX_CONCURRENT_CLOSES="${WORKER_MAX_CONCURRENT_CLOSES:-16}"
WORKER_MAX_CONCURRENT_RESETS="${WORKER_MAX_CONCURRENT_RESETS:-16}"
WORKER_RESET_ADMISSION_TIMEOUT="${WORKER_RESET_ADMISSION_TIMEOUT:-30}"
WORKER_RESET_BACKLOG_RETRY_AFTER="${WORKER_RESET_BACKLOG_RETRY_AFTER:-10}"
WORKER_RESET_CANCEL_JOIN_TIMEOUT="${WORKER_RESET_CANCEL_JOIN_TIMEOUT:-15}"
WORKER_SHUTDOWN_RESET_JOIN_TIMEOUT="${WORKER_SHUTDOWN_RESET_JOIN_TIMEOUT:-20}"
ENV_SERVER_PORT="${ENV_SERVER_PORT:-18081}"
STOP_EXISTING_SERVER="${STOP_EXISTING_SERVER:-0}"
STOP_EXISTING_TIMEOUT="${STOP_EXISTING_TIMEOUT:-30}"
SKIP_PREFLIGHT_CLEANUP="${SKIP_PREFLIGHT_CLEANUP:-0}"
PREFLIGHT_KILL_ORPHAN_RUNNING="${PREFLIGHT_KILL_ORPHAN_RUNNING:-1}"
FINAL_DOCKER_CLEANUP="${FINAL_DOCKER_CLEANUP:-1}"
FINAL_DOCKER_CLEANUP_TIMEOUT="${FINAL_DOCKER_CLEANUP_TIMEOUT:-90}"
DOCKER_NETWORK_LIFECYCLE_LOCK="${DOCKER_NETWORK_LIFECYCLE_LOCK:-/tmp/lightrl_docker_network_lifecycle.lock}"
POOL_SERVER_SHUTDOWN_GRACE="${POOL_SERVER_SHUTDOWN_GRACE:-60}"
PROXY_ENV_FILE="${PROXY_ENV_FILE:-/etc/seta_build_proxy.env}"
SKIP_PROXY_ENV="${SKIP_PROXY_ENV:-0}"
DOCKER_DATA_ROOT="$(detect_docker_data_root)"
DOCKER_ROOT="${DOCKER_DATA_ROOT}"
WORKER_DISK_GUARD_ENABLED="${WORKER_DISK_GUARD_ENABLED:-1}"
WORKER_DISK_GUARD_STRICT="${WORKER_DISK_GUARD_STRICT:-0}"
WORKER_MIN_DOCKER_FREE_GB="${WORKER_MIN_DOCKER_FREE_GB:-50}"
WORKER_MAX_DOCKER_USED_PCT="${WORKER_MAX_DOCKER_USED_PCT:-95}"
WORKER_MAX_DOCKER_INODE_PCT="${WORKER_MAX_DOCKER_INODE_PCT:-80}"
# Host-wide prune/GC cannot distinguish another pool on the same Docker host.
# Cleanup stays opt-in; the startup disk check warns by default and can be made
# fail-closed with WORKER_DISK_GUARD_STRICT=1.
PREFLIGHT_DISK_CLEANUP="${PREFLIGHT_DISK_CLEANUP:-0}"
PREFLIGHT_DOCKER_STORAGE_GC="${PREFLIGHT_DOCKER_STORAGE_GC:-0}"
DOCKER_GC_TRIGGER_USED_PCT="${DOCKER_GC_TRIGGER_USED_PCT:-${WORKER_MAX_DOCKER_USED_PCT}}"
DOCKER_GC_TARGET_USED_PCT="${DOCKER_GC_TARGET_USED_PCT:-90}"
DOCKER_GC_MIN_FREE_GB="${DOCKER_GC_MIN_FREE_GB:-${WORKER_MIN_DOCKER_FREE_GB}}"
DOCKER_GC_KEEP_PATTERNS="${DOCKER_GC_KEEP_PATTERNS:-ghcr.io/laude-institute/t-bench/*,ubuntu:*,python:*}"
DOCKER_GC_PRUNE_VOLUMES="${DOCKER_GC_PRUNE_VOLUMES:-0}"
DOCKER_GC_DRY_RUN="${DOCKER_GC_DRY_RUN:-0}"
DOCKER_GC_DELETE_OLD_IMAGES="${DOCKER_GC_DELETE_OLD_IMAGES:-0}"
WORKER_MAX_CONCURRENT_BUILDS="${WORKER_MAX_CONCURRENT_BUILDS:-2}"
WORKER_PRESSURE_GUARD_ENABLED="${WORKER_PRESSURE_GUARD_ENABLED:-1}"
WORKER_CLOSE_TASK_TIMEOUT="${WORKER_CLOSE_TASK_TIMEOUT:-45}"
WORKER_CLOSE_QUEUE_TIMEOUT="${WORKER_CLOSE_QUEUE_TIMEOUT:-${WORKER_CLOSE_TASK_TIMEOUT}}"
WORKER_CLOSE_SESSION_TIMEOUT="${WORKER_CLOSE_SESSION_TIMEOUT:-60}"
WORKER_ALLOCATED_TTL="${WORKER_ALLOCATED_TTL:-60}"  # P0 FIX: 120→60 to prevent slot accumulation
ENSURE_IMAGE_TIMEOUT="${ENSURE_IMAGE_TIMEOUT:-1200}"
RESET_SESSION_TIMEOUT="${RESET_SESSION_TIMEOUT:-600}"
WORKER_RESET_OPERATION_TIMEOUT="${WORKER_RESET_OPERATION_TIMEOUT:-1920}"
WORKER_DOCKER_BUILD_QUEUE_TIMEOUT="${WORKER_DOCKER_BUILD_QUEUE_TIMEOUT:-90}"
WORKER_DOCKER_BUILD_BACKLOG_RETRY_AFTER="${WORKER_DOCKER_BUILD_BACKLOG_RETRY_AFTER:-15}"
WORKER_RESET_TIMEOUT_RETRY_AFTER="${WORKER_RESET_TIMEOUT_RETRY_AFTER:-15}"
WORKER_RESETTING_TTL="${WORKER_RESETTING_TTL:-2100}"
WORKER_CLOSING_REQUESTED_TTL="${WORKER_CLOSING_REQUESTED_TTL:-300}"
WORKER_PIDS_PAUSE_ALLOCATE_PCT="${WORKER_PIDS_PAUSE_ALLOCATE_PCT:-60}"
WORKER_PIDS_REJECT_RESET_PCT="${WORKER_PIDS_REJECT_RESET_PCT:-70}"
WORKER_PIDS_MIN_FREE_ALLOCATE="${WORKER_PIDS_MIN_FREE_ALLOCATE:-6000}"
WORKER_PIDS_MIN_FREE_RESET="${WORKER_PIDS_MIN_FREE_RESET:-4000}"
WORKER_SHIM_PAUSE_ALLOCATE="${WORKER_SHIM_PAUSE_ALLOCATE:-160}"  # P1 FIX: align with start_server.sh
WORKER_SHIM_REJECT_RESET="${WORKER_SHIM_REJECT_RESET:-200}"  # P1 FIX: align with start_server.sh
WORKER_PENDING_CLOSES_PAUSE_ALLOCATE="${WORKER_PENDING_CLOSES_PAUSE_ALLOCATE:-32}"
WORKER_PENDING_CLOSES_REJECT_RESET="${WORKER_PENDING_CLOSES_REJECT_RESET:-64}"
WORKER_DOCKER_CLI_TIMEOUT="${WORKER_DOCKER_CLI_TIMEOUT:-3}"
WORKER_DOCKER_DEGRADED_FAIL_STREAK="${WORKER_DOCKER_DEGRADED_FAIL_STREAK:-2}"
WORKER_DOCKER_DEGRADED_COOLDOWN="${WORKER_DOCKER_DEGRADED_COOLDOWN:-120}"
WORKER_PRESSURE_CACHE_TTL="${WORKER_PRESSURE_CACHE_TTL:-5}"
WORKER_RESET_STORM_GUARD="${WORKER_RESET_STORM_GUARD:-1}"
WORKER_RESET_STORM_BLOCK_ALLOCATE="${WORKER_RESET_STORM_BLOCK_ALLOCATE:-1}"
WORKER_RESET_STORM_MIN_RESETTING="${WORKER_RESET_STORM_MIN_RESETTING:-32}"
WORKER_RESET_STORM_MIN_AGE="${WORKER_RESET_STORM_MIN_AGE:-180}"
WORKER_RESET_STORM_RATIO_PCT="${WORKER_RESET_STORM_RATIO_PCT:-50}"
TERMINAL_ENV_FORCE_DOCKER_CLEANUP="${TERMINAL_ENV_FORCE_DOCKER_CLEANUP:-1}"
TERMINAL_ENV_FORCE_DOCKER_CLEANUP_BROAD="${TERMINAL_ENV_FORCE_DOCKER_CLEANUP_BROAD:-1}"
TERMINAL_ENV_FORCE_DOCKER_CLEANUP_ALWAYS="${TERMINAL_ENV_FORCE_DOCKER_CLEANUP_ALWAYS:-1}"
TERMINAL_ENV_FORCE_DOCKER_CLEANUP_TIMEOUT="${TERMINAL_ENV_FORCE_DOCKER_CLEANUP_TIMEOUT:-20}"
TERMINAL_ENV_DOCKER_CLEANUP_WORKERS="${TERMINAL_ENV_DOCKER_CLEANUP_WORKERS:-8}"
WORKER_SHUTDOWN_FORCE_CLEANUP_TIMEOUT="${WORKER_SHUTDOWN_FORCE_CLEANUP_TIMEOUT:-120}"
TERMINAL_ENV_FAST_CLOSE="${TERMINAL_ENV_FAST_CLOSE:-1}"
TERMINAL_ENV_SKIP_UNBOUNDED_STOP="${TERMINAL_ENV_SKIP_UNBOUNDED_STOP:-1}"
TERMINAL_ENV_FAST_CLOSE_STOP_TIMEOUT="${TERMINAL_ENV_FAST_CLOSE_STOP_TIMEOUT:-5}"
WORKER_REPAIR_PENDING_CLOSES="${WORKER_REPAIR_PENDING_CLOSES:-1}"
WORKER_REPAIR_PENDING_CLOSES_MAX_ACTIVE_RUNS="${WORKER_REPAIR_PENDING_CLOSES_MAX_ACTIVE_RUNS:--1}"
WORKER_REPAIR_PENDING_CLOSES_CANCEL_TIMEOUT="${WORKER_REPAIR_PENDING_CLOSES_CANCEL_TIMEOUT:-5}"
WORKER_REPAIR_PENDING_CLOSES_MIN_AGE="${WORKER_REPAIR_PENDING_CLOSES_MIN_AGE:-45}"
WORKER_REPAIR_STALE_RUNS="${WORKER_REPAIR_STALE_RUNS:-1}"
WORKER_REPAIR_STALE_RUNS_MIN_AGE="${WORKER_REPAIR_STALE_RUNS_MIN_AGE:-0}"
WORKER_REPAIR_STALE_RUNS_MAX_REPAIRS="${WORKER_REPAIR_STALE_RUNS_MAX_REPAIRS:-20}"
WORKER_REPAIR_CLOSE_REQUESTED_RUNS="${WORKER_REPAIR_CLOSE_REQUESTED_RUNS:-1}"
WORKER_REPAIR_CLOSE_REQUESTED_MIN_AGE="${WORKER_REPAIR_CLOSE_REQUESTED_MIN_AGE:-0}"
WORKER_REPAIR_CLOSE_REQUESTED_MAX_REPAIRS="${WORKER_REPAIR_CLOSE_REQUESTED_MAX_REPAIRS:-20}"
WORKER_CLOSE_REQUESTED_FORCE_RELEASE="${WORKER_CLOSE_REQUESTED_FORCE_RELEASE:-1}"
WORKER_CLOSE_REQUESTED_FORCE_RELEASE_AFTER="${WORKER_CLOSE_REQUESTED_FORCE_RELEASE_AFTER:-120}"
WORKER_ORPHAN_DOCKER_SWEEP="${WORKER_ORPHAN_DOCKER_SWEEP:-1}"
WORKER_ORPHAN_DOCKER_SWEEP_INTERVAL="${WORKER_ORPHAN_DOCKER_SWEEP_INTERVAL:-180}"
WORKER_ORPHAN_DOCKER_SWEEP_MIN_AGE="${WORKER_ORPHAN_DOCKER_SWEEP_MIN_AGE:-600}"
WORKER_ORPHAN_DOCKER_SWEEP_MAX_REMOVE="${WORKER_ORPHAN_DOCKER_SWEEP_MAX_REMOVE:-64}"
WORKER_ORPHAN_DOCKER_SWEEP_TIMEOUT="${WORKER_ORPHAN_DOCKER_SWEEP_TIMEOUT:-15}"
WORKER_ORPHAN_DOCKER_SWEEP_BACKOFF_BASE="${WORKER_ORPHAN_DOCKER_SWEEP_BACKOFF_BASE:-120}"
WORKER_ORPHAN_DOCKER_SWEEP_BACKOFF_MAX="${WORKER_ORPHAN_DOCKER_SWEEP_BACKOFF_MAX:-900}"
WORKER_AUTO_REPAIR_ON_CAPACITY="${WORKER_AUTO_REPAIR_ON_CAPACITY:-1}"
WORKER_AUTO_REPAIR_CLOSE_REQUESTED_MIN_AGE="${WORKER_AUTO_REPAIR_CLOSE_REQUESTED_MIN_AGE:-0}"
WORKER_AUTO_REPAIR_STALE_MIN_AGE="${WORKER_AUTO_REPAIR_STALE_MIN_AGE:-0}"
WORKER_AUTO_REPAIR_MAX_REPAIRS="${WORKER_AUTO_REPAIR_MAX_REPAIRS:-40}"  # P1 FIX: 20→40 for aggressive cleanup
WORKER_REPAIR_RESETTING_MIN_AGE="${WORKER_REPAIR_RESETTING_MIN_AGE:-2100}"  # match WORKER_RESETTING_TTL
WORKER_DOCKER_BUILD_DEDUP="${WORKER_DOCKER_BUILD_DEDUP:-1}"
WORKER_DOCKER_BUILD_SKIP_EXISTING="${WORKER_DOCKER_BUILD_SKIP_EXISTING:-1}"
WORKER_DOCKER_BUILD_FAILED_TTL="${WORKER_DOCKER_BUILD_FAILED_TTL:-3600}"
WORKER_DOCKER_BUILD_TRANSIENT_TTL="${WORKER_DOCKER_BUILD_TRANSIENT_TTL:-60}"
WORKER_DOCKER_BUILD_TRANSIENT_RETRY_AFTER="${WORKER_DOCKER_BUILD_TRANSIENT_RETRY_AFTER:-60}"
WORKER_DOCKER_TASK_BLACKLIST_TTL="${WORKER_DOCKER_TASK_BLACKLIST_TTL:-86400}"
WORKER_DOCKERFILE_PRECHECK="${WORKER_DOCKERFILE_PRECHECK:-1}"
WORKER_TASK_IMAGE_RETRY_AFTER="${WORKER_TASK_IMAGE_RETRY_AFTER:-300}"
CONTAINER_PIDS_LIMIT="${CONTAINER_PIDS_LIMIT:-512}"
CONTAINER_MEMORY_LIMIT="${CONTAINER_MEMORY_LIMIT:-16g}"
if [[ "${WORKER_DISK_GUARD_STRICT}" != "0" && "${WORKER_DISK_GUARD_STRICT}" != "1" ]]; then
    echo "[ERROR] WORKER_DISK_GUARD_STRICT must be 0 or 1." >&2
    exit 1
fi
if [[ ! "${CONTAINER_PIDS_LIMIT}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] CONTAINER_PIDS_LIMIT must be a non-negative integer." >&2
    exit 1
fi
visible_cpu_count="$(nproc 2>/dev/null || echo 1)"
if (( CONTAINER_PIDS_LIMIT > 0 && CONTAINER_PIDS_LIMIT <= visible_cpu_count )); then
    echo "[ERROR] CONTAINER_PIDS_LIMIT=${CONTAINER_PIDS_LIMIT} must exceed visible CPUs=${visible_cpu_count}; SETA uv graders create a CPU-sized thread pool." >&2
    exit 1
fi
CPU_POOL_LOG_MAX_BYTES="${CPU_POOL_LOG_MAX_BYTES:-209715200}"
CPU_POOL_LOG_TAIL_BYTES="${CPU_POOL_LOG_TAIL_BYTES:-52428800}"
CPU_ERR_SCAN_LINES="${CPU_ERR_SCAN_LINES:-5000}"
POOL_SERVER_SUPERVISE="${POOL_SERVER_SUPERVISE:-1}"
POOL_SERVER_MAX_RESTARTS="${POOL_SERVER_MAX_RESTARTS:-20}"
POOL_SERVER_RESTART_BACKOFF_INITIAL="${POOL_SERVER_RESTART_BACKOFF_INITIAL:-5}"
POOL_SERVER_RESTART_BACKOFF_MAX="${POOL_SERVER_RESTART_BACKOFF_MAX:-300}"
POOL_SERVER_RESTART_RESET_WINDOW="${POOL_SERVER_RESTART_RESET_WINDOW:-1800}"
POOL_SERVER_CHILD_EXIT_CLEANUP="${POOL_SERVER_CHILD_EXIT_CLEANUP:-1}"

# ── Log paths ─────────────────────────────────────────────────────────────────
RUNS_ROOT="${RUNS_ROOT:-${REPO_ROOT}/runs}"
if [[ -n "${RUN_DIR:-}" ]]; then
    DEFAULT_REMOTE_LOG_ROOT="${RUN_DIR}/remote_logs"
elif [[ -n "${RUN_ID:-}" ]]; then
    DEFAULT_REMOTE_LOG_ROOT="${RUNS_ROOT}/${RUN_ID}/remote_logs"
else
    DEFAULT_REMOTE_LOG_ROOT="${REPO_ROOT}/deploy/workers/logs"
fi
REMOTE_LOG_ROOT="${REMOTE_LOG_ROOT:-${DEFAULT_REMOTE_LOG_ROOT}}"
CPU_WORKER_ID="${CPU_WORKER_ID:-$(hostname -f 2>/dev/null || hostname 2>/dev/null || echo unknown-worker)}"
CPU_WORKER_ID="$(printf '%s' "${CPU_WORKER_ID}" | tr -c 'A-Za-z0-9_.-' '_')"
OPENCLAW_REMOTE_RUN_ID="${OPENCLAW_REMOTE_RUN_ID:-$(date +%Y%m%d_%H%M%S)_pid$$}"
OPENCLAW_REMOTE_LOG_DIR="${OPENCLAW_REMOTE_LOG_DIR:-${REMOTE_LOG_ROOT}/${CPU_WORKER_ID}/${OPENCLAW_REMOTE_RUN_ID}}"
CPU_POOL_LOG="${CPU_POOL_LOG:-${OPENCLAW_REMOTE_LOG_DIR}/cpu_pool.log}"
CPU_ERR_LOG="${CPU_ERR_LOG:-${OPENCLAW_REMOTE_LOG_DIR}/cpu_err.log}"
export RUNS_ROOT REMOTE_LOG_ROOT CPU_WORKER_ID OPENCLAW_REMOTE_RUN_ID OPENCLAW_REMOTE_LOG_DIR
mkdir -p "${OPENCLAW_REMOTE_LOG_DIR}" "$(dirname "${CPU_POOL_LOG}")" "$(dirname "${CPU_ERR_LOG}")" "${REMOTE_LOG_ROOT}/${CPU_WORKER_ID}"
ln -sfnT "${OPENCLAW_REMOTE_LOG_DIR}" "${REMOTE_LOG_ROOT}/${CPU_WORKER_ID}/latest_server" 2>/dev/null || true

rotate_file_in_place() {
    local file="$1"
    local max_bytes="$2"
    local tail_bytes="$3"
    local size tmp
    [ -f "${file}" ] || return 0
    size=$(stat -c%s "${file}" 2>/dev/null || echo 0)
    [ -n "${size}" ] && [ "${size}" -ge 0 ] 2>/dev/null || size=0
    [ "${size}" -gt "${max_bytes}" ] || return 0
    tmp="$(mktemp "${OPENCLAW_REMOTE_LOG_DIR}/rotate.XXXXXX")"
    tail -c "${tail_bytes}" "${file}" > "${tmp}" 2>/dev/null || true
    : > "${file}"
    cat "${tmp}" >> "${file}" 2>/dev/null || true
    rm -f "${tmp}" 2>/dev/null || true
    log "  rotated ${file}: kept last ${tail_bytes} bytes from ${size} bytes"
}

rotate_file_in_place "${CPU_POOL_LOG}" "${CPU_POOL_LOG_MAX_BYTES}" "${CPU_POOL_LOG_TAIL_BYTES}"
exec > >(tee -a "${CPU_POOL_LOG}") 2>&1

log "=== pool_server starting ==="
log "  worker id: ${CPU_WORKER_ID}"
log "  run id:    ${OPENCLAW_REMOTE_RUN_ID}"
log "  log dir:   ${OPENCLAW_REMOTE_LOG_DIR}"
log "  full log: ${CPU_POOL_LOG}"
log "  err log:  ${CPU_ERR_LOG}"
log "  max_tasks=${WORKER_MAX_TASKS}  max_runs_per_task=${WORKER_MAX_RUNS_PER_TASK}"
log "  serial_task_ids=${WORKER_SERIAL_TASK_IDS} task_run_overrides=${WORKER_TASK_MAX_RUNS_OVERRIDES:-<none>} auto_serial_compose=${WORKER_AUTO_SERIALIZE_UNSAFE_COMPOSE}"
log "  max_concurrent_closes=${WORKER_MAX_CONCURRENT_CLOSES}"
log "  max_concurrent_builds=${WORKER_MAX_CONCURRENT_BUILDS}"
log "  max_concurrent_resets=${WORKER_MAX_CONCURRENT_RESETS} reset_admission_timeout=${WORKER_RESET_ADMISSION_TIMEOUT}s retry_after=${WORKER_RESET_BACKLOG_RETRY_AFTER}s cancel_join=${WORKER_RESET_CANCEL_JOIN_TIMEOUT}s shutdown_join=${WORKER_SHUTDOWN_RESET_JOIN_TIMEOUT}s"
log "  build_queue_timeout=${WORKER_DOCKER_BUILD_QUEUE_TIMEOUT}s retry_after=${WORKER_DOCKER_BUILD_BACKLOG_RETRY_AFTER}s"
log "  close_timeout queue=${WORKER_CLOSE_QUEUE_TIMEOUT}s session=${WORKER_CLOSE_SESSION_TIMEOUT}s legacy=${WORKER_CLOSE_TASK_TIMEOUT}s"
log "  port=${ENV_SERVER_PORT}  skip_cleanup=${SKIP_PREFLIGHT_CLEANUP}"
log "  preflight_kill_orphan_running=${PREFLIGHT_KILL_ORPHAN_RUNNING} final_docker_cleanup=${FINAL_DOCKER_CLEANUP}"
log "  potential_capacity=$((WORKER_MAX_TASKS * WORKER_MAX_RUNS_PER_TASK)) leases effective_max_total=${WORKER_MAX_TOTAL_RUNS}"
log "  docker_data_root=${DOCKER_DATA_ROOT} disk_guard=${WORKER_DISK_GUARD_ENABLED} strict=${WORKER_DISK_GUARD_STRICT}"
log "  pressure_guard=${WORKER_PRESSURE_GUARD_ENABLED} pids_pause=${WORKER_PIDS_PAUSE_ALLOCATE_PCT}% pids_reset=${WORKER_PIDS_REJECT_RESET_PCT}% pids_free_allocate=${WORKER_PIDS_MIN_FREE_ALLOCATE} pids_free_reset=${WORKER_PIDS_MIN_FREE_RESET}"
log "  pressure_guard shim_pause=${WORKER_SHIM_PAUSE_ALLOCATE} shim_reset=${WORKER_SHIM_REJECT_RESET} pending_pause=${WORKER_PENDING_CLOSES_PAUSE_ALLOCATE} pending_reset=${WORKER_PENDING_CLOSES_REJECT_RESET}"
log "  container_limits pids=${CONTAINER_PIDS_LIMIT} memory=${CONTAINER_MEMORY_LIMIT}"
log "  stale_ttl allocated=${WORKER_ALLOCATED_TTL}s resetting=${WORKER_RESETTING_TTL}s closing_requested=${WORKER_CLOSING_REQUESTED_TTL}s reset_operation_timeout=${WORKER_RESET_OPERATION_TIMEOUT}s"
log "  reset_phase_timeouts ensure_image=${ENSURE_IMAGE_TIMEOUT}s reset_session=${RESET_SESSION_TIMEOUT}s"
log "  force_cleanup=${TERMINAL_ENV_FORCE_DOCKER_CLEANUP} broad=${TERMINAL_ENV_FORCE_DOCKER_CLEANUP_BROAD} always=${TERMINAL_ENV_FORCE_DOCKER_CLEANUP_ALWAYS} timeout=${TERMINAL_ENV_FORCE_DOCKER_CLEANUP_TIMEOUT}s workers=${TERMINAL_ENV_DOCKER_CLEANUP_WORKERS} batch_timeout=${WORKER_SHUTDOWN_FORCE_CLEANUP_TIMEOUT}s"
log "  fast_close=${TERMINAL_ENV_FAST_CLOSE} skip_unbounded_stop=${TERMINAL_ENV_SKIP_UNBOUNDED_STOP} stop_timeout=${TERMINAL_ENV_FAST_CLOSE_STOP_TIMEOUT}s"
log "  pending_close_repair=${WORKER_REPAIR_PENDING_CLOSES} max_active=${WORKER_REPAIR_PENDING_CLOSES_MAX_ACTIVE_RUNS} cancel_timeout=${WORKER_REPAIR_PENDING_CLOSES_CANCEL_TIMEOUT}s min_age=${WORKER_REPAIR_PENDING_CLOSES_MIN_AGE}s"
log "  stale_run_repair=${WORKER_REPAIR_STALE_RUNS} min_age=${WORKER_REPAIR_STALE_RUNS_MIN_AGE}s max_repairs=${WORKER_REPAIR_STALE_RUNS_MAX_REPAIRS}"
log "  close_requested_repair=${WORKER_REPAIR_CLOSE_REQUESTED_RUNS} min_age=${WORKER_REPAIR_CLOSE_REQUESTED_MIN_AGE}s max_repairs=${WORKER_REPAIR_CLOSE_REQUESTED_MAX_REPAIRS} force_release=${WORKER_CLOSE_REQUESTED_FORCE_RELEASE} after=${WORKER_CLOSE_REQUESTED_FORCE_RELEASE_AFTER}s"
log "  orphan_sweep=${WORKER_ORPHAN_DOCKER_SWEEP} interval=${WORKER_ORPHAN_DOCKER_SWEEP_INTERVAL}s min_age=${WORKER_ORPHAN_DOCKER_SWEEP_MIN_AGE}s max_remove=${WORKER_ORPHAN_DOCKER_SWEEP_MAX_REMOVE} timeout=${WORKER_ORPHAN_DOCKER_SWEEP_TIMEOUT}s"
log "  docker_degraded fail_streak=${WORKER_DOCKER_DEGRADED_FAIL_STREAK} cooldown=${WORKER_DOCKER_DEGRADED_COOLDOWN}s reset_storm=${WORKER_RESET_STORM_GUARD} min=${WORKER_RESET_STORM_MIN_RESETTING} age=${WORKER_RESET_STORM_MIN_AGE}s ratio=${WORKER_RESET_STORM_RATIO_PCT}%"
log "  auto_repair_on_capacity=${WORKER_AUTO_REPAIR_ON_CAPACITY} close_min_age=${WORKER_AUTO_REPAIR_CLOSE_REQUESTED_MIN_AGE}s stale_min_age=${WORKER_AUTO_REPAIR_STALE_MIN_AGE}s max_repairs=${WORKER_AUTO_REPAIR_MAX_REPAIRS}"
log "  docker_build_dedup=${WORKER_DOCKER_BUILD_DEDUP} skip_existing=${WORKER_DOCKER_BUILD_SKIP_EXISTING} failed_ttl=${WORKER_DOCKER_BUILD_FAILED_TTL}s transient_ttl=${WORKER_DOCKER_BUILD_TRANSIENT_TTL}s transient_retry=${WORKER_DOCKER_BUILD_TRANSIENT_RETRY_AFTER}s blacklist_ttl=${WORKER_DOCKER_TASK_BLACKLIST_TTL}s precheck=${WORKER_DOCKERFILE_PRECHECK}"
log "  supervise=${POOL_SERVER_SUPERVISE} max_restarts=${POOL_SERVER_MAX_RESTARTS} backoff=${POOL_SERVER_RESTART_BACKOFF_INITIAL}-${POOL_SERVER_RESTART_BACKOFF_MAX}s reset_window=${POOL_SERVER_RESTART_RESET_WINDOW}s child_exit_cleanup=${POOL_SERVER_CHILD_EXIT_CLEANUP}"

if [[ "${SKIP_PROXY_ENV}" != "1" && -f "${PROXY_ENV_FILE}" ]]; then
    # shellcheck disable=SC1090
    set -a; . "${PROXY_ENV_FILE}"; set +a
    log "  loaded proxy env: ${PROXY_ENV_FILE}"
elif [[ "${SKIP_PROXY_ENV}" != "1" ]]; then
    log "  proxy env not found at ${PROXY_ENV_FILE}; continuing without it"
fi

docker_disk_snapshot() {
    df -P -BG "${DOCKER_DATA_ROOT}" 2>/dev/null | awk 'NR==2 {gsub("%","",$5); gsub("G","",$4); print $5, $4}'
}

docker_inode_snapshot() {
    df -Pi "${DOCKER_DATA_ROOT}" 2>/dev/null | awk 'NR==2 {gsub("%","",$5); print $5}'
}

TASK_CONTAINER_REGEX="${TASK_CONTAINER_REGEX:-^[0-9]+[-_].*(slime[-_]?run|client|helper).*$}"
TASK_IMAGE_REGEX="${TASK_IMAGE_REGEX:-^tb__[0-9]+__.*(:|$)}"
if [[ -z "${WORKER_CLEANUP_LEGACY_UNLABELED+x}" ]]; then
    if [[ "${TERMINAL_RL_POOL_NAMESPACE}" == "default" ]]; then
        # Existing SETA task Compose files predate pool labels.
        WORKER_CLEANUP_LEGACY_UNLABELED=1
    else
        WORKER_CLEANUP_LEGACY_UNLABELED=0
    fi
fi
if [[ "${WORKER_CLEANUP_LEGACY_UNLABELED}" != "0" && "${WORKER_CLEANUP_LEGACY_UNLABELED}" != "1" ]]; then
    log "❌ WORKER_CLEANUP_LEGACY_UNLABELED must be 0 or 1"
    exit 1
fi

task_container_lines() {
    docker ps --format '{{.ID}}\t{{.Names}}\t{{.Image}}\t{{.Label "harborrl.pool-namespace"}}' 2>/dev/null \
        | awk -F '\t' -v ns="${TERMINAL_RL_POOL_NAMESPACE}" \
            -v legacy="${WORKER_CLEANUP_LEGACY_UNLABELED}" \
            -v name_re="${TASK_CONTAINER_REGEX}" -v image_re="${TASK_IMAGE_REGEX}" \
            '((ns == "default" && ($2 ~ name_re || $3 ~ image_re) && ($4 == "default" || (legacy == "1" && $4 == ""))) || (ns != "default" && $4 == ns)) {print $0}' || true
}

task_container_ids() {
    task_container_lines | awk -F '\t' 'NF >= 1 {print $1}' | sed '/^$/d' || true
}

stopped_pool_container_ids() {
    docker ps -a --filter "status=exited" --filter "status=dead" \
        --format '{{.ID}}\t{{.Names}}\t{{.Image}}\t{{.Label "harborrl.pool-namespace"}}' 2>/dev/null \
        | awk -F '\t' -v ns="${TERMINAL_RL_POOL_NAMESPACE}" \
            -v legacy="${WORKER_CLEANUP_LEGACY_UNLABELED}" \
            -v name_re="${TASK_CONTAINER_REGEX}" -v image_re="${TASK_IMAGE_REGEX}" \
            '((ns == "default" && ($2 ~ name_re || $3 ~ image_re) && ($4 == "default" || (legacy == "1" && $4 == ""))) || (ns != "default" && $4 == ns)) {print $1}' \
        | sed '/^$/d' || true
}

dangling_pool_network_ids() {
    docker network ls --filter "dangling=true" \
        --format '{{.ID}}\t{{.Label "harborrl.pool-namespace"}}\t{{.Label "com.docker.compose.project"}}' 2>/dev/null \
        | awk -F '\t' -v ns="${TERMINAL_RL_POOL_NAMESPACE}" \
            -v legacy="${WORKER_CLEANUP_LEGACY_UNLABELED}" \
            -v project_re="${TASK_CONTAINER_REGEX}" \
            '((ns == "default" && ($2 == "default" || (legacy == "1" && $2 == "")) && $3 ~ project_re) || (ns != "default" && $2 == ns)) {print $1}' \
        | sed '/^$/d' || true
}

cleanup_stopped_pool_objects() {
    local reason="$1"
    local ids count

    ids="$(stopped_pool_container_ids)"
    count=$(printf '%s\n' "${ids}" | sed '/^$/d' | wc -l || true)
    log "  Docker cleanup (${reason}): owned stopped containers=${count:-0} namespace=${TERMINAL_RL_POOL_NAMESPACE}"
    if [[ "${count:-0}" -gt 0 ]] 2>/dev/null; then
        printf '%s\n' "${ids}" \
            | xargs -r -n 20 timeout "${FINAL_DOCKER_CLEANUP_TIMEOUT}" docker rm -f >/dev/null 2>&1 || true
    fi

    ids="$(dangling_pool_network_ids)"
    count=$(printf '%s\n' "${ids}" | sed '/^$/d' | wc -l || true)
    log "  Docker cleanup (${reason}): owned dangling networks=${count:-0} namespace=${TERMINAL_RL_POOL_NAMESPACE}"
    if [[ "${count:-0}" -gt 0 ]] 2>/dev/null; then
        printf '%s\n' "${ids}" \
            | docker_network_rm_safe "${FINAL_DOCKER_CLEANUP_TIMEOUT}"
    fi
}

cleanup_task_docker_objects() {
    local reason="$1"
    local ids count

    cleanup_stopped_pool_objects "${reason}"

    ids="$(task_container_ids)"
    count=$(printf '%s\n' "${ids}" | sed '/^$/d' | wc -l || true)
    log "  Docker cleanup (${reason}): running task containers=${count:-0}"
    if [[ "${count:-0}" -gt 0 ]] 2>/dev/null; then
        printf '%s\n' "${ids}" \
            | xargs -r -n 20 timeout "${FINAL_DOCKER_CLEANUP_TIMEOUT}" docker rm -f >/dev/null 2>&1 || true
        log "  Docker cleanup (${reason}): removed matching running task containers"
    fi

}

preflight_disk_guard() {
    if [[ "${WORKER_DISK_GUARD_ENABLED}" == "0" ]]; then
        log "  disk guard disabled (WORKER_DISK_GUARD_ENABLED=0)"
        return 0
    fi
    if [[ ! -d "${DOCKER_DATA_ROOT}" ]]; then
        log "  ❌ Docker data root does not exist: ${DOCKER_DATA_ROOT}"
        exit 1
    fi

    local snap used_pct free_gb inode_pct
    snap="$(docker_disk_snapshot || true)"
    inode_pct="$(docker_inode_snapshot || true)"
    used_pct="${snap%% *}"
    free_gb="${snap##* }"
    log "  ${DOCKER_DATA_ROOT}: used=${used_pct:-?}% free=${free_gb:-?}GB inode=${inode_pct:-?}%"
    log "  thresholds: free>=${WORKER_MIN_DOCKER_FREE_GB}GB used<=${WORKER_MAX_DOCKER_USED_PCT}% inode<=${WORKER_MAX_DOCKER_INODE_PCT}%"

    if [[ -z "${used_pct}" || -z "${free_gb}" || -z "${inode_pct}" ]]; then
        log "  ❌ Failed to read Docker data-root disk stats"
        exit 1
    fi

    if [[ "${used_pct}" -gt "${WORKER_MAX_DOCKER_USED_PCT}" \
       || "${free_gb}" -lt "${WORKER_MIN_DOCKER_FREE_GB}" \
       || "${inode_pct}" -gt "${WORKER_MAX_DOCKER_INODE_PCT}" ]]; then
        log "  ⚠️  Docker data-root is above guard threshold."
        if [[ "${PREFLIGHT_DISK_CLEANUP}" == "1" && -x "${SCRIPT_DIR}/../ops/cleanup_docker.sh" ]]; then
            log "  Running conservative cleanup before refusing start..."
            DOCKER_DATA_ROOT="${DOCKER_DATA_ROOT}" RUN_HEAVY_DF=0 \
              bash "${SCRIPT_DIR}/../ops/cleanup_docker.sh" || true
            snap="$(docker_disk_snapshot || true)"
            inode_pct="$(docker_inode_snapshot || true)"
            used_pct="${snap%% *}"
            free_gb="${snap##* }"
            log "  after cleanup: used=${used_pct:-?}% free=${free_gb:-?}GB inode=${inode_pct:-?}%"
        fi
        if [[ "${PREFLIGHT_DOCKER_STORAGE_GC}" == "1" && -f "${SCRIPT_DIR}/../ops/docker_storage_gc.py" ]]; then
            log "  Running Docker storage LRU GC before refusing start..."
            DOCKER_DATA_ROOT="${DOCKER_DATA_ROOT}" \
            DOCKER_GC_TRIGGER_USED_PCT="${DOCKER_GC_TRIGGER_USED_PCT}" \
            DOCKER_GC_TARGET_USED_PCT="${DOCKER_GC_TARGET_USED_PCT}" \
            DOCKER_GC_MIN_FREE_GB="${DOCKER_GC_MIN_FREE_GB}" \
            DOCKER_GC_KEEP_PATTERNS="${DOCKER_GC_KEEP_PATTERNS}" \
            DOCKER_GC_PRUNE_VOLUMES="${DOCKER_GC_PRUNE_VOLUMES}" \
            DOCKER_GC_DRY_RUN="${DOCKER_GC_DRY_RUN}" \
            DOCKER_GC_DELETE_OLD_IMAGES="${DOCKER_GC_DELETE_OLD_IMAGES}" \
              python3 "${SCRIPT_DIR}/../ops/docker_storage_gc.py" || true
            snap="$(docker_disk_snapshot || true)"
            inode_pct="$(docker_inode_snapshot || true)"
            used_pct="${snap%% *}"
            free_gb="${snap##* }"
            log "  after LRU GC: used=${used_pct:-?}% free=${free_gb:-?}GB inode=${inode_pct:-?}%"
        fi
    fi

    if [[ "${used_pct}" -gt "${WORKER_MAX_DOCKER_USED_PCT}" \
       || "${free_gb}" -lt "${WORKER_MIN_DOCKER_FREE_GB}" \
       || "${inode_pct}" -gt "${WORKER_MAX_DOCKER_INODE_PCT}" ]]; then
        if [[ "${WORKER_DISK_GUARD_STRICT}" == "1" ]]; then
            log "  ❌ Refusing to start pool_server under Docker disk pressure (strict mode)."
            log "     Preview: DOCKER_GC_DRY_RUN=1 python3 deploy/ops/docker_storage_gc.py"
            log "     Run:     python3 deploy/ops/docker_storage_gc.py"
            log "     Legacy:  AGGRESSIVE=1 PRUNE_VOLUMES=1 bash deploy/ops/fix_docker_overlay2_no_space.sh"
            log "     If Docker objects are empty but /data is still full, use PURGE_DOCKER_ROOT_WHEN_EMPTY=1."
            exit 1
        fi
        # Starting the HTTP pool does not itself allocate Docker layers.  Keep
        # the service available on small/full local partitions and let each
        # task report a useful Docker error if an image build cannot proceed.
        # Operators that require fail-closed admission can set STRICT=1.
        log "  ⚠️  Docker data-root remains above guard threshold; continuing in warning mode."
        log "     Docker image builds may fail until storage is reclaimed."
        log "     Set WORKER_DISK_GUARD_STRICT=1 to refuse startup instead."
        return 0
    fi

    log "  ✅ Docker data-root capacity OK"
}

# ── Pre-flight: docker daemon health ─────────────────────────────────────────
log "Pre-flight [1/6]: Docker daemon health check"
if ! timeout 10 docker info >/dev/null 2>&1; then
    log "  ❌ Docker daemon not responding!"
    log "  Run repair: sudo bash deploy/ops/fix_dockerd_and_proxy.sh"
    log "  Or force restart only: sudo bash deploy/ops/restart_docker_force.sh"
    exit 1
fi
log "  ✅ Docker daemon OK"

stop_existing_server() {
    local -a listener_pids=() owners=()
    local pid parent cmd owner deadline
    mapfile -t listener_pids < <(
        ss -ltnpH "( sport = :${ENV_SERVER_PORT} )" 2>/dev/null \
          | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u
    )
    ((${#listener_pids[@]})) || return 0

    # The listening Python process is supervised by run_pool_server.sh.  Stop
    # that shell (and let its trap clean the child) rather than killing only
    # Python, which would otherwise be immediately restarted by the supervisor.
    for pid in "${listener_pids[@]}"; do
        owner="${pid}"
        while parent="$(ps -o ppid= -p "${owner}" 2>/dev/null | tr -d ' ')"; do
            [[ -n "${parent}" && "${parent}" != "0" ]] || break
            cmd="$(ps -o args= -p "${parent}" 2>/dev/null || true)"
            if [[ "${cmd}" == *run_pool_server.sh* ]]; then
                owner="${parent}"
            else
                break
            fi
        done
        owners+=("${owner}")
    done

    mapfile -t owners < <(printf '%s\n' "${owners[@]}" | sort -u)
    log "  stopping existing pool server on port ${ENV_SERVER_PORT}: PIDs ${owners[*]}"
    for pid in "${owners[@]}"; do
        kill -TERM "${pid}" 2>/dev/null || true
    done
    deadline=$(( $(date +%s) + STOP_EXISTING_TIMEOUT ))
    while (( $(date +%s) < deadline )); do
        if ! ss -ltnH "( sport = :${ENV_SERVER_PORT} )" 2>/dev/null | grep -q ":${ENV_SERVER_PORT}"; then
            log "  ✅ old pool server stopped"
            return 0
        fi
        sleep 1
    done
    log "  ⚠️  old pool server did not stop within ${STOP_EXISTING_TIMEOUT}s; sending KILL"
    for pid in "${owners[@]}"; do
        kill -KILL "${pid}" 2>/dev/null || true
    done
    sleep 1
    ss -ltnH "( sport = :${ENV_SERVER_PORT} )" 2>/dev/null \
      | grep -q ":${ENV_SERVER_PORT}" \
      && { log "  ❌ port ${ENV_SERVER_PORT} remains occupied"; return 1; }
}

log "Pre-flight [2/6]: Docker data-root disk/inode guard"
preflight_disk_guard

if command -v ss >/dev/null 2>&1 && ss -tln "( sport = :${ENV_SERVER_PORT} )" | grep -q ":${ENV_SERVER_PORT}"; then
    if [[ "${STOP_EXISTING_SERVER}" == "1" ]]; then
        stop_existing_server || exit 1
    else
        log "  ❌ Port ${ENV_SERVER_PORT} is already in use"
        log "     Inspect: ss -tlnp '( sport = :${ENV_SERVER_PORT} )'"
        log "     To replace it safely, set STOP_EXISTING_SERVER=1"
        exit 1
    fi
fi

# ── Pre-flight: nofile ulimit check (坑4) ────────────────────────────────────
log "Pre-flight [3/6]: nofile ulimit check (need ≥65536)"
NOFILE_SOFT=$(ulimit -Sn 2>/dev/null || echo 0)
NOFILE_HARD=$(ulimit -Hn 2>/dev/null || echo 0)
log "  current: soft=${NOFILE_SOFT} hard=${NOFILE_HARD}"
if [[ "${NOFILE_SOFT}" -lt 65536 ]]; then
    log "  ⚠️  soft limit ${NOFILE_SOFT} < 65536, attempting to raise..."
    # Try to raise soft limit inline
    if ulimit -Sn 65536 2>/dev/null; then
        NOFILE_SOFT=$(ulimit -Sn)
        log "  ✅ Raised soft limit to ${NOFILE_SOFT}"
    else
        log "  ⚠️  Could not raise via ulimit (may need /etc/security/limits.conf or systemd override)"
        log "     Continuing anyway, but evaluate may fail at ≥32 concurrent tasks"
    fi
else
    log "  ✅ nofile soft limit OK (${NOFILE_SOFT})"
fi

# ── Pre-flight: docker address pool (坑3) ────────────────────────────────────
log "Pre-flight [4/6]: Docker bridge network address pool check"
# Count existing bridge networks (each consumes a /24)
BRIDGE_COUNT=$(docker network ls --filter driver=bridge -q 2>/dev/null | wc -l)
log "  existing bridge networks: ${BRIDGE_COUNT}"
# Check daemon.json for expanded pool
if [[ -f /etc/docker/daemon.json ]]; then
    if grep -q "default-address-pools" /etc/docker/daemon.json 2>/dev/null; then
        POOL_BASE=$(python3 -c "
import json, sys
d = json.load(open('/etc/docker/daemon.json'))
pools = d.get('default-address-pools', [])
total = sum((1 << (p.get('size', 24) - (int(p['base'].split('/')[1]) if '/' in p.get('base','') else 16))) for p in pools if 'base' in p)
print(total)
" 2>/dev/null || echo "unknown")
        log "  daemon.json has custom pools (estimated /24 capacity: ${POOL_BASE})"
        if [[ "${POOL_BASE}" != "unknown" ]] && [[ "${POOL_BASE}" -lt 1024 ]] 2>/dev/null; then
            log "  ⚠️  Address pool capacity ${POOL_BASE} may be insufficient for ${WORKER_MAX_TASKS} concurrent tasks"
            log "     Recommend: see issue #3 §2.3 for daemon.json expansion"
        else
            log "  ✅ Address pool looks sufficient"
        fi
    else
        log "  ⚠️  No custom default-address-pools in /etc/docker/daemon.json"
        log "     Default (256 /24 subnets) may be exhausted at >64 concurrent tasks"
        log "     Recommend adding: {\"default-address-pools\": [{\"base\":\"10.200.0.0/12\",\"size\":24}]}"
    fi
else
    log "  ⚠️  /etc/docker/daemon.json not found — using docker defaults (limited /24 pool)"
fi

# ── Pre-flight: orphan container/network cleanup (坑5) ───────────────────────
log "Pre-flight [5/6]: Orphan container/network cleanup (SKIP_PREFLIGHT_CLEANUP=${SKIP_PREFLIGHT_CLEANUP})"
if [[ "${SKIP_PREFLIGHT_CLEANUP}" != "1" ]]; then
    if [[ "${PREFLIGHT_KILL_ORPHAN_RUNNING}" == "1" ]]; then
        cleanup_task_docker_objects "preflight"
    else
        cleanup_stopped_pool_objects "preflight"
        ORPHAN_RUNNING=$(task_container_ids | wc -l || true)
        if [[ "${ORPHAN_RUNNING:-0}" -gt 0 ]] 2>/dev/null; then
            log "  ⚠️  ${ORPHAN_RUNNING} orphan task containers still running"
            log "     Matching name regex: ${TASK_CONTAINER_REGEX}"
            log "     Matching image regex: ${TASK_IMAGE_REGEX}"
            log "     Set PREFLIGHT_KILL_ORPHAN_RUNNING=1 to remove them before start."
        fi
    fi
else
    log "  ⏭  Skipped (SKIP_PREFLIGHT_CLEANUP=1)"
fi

log "=== Pre-flight checks complete, starting pool_server ==="
log ""

# ── Background error filter (every 30s) ──────────────────────────────────────
(
  while true; do
    sleep 30
    tail -n "${CPU_ERR_SCAN_LINES}" "${CPU_POOL_LOG}" 2>/dev/null \
      | grep -E "Error|Exception|Traceback|500|502|PermissionError|docker|FAILED|Connection|SLOTS_EXHAUSTED|Too many open files|address pools|pending_closes" \
      | grep -v "DeprecationWarning" \
      | tail -n 300 \
      > "${CPU_ERR_LOG}" 2>/dev/null || true
  done
) &
ERR_FILTER_PID=$!
POOL_SERVER_PID=""
CLEANUP_STARTED=0

cleanup() {
  local rc="${1:-0}"
  if [[ "${CLEANUP_STARTED}" == "1" ]]; then
    return 0
  fi
  CLEANUP_STARTED=1
  trap - EXIT INT TERM

  set +e
  if [[ -n "${POOL_SERVER_PID:-}" ]] && kill -0 "${POOL_SERVER_PID}" 2>/dev/null; then
    log "Stopping pool_server child PID=${POOL_SERVER_PID}..."
    kill "${POOL_SERVER_PID}" 2>/dev/null || true
    for _ in $(seq 1 "${POOL_SERVER_SHUTDOWN_GRACE}"); do
      kill -0 "${POOL_SERVER_PID}" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "${POOL_SERVER_PID}" 2>/dev/null; then
      log "pool_server child did not stop within ${POOL_SERVER_SHUTDOWN_GRACE}s; sending SIGKILL"
      kill -9 "${POOL_SERVER_PID}" 2>/dev/null || true
    fi
    wait "${POOL_SERVER_PID}" 2>/dev/null || true
  fi

  if [[ -n "${ERR_FILTER_PID:-}" ]]; then
    kill "${ERR_FILTER_PID}" 2>/dev/null || true
    wait "${ERR_FILTER_PID}" 2>/dev/null || true
  fi

  # Final snapshot
  tail -n "${CPU_ERR_SCAN_LINES}" "${CPU_POOL_LOG}" 2>/dev/null \
    | grep -E "Error|Exception|Traceback|500|502|PermissionError|docker|FAILED|Connection|SLOTS_EXHAUSTED|Too many open files|address pools|pending_closes" \
    | grep -v "DeprecationWarning" \
    | tail -n 300 \
    > "${CPU_ERR_LOG}" 2>/dev/null || true

  if [[ "${FINAL_DOCKER_CLEANUP}" == "1" ]]; then
    cleanup_task_docker_objects "final"
  else
    log "Final Docker cleanup skipped (FINAL_DOCKER_CLEANUP=0)"
  fi

  log "pool_server stopped (rc=${rc})."
}

terminate() {
  local sig_rc="${1:-143}"
  cleanup "${sig_rc}"
  exit "${sig_rc}"
}

trap 'cleanup "$?"' EXIT
trap 'terminate 130' INT
trap 'terminate 143' TERM

# ── Capacity summary before start ────────────────────────────────────────────
echo "========================================"
echo "  Pool Server v2 Configuration"
echo "  max_tasks:             ${WORKER_MAX_TASKS}"
echo "  max_runs_per_task:     ${WORKER_MAX_RUNS_PER_TASK}"
echo "  serial_task_ids:       ${WORKER_SERIAL_TASK_IDS}"
echo "  task_run_overrides:    ${WORKER_TASK_MAX_RUNS_OVERRIDES:-<none>}"
echo "  auto_serial_compose:   ${WORKER_AUTO_SERIALIZE_UNSAFE_COMPOSE}"
echo "  potential_capacity:    $((WORKER_MAX_TASKS * WORKER_MAX_RUNS_PER_TASK)) leases"
echo "  effective_max_total:   ${WORKER_MAX_TOTAL_RUNS} leases"
echo "  max_concurrent_closes: ${WORKER_MAX_CONCURRENT_CLOSES}"
echo "  max_concurrent_builds: ${WORKER_MAX_CONCURRENT_BUILDS}"
echo "  max_concurrent_resets: ${WORKER_MAX_CONCURRENT_RESETS} admission_timeout=${WORKER_RESET_ADMISSION_TIMEOUT}s retry_after=${WORKER_RESET_BACKLOG_RETRY_AFTER}s"
echo "  build_queue_timeout:   ${WORKER_DOCKER_BUILD_QUEUE_TIMEOUT}s retry_after=${WORKER_DOCKER_BUILD_BACKLOG_RETRY_AFTER}s"
echo "  pressure_guard:        ${WORKER_PRESSURE_GUARD_ENABLED}"
echo "  port:                  ${ENV_SERVER_PORT}"
echo "  log:                   ${CPU_POOL_LOG}"
echo "  nofile soft:           $(ulimit -Sn)"
echo "========================================"
echo ""

# ── Start pool_server ─────────────────────────────────────────────────────────
cd "${REPO_ROOT}"

export DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/benchmarks/environments}"
export TBENCH_OUTPUT_ROOT="${TBENCH_OUTPUT_ROOT:-${OPENCLAW_REMOTE_LOG_DIR}/task_outputs}"
export TBENCH_DOCKER_IMAGE_SOURCE="${TBENCH_DOCKER_IMAGE_SOURCE:-build}"
export TBENCH_DOCKER_PULL_PREFIX="${TBENCH_DOCKER_PULL_PREFIX:-}"
export AGENT_SAFETYBENCH_ROOT="${AGENT_SAFETYBENCH_ROOT:-${REPO_ROOT}/../Agent-SafetyBench}"
export COMPOSE_OVERRIDE_PATH="${COMPOSE_OVERRIDE_PATH:-}"
export PYTHONUNBUFFERED=1
export DOCKER_DATA_ROOT
export WORKER_DISK_GUARD_ENABLED
export WORKER_DISK_GUARD_STRICT
export WORKER_MIN_DOCKER_FREE_GB
export WORKER_MAX_DOCKER_USED_PCT
export WORKER_MAX_DOCKER_INODE_PCT
export WORKER_SERIAL_TASK_IDS
export WORKER_TASK_MAX_RUNS_OVERRIDES
export WORKER_AUTO_SERIALIZE_UNSAFE_COMPOSE
export WORKER_MAX_CONCURRENT_BUILDS
export WORKER_MAX_CONCURRENT_RESETS
export WORKER_RESET_ADMISSION_TIMEOUT
export WORKER_RESET_BACKLOG_RETRY_AFTER
export WORKER_RESET_CANCEL_JOIN_TIMEOUT
export WORKER_SHUTDOWN_RESET_JOIN_TIMEOUT
export WORKER_DOCKER_BUILD_QUEUE_TIMEOUT
export WORKER_DOCKER_BUILD_BACKLOG_RETRY_AFTER
export WORKER_RESET_TIMEOUT_RETRY_AFTER
export WORKER_PRESSURE_GUARD_ENABLED
export WORKER_CLOSE_TASK_TIMEOUT
export WORKER_CLOSE_QUEUE_TIMEOUT
export WORKER_CLOSE_SESSION_TIMEOUT
export WORKER_ALLOCATED_TTL
export ENSURE_IMAGE_TIMEOUT
export RESET_SESSION_TIMEOUT
export WORKER_RESET_OPERATION_TIMEOUT
export WORKER_RESETTING_TTL
export WORKER_CLOSING_REQUESTED_TTL
export WORKER_PIDS_PAUSE_ALLOCATE_PCT
export WORKER_PIDS_REJECT_RESET_PCT
export WORKER_PIDS_MIN_FREE_ALLOCATE
export WORKER_PIDS_MIN_FREE_RESET
export WORKER_SHIM_PAUSE_ALLOCATE
export WORKER_SHIM_REJECT_RESET
export WORKER_PENDING_CLOSES_PAUSE_ALLOCATE
export WORKER_PENDING_CLOSES_REJECT_RESET
export WORKER_SHUTDOWN_FORCE_CLEANUP_TIMEOUT
export WORKER_DOCKER_CLI_TIMEOUT
export WORKER_DOCKER_DEGRADED_FAIL_STREAK
export WORKER_DOCKER_DEGRADED_COOLDOWN
export WORKER_PRESSURE_CACHE_TTL
export WORKER_RESET_STORM_GUARD
export WORKER_RESET_STORM_BLOCK_ALLOCATE
export WORKER_RESET_STORM_MIN_RESETTING
export WORKER_RESET_STORM_MIN_AGE
export WORKER_RESET_STORM_RATIO_PCT
export TERMINAL_ENV_FORCE_DOCKER_CLEANUP
export TERMINAL_ENV_FORCE_DOCKER_CLEANUP_BROAD
export TERMINAL_ENV_FORCE_DOCKER_CLEANUP_ALWAYS
export TERMINAL_ENV_FORCE_DOCKER_CLEANUP_TIMEOUT
export TERMINAL_ENV_DOCKER_CLEANUP_WORKERS
export TERMINAL_ENV_FAST_CLOSE
export TERMINAL_ENV_SKIP_UNBOUNDED_STOP
export TERMINAL_ENV_FAST_CLOSE_STOP_TIMEOUT
export WORKER_REPAIR_PENDING_CLOSES
export WORKER_REPAIR_PENDING_CLOSES_MAX_ACTIVE_RUNS
export WORKER_REPAIR_PENDING_CLOSES_CANCEL_TIMEOUT
export WORKER_REPAIR_PENDING_CLOSES_MIN_AGE
export WORKER_REPAIR_STALE_RUNS
export WORKER_REPAIR_STALE_RUNS_MIN_AGE
export WORKER_REPAIR_STALE_RUNS_MAX_REPAIRS
export WORKER_REPAIR_CLOSE_REQUESTED_RUNS
export WORKER_REPAIR_CLOSE_REQUESTED_MIN_AGE
export WORKER_REPAIR_CLOSE_REQUESTED_MAX_REPAIRS
export WORKER_CLOSE_REQUESTED_FORCE_RELEASE
export WORKER_CLOSE_REQUESTED_FORCE_RELEASE_AFTER
export WORKER_ORPHAN_DOCKER_SWEEP
export WORKER_ORPHAN_DOCKER_SWEEP_INTERVAL
export WORKER_ORPHAN_DOCKER_SWEEP_MIN_AGE
export WORKER_ORPHAN_DOCKER_SWEEP_MAX_REMOVE
export WORKER_ORPHAN_DOCKER_SWEEP_TIMEOUT
export WORKER_ORPHAN_DOCKER_SWEEP_BACKOFF_BASE
export WORKER_ORPHAN_DOCKER_SWEEP_BACKOFF_MAX
export WORKER_AUTO_REPAIR_ON_CAPACITY
export WORKER_AUTO_REPAIR_CLOSE_REQUESTED_MIN_AGE
export WORKER_AUTO_REPAIR_STALE_MIN_AGE
export WORKER_AUTO_REPAIR_MAX_REPAIRS
export WORKER_DOCKER_BUILD_DEDUP
export WORKER_DOCKER_BUILD_SKIP_EXISTING
export WORKER_DOCKER_BUILD_FAILED_TTL
export WORKER_DOCKER_BUILD_TRANSIENT_TTL
export WORKER_DOCKER_BUILD_TRANSIENT_RETRY_AFTER
export WORKER_DOCKER_TASK_BLACKLIST_TTL
export WORKER_DOCKERFILE_PRECHECK
export WORKER_TASK_IMAGE_RETRY_AFTER
export CONTAINER_PIDS_LIMIT
export CONTAINER_MEMORY_LIMIT

pool_server_python_ok() {
    local python_bin="$1"
    [[ -x "${python_bin}" ]] || return 1
    PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
      "${python_bin}" - <<'PY' >/dev/null 2>&1
import sys

if sys.version_info < (3, 12):
    raise SystemExit(1)

# Import the actual worker entrypoint, not only top-level package specs.
# The worker imports terminal_bench/camel.toolkits lazily at first task reset
# (their package __init__ pulls the harness stack and is slow on shared
# storage), so this deep import no longer executes them: a partial terminal
# harness install now surfaces as a clear per-task reset error instead of a
# pre-uvicorn crash loop, and this check stays fast.
import harborrl.platform.worker_cli  # noqa: F401
import camel  # noqa: F401
PY
}

SHARED_CONDA_POOL_SERVER_VENV="${SHARED_CONDA_POOL_SERVER_VENV:-$(cd "${REPO_ROOT}/../.." && pwd)/.cache/lightrl-seta-worker-py312}"
LEGACY_SHARED_CONDA_POOL_SERVER_VENV="${LEGACY_SHARED_CONDA_POOL_SERVER_VENV:-$(cd "${REPO_ROOT}/.." && pwd)/conda_envs/openclaw-worker-py312}"
POOL_SERVER_PYTHON="${POOL_SERVER_PYTHON:-}"
if [[ -n "${POOL_SERVER_PYTHON}" ]]; then
    if ! pool_server_python_ok "${POOL_SERVER_PYTHON}"; then
        echo "[ERROR] explicit POOL_SERVER_PYTHON lacks required worker dependencies: ${POOL_SERVER_PYTHON}" >&2
        exit 1
    fi
else
    python_candidates=(
        "${POOL_SERVER_VENV:+${POOL_SERVER_VENV}/bin/python}"
        "${SHARED_CONDA_POOL_SERVER_VENV}/bin/python"
        "${LEGACY_SHARED_CONDA_POOL_SERVER_VENV}/bin/python"
        "${REPO_ROOT}/.venv/bin/python"
        "$(command -v python3 || command -v python)"
    )
    for candidate in "${python_candidates[@]}"; do
        [[ -n "${candidate}" ]] || continue
        if pool_server_python_ok "${candidate}"; then
            POOL_SERVER_PYTHON="${candidate}"
            break
        fi
        if [[ -x "${candidate}" ]]; then
            log "  rejecting pool_server Python with missing/incompatible deps: ${candidate}"
        fi
    done
fi

if [[ -z "${POOL_SERVER_PYTHON}" ]]; then
    echo "[ERROR] no compatible pool_server Python found." >&2
    echo "        Required: Python >=3.12 with a deep-importable worker_cli and camel." >&2
    echo "        Set POOL_SERVER_PYTHON or POOL_SERVER_VENV to a prepared environment." >&2
    exit 1
fi

POOL_SERVER_VENV="${POOL_SERVER_VENV:-$(cd -- "$(dirname -- "${POOL_SERVER_PYTHON}")/.." &>/dev/null && pwd)}"

if [ -f "${POOL_SERVER_VENV}/bin/activate" ]; then
    # shellcheck disable=SC1090
    source "${POOL_SERVER_VENV}/bin/activate"
else
    export PATH="${POOL_SERVER_VENV}/bin:${PATH}"
fi

log "  pool_server python: ${POOL_SERVER_PYTHON}"
log "  pool_server env: ${POOL_SERVER_VENV}"
"${POOL_SERVER_PYTHON}" - <<'PY'
import sys
print("  pool_server python version:", sys.version.replace("\n", " "))
PY

start_pool_server_child() {
  # Use stdbuf for line-buffered output (real-time log visibility). Do not use
  # exec here: the launcher owns cleanup traps and must survive the child process.
  stdbuf -oL -eL \
      "${POOL_SERVER_PYTHON}" -m harborrl.platform.worker_cli \
      --host 0.0.0.0 \
      --port "${ENV_SERVER_PORT}" \
      --max-tasks "${WORKER_MAX_TASKS}" \
      --max-runs-per-task "${WORKER_MAX_RUNS_PER_TASK}" \
      --max-total-runs "${WORKER_MAX_TOTAL_RUNS}" \
      --max-concurrent-closes "${WORKER_MAX_CONCURRENT_CLOSES}" \
      --output-root "${TBENCH_OUTPUT_ROOT}" &
  POOL_SERVER_PID=$!
  log "pool_server child started PID=${POOL_SERVER_PID}"
}

restart_count=0
backoff="${POOL_SERVER_RESTART_BACKOFF_INITIAL}"

while true; do
  child_started_ts="$(date +%s)"
  start_pool_server_child

  set +e
  wait "${POOL_SERVER_PID}"
  POOL_SERVER_RC=$?
  set -e
  POOL_SERVER_PID=""

  child_stopped_ts="$(date +%s)"
  child_runtime=$((child_stopped_ts - child_started_ts))
  log "pool_server child exited rc=${POOL_SERVER_RC} runtime=${child_runtime}s"

  if [[ "${POOL_SERVER_SUPERVISE}" != "1" ]]; then
    exit "${POOL_SERVER_RC}"
  fi

  if [[ "${POOL_SERVER_CHILD_EXIT_CLEANUP}" == "1" ]]; then
    cleanup_task_docker_objects "child_exit_rc_${POOL_SERVER_RC}"
  fi

  if [[ "${child_runtime}" -ge "${POOL_SERVER_RESTART_RESET_WINDOW}" ]] 2>/dev/null; then
    restart_count=0
    backoff="${POOL_SERVER_RESTART_BACKOFF_INITIAL}"
  fi

  restart_count=$((restart_count + 1))
  if [[ "${POOL_SERVER_MAX_RESTARTS}" -gt 0 ]] 2>/dev/null \
     && [[ "${restart_count}" -gt "${POOL_SERVER_MAX_RESTARTS}" ]]; then
    log "pool_server restart limit exceeded (${restart_count}/${POOL_SERVER_MAX_RESTARTS}); exiting rc=${POOL_SERVER_RC}"
    exit "${POOL_SERVER_RC}"
  fi

  log "Restarting pool_server after ${backoff}s (attempt=${restart_count}/${POOL_SERVER_MAX_RESTARTS})"
  sleep "${backoff}"
  backoff=$((backoff * 2))
  if [[ "${backoff}" -gt "${POOL_SERVER_RESTART_BACKOFF_MAX}" ]] 2>/dev/null; then
    backoff="${POOL_SERVER_RESTART_BACKOFF_MAX}"
  fi
done
