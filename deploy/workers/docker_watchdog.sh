#!/usr/bin/env bash
# docker_watchdog.sh — 加固版守护进程（针对 DinD/DooD + agentic-RL 场景）
#
# 设计目标：在 14 h 长跑中拦截 issue #3 §3/§4/§5 描述的环境层崩溃
#   §3 docker bridge address-pool 耗尽
#   §4 nofile=1024 触发 "Too many open files"
#   §5 长跑后 dockerd 状态污染（孤儿容器/网络残留）
#
# 修复了 v2 早期版本的 9 个问题（详见 ../runs/.../analysis/REPORT.md → docker_watchdog 修复方案）：
#   P0-1: 删除 restart_docker 中的 systemctl restart 调用（D state 时会无限挂起）
#   P0-2: pkill 前先 stop docker.socket，阻断 systemd auto-restart race
#   P0-3: emergency_pressure_relief 加 60 s 冷却 + foreground+timeout，避免并发拖死 dockerd
#   P1-1: 新增 pool_server /healthz + /status + bridge 网络数 监控（这次崩溃的真正信号）
#   P1-2: cgroup v2 detection（统一 hierarchy 路径解析）
#   P1-3: 检测 host pid namespace；不在 host ns 时不擦容器 state
#   P1-4: 周期性深度探活（network create/rm 模拟 pool 真实路径）
#   P1-5: 日志 rotate 改为 truncate-in-place，nohup fd 不丢
#   P1-6: enforce_container_limit 排除 pool_server 容器（杀候选改用 task 容器 pattern）
#   P1-7: 启动时打印 namespace 信息便于诊断
#
# 用法（推荐 systemd 起，见 docker-watchdog.service）：
#   systemctl enable --now docker-watchdog
# 或临时：
#   nohup bash docker_watchdog.sh > /tmp/docker_watchdog.log 2>&1 &

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"

# ── 可调参数 ──────────────────────────────────────────────────────────
MAX_RUNNING_CONTAINERS="${MAX_RUNNING_CONTAINERS:-80}"
HARD_KILL_THRESHOLD="${HARD_KILL_THRESHOLD:-128}"
CLEANUP_INTERVAL="${CLEANUP_INTERVAL:-60}"
HEALTH_CHECK_INTERVAL="${HEALTH_CHECK_INTERVAL:-30}"
CGROUP_MONITOR_INTERVAL="${CGROUP_MONITOR_INTERVAL:-15}"
PROC_MONITOR_INTERVAL="${PROC_MONITOR_INTERVAL:-15}"
DOCKER_CLI_CHECK_INTERVAL="${DOCKER_CLI_CHECK_INTERVAL:-30}"
PROXY_CHECK_INTERVAL="${PROXY_CHECK_INTERVAL:-300}"
POOL_CHECK_INTERVAL="${POOL_CHECK_INTERVAL:-30}"
DEEP_PROBE_INTERVAL="${DEEP_PROBE_INTERVAL:-300}"
PIDS_WARN_PCT="${PIDS_WARN_PCT:-55}"
PIDS_EMERGENCY_PCT="${PIDS_EMERGENCY_PCT:-70}"
PIDS_EMERGENCY_MIN_FREE="${PIDS_EMERGENCY_MIN_FREE:-3500}"
PROC_WARN_COOLDOWN_S="${PROC_WARN_COOLDOWN_S:-60}"
PIDS_RELIEF_COOLDOWN_S="${PIDS_RELIEF_COOLDOWN_S:-30}"
DOCKER_PROC_WARN="${DOCKER_PROC_WARN:-512}"
DOCKER_PROC_EMERGENCY="${DOCKER_PROC_EMERGENCY:-900}"
SHIM_PROC_WARN="${SHIM_PROC_WARN:-256}"
SHIM_PROC_EMERGENCY="${SHIM_PROC_EMERGENCY:-512}"
DOCKER_DOWN_SHIM_RELIEF="${DOCKER_DOWN_SHIM_RELIEF:-128}"
RUNC_PROC_WARN="${RUNC_PROC_WARN:-50}"
RUNC_PROC_EMERGENCY="${RUNC_PROC_EMERGENCY:-150}"
ZOMBIE_WARN="${ZOMBIE_WARN:-50}"
ZOMBIE_EMERGENCY="${ZOMBIE_EMERGENCY:-200}"
MEM_WARN_PCT="${MEM_WARN_PCT:-80}"
MEM_EMERGENCY_PCT="${MEM_EMERGENCY_PCT:-92}"
MAX_CONSECUTIVE_HEALTH_FAILS="${MAX_CONSECUTIVE_HEALTH_FAILS:-3}"
MAX_CONSECUTIVE_DOCKER_CLI_FAILS="${MAX_CONSECUTIVE_DOCKER_CLI_FAILS:-2}"
DOCKER_CLI_TIMEOUT="${DOCKER_CLI_TIMEOUT:-5}"
LOG_FILE="${LOG_FILE:-/tmp/docker_watchdog.log}"
LOG_MAX_BYTES="${LOG_MAX_BYTES:-209715200}"            # 200 MiB
DOCKER_SOCK="${DOCKER_SOCK:-/var/run/docker.sock}"
DOCKER_DATA_ROOT="${DOCKER_DATA_ROOT:-${DOCKER_ROOT:-/data}}"
PROXY_URL="${PROXY_URL:-http://httpproxy-headless.kubebrain.svc.pjlab.local:3128}"
NO_PROXY_LIST="${NO_PROXY_LIST:-localhost,127.0.0.1,10.0.0.0/8,100.96.0.0/12,.pjlab.org.cn,.pjlab.local,.svc}"
PROXY_ENV_FILE="${PROXY_ENV_FILE:-/etc/seta_build_proxy.env}"
FIX_SCRIPT="${FIX_SCRIPT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)/fix_dockerd_and_proxy.sh}"
WATCHDOG_AUTO_REPAIR="${WATCHDOG_AUTO_REPAIR:-1}"
WATCHDOG_REPAIR_MODE="${WATCHDOG_REPAIR_MODE:-restart}"  # restart | full-fix
WATCHDOG_FULL_FIX_ALLOW_SELF_STOP="${WATCHDOG_FULL_FIX_ALLOW_SELF_STOP:-0}"
WATCHDOG_KILL_SHIMS_ON_DOCKER_DOWN="${WATCHDOG_KILL_SHIMS_ON_DOCKER_DOWN:-1}"
REPAIR_LOCK_DIR="${REPAIR_LOCK_DIR:-/run/docker_watchdog_repair.lock}"
REPAIR_COOLDOWN_S="${REPAIR_COOLDOWN_S:-300}"

POOL_HOST="${POOL_HOST:-127.0.0.1}"
POOL_PORT="${POOL_PORT:-18081}"
POOL_PENDING_CLOSES_WARN="${POOL_PENDING_CLOSES_WARN:-50}"
POOL_PENDING_CLOSES_REPAIR="${POOL_PENDING_CLOSES_REPAIR:-1}"
POOL_PENDING_CLOSES_REPAIR_THRESHOLD="${POOL_PENDING_CLOSES_REPAIR_THRESHOLD:-${POOL_PENDING_CLOSES_WARN}}"
POOL_PENDING_CLOSES_STUCK_CHECKS="${POOL_PENDING_CLOSES_STUCK_CHECKS:-5}"
POOL_PENDING_CLOSES_ACTIVE_MAX="${POOL_PENDING_CLOSES_ACTIVE_MAX:--1}"
POOL_PENDING_CLOSES_REAP_LIMIT="${POOL_PENDING_CLOSES_REAP_LIMIT:-0}"
POOL_PENDING_CLOSES_REPAIR_COOLDOWN_S="${POOL_PENDING_CLOSES_REPAIR_COOLDOWN_S:-300}"
POOL_PENDING_CLOSES_CANCEL_API="${POOL_PENDING_CLOSES_CANCEL_API:-1}"
POOL_PENDING_CLOSES_CANCEL_TIMEOUT="${POOL_PENDING_CLOSES_CANCEL_TIMEOUT:-5}"
POOL_PENDING_CLOSES_CANCEL_MIN_AGE="${POOL_PENDING_CLOSES_CANCEL_MIN_AGE:-90}"
POOL_PENDING_CLOSES_KILL_CONTAINERS_WHEN_ACTIVE="${POOL_PENDING_CLOSES_KILL_CONTAINERS_WHEN_ACTIVE:-0}"
POOL_READY_FAILS_RESTART="${POOL_READY_FAILS_RESTART:-6}"
POOL_RESTART_ACTIVE_MAX="${POOL_RESTART_ACTIVE_MAX:-0}"
POOL_RESTART_COOLDOWN_S="${POOL_RESTART_COOLDOWN_S:-300}"
WATCHDOG_STOP_POOL_LAUNCHER="${WATCHDOG_STOP_POOL_LAUNCHER:-0}"
POOL_E2E_PROBE_INTERVAL="${POOL_E2E_PROBE_INTERVAL:-0}"
POOL_E2E_PROBE_PAYLOAD_FILE="${POOL_E2E_PROBE_PAYLOAD_FILE:-}"
POOL_E2E_PROBE_TIMEOUT="${POOL_E2E_PROBE_TIMEOUT:-600}"
POOL_E2E_PROBE_FAILS_RESTART="${POOL_E2E_PROBE_FAILS_RESTART:-2}"
POOL_RESET_STORM_REPAIR="${POOL_RESET_STORM_REPAIR:-1}"
POOL_RESET_STORM_MIN_RESETTING="${POOL_RESET_STORM_MIN_RESETTING:-32}"
POOL_RESET_STORM_RATIO_PCT="${POOL_RESET_STORM_RATIO_PCT:-80}"
POOL_RESET_STORM_MIN_AGE="${POOL_RESET_STORM_MIN_AGE:-2100}"
POOL_RESET_STORM_STUCK_CHECKS="${POOL_RESET_STORM_STUCK_CHECKS:-2}"
POOL_RESET_STORM_REPAIR_LIMIT="${POOL_RESET_STORM_REPAIR_LIMIT:-64}"
POOL_RESET_STORM_REPAIR_COOLDOWN_S="${POOL_RESET_STORM_REPAIR_COOLDOWN_S:-120}"
BRIDGE_NETS_WARN="${BRIDGE_NETS_WARN:-200}"
EMERGENCY_COOLDOWN_S="${EMERGENCY_COOLDOWN_S:-60}"
POOL_SERVER_NAME_REGEX="${POOL_SERVER_NAME_REGEX:-openclaw_pool_server}"
TASK_CONTAINER_REGEX="${TASK_CONTAINER_REGEX:-^[0-9]+[-_].*(slime[-_]?run|client|helper).*$}"
TASK_IMAGE_REGEX="${TASK_IMAGE_REGEX:-^tb__[0-9]+__.*(:|$)}"
WATCHDOG_REAP_HEADROOM="${WATCHDOG_REAP_HEADROOM:-32}"
WATCHDOG_SOFT_REAP_BATCH="${WATCHDOG_SOFT_REAP_BATCH:-16}"
WATCHDOG_HARD_REAP_BATCH="${WATCHDOG_HARD_REAP_BATCH:-64}"
WATCHDOG_STALE_MIN_AGE_SOFT="${WATCHDOG_STALE_MIN_AGE_SOFT:-900}"
WATCHDOG_STALE_MIN_AGE_PRESSURE="${WATCHDOG_STALE_MIN_AGE_PRESSURE:-300}"
WATCHDOG_STALE_MIN_AGE_HARD="${WATCHDOG_STALE_MIN_AGE_HARD:-120}"
WATCHDOG_STALE_STATUS_MIN_AGE="${WATCHDOG_STALE_STATUS_MIN_AGE:-3600}"
WATCHDOG_IDLE_REAP_ENABLED="${WATCHDOG_IDLE_REAP_ENABLED:-1}"
WATCHDOG_IDLE_REAP_MIN_CONTAINERS="${WATCHDOG_IDLE_REAP_MIN_CONTAINERS:-48}"
WATCHDOG_IDLE_REAP_MIN_GAP="${WATCHDOG_IDLE_REAP_MIN_GAP:-24}"
WATCHDOG_IDLE_REAP_BATCH="${WATCHDOG_IDLE_REAP_BATCH:-16}"
WATCHDOG_IDLE_REAP_MIN_AGE="${WATCHDOG_IDLE_REAP_MIN_AGE:-900}"
WATCHDOG_IDLE_REAP_COOLDOWN_S="${WATCHDOG_IDLE_REAP_COOLDOWN_S:-300}"
WATCHDOG_RESET_STORM_ORPHAN_REAP_ENABLED="${WATCHDOG_RESET_STORM_ORPHAN_REAP_ENABLED:-1}"
WATCHDOG_RESET_STORM_ORPHAN_REAP_ACTIVE_GAP="${WATCHDOG_RESET_STORM_ORPHAN_REAP_ACTIVE_GAP:-16}"
WATCHDOG_RESET_STORM_ORPHAN_REAP_BATCH="${WATCHDOG_RESET_STORM_ORPHAN_REAP_BATCH:-16}"
WATCHDOG_RESET_STORM_ORPHAN_REAP_MIN_AGE="${WATCHDOG_RESET_STORM_ORPHAN_REAP_MIN_AGE:-3600}"
WATCHDOG_RESET_STORM_ORPHAN_REAP_MIN_RESET_AGE="${WATCHDOG_RESET_STORM_ORPHAN_REAP_MIN_RESET_AGE:-900}"
WATCHDOG_RESET_STORM_ORPHAN_REAP_COOLDOWN_S="${WATCHDOG_RESET_STORM_ORPHAN_REAP_COOLDOWN_S:-300}"
WATCHDOG_STALE_LOW_CPU_PCT="${WATCHDOG_STALE_LOW_CPU_PCT:-1.0}"
WATCHDOG_STALE_LOW_MEM_MB="${WATCHDOG_STALE_LOW_MEM_MB:-1024}"
WATCHDOG_STATS_TIMEOUT="${WATCHDOG_STATS_TIMEOUT:-10}"
WATCHDOG_PROTECTED_IDS_FILE="${WATCHDOG_PROTECTED_IDS_FILE:-/tmp/docker_watchdog_protected_ids.$$}"
WATCHDOG_PROTECTED_NAMES_FILE="${WATCHDOG_PROTECTED_NAMES_FILE:-/tmp/docker_watchdog_protected_names.$$}"
WATCHDOG_PROTECTED_TRIALS_FILE="${WATCHDOG_PROTECTED_TRIALS_FILE:-/tmp/docker_watchdog_protected_trials.$$}"
HEARTBEAT_INTERVAL="${HEARTBEAT_INTERVAL:-600}"  # "I'm alive" line every 10 min

DISK_CHECK_INTERVAL="${DISK_CHECK_INTERVAL:-60}"
DISK_WARN_PCT="${DISK_WARN_PCT:-80}"
DISK_EMERGENCY_PCT="${DISK_EMERGENCY_PCT:-92}"
DISK_MIN_FREE_GB="${DISK_MIN_FREE_GB:-20}"
DISK_INODE_WARN_PCT="${DISK_INODE_WARN_PCT:-80}"
DISK_INODE_EMERGENCY_PCT="${DISK_INODE_EMERGENCY_PCT:-90}"
DISK_PRUNE_COOLDOWN_S="${DISK_PRUNE_COOLDOWN_S:-300}"
DISK_BUILD_CACHE_UNTIL="${DISK_BUILD_CACHE_UNTIL:-12h}"
WATCHDOG_AGGRESSIVE_IMAGE_PRUNE="${WATCHDOG_AGGRESSIVE_IMAGE_PRUNE:-0}"
WATCHDOG_DOCKER_STORAGE_GC="${WATCHDOG_DOCKER_STORAGE_GC:-1}"
DOCKER_GC_TRIGGER_USED_PCT="${DOCKER_GC_TRIGGER_USED_PCT:-85}"
DOCKER_GC_TARGET_USED_PCT="${DOCKER_GC_TARGET_USED_PCT:-70}"
DOCKER_GC_MIN_FREE_GB="${DOCKER_GC_MIN_FREE_GB:-${DISK_MIN_FREE_GB}}"
DOCKER_GC_KEEP_PATTERNS="${DOCKER_GC_KEEP_PATTERNS:-ghcr.io/laude-institute/t-bench/*,ubuntu:*,python:*}"
DOCKER_GC_PRUNE_VOLUMES="${DOCKER_GC_PRUNE_VOLUMES:-1}"
DOCKER_GC_DRY_RUN="${DOCKER_GC_DRY_RUN:-0}"
DOCKER_GC_DELETE_OLD_IMAGES="${DOCKER_GC_DELETE_OLD_IMAGES:-0}"
DOCKER_GC_TIMEOUT="${DOCKER_GC_TIMEOUT:-900}"
WATCHDOG_PRUNE_TIMEOUT="${WATCHDOG_PRUNE_TIMEOUT:-120}"
DOCKER_NETWORK_LIFECYCLE_LOCK="${DOCKER_NETWORK_LIFECYCLE_LOCK:-/tmp/lightrl_docker_network_lifecycle.lock}"
POOL_STOP_ON_DISK_EMERGENCY="${POOL_STOP_ON_DISK_EMERGENCY:-1}"
POOL_STOP_COOLDOWN_S="${POOL_STOP_COOLDOWN_S:-300}"

LOG_PREFIX="[docker-watchdog]"

# ── 自身防 OOM ────────────────────────────────────────────────────────
echo -900 > /proc/self/oom_score_adj 2>/dev/null || true

# ── 状态 ──────────────────────────────────────────────────────────────
LAST_EMERGENCY_TS=0
LAST_DEEP_PROBE_TS=0
LAST_DISK_PRUNE_TS=0
LAST_POOL_STOP_TS=0
LAST_REPAIR_TS=0
LAST_PROC_WARN_TS=0
LAST_PIDS_RELIEF_TS=0
LAST_POOL_PENDING_REPAIR_TS=0
LAST_IDLE_REAP_TS=0
LAST_RESET_STORM_ORPHAN_REAP_TS=0
POOL_PENDING_HIGH_COUNT=0

cleanup_watchdog_tmp() {
    rm -f \
        "${WATCHDOG_PROTECTED_IDS_FILE}" \
        "${WATCHDOG_PROTECTED_NAMES_FILE}" \
        "${WATCHDOG_PROTECTED_TRIALS_FILE}" \
        2>/dev/null || true
}
trap cleanup_watchdog_tmp EXIT

# ── namespace 检测 ────────────────────────────────────────────────────
HOST_PID_NS=0
detect_pid_namespace() {
    # 与 PID 1 共享 mnt namespace 的进程基本就是 host
    local self_pid_ns host_pid_ns
    self_pid_ns=$(readlink /proc/self/ns/pid 2>/dev/null || echo "?")
    host_pid_ns=$(readlink /proc/1/ns/pid 2>/dev/null || echo "?")
    if [ -n "$self_pid_ns" ] && [ "$self_pid_ns" = "$host_pid_ns" ] && [ "$self_pid_ns" != "?" ]; then
        HOST_PID_NS=1
    else
        HOST_PID_NS=0
    fi
}

# ── 工具函数 ──────────────────────────────────────────────────────────
# 用 truncate-in-place 而不是 mv，否则 nohup 重定向的 fd 会丢
rotate_log_if_big() {
    [ -f "${LOG_FILE}" ] || return 0
    local sz tmp_file
    sz=$(stat -c%s "${LOG_FILE}" 2>/dev/null || echo 0)
    [ -n "${sz}" ] && [ "${sz}" -ge 0 ] 2>/dev/null || sz=0
    [ "${sz}" -gt "${LOG_MAX_BYTES}" ] || return 0
    local tail_bytes=52428800   # 保留尾部 50 MB
    tmp_file="$(mktemp /tmp/docker_watchdog_rotate.XXXXXX 2>/dev/null || echo /tmp/docker_watchdog_rotate.$$)"
    tail -c "$tail_bytes" "${LOG_FILE}" > "${tmp_file}" 2>/dev/null || true
    : > "${LOG_FILE}"
    cat "${tmp_file}" >> "${LOG_FILE}" 2>/dev/null || true
    rm -f "${tmp_file}" 2>/dev/null || true
}

log() {
    echo "$(date '+%F %T') ${LOG_PREFIX} $*"
    rotate_log_if_big
}

docker_network_prune_safe() {
    local timeout_s="${1:-30}"
    if ! command -v flock >/dev/null 2>&1; then
        log "WARN: flock is unavailable; skipping unsafe docker network prune"
        return 0
    fi
    timeout "${timeout_s}" flock -w "${timeout_s}" \
        "${DOCKER_NETWORK_LIFECYCLE_LOCK}" \
        docker network prune -f >/dev/null 2>&1 || true
}

positive_int_or_default() {
    local name="$1"
    local value="$2"
    local default="$3"
    if [[ "${value}" =~ ^[0-9]+$ ]] && [ "${value}" -gt 0 ] 2>/dev/null; then
        printf '%s' "${value}"
    else
        echo "$(date '+%F %T') ${LOG_PREFIX} WARN: invalid ${name}=${value}; using ${default}" >&2
        printf '%s' "${default}"
    fi
}

nonnegative_int_or_default() {
    local name="$1"
    local value="$2"
    local default="$3"
    if [[ "${value}" =~ ^[0-9]+$ ]]; then
        printf '%s' "${value}"
    else
        echo "$(date '+%F %T') ${LOG_PREFIX} WARN: invalid ${name}=${value}; using ${default}" >&2
        printf '%s' "${default}"
    fi
}

integer_or_default() {
    local name="$1"
    local value="$2"
    local default="$3"
    if [[ "${value}" =~ ^-?[0-9]+$ ]]; then
        printf '%s' "${value}"
    else
        echo "$(date '+%F %T') ${LOG_PREFIX} WARN: invalid ${name}=${value}; using ${default}" >&2
        printf '%s' "${default}"
    fi
}

POOL_PENDING_CLOSES_WARN="$(positive_int_or_default POOL_PENDING_CLOSES_WARN "${POOL_PENDING_CLOSES_WARN}" 50)"
POOL_PENDING_CLOSES_REPAIR_THRESHOLD="$(positive_int_or_default POOL_PENDING_CLOSES_REPAIR_THRESHOLD "${POOL_PENDING_CLOSES_REPAIR_THRESHOLD}" "${POOL_PENDING_CLOSES_WARN}")"
POOL_PENDING_CLOSES_STUCK_CHECKS="$(positive_int_or_default POOL_PENDING_CLOSES_STUCK_CHECKS "${POOL_PENDING_CLOSES_STUCK_CHECKS}" 5)"
POOL_PENDING_CLOSES_ACTIVE_MAX="$(integer_or_default POOL_PENDING_CLOSES_ACTIVE_MAX "${POOL_PENDING_CLOSES_ACTIVE_MAX}" -1)"
POOL_PENDING_CLOSES_REAP_LIMIT="$(nonnegative_int_or_default POOL_PENDING_CLOSES_REAP_LIMIT "${POOL_PENDING_CLOSES_REAP_LIMIT}" 0)"
POOL_PENDING_CLOSES_REPAIR_COOLDOWN_S="$(nonnegative_int_or_default POOL_PENDING_CLOSES_REPAIR_COOLDOWN_S "${POOL_PENDING_CLOSES_REPAIR_COOLDOWN_S}" 300)"
POOL_PENDING_CLOSES_CANCEL_TIMEOUT="$(positive_int_or_default POOL_PENDING_CLOSES_CANCEL_TIMEOUT "${POOL_PENDING_CLOSES_CANCEL_TIMEOUT}" 5)"
POOL_PENDING_CLOSES_CANCEL_MIN_AGE="$(nonnegative_int_or_default POOL_PENDING_CLOSES_CANCEL_MIN_AGE "${POOL_PENDING_CLOSES_CANCEL_MIN_AGE}" 90)"
POOL_READY_FAILS_RESTART="$(positive_int_or_default POOL_READY_FAILS_RESTART "${POOL_READY_FAILS_RESTART}" 6)"
POOL_RESTART_ACTIVE_MAX="$(nonnegative_int_or_default POOL_RESTART_ACTIVE_MAX "${POOL_RESTART_ACTIVE_MAX}" 0)"
POOL_RESTART_COOLDOWN_S="$(nonnegative_int_or_default POOL_RESTART_COOLDOWN_S "${POOL_RESTART_COOLDOWN_S}" 300)"
POOL_E2E_PROBE_INTERVAL="$(nonnegative_int_or_default POOL_E2E_PROBE_INTERVAL "${POOL_E2E_PROBE_INTERVAL}" 0)"
POOL_E2E_PROBE_TIMEOUT="$(positive_int_or_default POOL_E2E_PROBE_TIMEOUT "${POOL_E2E_PROBE_TIMEOUT}" 600)"
POOL_E2E_PROBE_FAILS_RESTART="$(positive_int_or_default POOL_E2E_PROBE_FAILS_RESTART "${POOL_E2E_PROBE_FAILS_RESTART}" 2)"
POOL_RESET_STORM_MIN_RESETTING="$(positive_int_or_default POOL_RESET_STORM_MIN_RESETTING "${POOL_RESET_STORM_MIN_RESETTING}" 32)"
POOL_RESET_STORM_RATIO_PCT="$(positive_int_or_default POOL_RESET_STORM_RATIO_PCT "${POOL_RESET_STORM_RATIO_PCT}" 80)"
POOL_RESET_STORM_MIN_AGE="$(positive_int_or_default POOL_RESET_STORM_MIN_AGE "${POOL_RESET_STORM_MIN_AGE}" 2100)"
POOL_RESET_STORM_STUCK_CHECKS="$(positive_int_or_default POOL_RESET_STORM_STUCK_CHECKS "${POOL_RESET_STORM_STUCK_CHECKS}" 2)"
POOL_RESET_STORM_REPAIR_LIMIT="$(positive_int_or_default POOL_RESET_STORM_REPAIR_LIMIT "${POOL_RESET_STORM_REPAIR_LIMIT}" 64)"
POOL_RESET_STORM_REPAIR_COOLDOWN_S="$(positive_int_or_default POOL_RESET_STORM_REPAIR_COOLDOWN_S "${POOL_RESET_STORM_REPAIR_COOLDOWN_S}" 120)"
WATCHDOG_IDLE_REAP_MIN_CONTAINERS="$(positive_int_or_default WATCHDOG_IDLE_REAP_MIN_CONTAINERS "${WATCHDOG_IDLE_REAP_MIN_CONTAINERS}" 48)"
WATCHDOG_IDLE_REAP_MIN_GAP="$(positive_int_or_default WATCHDOG_IDLE_REAP_MIN_GAP "${WATCHDOG_IDLE_REAP_MIN_GAP}" 24)"
WATCHDOG_IDLE_REAP_BATCH="$(positive_int_or_default WATCHDOG_IDLE_REAP_BATCH "${WATCHDOG_IDLE_REAP_BATCH}" 16)"
WATCHDOG_IDLE_REAP_MIN_AGE="$(positive_int_or_default WATCHDOG_IDLE_REAP_MIN_AGE "${WATCHDOG_IDLE_REAP_MIN_AGE}" 900)"
WATCHDOG_IDLE_REAP_COOLDOWN_S="$(nonnegative_int_or_default WATCHDOG_IDLE_REAP_COOLDOWN_S "${WATCHDOG_IDLE_REAP_COOLDOWN_S}" 300)"
WATCHDOG_RESET_STORM_ORPHAN_REAP_ACTIVE_GAP="$(positive_int_or_default WATCHDOG_RESET_STORM_ORPHAN_REAP_ACTIVE_GAP "${WATCHDOG_RESET_STORM_ORPHAN_REAP_ACTIVE_GAP}" 16)"
WATCHDOG_RESET_STORM_ORPHAN_REAP_BATCH="$(positive_int_or_default WATCHDOG_RESET_STORM_ORPHAN_REAP_BATCH "${WATCHDOG_RESET_STORM_ORPHAN_REAP_BATCH}" 16)"
WATCHDOG_RESET_STORM_ORPHAN_REAP_MIN_AGE="$(positive_int_or_default WATCHDOG_RESET_STORM_ORPHAN_REAP_MIN_AGE "${WATCHDOG_RESET_STORM_ORPHAN_REAP_MIN_AGE}" 3600)"
WATCHDOG_RESET_STORM_ORPHAN_REAP_MIN_RESET_AGE="$(positive_int_or_default WATCHDOG_RESET_STORM_ORPHAN_REAP_MIN_RESET_AGE "${WATCHDOG_RESET_STORM_ORPHAN_REAP_MIN_RESET_AGE}" 900)"
WATCHDOG_RESET_STORM_ORPHAN_REAP_COOLDOWN_S="$(nonnegative_int_or_default WATCHDOG_RESET_STORM_ORPHAN_REAP_COOLDOWN_S "${WATCHDOG_RESET_STORM_ORPHAN_REAP_COOLDOWN_S}" 300)"

docker_alive() {
    timeout 3 curl -fsS --max-time 2 \
        --unix-socket "${DOCKER_SOCK}" \
        http://./_ping >/dev/null 2>&1
}

docker_cli_alive() {
    timeout "${DOCKER_CLI_TIMEOUT}" docker ps -q >/dev/null 2>&1
}

proxy_alive() {
    timeout 5 curl -fsS --max-time 4 --noproxy "" -x "${PROXY_URL}" http://example.com >/dev/null 2>&1
}

# 深度探活：模拟 pool_server 真实 reset 路径——能创建+删 bridge 网络
docker_deep_alive() {
    local netname="wd_probe_$(date +%s)_$$"
    command -v flock >/dev/null 2>&1 || return 0
    if ! timeout 15 flock -w 10 "${DOCKER_NETWORK_LIFECYCLE_LOCK}" \
        sh -c 'docker network create --driver bridge "$1" >/dev/null 2>&1 &&
               docker network rm "$1" >/dev/null 2>&1' sh "${netname}"; then
        return 1
    fi
    return 0
}

# ── repair 防抖和互斥 ────────────────────────────────────────────────
acquire_repair_lock() {
    local now owner_pid owner_ts age
    now=$(date +%s)
    if mkdir "${REPAIR_LOCK_DIR}" 2>/dev/null; then
        printf '%s %s\n' "$$" "$now" > "${REPAIR_LOCK_DIR}/owner" 2>/dev/null || true
        return 0
    fi

    if [ -f "${REPAIR_LOCK_DIR}/owner" ]; then
        read -r owner_pid owner_ts < "${REPAIR_LOCK_DIR}/owner" 2>/dev/null || true
        age=$((now - ${owner_ts:-0}))
        if [ -n "${owner_pid:-}" ] && ! kill -0 "${owner_pid}" 2>/dev/null && [ "$age" -gt 600 ]; then
            log "REPAIR: removing stale lock ${REPAIR_LOCK_DIR} (owner=${owner_pid}, age=${age}s)"
            rm -rf "${REPAIR_LOCK_DIR}" 2>/dev/null || true
            if mkdir "${REPAIR_LOCK_DIR}" 2>/dev/null; then
                printf '%s %s\n' "$$" "$now" > "${REPAIR_LOCK_DIR}/owner" 2>/dev/null || true
                return 0
            fi
        fi
        log "REPAIR suppressed: another repair owns ${REPAIR_LOCK_DIR} (owner=${owner_pid:-?}, age=${age:-?}s)"
    else
        log "REPAIR suppressed: another repair owns ${REPAIR_LOCK_DIR}"
    fi
    return 1
}

release_repair_lock() {
    rm -rf "${REPAIR_LOCK_DIR}" 2>/dev/null || true
}

repair_snapshot() {
    log "REPAIR snapshot: pids=${LAST_PIDS_CUR:-?}/${LAST_PIDS_MAX:-?} (${LAST_PIDS_PCT:-?}%) tasks=${LAST_PROC_TASKS:-?} procs=${LAST_PROC_TOTAL:-?} zombies=${LAST_ZOMBIES:-?} dockerd=${LAST_DOCKERD_PROCS:-?} containerd=${LAST_CONTAINERD_PROCS:-?} shim=${LAST_SHIM_PROCS:-?} runc=${LAST_RUNC_PROCS:-?} docker_cli_fails=${DOCKER_CLI_FAILS:-0}"
}

run_full_fix_script() {
    if [ "${WATCHDOG_FULL_FIX_ALLOW_SELF_STOP}" != "1" ]; then
        log "REPAIR: full-fix requested but disabled because fix_dockerd_and_proxy.sh stops docker-watchdog; falling back to internal restart"
        restart_docker
        return $?
    fi
    if [ ! -f "${FIX_SCRIPT}" ]; then
        log "REPAIR: full-fix script not found: ${FIX_SCRIPT}; falling back to internal restart"
        restart_docker
        return $?
    fi
    log "REPAIR: running full fix script with START_WATCHDOG=0 SKIP_VERIFY=1 (this may stop this watchdog service)"
    DOCKER_DATA_ROOT="${DOCKER_DATA_ROOT}" PROXY_URL="${PROXY_URL}" START_WATCHDOG=0 SKIP_VERIFY=1 \
        bash "${FIX_SCRIPT}"
}

trigger_repair() {
    local reason="$1"
    local force="${2:-0}"
    local now
    now=$(date +%s)
    if [ "${WATCHDOG_AUTO_REPAIR}" != "1" ]; then
        log "REPAIR disabled (WATCHDOG_AUTO_REPAIR=0): ${reason}"
        repair_snapshot
        return 0
    fi
    if [ "${force}" != "1" ] && [ $((now - LAST_REPAIR_TS)) -lt "${REPAIR_COOLDOWN_S}" ]; then
        log "REPAIR suppressed (cooldown ${REPAIR_COOLDOWN_S}s active): ${reason}"
        repair_snapshot
        return 0
    fi
    if ! acquire_repair_lock; then
        return 0
    fi

    LAST_REPAIR_TS="$now"
    log "REPAIR trigger: ${reason}$([ "${force}" = "1" ] && echo " (forced)" || true)"
    repair_snapshot
    case "${WATCHDOG_REPAIR_MODE}" in
        full-fix)
            run_full_fix_script || log "REPAIR: full-fix/restart path failed"
            ;;
        restart|*)
            restart_docker || log "REPAIR: internal dockerd restart failed"
            ;;
    esac
    release_repair_lock
}

task_container_lines() {
    timeout "${DOCKER_CLI_TIMEOUT}" docker ps --format '{{.ID}}\t{{.Names}}\t{{.Image}}' 2>/dev/null \
        | awk -F '\t' -v name_re="${TASK_CONTAINER_REGEX}" -v image_re="${TASK_IMAGE_REGEX}" '
            $2 ~ name_re || $3 ~ image_re {print $0}
        '
}

task_container_ids() {
    task_container_lines | awk -F '\t' '{print $1}'
}

task_container_ids_oldest_first() {
    timeout "${DOCKER_CLI_TIMEOUT}" docker ps --format '{{.CreatedAt}}\t{{.ID}}\t{{.Names}}\t{{.Image}}' 2>/dev/null \
        | awk -F '\t' -v name_re="${TASK_CONTAINER_REGEX}" -v image_re="${TASK_IMAGE_REGEX}" '
            $3 ~ name_re || $4 ~ image_re {print $0}
        ' \
        | sort \
        | awk -F '\t' '{print $2}'
}

task_container_count() {
    task_container_lines | wc -l
}

proc_cmdline_text() {
    local proc_dir="$1"
    tr '\0' ' ' < "${proc_dir}/cmdline" 2>/dev/null || true
}

stop_pool_server_for_pressure() {
    local reason="$1"
    local include_launcher="${2:-${WATCHDOG_STOP_POOL_LAUNCHER}}"
    local proc_dir pid cmdline pids killed=0

    for proc_dir in /proc/[0-9]*; do
        [ -r "${proc_dir}/cmdline" ] || continue
        pid="${proc_dir##*/}"
        cmdline="$(proc_cmdline_text "${proc_dir}")"
        case "${cmdline}" in
            *harborrl.platform.worker_cli*|*remote.pool_server*|*pool_server.py*)
                log "PRESSURE: stopping pool_server pid=${pid} reason=${reason}"
                kill "${pid}" 2>/dev/null || true
                killed=$((killed + 1))
                ;;
            *run_pool_server.sh*)
                if [ "${include_launcher}" = "1" ]; then
                    log "PRESSURE: stopping pool_server launcher pid=${pid} reason=${reason}"
                    kill "${pid}" 2>/dev/null || true
                    killed=$((killed + 1))
                fi
                ;;
        esac
    done

    if [ "${killed}" -eq 0 ]; then
        log "PRESSURE: no pool_server process matched for stop (reason=${reason})"
        return 0
    fi

    sleep 5
    pids=""
    for proc_dir in /proc/[0-9]*; do
        [ -r "${proc_dir}/cmdline" ] || continue
        pid="${proc_dir##*/}"
        cmdline="$(proc_cmdline_text "${proc_dir}")"
        case "${cmdline}" in
            *harborrl.platform.worker_cli*|*remote.pool_server*|*pool_server.py*)
                pids="${pids} ${pid}"
                ;;
            *run_pool_server.sh*)
                if [ "${include_launcher}" = "1" ]; then
                    pids="${pids} ${pid}"
                fi
                ;;
        esac
    done
    if [ -n "${pids}" ]; then
        log "PRESSURE: pool_server still alive after SIGTERM; sending SIGKILL (reason=${reason})"
        printf '%s\n' "${pids}" | xargs -r kill -9 2>/dev/null || true
    fi
}

kill_task_containers_for_pressure() {
    local reason="$1"
    local limit="${2:-0}"
    local ids n

    if [ "${limit}" -gt 0 ] 2>/dev/null; then
        ids="$(task_container_ids_oldest_first | head -n "${limit}" 2>/dev/null || true)"
    else
        ids="$(task_container_ids 2>/dev/null || true)"
    fi
    if [ -z "${ids}" ]; then
        log "PRESSURE: no task containers matched for kill (reason=${reason}, name_re=${TASK_CONTAINER_REGEX}, image_re=${TASK_IMAGE_REGEX})"
        return 1
    fi

    n="$(printf '%s\n' "${ids}" | wc -l)"
    log "PRESSURE: removing ${n} task containers (reason=${reason})"
    printf '%s\n' "${ids}" | xargs -r -n 10 timeout 30 docker rm -f >/dev/null 2>&1 || true
    return 0
}

pids_pressure_relief() {
    local reason="$1"
    local now
    now=$(date +%s)
    if [ $((now - LAST_PIDS_RELIEF_TS)) -lt "${PIDS_RELIEF_COOLDOWN_S}" ]; then
        log "PRESSURE suppressed (cooldown ${PIDS_RELIEF_COOLDOWN_S}s active): ${reason}"
        return 0
    fi
    LAST_PIDS_RELIEF_TS="$now"

    log "PRESSURE: pids emergency relief: ${reason}"
    repair_snapshot
    stop_pool_server_for_pressure "${reason}"
    kill_task_containers_for_pressure "${reason}" 0 || true
}

repair_stuck_pool_pending_closes() {
    local pending="$1"
    local active="$2"
    local now reason matched repair_tmp repair_code

    [ "${POOL_PENDING_CLOSES_REPAIR}" = "1" ] || return 0
    now=$(date +%s)
    if [ $((now - LAST_POOL_PENDING_REPAIR_TS)) -lt "${POOL_PENDING_CLOSES_REPAIR_COOLDOWN_S}" ]; then
        log "POOL_REPAIR suppressed (cooldown ${POOL_PENDING_CLOSES_REPAIR_COOLDOWN_S}s active): pending_closes=${pending} active=${active}"
        return 0
    fi
    LAST_POOL_PENDING_REPAIR_TS="$now"

    reason="stuck pool pending_closes=${pending} active=${active} high_count=${POOL_PENDING_HIGH_COUNT}"
    matched=0
    if [ "${active}" -eq 0 ] 2>/dev/null \
       || [ "${POOL_PENDING_CLOSES_KILL_CONTAINERS_WHEN_ACTIVE}" = "1" ]; then
        log "POOL_REPAIR: ${reason}; reaping task containers with broad matcher"
        if kill_task_containers_for_pressure "${reason}" "${POOL_PENDING_CLOSES_REAP_LIMIT}"; then
            matched=1
        fi
        timeout 30 docker container prune -f --filter "until=0s" >/dev/null 2>&1 || true
        docker_network_prune_safe 30
    else
        log "POOL_REPAIR: ${reason}; active rollouts exist, skipping container kill/prune and using pending-close API only"
    fi

    if [ "${POOL_PENDING_CLOSES_CANCEL_API}" = "1" ]; then
        repair_tmp="$(mktemp /tmp/pool_pending_repair.XXXXXX 2>/dev/null || echo /tmp/pool_pending_repair.$$)"
        repair_code=$(timeout 10 curl -sS --noproxy '*' -o "${repair_tmp}" -w '%{http_code}' \
            -X POST -H 'Content-Type: application/json' \
            --data "{\"reason\":\"watchdog_pending_closes_repair\",\"max_active_runs\":${POOL_PENDING_CLOSES_ACTIVE_MAX},\"cancel_timeout\":${POOL_PENDING_CLOSES_CANCEL_TIMEOUT},\"min_age\":${POOL_PENDING_CLOSES_CANCEL_MIN_AGE}}" \
            "http://${POOL_HOST}:${POOL_PORT}/repair/pending_closes" 2>/dev/null || echo "000")
        if [ "${repair_code}" = "200" ]; then
            log "POOL_REPAIR: pending-close API response: $(head -c 300 "${repair_tmp}" 2>/dev/null)"
        else
            log "POOL_REPAIR: pending-close API failed HTTP ${repair_code}: $(head -c 300 "${repair_tmp}" 2>/dev/null)"
            if [ "${matched}" = "0" ] && [ "${active}" -eq 0 ] 2>/dev/null; then
                log "POOL_REPAIR: no task containers matched and active=0; stop/restart pool_server manually if pending_closes remains high"
            fi
        fi
        rm -f "${repair_tmp}" 2>/dev/null || true
    fi
}

repair_pool_reset_storm() {
    local resetting="$1"
    local active="$2"
    local max_age="$3"
    local now repair_tmp repair_code

    [ "${POOL_RESET_STORM_REPAIR}" = "1" ] || return 0
    now=$(date +%s)
    if [ $((now - LAST_POOL_RESET_STORM_REPAIR_TS)) -lt "${POOL_RESET_STORM_REPAIR_COOLDOWN_S}" ]; then
        log "POOL_RESET_REPAIR suppressed (cooldown ${POOL_RESET_STORM_REPAIR_COOLDOWN_S}s active): resetting=${resetting} active=${active} max_age=${max_age}s"
        return 0
    fi
    LAST_POOL_RESET_STORM_REPAIR_TS="$now"

    repair_tmp="$(mktemp /tmp/pool_reset_repair.XXXXXX 2>/dev/null || echo /tmp/pool_reset_repair.$$)"
    repair_code=$(timeout 15 curl -sS --noproxy '*' -o "${repair_tmp}" -w '%{http_code}' \
        -X POST -H 'Content-Type: application/json' \
        --data "{\"reason\":\"watchdog_reset_storm\",\"min_age\":${POOL_RESET_STORM_MIN_AGE},\"max_repairs\":${POOL_RESET_STORM_REPAIR_LIMIT},\"wait_for_cleanup\":false}" \
        "http://${POOL_HOST}:${POOL_PORT}/repair/resetting_runs" 2>/dev/null || echo "000")
    if [ "${repair_code}" = "200" ]; then
        log "POOL_RESET_REPAIR: response: $(head -c 500 "${repair_tmp}" 2>/dev/null)"
    else
        log "POOL_RESET_REPAIR: failed HTTP ${repair_code}: $(head -c 500 "${repair_tmp}" 2>/dev/null)"
    fi
    rm -f "${repair_tmp}" 2>/dev/null || true
}

# ── 紧急泄压（带冷却 + foreground + timeout）─────────────────────────
emergency_pressure_relief() {
    local reason="$1"
    local now
    now=$(date +%s)
    if [ $((now - LAST_EMERGENCY_TS)) -lt "${EMERGENCY_COOLDOWN_S}" ]; then
        log "EMERGENCY suppressed (cooldown ${EMERGENCY_COOLDOWN_S}s active): ${reason}"
        return
    fi
    LAST_EMERGENCY_TS="$now"
    log "EMERGENCY: ${reason} — kill task containers + prune"

    kill_task_containers_for_pressure "${reason}" 30 || true

    # 清理 stopped + dangling network（foreground，防并发拖死 dockerd）
    timeout 30 docker container prune -f >/dev/null 2>&1 || true
    docker_network_prune_safe 30
}

# ── cgroup 检测（v1 + v2）────────────────────────────────────────────
CGROUP_VERSION=""
CGROUP_PIDS_DIR=""
CGROUP_MEM_DIR=""
CGROUP_PIDS_MAX_VAL=""
CGROUP_MEM_MAX_VAL=""
CGROUP_PIDS_CUR_FILE=""
CGROUP_MEM_CUR_FILE=""

# 沿 cgroup 路径从深到浅扫描，找到"最严有限限制"所在的目录
# 参数: $1=控制器挂载点  $2=cgroup 相对路径  $3=current 文件名  $4=max 文件名
find_tightest_limit() {
    local mount="$1" rel="$2" cur_name="$3" max_name="$4"
    local best_dir="" best_val=""
    local path="$rel"
    while [ -n "$path" ] && [ "$path" != "/" ]; do
        local full="${mount}${path}"
        if [ -f "${full}/${max_name}" ]; then
            local v
            v=$(cat "${full}/${max_name}" 2>/dev/null)
            if [ -n "$v" ] && [ "$v" != "max" ] \
               && [ "$v" -gt 0 ] 2>/dev/null \
               && [ "$v" -lt 9000000000000000000 ] 2>/dev/null; then
                if [ -z "$best_val" ] || [ "$v" -lt "$best_val" ] 2>/dev/null; then
                    best_val="$v"
                    best_dir="$full"
                fi
            fi
        fi
        path="${path%/*}"
    done
    [ -n "$best_dir" ] && echo "${best_dir}|${best_val}"
}

detect_cgroup_v2() {
    [ -f /sys/fs/cgroup/cgroup.controllers ] || return 1
    local rel
    rel=$(awk -F: '$1=="0"{print $3}' /proc/self/cgroup 2>/dev/null)
    [ -n "$rel" ] || return 1
    # v2: pids.max / memory.max 在同一 unified hierarchy
    local r
    r=$(find_tightest_limit /sys/fs/cgroup "$rel" pids.current pids.max)
    if [ -n "$r" ]; then
        CGROUP_PIDS_DIR="${r%|*}"
        CGROUP_PIDS_MAX_VAL="${r#*|}"
        CGROUP_PIDS_CUR_FILE="${CGROUP_PIDS_DIR}/pids.current"
    fi
    r=$(find_tightest_limit /sys/fs/cgroup "$rel" memory.current memory.max)
    if [ -n "$r" ]; then
        CGROUP_MEM_DIR="${r%|*}"
        CGROUP_MEM_MAX_VAL="${r#*|}"
        CGROUP_MEM_CUR_FILE="${CGROUP_MEM_DIR}/memory.current"
    fi
    if [ -n "$CGROUP_PIDS_DIR" ] || [ -n "$CGROUP_MEM_DIR" ]; then
        CGROUP_VERSION="v2"
        return 0
    fi
    return 1
}

detect_cgroup_v1() {
    local pids_rel mem_rel
    pids_rel=$(awk -F: '$2 ~ /(^|,)pids(,|$)/  {print $3}'   /proc/self/cgroup 2>/dev/null | head -1)
    mem_rel=$(awk  -F: '$2 ~ /(^|,)memory(,|$)/{print $3}'   /proc/self/cgroup 2>/dev/null | head -1)
    if [ -d /sys/fs/cgroup/pids ] && [ -n "$pids_rel" ]; then
        local r
        r=$(find_tightest_limit /sys/fs/cgroup/pids "$pids_rel" pids.current pids.max)
        if [ -n "$r" ]; then
            CGROUP_PIDS_DIR="${r%|*}"
            CGROUP_PIDS_MAX_VAL="${r#*|}"
            CGROUP_PIDS_CUR_FILE="${CGROUP_PIDS_DIR}/pids.current"
        fi
    fi
    if [ -d /sys/fs/cgroup/memory ] && [ -n "$mem_rel" ]; then
        local r
        r=$(find_tightest_limit /sys/fs/cgroup/memory "$mem_rel" memory.usage_in_bytes memory.limit_in_bytes)
        if [ -n "$r" ]; then
            CGROUP_MEM_DIR="${r%|*}"
            CGROUP_MEM_MAX_VAL="${r#*|}"
            CGROUP_MEM_CUR_FILE="${CGROUP_MEM_DIR}/memory.usage_in_bytes"
        fi
    fi
    if [ -n "$CGROUP_PIDS_DIR" ] || [ -n "$CGROUP_MEM_DIR" ]; then
        CGROUP_VERSION="v1"
        return 0
    fi
    return 1
}

detect_cgroup() {
    detect_cgroup_v2 || detect_cgroup_v1 || return 1
}

read_effective_pids() {
    local cur="" max=""
    if [ -n "${CGROUP_PIDS_CUR_FILE}" ] && [ -f "${CGROUP_PIDS_CUR_FILE}" ]; then
        read -r cur < "${CGROUP_PIDS_CUR_FILE}" 2>/dev/null || cur=""
        max="${CGROUP_PIDS_MAX_VAL}"
    fi
    if [ -z "$cur" ]; then
        cur="${LAST_PROC_TASKS:-}"
    fi
    if [ -z "$max" ] && [ -f /proc/sys/kernel/threads-max ]; then
        read -r max < /proc/sys/kernel/threads-max 2>/dev/null || max=""
    fi
    [ -n "$cur" ] && [ -n "$max" ] && echo "${cur} ${max}"
}

collect_proc_metrics() {
    local proc_dir task_dir stat_line rest state name
    local total=0 tasks=0 zombies=0 dockerd=0 containerd=0 shim=0 runc=0 docker_cli=0

    for proc_dir in /proc/[0-9]*; do
        [ -d "$proc_dir" ] || continue
        total=$((total + 1))
        if IFS= read -r name < "${proc_dir}/comm" 2>/dev/null; then
            :
        else
            name="?"
        fi
        if IFS= read -r stat_line < "${proc_dir}/stat" 2>/dev/null; then
            rest="${stat_line#*) }"
            state="${rest%% *}"
            [ "$state" = "Z" ] && zombies=$((zombies + 1))
        fi
        for task_dir in "${proc_dir}"/task/[0-9]*; do
            [ -d "$task_dir" ] && tasks=$((tasks + 1))
        done
        case "$name" in
            dockerd) dockerd=$((dockerd + 1)) ;;
            containerd) containerd=$((containerd + 1)) ;;
            containerd-shim*) shim=$((shim + 1)) ;;
            runc) runc=$((runc + 1)) ;;
            docker) docker_cli=$((docker_cli + 1)) ;;
        esac
    done

    LAST_PROC_TOTAL="$total"
    LAST_PROC_TASKS="$tasks"
    LAST_ZOMBIES="$zombies"
    LAST_DOCKERD_PROCS="$dockerd"
    LAST_CONTAINERD_PROCS="$containerd"
    LAST_SHIM_PROCS="$shim"
    LAST_RUNC_PROCS="$runc"
    LAST_DOCKER_CLI_PROCS="$docker_cli"

    local pids cur max pct
    pids="$(read_effective_pids 2>/dev/null || true)"
    cur="${pids%% *}"
    max="${pids##* }"
    pct="?"
    if [ -n "$cur" ] && [ -n "$max" ] && [ "$max" -gt 0 ] 2>/dev/null; then
        pct=$((cur * 100 / max))
    fi
    LAST_PIDS_CUR="$cur"
    LAST_PIDS_MAX="$max"
    LAST_PIDS_PCT="$pct"
}

proc_warn_log() {
    local now msg
    msg="$1"
    now=$(date +%s)
    if [ $((now - LAST_PROC_WARN_TS)) -ge "${PROC_WARN_COOLDOWN_S}" ]; then
        log "$msg"
        LAST_PROC_WARN_TS="$now"
    fi
}

monitor_proc_pressure() {
    collect_proc_metrics

    local docker_related
    docker_related=$((LAST_DOCKERD_PROCS + LAST_CONTAINERD_PROCS + LAST_SHIM_PROCS + LAST_RUNC_PROCS + LAST_DOCKER_CLI_PROCS))

    local pids_free
    pids_free=-1
    if [ -n "${LAST_PIDS_CUR}" ] && [ -n "${LAST_PIDS_MAX}" ] && [ "${LAST_PIDS_MAX}" -gt 0 ] 2>/dev/null; then
        pids_free=$((LAST_PIDS_MAX - LAST_PIDS_CUR))
    fi

    if [ "${LAST_PIDS_PCT}" != "?" ] && [ "${LAST_PIDS_PCT}" -ge "${PIDS_EMERGENCY_PCT}" ] 2>/dev/null; then
        pids_pressure_relief "pids pressure ${LAST_PIDS_CUR}/${LAST_PIDS_MAX} (${LAST_PIDS_PCT}%) before fork failure"
        trigger_repair "pids pressure ${LAST_PIDS_CUR}/${LAST_PIDS_MAX} (${LAST_PIDS_PCT}%) before fork failure"
        return 0
    fi
    if [ "${pids_free}" -ge 0 ] 2>/dev/null && [ "${pids_free}" -lt "${PIDS_EMERGENCY_MIN_FREE}" ] 2>/dev/null; then
        pids_pressure_relief "pids free headroom ${pids_free}<${PIDS_EMERGENCY_MIN_FREE} (${LAST_PIDS_CUR}/${LAST_PIDS_MAX}, ${LAST_PIDS_PCT}%) before fork failure"
        trigger_repair "pids free headroom ${pids_free}<${PIDS_EMERGENCY_MIN_FREE} (${LAST_PIDS_CUR}/${LAST_PIDS_MAX}, ${LAST_PIDS_PCT}%) before fork failure"
        return 0
    fi
    if [ "${LAST_ZOMBIES}" -ge "${ZOMBIE_EMERGENCY}" ] 2>/dev/null; then
        trigger_repair "zombie process pressure zombies=${LAST_ZOMBIES}"
        return 0
    fi
    if [ "${LAST_SHIM_PROCS}" -ge "${SHIM_PROC_EMERGENCY}" ] 2>/dev/null; then
        trigger_repair "containerd-shim process pressure shim=${LAST_SHIM_PROCS}"
        return 0
    fi
    if [ "${LAST_RUNC_PROCS}" -ge "${RUNC_PROC_EMERGENCY}" ] 2>/dev/null; then
        trigger_repair "runc process pressure runc=${LAST_RUNC_PROCS}"
        return 0
    fi
    if [ "${docker_related}" -ge "${DOCKER_PROC_EMERGENCY}" ] 2>/dev/null; then
        trigger_repair "Docker-related process pressure docker_related=${docker_related}"
        return 0
    fi

    if [ "${LAST_PIDS_PCT}" != "?" ] && [ "${LAST_PIDS_PCT}" -ge "${PIDS_WARN_PCT}" ] 2>/dev/null; then
        proc_warn_log "WARN: pids ${LAST_PIDS_CUR}/${LAST_PIDS_MAX} (${LAST_PIDS_PCT}%) tasks=${LAST_PROC_TASKS} procs=${LAST_PROC_TOTAL} zombies=${LAST_ZOMBIES} shim=${LAST_SHIM_PROCS} runc=${LAST_RUNC_PROCS}"
    elif [ "${LAST_ZOMBIES}" -ge "${ZOMBIE_WARN}" ] 2>/dev/null \
       || [ "${LAST_SHIM_PROCS}" -ge "${SHIM_PROC_WARN}" ] 2>/dev/null \
       || [ "${LAST_RUNC_PROCS}" -ge "${RUNC_PROC_WARN}" ] 2>/dev/null \
       || [ "${docker_related}" -ge "${DOCKER_PROC_WARN}" ] 2>/dev/null; then
        proc_warn_log "WARN: process pressure tasks=${LAST_PROC_TASKS} procs=${LAST_PROC_TOTAL} zombies=${LAST_ZOMBIES} docker_related=${docker_related} dockerd=${LAST_DOCKERD_PROCS} containerd=${LAST_CONTAINERD_PROCS} shim=${LAST_SHIM_PROCS} runc=${LAST_RUNC_PROCS} docker_cli=${LAST_DOCKER_CLI_PROCS}"
    fi
}

monitor_pod_cgroup() {
    [ -z "$CGROUP_VERSION" ] && return 0

    if [ -n "$CGROUP_PIDS_DIR" ] && [ -f "$CGROUP_PIDS_CUR_FILE" ]; then
        local cur
        read -r cur < "$CGROUP_PIDS_CUR_FILE" 2>/dev/null
        if [ -n "$cur" ] && [ "$cur" -ge 0 ] 2>/dev/null; then
            local pct=$(( cur * 100 / CGROUP_PIDS_MAX_VAL ))
            local free=$((CGROUP_PIDS_MAX_VAL - cur))
            if [ "$pct" -ge "$PIDS_EMERGENCY_PCT" ]; then
                LAST_PIDS_CUR="$cur"
                LAST_PIDS_MAX="$CGROUP_PIDS_MAX_VAL"
                LAST_PIDS_PCT="$pct"
                pids_pressure_relief "cgroup PIDs ${cur}/${CGROUP_PIDS_MAX_VAL} (${pct}%)"
                trigger_repair "cgroup PIDs ${cur}/${CGROUP_PIDS_MAX_VAL} (${pct}%)"
            elif [ "$free" -lt "$PIDS_EMERGENCY_MIN_FREE" ] 2>/dev/null; then
                LAST_PIDS_CUR="$cur"
                LAST_PIDS_MAX="$CGROUP_PIDS_MAX_VAL"
                LAST_PIDS_PCT="$pct"
                pids_pressure_relief "cgroup PIDs free headroom ${free}<${PIDS_EMERGENCY_MIN_FREE} (${cur}/${CGROUP_PIDS_MAX_VAL}, ${pct}%)"
                trigger_repair "cgroup PIDs free headroom ${free}<${PIDS_EMERGENCY_MIN_FREE} (${cur}/${CGROUP_PIDS_MAX_VAL}, ${pct}%)"
            elif [ "$pct" -ge "$PIDS_WARN_PCT" ]; then
                log "WARN: PIDs ${cur}/${CGROUP_PIDS_MAX_VAL} (${pct}%) — aggressive cleanup"
                timeout 20 docker container prune -f --filter "until=30s" >/dev/null 2>&1 || true
            fi
        fi
    fi

    if [ -n "$CGROUP_MEM_DIR" ] && [ -f "$CGROUP_MEM_CUR_FILE" ]; then
        local cur
        read -r cur < "$CGROUP_MEM_CUR_FILE" 2>/dev/null
        if [ -n "$cur" ] && [ "$cur" -ge 0 ] 2>/dev/null; then
            local pct=$(( cur * 100 / CGROUP_MEM_MAX_VAL ))
            if [ "$pct" -ge "$MEM_EMERGENCY_PCT" ]; then
                emergency_pressure_relief "Memory ${cur}/${CGROUP_MEM_MAX_VAL} (${pct}%)"
            elif [ "$pct" -ge "$MEM_WARN_PCT" ]; then
                log "WARN: Memory ${cur}/${CGROUP_MEM_MAX_VAL} (${pct}%) — pruning"
                timeout 20 docker container prune -f --filter "until=30s" >/dev/null 2>&1 || true
            fi
        fi
    fi
}

# ── pool_server 监控（核心修复：捕捉 dockerd OK 但 /reset 500 的故障形态）──
# 副作用：每次成功调用会更新 LAST_POOL_ACTIVE / LAST_POOL_PENDING / LAST_BRIDGE_NETS
# 供 heartbeat 复用，不重新发起 HTTP/docker 调用。
LAST_POOL_ACTIVE="?"
LAST_POOL_PENDING="?"
LAST_POOL_RESETTING="?"
LAST_POOL_RESET_MAX_AGE="?"
LAST_POOL_PROTECTED_COUNT=0
LAST_BRIDGE_NETS="?"
LAST_POOL_STATUS_TS=0
POOL_RESET_STORM_HIGH_COUNT=0
LAST_POOL_RESET_STORM_REPAIR_TS=0
check_pool_server() {
    local ready_tmp ready_code ready_path ready_failed=0
    ready_path="/readyz"
    ready_tmp="$(mktemp /tmp/pool_ready.XXXXXX 2>/dev/null || echo /tmp/pool_ready.$$)"
    ready_code=$(timeout 5 curl -sS --noproxy '*' -o "$ready_tmp" -w '%{http_code}' \
        "http://${POOL_HOST}:${POOL_PORT}${ready_path}" 2>/dev/null || echo "000")
    if [ "$ready_code" = "404" ]; then
        ready_path="/healthz"
        ready_code=$(timeout 5 curl -sS --noproxy '*' -o "$ready_tmp" -w '%{http_code}' \
            "http://${POOL_HOST}:${POOL_PORT}${ready_path}" 2>/dev/null || echo "000")
    fi
    if [ "$ready_code" = "000" ]; then
        log "WARN: pool_server ${ready_path} unreachable"
        LAST_POOL_ACTIVE="down"
        LAST_POOL_PENDING="down"
        LAST_POOL_RESETTING="down"
        LAST_POOL_RESET_MAX_AGE="down"
        LAST_POOL_PROTECTED_COUNT=0
        LAST_POOL_STATUS_TS=0
        : > "${WATCHDOG_PROTECTED_IDS_FILE}" 2>/dev/null || true
        : > "${WATCHDOG_PROTECTED_NAMES_FILE}" 2>/dev/null || true
        : > "${WATCHDOG_PROTECTED_TRIALS_FILE}" 2>/dev/null || true
        rm -f "$ready_tmp" 2>/dev/null || true
        return 1
    fi
    if [ "$ready_code" -ge 400 ] 2>/dev/null; then
        ready_failed=1
        log "WARN: pool_server ${ready_path} returned HTTP ${ready_code}: $(head -c 300 "$ready_tmp" 2>/dev/null)"
    fi
    rm -f "$ready_tmp" 2>/dev/null || true

    local body pending=0 active=0 active_tasks=0 active_runs=0 protected_count=0 resetting=0 reset_max_age=0
    body=$(timeout 3 curl -fsS --noproxy '*' "http://${POOL_HOST}:${POOL_PORT}/status" 2>/dev/null)
    if [ -n "$body" ]; then
        local status_tmp ids_tmp names_tmp trials_tmp parsed
        status_tmp="$(mktemp /tmp/pool_status.XXXXXX 2>/dev/null || echo /tmp/pool_status.$$)"
        ids_tmp="$(mktemp /tmp/pool_active_ids.XXXXXX 2>/dev/null || echo /tmp/pool_active_ids.$$)"
        names_tmp="$(mktemp /tmp/pool_active_names.XXXXXX 2>/dev/null || echo /tmp/pool_active_names.$$)"
        trials_tmp="$(mktemp /tmp/pool_active_trials.XXXXXX 2>/dev/null || echo /tmp/pool_active_trials.$$)"
        printf '%s' "$body" > "$status_tmp"
        parsed=$(python3 - "$status_tmp" "$ids_tmp" "$names_tmp" "$trials_tmp" <<'PY' 2>/dev/null || true
import json
import sys

status_path, ids_path, names_path, trials_path = sys.argv[1:5]

def clean_values(values):
    out = set()
    for value in values:
        if isinstance(value, str):
            value = value.strip()
            if value:
                out.add(value)
    return out

try:
    with open(status_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    pool = data.get("pool", data)
    pending = int(pool.get("pending_closes", 0) or 0)
    active_tasks = int(pool.get("active_tasks", 0) or 0)
    active_runs = int(pool.get("total_active_runs", 0) or 0)
    phase_counts = pool.get("phase_counts", {})
    resetting = int(phase_counts.get("resetting", 0) or 0) if isinstance(phase_counts, dict) else 0
    reset_max_age = 0.0

    ids = clean_values(pool.get("active_container_ids", []))
    names = clean_values(pool.get("active_container_names", []))
    trials = clean_values(pool.get("active_trial_names", []))
    tasks = pool.get("tasks", {})
    if isinstance(tasks, dict):
        for task_info in tasks.values():
            if not isinstance(task_info, dict):
                continue
            runs = task_info.get("runs", {})
            if not isinstance(runs, dict):
                continue
            for run_info in runs.values():
                if not isinstance(run_info, dict):
                    continue
                container = run_info.get("container", {})
                if run_info.get("phase") == "resetting":
                    try:
                        reset_max_age = max(reset_max_age, float(run_info.get("reset_age_sec", 0) or 0))
                    except Exception:
                        pass
                if not isinstance(container, dict):
                    continue
                ids.update(clean_values([container.get("id"), container.get("short_id")]))
                names.update(clean_values([container.get("name")]))
                trials.update(clean_values([container.get("trial_name")]))

    for path, values in ((ids_path, ids), (names_path, names), (trials_path, trials)):
        with open(path, "w", encoding="utf-8") as fh:
            for value in sorted(values):
                fh.write(value + "\n")
    protected_count = max(len(ids), len(names), len(trials), active_runs)
    print("OK", pending, active_tasks, active_runs, protected_count, resetting, int(reset_max_age))
except Exception:
    for path in (ids_path, names_path, trials_path):
        with open(path, "w", encoding="utf-8") as fh:
            pass
    print("ERR 0 0 0 0 0 0")
PY
)
        set -- ${parsed}
        if [ "${1:-ERR}" = "OK" ]; then
            pending="${2:-0}"
            active_tasks="${3:-0}"
            active_runs="${4:-0}"
            protected_count="${5:-0}"
            resetting="${6:-0}"
            reset_max_age="${7:-0}"
            active="${active_runs}"
            pending="${pending:-0}"
            active="${active:-0}"
            mv -f "$ids_tmp" "${WATCHDOG_PROTECTED_IDS_FILE}" 2>/dev/null || cp "$ids_tmp" "${WATCHDOG_PROTECTED_IDS_FILE}" 2>/dev/null || true
            mv -f "$names_tmp" "${WATCHDOG_PROTECTED_NAMES_FILE}" 2>/dev/null || cp "$names_tmp" "${WATCHDOG_PROTECTED_NAMES_FILE}" 2>/dev/null || true
            mv -f "$trials_tmp" "${WATCHDOG_PROTECTED_TRIALS_FILE}" 2>/dev/null || cp "$trials_tmp" "${WATCHDOG_PROTECTED_TRIALS_FILE}" 2>/dev/null || true
            LAST_POOL_PENDING="$pending"
            LAST_POOL_ACTIVE="$active"
            LAST_POOL_RESETTING="$resetting"
            LAST_POOL_RESET_MAX_AGE="$reset_max_age"
            LAST_POOL_PROTECTED_COUNT="$protected_count"
            LAST_POOL_STATUS_TS="$(date +%s)"
        else
            : > "${WATCHDOG_PROTECTED_IDS_FILE}" 2>/dev/null || true
            : > "${WATCHDOG_PROTECTED_NAMES_FILE}" 2>/dev/null || true
            : > "${WATCHDOG_PROTECTED_TRIALS_FILE}" 2>/dev/null || true
            LAST_POOL_ACTIVE="unknown"
            LAST_POOL_PENDING="unknown"
            LAST_POOL_RESETTING="unknown"
            LAST_POOL_RESET_MAX_AGE="unknown"
            LAST_POOL_PROTECTED_COUNT=0
            LAST_POOL_STATUS_TS=0
        fi
        rm -f "$status_tmp" "$ids_tmp" "$names_tmp" "$trials_tmp" 2>/dev/null || true
        if [ "$pending" -gt "$POOL_PENDING_CLOSES_WARN" ] 2>/dev/null; then
            local active_allows_repair=0
            if [ "$POOL_PENDING_CLOSES_ACTIVE_MAX" -lt 0 ] 2>/dev/null \
               || [ "$active" -le "$POOL_PENDING_CLOSES_ACTIVE_MAX" ] 2>/dev/null; then
                active_allows_repair=1
            fi
            POOL_PENDING_HIGH_COUNT=$((POOL_PENDING_HIGH_COUNT + 1))
            log "WARN: pool_server pending_closes=${pending} (active_runs=${active_runs}, active_tasks=${active_tasks}, protected=${protected_count}, high_count=${POOL_PENDING_HIGH_COUNT}/${POOL_PENDING_CLOSES_STUCK_CHECKS})"
            if [ "$pending" -ge "$POOL_PENDING_CLOSES_REPAIR_THRESHOLD" ] 2>/dev/null \
               && [ "$POOL_PENDING_HIGH_COUNT" -ge "$POOL_PENDING_CLOSES_STUCK_CHECKS" ] 2>/dev/null \
               && [ "$active_allows_repair" = "1" ]; then
                repair_stuck_pool_pending_closes "$pending" "$active"
                POOL_PENDING_HIGH_COUNT=0
            fi
        else
            POOL_PENDING_HIGH_COUNT=0
        fi
        if [ "${POOL_RESET_STORM_REPAIR}" = "1" ] \
           && [ "$active" -gt 0 ] 2>/dev/null \
           && [ "$resetting" -ge "$POOL_RESET_STORM_MIN_RESETTING" ] 2>/dev/null \
           && [ "$reset_max_age" -ge "$POOL_RESET_STORM_MIN_AGE" ] 2>/dev/null \
           && [ $((resetting * 100 / active)) -ge "$POOL_RESET_STORM_RATIO_PCT" ] 2>/dev/null; then
            POOL_RESET_STORM_HIGH_COUNT=$((POOL_RESET_STORM_HIGH_COUNT + 1))
            log "WARN: pool_server reset storm resetting=${resetting}/${active} max_age=${reset_max_age}s high_count=${POOL_RESET_STORM_HIGH_COUNT}/${POOL_RESET_STORM_STUCK_CHECKS}"
            if [ "$POOL_RESET_STORM_HIGH_COUNT" -ge "$POOL_RESET_STORM_STUCK_CHECKS" ] 2>/dev/null; then
                repair_pool_reset_storm "$resetting" "$active" "$reset_max_age"
                POOL_RESET_STORM_HIGH_COUNT=0
            fi
        else
            POOL_RESET_STORM_HIGH_COUNT=0
        fi
    else
        LAST_POOL_ACTIVE="unknown"
        LAST_POOL_PENDING="unknown"
        LAST_POOL_RESETTING="unknown"
        LAST_POOL_RESET_MAX_AGE="unknown"
        LAST_POOL_PROTECTED_COUNT=0
        LAST_POOL_STATUS_TS=0
        : > "${WATCHDOG_PROTECTED_IDS_FILE}" 2>/dev/null || true
        : > "${WATCHDOG_PROTECTED_NAMES_FILE}" 2>/dev/null || true
        : > "${WATCHDOG_PROTECTED_TRIALS_FILE}" 2>/dev/null || true
    fi

    local nets
    nets=$(docker network ls --filter driver=bridge -q 2>/dev/null | wc -l)
    LAST_BRIDGE_NETS="$nets"
    if [ "$nets" -gt "$BRIDGE_NETS_WARN" ] 2>/dev/null; then
        log "WARN: ${nets} bridge networks, address-pool risk; pruning"
        docker_network_prune_safe 30
    fi
    return "${ready_failed}"
}

check_pool_e2e_probe() {
    local probe_tmp probe_code
    if [ "${POOL_E2E_PROBE_INTERVAL}" -le 0 ] 2>/dev/null; then
        return 0
    fi
    if [ -z "${POOL_E2E_PROBE_PAYLOAD_FILE}" ] || [ ! -r "${POOL_E2E_PROBE_PAYLOAD_FILE}" ]; then
        log "WARN: pool E2E probe enabled but payload file is not readable: ${POOL_E2E_PROBE_PAYLOAD_FILE:-<unset>}"
        return 1
    fi

    probe_tmp="$(mktemp /tmp/pool_e2e_probe.XXXXXX 2>/dev/null || echo /tmp/pool_e2e_probe.$$)"
    probe_code=$(timeout "${POOL_E2E_PROBE_TIMEOUT}" curl -sS --noproxy '*' \
        -o "${probe_tmp}" -w '%{http_code}' \
        -X POST -H 'Content-Type: application/json' \
        --data-binary @"${POOL_E2E_PROBE_PAYLOAD_FILE}" \
        "http://${POOL_HOST}:${POOL_PORT}/probe/rollout" 2>/dev/null || echo "000")
    if [ "${probe_code}" -ge 200 ] 2>/dev/null && [ "${probe_code}" -lt 300 ] 2>/dev/null; then
        log "POOL_E2E: probe ok: $(head -c 300 "${probe_tmp}" 2>/dev/null)"
        rm -f "${probe_tmp}" 2>/dev/null || true
        return 0
    fi
    log "WARN: pool E2E probe failed HTTP ${probe_code}: $(head -c 500 "${probe_tmp}" 2>/dev/null)"
    rm -f "${probe_tmp}" 2>/dev/null || true
    return 1
}

monitor_docker_cli() {
    if docker_cli_alive; then
        DOCKER_CLI_FAILS=0
        LAST_DOCKER_CLI_STATUS="ok"
        return 0
    fi

    DOCKER_CLI_FAILS=$((DOCKER_CLI_FAILS + 1))
    LAST_DOCKER_CLI_STATUS="fail"
    log "WARN: docker CLI probe timed out or failed (${DOCKER_CLI_FAILS}/${MAX_CONSECUTIVE_DOCKER_CLI_FAILS}, timeout=${DOCKER_CLI_TIMEOUT}s)"
    if [ "${DOCKER_CLI_FAILS}" -ge "${MAX_CONSECUTIVE_DOCKER_CLI_FAILS}" ]; then
        trigger_repair "docker CLI timeout/failure while dockerd ping may still be ambiguous"
        DOCKER_CLI_FAILS=0
    fi
}

monitor_proxy() {
    if proxy_alive; then
        LAST_PROXY_STATUS="ok"
    else
        LAST_PROXY_STATUS="fail"
        log "WARN: proxy probe failed: ${PROXY_URL}"
    fi
}

stop_pool_server_for_disk_pressure() {
    [ "${POOL_STOP_ON_DISK_EMERGENCY}" = "1" ] || return 0
    local reason="${1:-disk pressure}"
    local now pids
    now=$(date +%s)
    if [ $((now - LAST_POOL_STOP_TS)) -lt "${POOL_STOP_COOLDOWN_S}" ]; then
        log "DISK: pool_server protective stop suppressed by cooldown"
        return 0
    fi
    LAST_POOL_STOP_TS="$now"

    # P1 fix: Use exact match with pgrep -fx to avoid killing unrelated processes
    pids=$(pgrep -f "python.*pool_server\.py" 2>/dev/null || true)
    if [ -z "$pids" ]; then
        log "DISK: pool_server already stopped or not found"
        return 0
    fi

    log "DISK: protective stop of pool_server due to persistent Docker data-root pressure (${reason}): pid(s) ${pids}"
    echo "$pids" | xargs -r kill 2>/dev/null || true
    sleep 5
    echo "$pids" | xargs -r kill -9 2>/dev/null || true
}

# ── stopped 容器清理 ─────────────────────────────────────────────────
cleanup_stopped() {
    local stopped
    stopped=$(docker ps -aq --filter "status=exited" --filter "status=dead" 2>/dev/null | wc -l)
    if [ "${stopped}" -gt 5 ]; then
        log "Cleaning ${stopped} stopped containers..."
        timeout 30 docker container prune -f --filter "until=2m" >/dev/null 2>&1 || true
    fi
    local dn
    dn=$(docker network ls --filter "dangling=true" -q 2>/dev/null | wc -l)
    if [ "${dn}" -gt 0 ]; then
        docker_network_prune_safe 20
    fi
}

# ── Docker data-root 磁盘压力监控 ─────────────────────────────────────
# 不调用 docker system df：在 image/cache 很多或 dockerd 元数据锁竞争时它会卡很久。
# 这里只用 df 快速判断 /data 是否接近爆盘，再做带 timeout 的渐进清理。
docker_disk_stats() {
    local line used avail inode_line inode_used
    line=$(df -P -BG "${DOCKER_DATA_ROOT}" 2>/dev/null | awk 'NR==2 {print $5, $4}')
    [ -n "$line" ] || return 1
    used="${line% *}"
    avail="${line#* }"
    used="${used%\%}"
    avail="${avail%G}"
    inode_line=$(df -Pi "${DOCKER_DATA_ROOT}" 2>/dev/null | awk 'NR==2 {print $5}')
    inode_used="${inode_line%\%}"
    [ -n "$used" ] && [ -n "$avail" ] && [ -n "$inode_used" ] || return 1
    echo "${used} ${avail} ${inode_used}"
}

disk_prune_light() {
    log "DISK: light cleanup: stopped containers + networks + build cache older than ${DISK_BUILD_CACHE_UNTIL}"
    timeout 30 docker container prune -f --filter "until=2m" >/dev/null 2>&1 || true
    docker_network_prune_safe 30
    timeout "${WATCHDOG_PRUNE_TIMEOUT}" docker builder prune -af --filter "until=${DISK_BUILD_CACHE_UNTIL}" >/dev/null 2>&1 || true
    timeout 60 docker image prune -f >/dev/null 2>&1 || true
    docker_storage_gc warn
}

docker_storage_gc() {
    local reason="${1:-disk-pressure}"
    [ "${WATCHDOG_DOCKER_STORAGE_GC}" = "1" ] || return 0
    [ -f "${SCRIPT_DIR}/../ops/docker_storage_gc.py" ] || {
        log "DISK: docker_storage_gc.py not found; skipping LRU image GC"
        return 0
    }
    log "DISK: running Docker storage LRU GC (${reason}); trigger=${DOCKER_GC_TRIGGER_USED_PCT}% target=${DOCKER_GC_TARGET_USED_PCT}% min_free=${DOCKER_GC_MIN_FREE_GB}GB dry_run=${DOCKER_GC_DRY_RUN}"
    timeout "${DOCKER_GC_TIMEOUT}" env \
        DOCKER_DATA_ROOT="${DOCKER_DATA_ROOT}" \
        DOCKER_GC_TRIGGER_USED_PCT="${DOCKER_GC_TRIGGER_USED_PCT}" \
        DOCKER_GC_TARGET_USED_PCT="${DOCKER_GC_TARGET_USED_PCT}" \
        DOCKER_GC_MIN_FREE_GB="${DOCKER_GC_MIN_FREE_GB}" \
        DOCKER_GC_KEEP_PATTERNS="${DOCKER_GC_KEEP_PATTERNS}" \
        DOCKER_GC_PRUNE_VOLUMES="${DOCKER_GC_PRUNE_VOLUMES}" \
        DOCKER_GC_BUILDER_CACHE_UNTIL="${DISK_BUILD_CACHE_UNTIL}" \
        DOCKER_GC_DRY_RUN="${DOCKER_GC_DRY_RUN}" \
        DOCKER_GC_DELETE_OLD_IMAGES="${DOCKER_GC_DELETE_OLD_IMAGES}" \
        python3 "${SCRIPT_DIR}/../ops/docker_storage_gc.py" || \
        log "WARN: Docker storage LRU GC timed out or failed"
}

disk_prune_emergency() {
    local reason="$1"
    emergency_pressure_relief "Docker data-root disk pressure: ${reason}"
    disk_prune_light
    if [ "${WATCHDOG_AGGRESSIVE_IMAGE_PRUNE}" = "1" ] && [ "${WATCHDOG_DOCKER_STORAGE_GC}" != "1" ]; then
        log "DISK: WATCHDOG_AGGRESSIVE_IMAGE_PRUNE=1 and LRU GC disabled, pruning all unused images"
        timeout "${WATCHDOG_PRUNE_TIMEOUT}" docker image prune -af >/dev/null 2>&1 || true
    else
        log "DISK: old-image cleanup handled by LRU GC; set WATCHDOG_DOCKER_STORAGE_GC=0 WATCHDOG_AGGRESSIVE_IMAGE_PRUNE=1 for legacy prune -af"
    fi

    local stats used_pct avail_gb inode_pct
    stats=$(docker_disk_stats 2>/dev/null || true)
    if [ -z "$stats" ]; then
        log "DISK: cannot read stats after emergency cleanup; stopping pool_server defensively"
        stop_pool_server_for_disk_pressure
        return 0
    fi
    used_pct=$(echo "$stats" | awk '{print $1}')
    avail_gb=$(echo "$stats" | awk '{print $2}')
    inode_pct=$(echo "$stats" | awk '{print $3}')
    if [ "${used_pct}" -ge "${DISK_EMERGENCY_PCT}" ] 2>/dev/null \
       || [ "${avail_gb}" -le "${DISK_MIN_FREE_GB}" ] 2>/dev/null \
       || [ "${inode_pct}" -ge "${DISK_INODE_EMERGENCY_PCT}" ] 2>/dev/null; then
        log "DISK: pressure persists after cleanup (${used_pct}% used, ${avail_gb}GB free, inode ${inode_pct}%); stopping pool_server"
        stop_pool_server_for_disk_pressure
    fi
}

monitor_docker_disk() {
    local stats used_pct avail_gb inode_pct now
    stats=$(docker_disk_stats) || {
        log "WARN: cannot read disk stats for ${DOCKER_DATA_ROOT}"
        return 0
    }
    used_pct=$(echo "$stats" | awk '{print $1}')
    avail_gb=$(echo "$stats" | awk '{print $2}')
    inode_pct=$(echo "$stats" | awk '{print $3}')

    if [ "${used_pct}" -ge "${DISK_EMERGENCY_PCT}" ] 2>/dev/null \
       || [ "${avail_gb}" -le "${DISK_MIN_FREE_GB}" ] 2>/dev/null \
       || [ "${inode_pct}" -ge "${DISK_INODE_EMERGENCY_PCT}" ] 2>/dev/null; then
        now=$(date +%s)
        if [ $((now - LAST_DISK_PRUNE_TS)) -lt "${DISK_PRUNE_COOLDOWN_S}" ]; then
            log "DISK: emergency condition persists (${used_pct}% used, ${avail_gb}GB free, inode ${inode_pct}%), cleanup cooldown active"
            stop_pool_server_for_disk_pressure
            return 0
        fi
        LAST_DISK_PRUNE_TS="$now"
        disk_prune_emergency "${used_pct}% used, ${avail_gb}GB free, inode ${inode_pct}%"
        return 0
    fi

    if [ "${used_pct}" -ge "${DISK_WARN_PCT}" ] 2>/dev/null \
       || [ "${inode_pct}" -ge "${DISK_INODE_WARN_PCT}" ] 2>/dev/null; then
        now=$(date +%s)
        if [ $((now - LAST_DISK_PRUNE_TS)) -lt "${DISK_PRUNE_COOLDOWN_S}" ]; then
            log "DISK: warn ${used_pct}% used, ${avail_gb}GB free, inode ${inode_pct}%; cleanup cooldown active"
            return 0
        fi
        LAST_DISK_PRUNE_TS="$now"
        log "DISK: warn ${used_pct}% used, ${avail_gb}GB free, inode ${inode_pct}%"
        disk_prune_light
    fi
}

# ── 运行容器数上限（双闸门，排除 pool_server）─────────────────────────
# 副作用：更新 LAST_RUNNING_TASKS 供 heartbeat 复用
LAST_RUNNING_TASKS="?"
pool_status_age_seconds() {
    local now
    now=$(date +%s)
    if [ "${LAST_POOL_STATUS_TS:-0}" -le 0 ] 2>/dev/null; then
        echo "unknown"
        return 0
    fi
    echo $((now - LAST_POOL_STATUS_TS))
}

pool_status_is_fresh() {
    local age max_age
    [[ "${LAST_POOL_ACTIVE}" =~ ^[0-9]+$ ]] || return 1
    age="$(pool_status_age_seconds)"
    [[ "${age}" =~ ^[0-9]+$ ]] || return 1
    max_age=$((POOL_CHECK_INTERVAL * 3))
    [ "${age}" -le "${max_age}" ] 2>/dev/null
}

min_int() {
    local a="$1"
    local b="$2"
    if [ "$a" -lt "$b" ] 2>/dev/null; then
        echo "$a"
    else
        echo "$b"
    fi
}

task_container_reap_target() {
    local target active_floor
    target="${MAX_RUNNING_CONTAINERS}"
    if [[ "${LAST_POOL_ACTIVE}" =~ ^[0-9]+$ ]]; then
        active_floor=$((LAST_POOL_ACTIVE + WATCHDOG_REAP_HEADROOM))
        if [ "$active_floor" -gt "$target" ] 2>/dev/null; then
            target="$active_floor"
        fi
    fi
    echo "$target"
}

reap_unprotected_task_containers() {
    local reason="$1"
    local limit="${2:-0}"
    local min_age="${3:-900}"
    local mode="${4:-soft}"
    local require_idle="${5:-1}"
    local ps_tmp stats_tmp kill_tmp sample_tmp summary rc total protected candidates selected sample

    ps_tmp="$(mktemp /tmp/watchdog_task_ps.XXXXXX 2>/dev/null || echo /tmp/watchdog_task_ps.$$)"
    stats_tmp="$(mktemp /tmp/watchdog_task_stats.XXXXXX 2>/dev/null || echo /tmp/watchdog_task_stats.$$)"
    kill_tmp="$(mktemp /tmp/watchdog_task_kill.XXXXXX 2>/dev/null || echo /tmp/watchdog_task_kill.$$)"
    sample_tmp="$(mktemp /tmp/watchdog_task_sample.XXXXXX 2>/dev/null || echo /tmp/watchdog_task_sample.$$)"

    if ! timeout "${DOCKER_CLI_TIMEOUT}" docker ps -a --format '{{.CreatedAt}}\t{{.ID}}\t{{.Names}}\t{{.Image}}\t{{.Status}}' > "$ps_tmp" 2>/dev/null; then
        log "REAP: docker ps failed; cannot build task container inventory (reason=${reason})"
        rm -f "$ps_tmp" "$stats_tmp" "$kill_tmp" "$sample_tmp" 2>/dev/null || true
        return 1
    fi
    timeout "${WATCHDOG_STATS_TIMEOUT}" docker stats --no-stream --format '{{.Container}}\t{{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}' > "$stats_tmp" 2>/dev/null || true

    summary=$(python3 - "$ps_tmp" "$stats_tmp" \
        "${WATCHDOG_PROTECTED_IDS_FILE}" \
        "${WATCHDOG_PROTECTED_NAMES_FILE}" \
        "${WATCHDOG_PROTECTED_TRIALS_FILE}" \
        "$kill_tmp" "$sample_tmp" \
        "$limit" "$min_age" "$mode" "$require_idle" \
        "$TASK_CONTAINER_REGEX" "$TASK_IMAGE_REGEX" \
        "$WATCHDOG_STALE_LOW_CPU_PCT" "$WATCHDOG_STALE_LOW_MEM_MB" <<'PY' 2>/dev/null
import datetime as _dt
import re
import sys
import time

(
    ps_path,
    stats_path,
    protected_ids_path,
    protected_names_path,
    protected_trials_path,
    kill_path,
    sample_path,
    limit_raw,
    min_age_raw,
    mode,
    require_idle_raw,
    name_re_raw,
    image_re_raw,
    low_cpu_raw,
    low_mem_raw,
) = sys.argv[1:16]

def read_set(path):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return {line.strip() for line in fh if line.strip()}
    except OSError:
        return set()

def docker_name_variants(value):
    if not value:
        return set()
    raw = value.strip()
    if not raw:
        return set()
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", raw).strip("-_.")
    variants = {
        raw,
        cleaned,
        cleaned.replace(".", "-"),
        cleaned.replace("_", "-"),
        cleaned.replace(".", "_"),
    }
    return {v for v in variants if v and "slime-run" in v}

def compile_re(raw):
    try:
        return re.compile(raw)
    except re.error:
        return re.compile(r"a^")

def parse_int(raw, default):
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return default

def parse_float(raw, default=None):
    try:
        return float(str(raw).strip().rstrip("%"))
    except (TypeError, ValueError):
        return default

def parse_mem_mb(raw):
    left = str(raw or "").split("/", 1)[0].strip()
    m = re.match(r"^([0-9.]+)\s*([KMGT]?i?B)?$", left, re.I)
    if not m:
        return None
    value = parse_float(m.group(1), None)
    if value is None:
        return None
    unit = (m.group(2) or "B").lower()
    factors = {
        "b": 1.0 / (1024 * 1024),
        "kb": 1.0 / 1024,
        "kib": 1.0 / 1024,
        "mb": 1.0,
        "mib": 1.0,
        "gb": 1024.0,
        "gib": 1024.0,
        "tb": 1024.0 * 1024.0,
        "tib": 1024.0 * 1024.0,
    }
    return value * factors.get(unit, 1.0)

def parse_created_epoch(raw):
    parts = str(raw or "").split()
    if len(parts) >= 3:
        try:
            return int(_dt.datetime.strptime(" ".join(parts[:3]), "%Y-%m-%d %H:%M:%S %z").timestamp())
        except (ValueError, OverflowError):
            return 0
    return 0

def id_is_protected(container_id, protected_ids):
    if not container_id:
        return False
    for protected_id in protected_ids:
        if container_id == protected_id or container_id.startswith(protected_id) or protected_id.startswith(container_id):
            return True
    return False

def name_is_project_match(name, projects):
    if not name:
        return False
    for project in projects:
        if name == project or name.startswith(f"{project}-") or name.startswith(f"{project}_") or name.startswith(project):
            return True
    return False

limit = parse_int(limit_raw, 0)
min_age = parse_int(min_age_raw, 900)
require_idle = str(require_idle_raw) == "1"
low_cpu = parse_float(low_cpu_raw, 1.0)
low_mem = parse_float(low_mem_raw, 1024.0)
name_re = compile_re(name_re_raw)
image_re = compile_re(image_re_raw)
now = int(time.time())

protected_ids = read_set(protected_ids_path)
protected_names = read_set(protected_names_path)
protected_trials = read_set(protected_trials_path)
protected_projects = set()
for value in protected_names | protected_trials:
    protected_projects.update(docker_name_variants(value))

stats = {}
try:
    with open(stats_path, "r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            cid, name, cpu_raw, mem_raw = parts[:4]
            item = (parse_float(cpu_raw, None), parse_mem_mb(mem_raw), mem_raw)
            if cid:
                stats[cid] = item
            if name:
                stats[name] = item
except OSError:
    pass

total = 0
protected = 0
candidates = []

try:
    lines = open(ps_path, "r", encoding="utf-8").read().splitlines()
except OSError:
    lines = []

for line in lines:
    parts = line.split("\t", 4)
    if len(parts) < 5:
        continue
    created_raw, cid, name, image, status = parts
    if not (name_re.search(name or "") or image_re.search(image or "")):
        continue
    total += 1
    if id_is_protected(cid, protected_ids) or name in protected_names or name_is_project_match(name, protected_projects):
        protected += 1
        continue

    created_epoch = parse_created_epoch(created_raw)
    age = max(0, now - created_epoch) if created_epoch else 0
    stat = stats.get(cid) or stats.get(name) or (None, None, "")
    cpu_pct, mem_mb, mem_raw = stat
    is_low_cpu = cpu_pct is not None and cpu_pct <= low_cpu
    is_low_mem = mem_mb is not None and mem_mb <= low_mem
    is_idle = is_low_cpu and is_low_mem
    status_l = (status or "").lower()
    priority = None
    why = ""

    if "dead" in status_l or "exited" in status_l or "removing" in status_l:
        priority = 0
        why = "stopped"
    elif "created" in status_l:
        if age >= min_age:
            priority = 1
            why = "created_old"
    elif status_l.startswith("up"):
        if age >= min_age and (not require_idle or is_idle):
            priority = 2 if is_idle else 3
            why = "running_idle" if is_idle else "running_old"
    elif age >= min_age and (not require_idle or is_idle):
        priority = 4
        why = "unknown_old"

    if priority is None:
        continue
    cpu_text = "?" if cpu_pct is None else f"{cpu_pct:.2f}%"
    mem_text = "?" if mem_mb is None else f"{mem_mb:.1f}MiB"
    sample = f"{cid} name={name} age={age}s status={status} cpu={cpu_text} mem={mem_text} why={why}"
    candidates.append((priority, created_epoch or now, cid, sample))

candidates.sort(key=lambda item: (item[0], item[1], item[2]))
selected = candidates[:limit] if limit > 0 else candidates
with open(kill_path, "w", encoding="utf-8") as fh:
    for _, _, cid, _ in selected:
        fh.write(cid + "\n")
with open(sample_path, "w", encoding="utf-8") as fh:
    for _, _, _, sample in selected[:8]:
        fh.write(sample + "\n")
print(total, protected, len(candidates), len(selected))
PY
)
    rc=$?
    if [ "$rc" -ne 0 ] || [ -z "$summary" ]; then
        log "REAP: inventory parser failed (reason=${reason}, mode=${mode})"
        rm -f "$ps_tmp" "$stats_tmp" "$kill_tmp" "$sample_tmp" 2>/dev/null || true
        return 1
    fi

    set -- ${summary}
    total="${1:-0}"
    protected="${2:-0}"
    candidates="${3:-0}"
    selected="${4:-0}"
    if [ "$selected" -le 0 ] 2>/dev/null; then
        log "REAP: no eligible unprotected task containers (reason=${reason}, mode=${mode}, total=${total}, protected=${protected}, candidates=${candidates}, min_age=${min_age}s, require_idle=${require_idle})"
        rm -f "$ps_tmp" "$stats_tmp" "$kill_tmp" "$sample_tmp" 2>/dev/null || true
        return 1
    fi

    sample="$(tr '\n' ';' < "$sample_tmp" 2>/dev/null | head -c 700)"
    log "REAP: removing ${selected} unprotected task containers (reason=${reason}, mode=${mode}, total=${total}, protected=${protected}, candidates=${candidates}, min_age=${min_age}s, require_idle=${require_idle}) sample=${sample}"
    xargs -r -n 10 timeout 30 docker rm -f < "$kill_tmp" >/dev/null 2>&1 || true
    rm -f "$ps_tmp" "$stats_tmp" "$kill_tmp" "$sample_tmp" 2>/dev/null || true
    return 0
}

reap_reset_storm_orphan_task_containers() {
    local running="$1"
    local active_gap ratio limit now

    [ "${WATCHDOG_RESET_STORM_ORPHAN_REAP_ENABLED}" = "1" ] || return 0
    pool_status_is_fresh || return 0
    [[ "${LAST_POOL_ACTIVE}" =~ ^[0-9]+$ ]] || return 0
    [[ "${LAST_POOL_RESETTING}" =~ ^[0-9]+$ ]] || return 0
    [[ "${LAST_POOL_RESET_MAX_AGE}" =~ ^[0-9]+$ ]] || return 0
    [ "${LAST_POOL_ACTIVE}" -gt 0 ] 2>/dev/null || return 0
    [ "${LAST_POOL_RESETTING}" -ge "${POOL_RESET_STORM_MIN_RESETTING}" ] 2>/dev/null || return 0
    [ "${LAST_POOL_RESET_MAX_AGE}" -ge "${WATCHDOG_RESET_STORM_ORPHAN_REAP_MIN_RESET_AGE}" ] 2>/dev/null || return 0

    ratio=$((LAST_POOL_RESETTING * 100 / LAST_POOL_ACTIVE))
    [ "${ratio}" -ge "${POOL_RESET_STORM_RATIO_PCT}" ] 2>/dev/null || return 0

    active_gap=$((LAST_POOL_ACTIVE - running))
    [ "${active_gap}" -ge "${WATCHDOG_RESET_STORM_ORPHAN_REAP_ACTIVE_GAP}" ] 2>/dev/null || return 0

    now=$(date +%s)
    if [ $((now - LAST_RESET_STORM_ORPHAN_REAP_TS)) -lt "${WATCHDOG_RESET_STORM_ORPHAN_REAP_COOLDOWN_S}" ]; then
        log "Reset-storm orphan reap suppressed: running=${running} active=${LAST_POOL_ACTIVE} resetting=${LAST_POOL_RESETTING} reset_max_age=${LAST_POOL_RESET_MAX_AGE}s active_gap=${active_gap} cooldown=${WATCHDOG_RESET_STORM_ORPHAN_REAP_COOLDOWN_S}s"
        return 0
    fi
    LAST_RESET_STORM_ORPHAN_REAP_TS="${now}"

    limit="$(min_int "${active_gap}" "${WATCHDOG_RESET_STORM_ORPHAN_REAP_BATCH}")"
    log "Reset-storm orphan reap: running=${running} active=${LAST_POOL_ACTIVE} resetting=${LAST_POOL_RESETTING} reset_max_age=${LAST_POOL_RESET_MAX_AGE}s active_gap=${active_gap}; reaping up to ${limit} old idle unprotected containers"
    reap_unprotected_task_containers \
        "reset storm orphan gap running=${running} active=${LAST_POOL_ACTIVE} resetting=${LAST_POOL_RESETTING}" \
        "${limit}" \
        "${WATCHDOG_RESET_STORM_ORPHAN_REAP_MIN_AGE}" \
        "reset_storm_orphan" \
        1 || true
}

reap_idle_orphan_task_containers() {
    local running="$1"
    local idle_target idle_excess limit now

    [ "${WATCHDOG_IDLE_REAP_ENABLED}" = "1" ] || return 0
    pool_status_is_fresh || return 0
    [ "${running}" -ge "${WATCHDOG_IDLE_REAP_MIN_CONTAINERS}" ] 2>/dev/null || return 0

    idle_target=$((LAST_POOL_ACTIVE + WATCHDOG_REAP_HEADROOM))
    if [ "${idle_target}" -lt "${WATCHDOG_IDLE_REAP_MIN_CONTAINERS}" ] 2>/dev/null; then
        idle_target="${WATCHDOG_IDLE_REAP_MIN_CONTAINERS}"
    fi

    idle_excess=$((running - idle_target))
    [ "${idle_excess}" -ge "${WATCHDOG_IDLE_REAP_MIN_GAP}" ] 2>/dev/null || return 0

    now=$(date +%s)
    if [ $((now - LAST_IDLE_REAP_TS)) -lt "${WATCHDOG_IDLE_REAP_COOLDOWN_S}" ]; then
        log "Idle orphan reap suppressed: running=${running} active=${LAST_POOL_ACTIVE} target=${idle_target} excess=${idle_excess} cooldown=${WATCHDOG_IDLE_REAP_COOLDOWN_S}s"
        return 0
    fi
    LAST_IDLE_REAP_TS="${now}"

    limit="$(min_int "${idle_excess}" "${WATCHDOG_IDLE_REAP_BATCH}")"
    log "Idle orphan reap: running=${running} active=${LAST_POOL_ACTIVE} protected=${LAST_POOL_PROTECTED_COUNT} target=${idle_target} excess=${idle_excess}; reaping up to ${limit} old idle unprotected containers"
    reap_unprotected_task_containers \
        "idle orphan gap running=${running} active=${LAST_POOL_ACTIVE} target=${idle_target}" \
        "${limit}" \
        "${WATCHDOG_IDLE_REAP_MIN_AGE}" \
        "idle_orphan" \
        1 || true
}

enforce_container_limit() {
    local running status_age target excess limit pressure_reason reap_min_age
    # 只统计 task 容器（带数字前缀 + client/helper 后缀），不算 pool_server 等基础容器
    running=$(task_container_count 2>/dev/null || echo 0)
    LAST_RUNNING_TASKS="$running"

    if [ "${running}" -gt "${HARD_KILL_THRESHOLD}" ] 2>/dev/null \
       || [ "${LAST_SHIM_PROCS:-0}" -ge "${SHIM_PROC_WARN}" ] 2>/dev/null; then
        if [ "${running}" -gt "${HARD_KILL_THRESHOLD}" ] 2>/dev/null; then
            pressure_reason="hard task container limit ${running}>${HARD_KILL_THRESHOLD}"
            reap_min_age="${WATCHDOG_STALE_MIN_AGE_HARD}"
            if [ "${LAST_SHIM_PROCS:-0}" -ge "${SHIM_PROC_WARN}" ] 2>/dev/null; then
                pressure_reason="${pressure_reason}; shim pressure ${LAST_SHIM_PROCS:-0}>=${SHIM_PROC_WARN}"
            fi
        else
            pressure_reason="shim pressure ${LAST_SHIM_PROCS:-0}>=${SHIM_PROC_WARN}"
            reap_min_age="${WATCHDOG_STALE_MIN_AGE_PRESSURE}"
        fi
        if pool_status_is_fresh; then
            target="$(task_container_reap_target)"
            excess=$((running - target))
            if [ "$excess" -le 0 ] 2>/dev/null; then
                log "HARD/PRESSURE: ${pressure_reason}, but running=${running} <= protected target=${target} (pool active=${LAST_POOL_ACTIVE}, protected=${LAST_POOL_PROTECTED_COUNT}); no task reap"
                return
            fi
            limit="$(min_int "$excess" "$WATCHDOG_HARD_REAP_BATCH")"
            log "HARD/PRESSURE: ${pressure_reason}; running=${running} target=${target} pool active=${LAST_POOL_ACTIVE} protected=${LAST_POOL_PROTECTED_COUNT}; reaping up to ${limit}"
            reap_unprotected_task_containers "${pressure_reason}" "${limit}" "${reap_min_age}" "hard" 0 || true
        else
            status_age="$(pool_status_age_seconds)"
            excess=$((running - MAX_RUNNING_CONTAINERS))
            if [ "$excess" -le 0 ] 2>/dev/null; then
                log "HARD/PRESSURE suppressed: ${pressure_reason}, pool status stale active=${LAST_POOL_ACTIVE} status_age=${status_age}s and running=${running} <= ${MAX_RUNNING_CONTAINERS}"
                return
            fi
            limit="$(min_int "$excess" "$WATCHDOG_HARD_REAP_BATCH")"
            log "HARD/PRESSURE conservative reap: ${pressure_reason}; pool status stale active=${LAST_POOL_ACTIVE} status_age=${status_age}s; only very old idle unprotected containers are eligible (limit=${limit})"
            reap_unprotected_task_containers "${pressure_reason}; stale pool status" "${limit}" "${WATCHDOG_STALE_STATUS_MIN_AGE}" "stale_status" 1 || true
        fi
        return
    fi

    reap_reset_storm_orphan_task_containers "${running}"

    if [ "${running}" -le "${MAX_RUNNING_CONTAINERS}" ] 2>/dev/null; then
        reap_idle_orphan_task_containers "${running}"
    fi

    if [ "${running}" -gt "${MAX_RUNNING_CONTAINERS}" ]; then
        if pool_status_is_fresh; then
            target="$(task_container_reap_target)"
            excess=$((running - target))
            if [ "$excess" -le 0 ] 2>/dev/null; then
                log "Soft limit: ${running} task containers > ${MAX_RUNNING_CONTAINERS}, but running <= protected target=${target} (pool active=${LAST_POOL_ACTIVE}, protected=${LAST_POOL_PROTECTED_COUNT}); no task reap"
                return
            fi
            limit="$(min_int "$excess" "$WATCHDOG_SOFT_REAP_BATCH")"
            log "Soft limit: ${running} task containers > ${MAX_RUNNING_CONTAINERS}; running=${running} target=${target} pool active=${LAST_POOL_ACTIVE} protected=${LAST_POOL_PROTECTED_COUNT}; reaping up to ${limit} old idle unprotected containers"
            reap_unprotected_task_containers "soft task container limit ${running}>${MAX_RUNNING_CONTAINERS}" "${limit}" "${WATCHDOG_STALE_MIN_AGE_SOFT}" "soft" 1 || true
        else
            status_age="$(pool_status_age_seconds)"
            log "Soft limit suppressed: ${running} task containers > ${MAX_RUNNING_CONTAINERS}, pool status stale active=${LAST_POOL_ACTIVE} status_age=${status_age}s; not killing possible active rollout containers"
        fi
    fi
}

# ── dockerd 重启（绕过 systemctl restart，沿用 restart_docker_force.sh 模式）──
restart_docker() {
    log "Docker daemon is DOWN. Attempting forced restart (no systemctl restart)..."
    collect_proc_metrics
    repair_snapshot

    if [ "${LAST_SHIM_PROCS:-0}" -ge "${DOCKER_DOWN_SHIM_RELIEF}" ] 2>/dev/null; then
        log "Docker is down with ${LAST_SHIM_PROCS} containerd-shim processes (threshold=${DOCKER_DOWN_SHIM_RELIEF}); stopping pool_server before restart"
        stop_pool_server_for_pressure "dockerd down with shim pressure"
    fi

    # 1) 阻断 systemd auto-restart：reset-failed + stop docker.socket
    timeout 5 systemctl reset-failed docker.service docker.socket 2>/dev/null || true
    timeout 5 systemctl stop docker.socket 2>/dev/null || true

    # 2) pkill -9 dockerd; optionally clear shim processes once Docker is confirmed down.
    pkill -9 -x dockerd 2>/dev/null || true
    sleep 2
    if pgrep -x dockerd >/dev/null 2>&1; then
        log "WARN: dockerd still alive after SIGKILL (D state?), aborting state cleanup"
        return 1
    fi

    if [ "${WATCHDOG_KILL_SHIMS_ON_DOCKER_DOWN}" = "1" ] \
       && [ "${LAST_SHIM_PROCS:-0}" -ge "${DOCKER_DOWN_SHIM_RELIEF}" ] 2>/dev/null; then
        local shim_pids shim_n
        shim_pids="$(pgrep -f containerd-shim 2>/dev/null || true)"
        if [ -n "${shim_pids}" ]; then
            shim_n="$(printf '%s\n' "${shim_pids}" | wc -l)"
            log "Killing ${shim_n} containerd-shim processes because dockerd is down and shim pressure is high"
            printf '%s\n' "${shim_pids}" | xargs -r kill -9 2>/dev/null || true
            sleep 1
        fi
    fi

    rm -f /var/run/docker.pid "${DOCKER_SOCK}"

    # 3) 清理 container state。正常情况下仅在 no-shim 时清；Docker 已 down 且 shim 压力高时，
    #    这些 state 已无法安全恢复，清理后让 dockerd 干净启动。
    if [ "$HOST_PID_NS" = "1" ]; then
        if ! pgrep -f containerd-shim >/dev/null 2>&1 \
           || { [ "${WATCHDOG_KILL_SHIMS_ON_DOCKER_DOWN}" = "1" ] \
                && [ "${LAST_SHIM_PROCS:-0}" -ge "${DOCKER_DOWN_SHIM_RELIEF}" ] 2>/dev/null; }; then
            if [ -d "${DOCKER_DATA_ROOT}/containers" ]; then
                local n
                n=$(find "${DOCKER_DATA_ROOT}/containers" -maxdepth 1 -mindepth 1 2>/dev/null | wc -l)
                if [ "${n}" -gt 0 ]; then
                    log "Clearing ${n} stale container states before dockerd restart"
                    rm -rf "${DOCKER_DATA_ROOT}/containers"/* 2>/dev/null || true
                fi
            fi
            rm -f "${DOCKER_DATA_ROOT}/network/files/local-kv.db" 2>/dev/null || true
        else
            log "containerd-shim still alive, preserving container state"
        fi
    else
        log "Not in host PID namespace; cannot reliably enumerate shim — skipping state cleanup"
    fi

    # 4) 启动 dockerd（直接 nohup，不走 systemd）
    if [ -f "${PROXY_ENV_FILE}" ]; then
        # shellcheck disable=SC1090
        set -a; . "${PROXY_ENV_FILE}"; set +a
        log "Loaded proxy env from ${PROXY_ENV_FILE} before dockerd restart"
    else
        export HTTP_PROXY="${PROXY_URL}" HTTPS_PROXY="${PROXY_URL}"
        export http_proxy="${PROXY_URL}" https_proxy="${PROXY_URL}"
        export NO_PROXY="${NO_PROXY_LIST}" no_proxy="${NO_PROXY_LIST}"
        log "Proxy env file missing; exported PROXY_URL for dockerd restart: ${PROXY_URL}"
    fi
    nohup dockerd --containerd=/run/containerd/containerd.sock \
        > /tmp/dockerd_watchdog_restart.log 2>&1 &
    local pid=$!
    log "Started dockerd PID=${pid}, waiting for API..."

    local i
    for i in $(seq 1 60); do
        if docker_alive; then
            log "Docker API ready after ${i} attempts (~$((i*5))s)"
            return 0
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            log "ERROR: dockerd died during startup; tail log:"
            tail -20 /tmp/dockerd_watchdog_restart.log 2>/dev/null | sed 's/^/  /' || true
            return 1
        fi
        sleep 5
    done
    log "ERROR: Docker failed to start after 5 min. See /tmp/dockerd_watchdog_restart.log"
    return 1
}

# ── Main ─────────────────────────────────────────────────────────────
log "========================================"
log "Starting docker_watchdog PID=$$"
log "  MAX_RUNNING=${MAX_RUNNING_CONTAINERS}  HARD_KILL=${HARD_KILL_THRESHOLD}"
log "  health every ${HEALTH_CHECK_INTERVAL}s; cgroup every ${CGROUP_MONITOR_INTERVAL}s; proc every ${PROC_MONITOR_INTERVAL}s"
log "  docker-cli every ${DOCKER_CLI_CHECK_INTERVAL}s timeout=${DOCKER_CLI_TIMEOUT}s fail_trigger=${MAX_CONSECUTIVE_DOCKER_CLI_FAILS}; proxy every ${PROXY_CHECK_INTERVAL}s"
log "  pool every ${POOL_CHECK_INTERVAL}s; deep probe every ${DEEP_PROBE_INTERVAL}s"
log "  heartbeat every ${HEARTBEAT_INTERVAL}s"
log "  disk every ${DISK_CHECK_INTERVAL}s; warn=${DISK_WARN_PCT}% emerg=${DISK_EMERGENCY_PCT}% min_free=${DISK_MIN_FREE_GB}GB inode_warn=${DISK_INODE_WARN_PCT}% inode_emerg=${DISK_INODE_EMERGENCY_PCT}%"
log "  pool_stop_on_disk_emergency=${POOL_STOP_ON_DISK_EMERGENCY}"
log "  PIDs warn=${PIDS_WARN_PCT}% emerg=${PIDS_EMERGENCY_PCT}% emerg_min_free=${PIDS_EMERGENCY_MIN_FREE}"
log "  proc warn: docker_related=${DOCKER_PROC_WARN} shim=${SHIM_PROC_WARN} runc=${RUNC_PROC_WARN} zombies=${ZOMBIE_WARN}"
log "  proc emerg: docker_related=${DOCKER_PROC_EMERGENCY} shim=${SHIM_PROC_EMERGENCY} runc=${RUNC_PROC_EMERGENCY} zombies=${ZOMBIE_EMERGENCY}"
log "  docker_down_shim_relief=${DOCKER_DOWN_SHIM_RELIEF} kill_shims_on_docker_down=${WATCHDOG_KILL_SHIMS_ON_DOCKER_DOWN}"
log "  Mem  warn=${MEM_WARN_PCT}% emerg=${MEM_EMERGENCY_PCT}%"
log "  pool=${POOL_HOST}:${POOL_PORT}  pool_server_regex=${POOL_SERVER_NAME_REGEX}"
log "  pool_ready restart_failures=${POOL_READY_FAILS_RESTART} restart_active_max=${POOL_RESTART_ACTIVE_MAX} restart_cooldown=${POOL_RESTART_COOLDOWN_S}s stop_launcher=${WATCHDOG_STOP_POOL_LAUNCHER}"
log "  pool_e2e interval=${POOL_E2E_PROBE_INTERVAL}s timeout=${POOL_E2E_PROBE_TIMEOUT}s fail_trigger=${POOL_E2E_PROBE_FAILS_RESTART} payload=${POOL_E2E_PROBE_PAYLOAD_FILE:-<unset>}"
log "  pool_pending repair=${POOL_PENDING_CLOSES_REPAIR} warn=${POOL_PENDING_CLOSES_WARN} threshold=${POOL_PENDING_CLOSES_REPAIR_THRESHOLD} stuck_checks=${POOL_PENDING_CLOSES_STUCK_CHECKS} active_max=${POOL_PENDING_CLOSES_ACTIVE_MAX} reap_limit=${POOL_PENDING_CLOSES_REAP_LIMIT} cooldown=${POOL_PENDING_CLOSES_REPAIR_COOLDOWN_S}s cancel_api=${POOL_PENDING_CLOSES_CANCEL_API} cancel_timeout=${POOL_PENDING_CLOSES_CANCEL_TIMEOUT}s kill_when_active=${POOL_PENDING_CLOSES_KILL_CONTAINERS_WHEN_ACTIVE}"
log "  pool_resetstorm repair=${POOL_RESET_STORM_REPAIR} min_resetting=${POOL_RESET_STORM_MIN_RESETTING} ratio=${POOL_RESET_STORM_RATIO_PCT}% min_age=${POOL_RESET_STORM_MIN_AGE}s stuck_checks=${POOL_RESET_STORM_STUCK_CHECKS} limit=${POOL_RESET_STORM_REPAIR_LIMIT} cooldown=${POOL_RESET_STORM_REPAIR_COOLDOWN_S}s"
log "  task_container_regex=${TASK_CONTAINER_REGEX}"
log "  task_image_regex=${TASK_IMAGE_REGEX}"
log "  task_reap headroom=${WATCHDOG_REAP_HEADROOM} soft_batch=${WATCHDOG_SOFT_REAP_BATCH} hard_batch=${WATCHDOG_HARD_REAP_BATCH} soft_age=${WATCHDOG_STALE_MIN_AGE_SOFT}s pressure_age=${WATCHDOG_STALE_MIN_AGE_PRESSURE}s hard_age=${WATCHDOG_STALE_MIN_AGE_HARD}s stale_status_age=${WATCHDOG_STALE_STATUS_MIN_AGE}s low_cpu=${WATCHDOG_STALE_LOW_CPU_PCT}% low_mem=${WATCHDOG_STALE_LOW_MEM_MB}MiB stats_timeout=${WATCHDOG_STATS_TIMEOUT}s"
log "  idle_orphan_reap enabled=${WATCHDOG_IDLE_REAP_ENABLED} min_containers=${WATCHDOG_IDLE_REAP_MIN_CONTAINERS} min_gap=${WATCHDOG_IDLE_REAP_MIN_GAP} batch=${WATCHDOG_IDLE_REAP_BATCH} min_age=${WATCHDOG_IDLE_REAP_MIN_AGE}s cooldown=${WATCHDOG_IDLE_REAP_COOLDOWN_S}s"
log "  reset_storm_orphan_reap enabled=${WATCHDOG_RESET_STORM_ORPHAN_REAP_ENABLED} active_gap=${WATCHDOG_RESET_STORM_ORPHAN_REAP_ACTIVE_GAP} batch=${WATCHDOG_RESET_STORM_ORPHAN_REAP_BATCH} min_age=${WATCHDOG_RESET_STORM_ORPHAN_REAP_MIN_AGE}s min_reset_age=${WATCHDOG_RESET_STORM_ORPHAN_REAP_MIN_RESET_AGE}s cooldown=${WATCHDOG_RESET_STORM_ORPHAN_REAP_COOLDOWN_S}s"
log "  docker_data_root=${DOCKER_DATA_ROOT}  proxy_url=${PROXY_URL}  proxy_env_file=${PROXY_ENV_FILE}"
log "  auto_repair=${WATCHDOG_AUTO_REPAIR} repair_mode=${WATCHDOG_REPAIR_MODE} repair_cooldown=${REPAIR_COOLDOWN_S}s repair_lock=${REPAIR_LOCK_DIR}"
log "  log_file=${LOG_FILE}  log_max=${LOG_MAX_BYTES}"

detect_pid_namespace
log "  pid_namespace: $([ "$HOST_PID_NS" = "1" ] && echo host || echo containerized)"

if detect_cgroup; then
    log "  cgroup: ${CGROUP_VERSION}"
    [ -n "$CGROUP_PIDS_DIR" ] && log "    pids: ${CGROUP_PIDS_DIR}  (max=${CGROUP_PIDS_MAX_VAL})"
    [ -n "$CGROUP_MEM_DIR"  ] && log "    mem : ${CGROUP_MEM_DIR}  (max=${CGROUP_MEM_MAX_VAL})"
else
    log "  cgroup: <NOT DETECTED — cgroup pressure disabled; /proc pressure still enabled>"
fi
log "========================================"

LAST_CLEANUP=0
LAST_CGROUP_CHECK=0
LAST_PROC_CHECK=0
LAST_DOCKER_CLI_CHECK=0
LAST_PROXY_CHECK=0
LAST_POOL_CHECK=0
LAST_POOL_E2E_PROBE_TS=0
LAST_DISK_CHECK=0
LAST_HEARTBEAT_TS=0
HEALTH_FAILS=0
DOCKER_CLI_FAILS=0
DEEP_PROBE_FAILS=0
POOL_READY_FAILS=0
POOL_E2E_PROBE_FAILS=0
LAST_POOL_RESTART_TS=0
LAST_DOCKER_CLI_STATUS="?"
LAST_PROXY_STATUS="?"

while true; do
    NOW=$(date +%s)

    # 1) 浅探活
    if docker_alive; then
        HEALTH_FAILS=0
    else
        HEALTH_FAILS=$((HEALTH_FAILS + 1))
        log "Health check failed (${HEALTH_FAILS}/${MAX_CONSECUTIVE_HEALTH_FAILS})"
        if [ "${HEALTH_FAILS}" -ge "${MAX_CONSECUTIVE_HEALTH_FAILS}" ]; then
            trigger_repair "dockerd unix-socket ping failed ${HEALTH_FAILS} consecutive times" 1
            HEALTH_FAILS=0
            sleep 10
            continue
        fi
        sleep "${HEALTH_CHECK_INTERVAL}"
        continue
    fi

    # 2) 深度探活（5 min 一次）—— 抓 address-pool 耗尽这种 ping OK 但 reset 500 的形态
    if [ $((NOW - LAST_DEEP_PROBE_TS)) -ge "${DEEP_PROBE_INTERVAL}" ]; then
        if docker_deep_alive; then
            LAST_DEEP_PROBE_TS="$NOW"
            DEEP_PROBE_FAILS=0
        else
            DEEP_PROBE_FAILS=$((DEEP_PROBE_FAILS + 1))
            log "WARN: deep probe failed (network create/rm, fails=${DEEP_PROBE_FAILS}) — likely address-pool exhausted or docker CLI/API wedged"
            docker_network_prune_safe 30
            if [ "${DEEP_PROBE_FAILS}" -ge 2 ]; then
                trigger_repair "deep docker network probe failed ${DEEP_PROBE_FAILS} consecutive times"
                DEEP_PROBE_FAILS=0
            fi
            LAST_DEEP_PROBE_TS="$NOW"
        fi
    fi

    # 3) /proc 进程压力监控：不依赖 docker CLI，尽量在 fork 失败前预警/修复
    if [ $((NOW - LAST_PROC_CHECK)) -ge "${PROC_MONITOR_INTERVAL}" ]; then
        monitor_proc_pressure
        LAST_PROC_CHECK="$NOW"
    fi

    # 4) cgroup 监控
    if [ $((NOW - LAST_CGROUP_CHECK)) -ge "${CGROUP_MONITOR_INTERVAL}" ]; then
        monitor_pod_cgroup
        LAST_CGROUP_CHECK="$NOW"
    fi

    # 5) Docker CLI 探针：dockerd _ping OK 但 CLI/daemon metadata 卡死时触发
    if [ $((NOW - LAST_DOCKER_CLI_CHECK)) -ge "${DOCKER_CLI_CHECK_INTERVAL}" ]; then
        monitor_docker_cli
        LAST_DOCKER_CLI_CHECK="$NOW"
    fi

    # 6) proxy 低频探测：只告警，restart_docker 会显式带上 PROXY_URL
    if [ $((NOW - LAST_PROXY_CHECK)) -ge "${PROXY_CHECK_INTERVAL}" ]; then
        monitor_proxy
        LAST_PROXY_CHECK="$NOW"
    fi

    # 7) pool_server 监控
    if [ $((NOW - LAST_POOL_CHECK)) -ge "${POOL_CHECK_INTERVAL}" ]; then
        if check_pool_server; then
            POOL_READY_FAILS=0
        else
            POOL_READY_FAILS=$((POOL_READY_FAILS + 1))
            log "WARN: pool_server readiness failed (${POOL_READY_FAILS}/${POOL_READY_FAILS_RESTART}) active=${LAST_POOL_ACTIVE} pending_closes=${LAST_POOL_PENDING}"
            if [ "${POOL_READY_FAILS}" -ge "${POOL_READY_FAILS_RESTART}" ] 2>/dev/null; then
                if [ "${LAST_POOL_ACTIVE}" != "down" ] \
                   && [ "${LAST_POOL_ACTIVE}" != "unknown" ] \
                   && [ "${LAST_POOL_ACTIVE}" -gt "${POOL_RESTART_ACTIVE_MAX}" ] 2>/dev/null; then
                    log "POOL_RESTART deferred: active_runs=${LAST_POOL_ACTIVE} > restart_active_max=${POOL_RESTART_ACTIVE_MAX}"
                elif [ $((NOW - LAST_POOL_RESTART_TS)) -lt "${POOL_RESTART_COOLDOWN_S}" ]; then
                    log "POOL_RESTART suppressed: cooldown ${POOL_RESTART_COOLDOWN_S}s active"
                else
                    log "POOL_RESTART: stopping pool_server child after ${POOL_READY_FAILS} readiness failures"
                    stop_pool_server_for_pressure "pool readiness failed ${POOL_READY_FAILS} consecutive checks" 0
                    LAST_POOL_RESTART_TS="$NOW"
                    POOL_READY_FAILS=0
                fi
            fi
        fi
        LAST_POOL_CHECK="$NOW"
    fi

    # 8) 可选 E2E rollout 探活：真实 allocate + reset + optional exec + close
    if [ "${POOL_E2E_PROBE_INTERVAL}" -gt 0 ] 2>/dev/null \
       && [ $((NOW - LAST_POOL_E2E_PROBE_TS)) -ge "${POOL_E2E_PROBE_INTERVAL}" ]; then
        if check_pool_e2e_probe; then
            POOL_E2E_PROBE_FAILS=0
        else
            POOL_E2E_PROBE_FAILS=$((POOL_E2E_PROBE_FAILS + 1))
            log "WARN: pool E2E probe failed (${POOL_E2E_PROBE_FAILS}/${POOL_E2E_PROBE_FAILS_RESTART}) active=${LAST_POOL_ACTIVE} pending_closes=${LAST_POOL_PENDING}"
            if [ "${POOL_E2E_PROBE_FAILS}" -ge "${POOL_E2E_PROBE_FAILS_RESTART}" ] 2>/dev/null; then
                if [ "${LAST_POOL_ACTIVE}" != "down" ] \
                   && [ "${LAST_POOL_ACTIVE}" != "unknown" ] \
                   && [ "${LAST_POOL_ACTIVE}" -gt "${POOL_RESTART_ACTIVE_MAX}" ] 2>/dev/null; then
                    log "POOL_E2E restart deferred: active_runs=${LAST_POOL_ACTIVE} > restart_active_max=${POOL_RESTART_ACTIVE_MAX}"
                elif [ $((NOW - LAST_POOL_RESTART_TS)) -lt "${POOL_RESTART_COOLDOWN_S}" ]; then
                    log "POOL_E2E restart suppressed: cooldown ${POOL_RESTART_COOLDOWN_S}s active"
                else
                    log "POOL_E2E restart: stopping pool_server child after ${POOL_E2E_PROBE_FAILS} E2E failures"
                    stop_pool_server_for_pressure "pool E2E probe failed ${POOL_E2E_PROBE_FAILS} consecutive checks" 0
                    LAST_POOL_RESTART_TS="$NOW"
                    POOL_E2E_PROBE_FAILS=0
                fi
            fi
        fi
        LAST_POOL_E2E_PROBE_TS="$NOW"
    fi

    # 9) 容器清理 + 上限
    if [ $((NOW - LAST_CLEANUP)) -ge "${CLEANUP_INTERVAL}" ]; then
        cleanup_stopped
        enforce_container_limit
        LAST_CLEANUP="$NOW"
    fi

    # 10) Docker data-root 磁盘压力监控
    if [ $((NOW - LAST_DISK_CHECK)) -ge "${DISK_CHECK_INTERVAL}" ]; then
        monitor_docker_disk
        LAST_DISK_CHECK="$NOW"
    fi

    # 11) 低频心跳（默认 10 min）—— 复用上面已采集的指标，不发起新的 docker / curl
    if [ $((NOW - LAST_HEARTBEAT_TS)) -ge "${HEARTBEAT_INTERVAL}" ]; then
        log "OK: dockerd alive | docker_cli=${LAST_DOCKER_CLI_STATUS} proxy=${LAST_PROXY_STATUS} | pids=${LAST_PIDS_CUR:-?}/${LAST_PIDS_MAX:-?} (${LAST_PIDS_PCT:-?}%) tasks=${LAST_PROC_TASKS:-?} zombies=${LAST_ZOMBIES:-?} shim=${LAST_SHIM_PROCS:-?} runc=${LAST_RUNC_PROCS:-?} | pool active=${LAST_POOL_ACTIVE} pending_closes=${LAST_POOL_PENDING} resetting=${LAST_POOL_RESETTING} reset_max_age=${LAST_POOL_RESET_MAX_AGE}s | bridges=${LAST_BRIDGE_NETS} | task_containers=${LAST_RUNNING_TASKS}"
        LAST_HEARTBEAT_TS="$NOW"
    fi

    sleep "${HEALTH_CHECK_INTERVAL}"
done
