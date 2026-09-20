# ── Router / worker URLs ─────────────────────────────────────────────
WORKER_URLS="${WORKER_URLS:-}"
WORKER_URLS_FROM_FILE=0
RUN_LOCAL_WORKER_URLS_FILE=""
if [[ -z "${WORKER_URLS_FILE:-}" && "${DATASET}" == "swesmith" ]]; then
  RUN_LOCAL_WORKER_URLS_FILE="${RUN_DIR}/config/worker_urls.txt"
  WORKER_URLS_FILE="${RUN_LOCAL_WORKER_URLS_FILE}"
else
  WORKER_URLS_FILE="${WORKER_URLS_FILE:-${RUNS_ROOT:-${REPO_ROOT}/runs}/cluster/worker_urls.txt}"
fi
WORKER_URLS_RELOAD_INTERVAL="${WORKER_URLS_RELOAD_INTERVAL:-120}"

read_worker_urls_from_file() {
  local file="$1"
  python3 - "$file" <<'PY'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    sys.exit(0)
urls = []
for raw_line in path.read_text(encoding="utf-8").splitlines():
    line = raw_line.split("#", 1)[0].strip()
    if not line:
        continue
    if line.startswith("export "):
        line = line[len("export "):].strip()
    if line.startswith("WORKER_URLS="):
        line = line.split("=", 1)[1].strip()
    line = line.strip().strip('"').strip("'")
    urls.extend(part.rstrip("/") for part in re.split(r"[,\s]+", line) if part)
print(",".join(urls))
PY
}

if [[ -z "${WORKER_URLS}" && -f "${WORKER_URLS_FILE}" ]]; then
  WORKER_URLS="$(read_worker_urls_from_file "${WORKER_URLS_FILE}")"
  WORKER_URLS_FROM_FILE=1
fi
if [[ "${DRY_RUN}" != "1" && "${NEEDS_ENV_ROUTER}" == "1" && -z "${WORKER_URLS}" ]]; then
  echo "[ERROR] WORKER_URLS is unset. Example:"
  if [[ "${DATASET}" == "swesmith" ]]; then
    echo "        export WORKER_URLS=http://<worker-ip>:18082"
  else
    echo "        export WORKER_URLS=http://<worker-ip>:18081"
  fi
  echo "        or write that URL into WORKER_URLS_FILE=${WORKER_URLS_FILE}"
  exit 1
fi
if [[ "${NEEDS_ENV_ROUTER}" == "1" && -n "${WORKER_URLS}" ]]; then
  mkdir -p "$(dirname "${WORKER_URLS_FILE}")"
  if [[ -n "${RUN_LOCAL_WORKER_URLS_FILE}" ]]; then
    printf "%s\n" "${WORKER_URLS}" > "${WORKER_URLS_FILE}"
  elif [[ ! -s "${WORKER_URLS_FILE}" ]]; then
    printf "%s\n" "${WORKER_URLS}" > "${WORKER_URLS_FILE}"
  fi
fi
export WORKER_URLS WORKER_URLS_FILE WORKER_URLS_RELOAD_INTERVAL

if [[ "${HARBORRL_GPU_LAYOUT:-split}" == colocate ]]; then
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:2048}"
  if [[ "${PYTORCH_CUDA_ALLOC_CONF}" == *"expandable_segments:True"* ]]; then
    echo "colocate memory saver does not support expandable_segments"; exit 1
  fi
else
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:2048,expandable_segments:True}"
fi
if [[ -z "${MASTER_ADDR:-}" ]]; then
  MASTER_ADDR="$(hostname -I 2>/dev/null | awk '{print $1}' || true)"
  MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
fi
export MASTER_ADDR
NODE_IP="${MASTER_ADDR}"

export USE_REMOTE_ENV="${USE_REMOTE_ENV:-${NEEDS_ENV_ROUTER}}"
export PROVIDER_NAME="${PROVIDER_NAME:-build}"
export ENV_SERVER_BIND_HOST="${ENV_SERVER_BIND_HOST:-0.0.0.0}"
export ENV_SERVER_PORT="${ENV_SERVER_PORT:-18080}"
export ENV_SERVER_HOST="${ENV_SERVER_HOST:-${MASTER_ADDR}}"
FIRST_WORKER_URL=""
if [[ -n "${WORKER_URLS}" ]]; then
  FIRST_WORKER_URL="${WORKER_URLS%%,*}"
fi
if [[ "${NEEDS_ENV_ROUTER}" == "1" && -n "${FIRST_WORKER_URL}" ]]; then
  # When explicit remote workers are provided, use them directly by default.
  # Set START_ENV_POOL_SERVER=1 to force launching a local fan-out router.
  export ENV_SERVER_URL="${ENV_SERVER_URL:-${FIRST_WORKER_URL}}"
  export START_ENV_POOL_SERVER="${START_ENV_POOL_SERVER:-0}"
else
  export ENV_SERVER_URL="${ENV_SERVER_URL:-http://${ENV_SERVER_HOST}:${ENV_SERVER_PORT}}"
  export START_ENV_POOL_SERVER="${START_ENV_POOL_SERVER:-${NEEDS_ENV_ROUTER}}"
fi
export AGENT_SAFETYBENCH_REMOTE_ENV
export AGENTHARM_REMOTE_ENV
export AGENTHARM_ROOT
export AGENTHARM_REWARD

ROUTER_HOST="${ROUTER_HOST:-0.0.0.0}"
ROUTER_PORT="${ROUTER_PORT:-${ENV_SERVER_PORT}}"
CHECK_HOST="${CHECK_HOST:-127.0.0.1}"
CHECK_WAIT_SECS="${CHECK_WAIT_SECS:-60}"
READY_PROBE_TIMEOUT="${READY_PROBE_TIMEOUT:-5}"
ROUTER_REQUIRE_READY="${ROUTER_REQUIRE_READY:-1}"
ROUTER_READY_WAIT_FOREVER="${ROUTER_READY_WAIT_FOREVER:-0}"
WORKER_PREFLIGHT_REQUIRE_READY="${WORKER_PREFLIGHT_REQUIRE_READY:-1}"
WORKER_PREFLIGHT_TIMEOUT="${WORKER_PREFLIGHT_TIMEOUT:-5}"
export ROUTER_READYZ_WORKER_TIMEOUT="${ROUTER_READYZ_WORKER_TIMEOUT:-${WORKER_PREFLIGHT_TIMEOUT}}"
AUTO_CLOSE_STALE_WORKER_RUNS="${AUTO_CLOSE_STALE_WORKER_RUNS:-1}"
STALE_WORKER_CLOSE_INTERVAL="${STALE_WORKER_CLOSE_INTERVAL:-10}"
STALE_WORKER_CLOSE_TIMEOUT="${STALE_WORKER_CLOSE_TIMEOUT:-10}"
STALE_WORKER_REPAIR_MIN_AGE="${STALE_WORKER_REPAIR_MIN_AGE:-0}"
STALE_WORKER_REPAIR_MAX_REPAIRS="${STALE_WORKER_REPAIR_MAX_REPAIRS:-20}"
if ! [[ "${STALE_WORKER_CLOSE_INTERVAL}" =~ ^[0-9]+$ ]] || [[ "${STALE_WORKER_CLOSE_INTERVAL}" -le 0 ]]; then
  STALE_WORKER_CLOSE_INTERVAL=10
fi

probe_ready_endpoint() {
  local base_url="$1"
  local label="$2"
  local timeout_s="${3:-${READY_PROBE_TIMEOUT}}"
  local tmp err_tmp code path body curl_rc curl_error

  tmp="$(mktemp /tmp/lightrl_ready.XXXXXX 2>/dev/null || printf '/tmp/lightrl_ready.%s' "$$")"
  err_tmp="$(mktemp /tmp/lightrl_ready_err.XXXXXX 2>/dev/null || printf '/tmp/lightrl_ready_err.%s' "$$")"
  path="/readyz"
  curl_rc=0
  code="$(curl -sS --max-time "${timeout_s}" --noproxy '*' \
    -o "${tmp}" -w '%{http_code}' "${base_url}${path}" 2>"${err_tmp}")" || curl_rc=$?
  if (( curl_rc != 0 )); then
    curl_error="$(head -c 300 "${err_tmp}" 2>/dev/null || true)"
    log "  [WARN] ${label}${path} transport failure (curl rc=${curl_rc}, HTTP ${code:-000})${curl_error:+: ${curl_error}}"
    rm -f "${tmp}" "${err_tmp}" 2>/dev/null || true
    return 2
  fi
  if [[ "${code}" == "404" ]]; then
    path="/healthz"
    curl_rc=0
    code="$(curl -sS --max-time "${timeout_s}" --noproxy '*' \
      -o "${tmp}" -w '%{http_code}' "${base_url}${path}" 2>"${err_tmp}")" || curl_rc=$?
    if (( curl_rc != 0 )); then
      curl_error="$(head -c 300 "${err_tmp}" 2>/dev/null || true)"
      log "  [WARN] ${label}${path} transport failure (curl rc=${curl_rc}, HTTP ${code:-000})${curl_error:+: ${curl_error}}"
      rm -f "${tmp}" "${err_tmp}" 2>/dev/null || true
      return 2
    fi
  fi

  if [[ "${code}" =~ ^2[0-9][0-9]$ ]]; then
    log "  [OK] ${label}${path}"
    rm -f "${tmp}" "${err_tmp}" 2>/dev/null || true
    return 0
  fi

  body="$(head -c 300 "${tmp}" 2>/dev/null || true)"
  log "  [WARN] ${label}${path} not ready HTTP ${code}${body:+: ${body}}"
  rm -f "${tmp}" "${err_tmp}" 2>/dev/null || true
  return 1
}

extract_stale_lease_ids() {
  local json_path="$1"
  python3 - "${json_path}" <<'PY'
import json
import sys

path = sys.argv[1]
try:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        data = json.load(f)
except Exception:
    sys.exit(0)

seen = set()
out = []

def walk(obj):
    if isinstance(obj, dict):
        stale_runs = obj.get("stale_runs")
        if isinstance(stale_runs, list):
            for item in stale_runs:
                if not isinstance(item, dict):
                    continue
                lease_id = item.get("lease_id")
                if isinstance(lease_id, str) and lease_id and lease_id not in seen:
                    seen.add(lease_id)
                    out.append(lease_id)
        for value in obj.values():
            walk(value)
    elif isinstance(obj, list):
        for value in obj:
            walk(value)

walk(data)
for lease_id in out:
    print(lease_id)
PY
}

close_stale_worker_runs() {
  local base_url="$1"
  local label="$2"
  local timeout_s="${3:-${STALE_WORKER_CLOSE_TIMEOUT}}"
  local tmp ids_tmp path code lease_id close_tmp close_code close_body count repair_tmp repair_code repair_body

  if [[ "${AUTO_CLOSE_STALE_WORKER_RUNS}" != "1" ]]; then
    return 1
  fi
  if ! command -v python3 >/dev/null 2>&1; then
    log "  [WARN] stale-run cleanup skipped for ${label}: python3 not found"
    return 1
  fi

  tmp="$(mktemp /tmp/lightrl_stale_ready.XXXXXX 2>/dev/null || printf '/tmp/lightrl_stale_ready.%s' "$$")"
  ids_tmp="$(mktemp /tmp/lightrl_stale_ids.XXXXXX 2>/dev/null || printf '/tmp/lightrl_stale_ids.%s' "$$")"
  : > "${ids_tmp}"
  for path in /readyz /status; do
    code="$(curl -sS --max-time "${timeout_s}" --noproxy '*' \
      -o "${tmp}" -w '%{http_code}' "${base_url}${path}" 2>/dev/null || true)"
    if [[ "${code}" =~ ^[0-9][0-9][0-9]$ ]]; then
      extract_stale_lease_ids "${tmp}" >> "${ids_tmp}" || true
    fi
  done

  if [[ ! -s "${ids_tmp}" ]]; then
    rm -f "${tmp}" "${ids_tmp}" 2>/dev/null || true
    return 1
  fi

  repair_tmp="$(mktemp /tmp/lightrl_stale_repair.XXXXXX 2>/dev/null || printf '/tmp/lightrl_stale_repair.%s' "$$")"
  repair_code="$(curl -sS --max-time "${timeout_s}" --noproxy '*' \
    -X POST -H 'Content-Type: application/json' \
    --data "{\"reason\":\"startup_readyz_repair\",\"min_age\":${STALE_WORKER_REPAIR_MIN_AGE},\"max_repairs\":${STALE_WORKER_REPAIR_MAX_REPAIRS}}" \
    -o "${repair_tmp}" -w '%{http_code}' "${base_url}/repair/stale_runs" 2>/dev/null || true)"
  repair_body="$(head -c 320 "${repair_tmp}" 2>/dev/null || true)"
  if [[ "${repair_code}" =~ ^2[0-9][0-9]$ ]]; then
    log "  [REPAIR] ${label}: repair stale runs HTTP ${repair_code}${repair_body:+: ${repair_body}}"
    rm -f "${tmp}" "${ids_tmp}" "${repair_tmp}" 2>/dev/null || true
    return 0
  elif [[ "${repair_code}" != "404" && "${repair_code}" != "000" ]]; then
    log "  [WARN] ${label}: repair stale runs endpoint HTTP ${repair_code}${repair_body:+: ${repair_body}}"
  else
    log "  [WARN] ${label}: /repair/stale_runs unavailable; falling back to duplicate /close requests. Restart worker to load the repair endpoint if stale in-flight runs persist."
  fi
  rm -f "${repair_tmp}" 2>/dev/null || true

  count=0
  while IFS= read -r lease_id; do
    [[ -n "${lease_id}" ]] || continue
    close_tmp="$(mktemp /tmp/lightrl_stale_close.XXXXXX 2>/dev/null || printf '/tmp/lightrl_stale_close.%s' "$$")"
    close_code="$(curl -sS --max-time "${timeout_s}" --noproxy '*' \
      -X POST -H 'Content-Type: application/json' \
      --data "{\"lease_id\":\"${lease_id}\"}" \
      -o "${close_tmp}" -w '%{http_code}' "${base_url}/close" 2>/dev/null || true)"
    close_body="$(head -c 240 "${close_tmp}" 2>/dev/null || true)"
    log "  [REPAIR] ${label}: close stale lease=${lease_id} HTTP ${close_code}${close_body:+: ${close_body}}"
    rm -f "${close_tmp}" 2>/dev/null || true
    count=$((count + 1))
  done < <(python3 - "${ids_tmp}" <<'PY'
import sys
seen = set()
for line in open(sys.argv[1], "r", encoding="utf-8", errors="ignore"):
    value = line.strip()
    if value and value not in seen:
        seen.add(value)
        print(value)
PY
  )

  rm -f "${tmp}" "${ids_tmp}" 2>/dev/null || true
  [[ "${count}" -gt 0 ]]
}

close_stale_runs_for_all_workers() {
  local reason="$1"
  local repaired=0
  local _w
  IFS=',' read -r -a _STALE_WORKERS <<< "${WORKER_URLS}"
  for _w in "${_STALE_WORKERS[@]}"; do
    [[ -n "${_w}" ]] || continue
    if close_stale_worker_runs "${_w}" "${_w} (${reason})" "${STALE_WORKER_CLOSE_TIMEOUT}"; then
      repaired=1
    fi
  done
  [[ "${repaired}" -eq 1 ]]
}

# ── Robustness knobs (informed by issue #3 postmortem) ───────────────
# Router forward to pool_server (tuned for burst of docker-compose down/up):
#   - ROUTER_FORWARD_TIMEOUT: raise 600 → 900 (issue #3 §1.X-E: 90s → 600s still
#     tight when pool is processing 32 concurrent closes; 15min is generous).
#   - ROUTER_FORWARD_RETRIES: 1 → 3 (matches agent_runner http_utils retries).
#   - ROUTER_FORWARD_RETRY_BACKOFF: 0.2 → 1.0 (exponential-ish, gives the pool
#     a real window to finish in-flight docker operations before the retry).
export ROUTER_FORWARD_TIMEOUT="${ROUTER_FORWARD_TIMEOUT:-900}"
export ROUTER_FORWARD_RETRIES="${ROUTER_FORWARD_RETRIES:-3}"
export ROUTER_FORWARD_RETRY_BACKOFF="${ROUTER_FORWARD_RETRY_BACKOFF:-1.0}"
export ROUTER_PRESSURE_COOLDOWN="${ROUTER_PRESSURE_COOLDOWN:-60}"

# ── Trajectory export (parallels swe-rl export/swe_rollouts) ─────────────────
# Trajectory export is now ON by default (writes to the categorized
# RUN_DIR/trajectories/).
# Set TERMINAL_SAVE_TRAJ_DIR="" to disable.
export TERMINAL_SAVE_TRAJ_DIR="${TERMINAL_SAVE_TRAJ_DIR}"
export TRAJECTORY_SAVE_INTERVAL="${TRAJECTORY_SAVE_INTERVAL}"
export TRAJECTORY_SAVE_INTERVAL_SETA
export TRAJECTORY_SAVE_INTERVAL_AGENT_SAFETYBENCH
export TRAJECTORY_SAVE_INTERVAL_AGENTHARM
export TRAJECTORY_SAVE_POLICY TRAJECTORY_TASK_SAVE_INTERVAL TRAJECTORY_TASK_MAX_PER_STEP
export TRAJECTORY_TASK_MAX_PER_TASK TRAJECTORY_MAX_TOTAL TRAJECTORY_SAVE_REWARD_STRATA
export TRAJECTORY_SAVE_LOG_DECISIONS

# Proxy bypass: some environments inject http_proxy/HTTPS_PROXY via shell rc.
# aiohttp + requests will then try to tunnel the internal router→worker traffic
# through a proxy, causing spurious connection failures. Explicitly list all
# hosts on the rollout datapath as NO_PROXY (matches swe-rl v1/v4 pattern).
ALL_WORKER_HOSTS=""
if [[ -n "${WORKER_URLS}" ]]; then
  ALL_WORKER_HOSTS="$(echo "${WORKER_URLS}" | tr ',' '\n' \
    | sed -E 's#https?://([^:/]+).*#\1#' | tr '\n' ',' | sed 's/,$//')"
fi
NO_PROXY_REQUIRED="localhost,127.0.0.1,${MASTER_ADDR}${ALL_WORKER_HOSTS:+,${ALL_WORKER_HOSTS}}"
NO_PROXY_EXISTING="${NO_PROXY:-${no_proxy:-}}"
if [[ -n "${NO_PROXY_EXISTING}" ]]; then
  export NO_PROXY="${NO_PROXY_EXISTING},${NO_PROXY_REQUIRED}"
else
  export NO_PROXY="${NO_PROXY_REQUIRED}"
fi
export no_proxy="${NO_PROXY}"

# Router uses `python3` which, after the PATH export above, resolves to
# lightrft_py312/bin/python. Override ROUTER_PYTHON if you want a different env.
ROUTER_PYTHON="${ROUTER_PYTHON:-python3}"

export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray_terminal_rl}"
mkdir -p "${RAY_TMPDIR}"
SLIME_RAY_PLACEMENT_GPU_PROBE="${SLIME_RAY_PLACEMENT_GPU_PROBE:-0}"
export SLIME_RAY_PLACEMENT_GPU_PROBE
