#!/usr/bin/env bash
# Diagnose and conservatively repair Docker data-root pressure on safevo/rlinfra.
#
# Default mode is read-only:
#   DOCKER_DATA_ROOT=/data bash deploy/ops/safevo_docker_storage_doctor.sh
#
# Conservative repair mode, still preserving tagged images:
#   MODE=repair APPLY=1 DOCKER_DATA_ROOT=/data bash deploy/ops/safevo_docker_storage_doctor.sh
#
# This script intentionally does not delete tagged images. Image deletion is a
# separate, explicit operation in docker_storage_gc.py with
# DOCKER_GC_DELETE_OLD_IMAGES=1.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"

DOCKER_DATA_ROOT="${DOCKER_DATA_ROOT:-${DOCKER_ROOT:-/data}}"
MODE="${MODE:-diagnose}"          # diagnose | repair
APPLY="${APPLY:-0}"               # repair commands are dry-run unless APPLY=1
DOCKER_TIMEOUT="${DOCKER_TIMEOUT:-30}"
DU_TIMEOUT="${DU_TIMEOUT:-240}"
PRUNE_TIMEOUT="${PRUNE_TIMEOUT:-180}"
TOP_N="${TOP_N:-30}"
IMAGE_LARGE_GB="${IMAGE_LARGE_GB:-1}"
CONTAINER_UNTIL="${CONTAINER_UNTIL:-1h}"
BUILDER_CACHE_UNTIL="${BUILDER_CACHE_UNTIL:-24h}"
PRUNE_VOLUMES="${PRUNE_VOLUMES:-0}"
TRUNCATE_LOGS="${TRUNCATE_LOGS:-0}"
LOG_TRUNCATE_THRESHOLD_MB="${LOG_TRUNCATE_THRESHOLD_MB:-1024}"
RUN_DOCKER_DF="${RUN_DOCKER_DF:-0}"
RUN_OVERLAY_DU="${RUN_OVERLAY_DU:-1}"
RUN_OVERLAY_SAMPLE="${RUN_OVERLAY_SAMPLE:-1}"
OVERLAY_SAMPLE_N="${OVERLAY_SAMPLE_N:-80}"
LOCK_FILE="${LOCK_FILE:-/tmp/openclaw_safevo_docker_storage_doctor.lock}"
RUNS_ROOT="${RUNS_ROOT:-${REPO_ROOT}/runs}"
HOST_ID="$(hostname -f 2>/dev/null || hostname 2>/dev/null || echo unknown-host)"
TRAIN_RUN_ID="${RUN_ID:-}"
DOCTOR_RUN_ID="${OPENCLAW_DOCTOR_RUN_ID:-$(date +%Y%m%d_%H%M%S)_pid$$}"
if [[ -z "${LOG_DIR:-}" ]]; then
    if [[ -n "${RUN_DIR:-}" ]]; then
        LOG_DIR="${RUN_DIR}/diagnostics/docker_storage_doctor/${HOST_ID}/${DOCTOR_RUN_ID}"
    elif [[ -n "${TRAIN_RUN_ID}" ]]; then
        LOG_DIR="${RUNS_ROOT}/${TRAIN_RUN_ID}/diagnostics/docker_storage_doctor/${HOST_ID}/${DOCTOR_RUN_ID}"
    else
        LOG_DIR="${RUNS_ROOT}/diagnostics/docker_storage_doctor/${HOST_ID}/${DOCTOR_RUN_ID}"
    fi
fi
LOG_FILE="${LOG_FILE:-${LOG_DIR}/doctor.log}"

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

log() {
    printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

die() {
    log "ERROR: $*"
    exit 1
}

run_ro() {
    log "+ $*"
    timeout "${DOCKER_TIMEOUT}" "$@" || true
}

run_repair() {
    log "+ $*"
    if [[ "${APPLY}" != "1" ]]; then
        log "DRY_RUN: skipped repair command; set APPLY=1 to execute"
        return 0
    fi
    timeout "${PRUNE_TIMEOUT}" "$@" || log "WARN: repair command failed: $*"
}

acquire_lock() {
    if command -v flock >/dev/null 2>&1; then
        exec 9>"${LOCK_FILE}"
        if ! flock -n 9; then
            log "Another doctor instance is running (lock=${LOCK_FILE}); exiting."
            exit 0
        fi
    fi
}

docker_t() {
    timeout "${DOCKER_TIMEOUT}" docker "$@"
}

docker_info_value() {
    docker_t info --format "$1" 2>/dev/null || true
}

print_header() {
    log "========================================"
    log "HarborRL Docker storage doctor"
    log "host=${HOST_ID}"
    log "repo=${REPO_ROOT}"
    log "mode=${MODE} apply=${APPLY}"
    log "docker_data_root=${DOCKER_DATA_ROOT}"
    log "timeouts: docker=${DOCKER_TIMEOUT}s du=${DU_TIMEOUT}s prune=${PRUNE_TIMEOUT}s"
    log "overlay scans: full_du=${RUN_OVERLAY_DU} sample=${RUN_OVERLAY_SAMPLE} sample_n=${OVERLAY_SAMPLE_N}"
    log "log_dir=${LOG_DIR}"
    log "log_file=${LOG_FILE}"
    log "========================================"
}

preflight() {
    [[ -d "${DOCKER_DATA_ROOT}" ]] || die "Docker data root does not exist: ${DOCKER_DATA_ROOT}"
    timeout "${DOCKER_TIMEOUT}" docker info >/dev/null 2>&1 || die "docker daemon is not responding"
    local real_root
    real_root="$(docker_info_value '{{.DockerRootDir}}')"
    if [[ -n "${real_root}" && "${real_root}" != "${DOCKER_DATA_ROOT}" ]]; then
        die "Docker Root Dir (${real_root}) != DOCKER_DATA_ROOT (${DOCKER_DATA_ROOT}); set DOCKER_DATA_ROOT correctly"
    fi
}

disk_report() {
    log "Filesystem usage:"
    df -h "${DOCKER_DATA_ROOT}" / 2>/dev/null || true
    log "Inode usage:"
    df -ih "${DOCKER_DATA_ROOT}" / 2>/dev/null || true
}

docker_summary() {
    log "Docker summary:"
    docker_t info 2>/dev/null | grep -E 'Docker Root Dir|Storage Driver|Containers:|Running:|Paused:|Stopped:|Images:' || true
    log "Docker object counts:"
    printf '  containers=%s running=%s images=%s unique_images=%s volumes=%s networks=%s\n' \
        "$(docker_t ps -aq 2>/dev/null | wc -l || echo '?')" \
        "$(docker_t ps -q 2>/dev/null | wc -l || echo '?')" \
        "$(docker_t image ls -q 2>/dev/null | wc -l || echo '?')" \
        "$(docker_t image ls -q 2>/dev/null | sort -u | wc -l || echo '?')" \
        "$(docker_t volume ls -q 2>/dev/null | wc -l || echo '?')" \
        "$(docker_t network ls -q 2>/dev/null | wc -l || echo '?')"
}

top_level_du() {
    log "Top-level ${DOCKER_DATA_ROOT} usage (timeout=${DU_TIMEOUT}s):"
    timeout "${DU_TIMEOUT}" du -xhd1 "${DOCKER_DATA_ROOT}" 2>"${LOG_DIR}/du_top.err" \
        | sort -h \
        | tail -40 \
        | tee "${LOG_DIR}/du_top.txt" || {
            log "WARN: top-level du timed out or failed"
            tail -20 "${LOG_DIR}/du_top.err" 2>/dev/null || true
        }
}

overlay_dir_count() {
    local overlay="${DOCKER_DATA_ROOT}/overlay2"
    [[ -d "${overlay}" ]] || return 0
    log "overlay2 directory count:"
    timeout "${DOCKER_TIMEOUT}" find "${overlay}" -mindepth 1 -maxdepth 1 -type d ! -name l 2>/dev/null \
        | wc -l || true
}

overlay_top_dirs() {
    local overlay="${DOCKER_DATA_ROOT}/overlay2"
    [[ -d "${overlay}" ]] || return 0
    [[ "${RUN_OVERLAY_DU}" == "1" ]] || {
        log "Skipping overlay2 du scan. Set RUN_OVERLAY_DU=1 to enable it."
        return 0
    }
    log "Largest overlay2 directories (timeout=${DU_TIMEOUT}s):"
    timeout "${DU_TIMEOUT}" bash -c '
        root="$1"
        top_n="$2"
        find "$root/overlay2" -mindepth 1 -maxdepth 1 -type d ! -name l -print0 2>/dev/null \
          | xargs -0 -r du -xsB1 2>/dev/null \
          | sort -nr \
          | head -n "$top_n"
    ' bash "${DOCKER_DATA_ROOT}" "${TOP_N}" \
        | tee "${LOG_DIR}/overlay2_top_bytes.tsv" \
        | awk '{printf "  %.2fG\t%s\n", $1/1024/1024/1024, $2}' || {
            log "WARN: overlay2 du scan timed out or failed"
        }
}

overlay_sample_dirs() {
    local overlay="${DOCKER_DATA_ROOT}/overlay2"
    [[ -d "${overlay}" ]] || return 0
    [[ "${RUN_OVERLAY_SAMPLE}" == "1" ]] || {
        log "Skipping overlay2 sampled du scan. Set RUN_OVERLAY_SAMPLE=1 to enable it."
        return 0
    }
    log "Sampled overlay2 directory sizes (n=${OVERLAY_SAMPLE_N}, timeout=${DU_TIMEOUT}s):"
    timeout "${DU_TIMEOUT}" bash -c '
        root="$1"
        sample_n="$2"
        sample_file="$3"
        find "$root/overlay2" -mindepth 1 -maxdepth 1 -type d ! -name l -print 2>/dev/null \
          | shuf -n "$sample_n" >"$sample_file"
        xargs -r du -xsB1 <"$sample_file" 2>/dev/null \
          | sort -nr
    ' bash "${DOCKER_DATA_ROOT}" "${OVERLAY_SAMPLE_N}" "${LOG_DIR}/overlay2_sample_dirs.txt" \
        | tee "${LOG_DIR}/overlay2_sample_bytes.tsv" \
        | awk '
            {sum += $1; n += 1; if (NR <= 20) printf "  %.2fG\t%s\n", $1/1024/1024/1024, $2}
            END {
                if (n > 0) printf "  sample_count=%d sample_avg=%.2fG\n", n, sum/n/1024/1024/1024;
                else print "  no sampled overlay2 directories";
            }
        ' || {
            log "WARN: overlay2 sampled du scan timed out or failed"
        }
}

container_size_report() {
    log "Container writable-layer sizes from docker inspect --size:"
    local ids_file="${LOG_DIR}/container_ids.txt"
    docker_t ps -aq --no-trunc >"${ids_file}" 2>/dev/null || true
    if [[ ! -s "${ids_file}" ]]; then
        log "  no containers"
        return 0
    fi
    local inspect_json="${LOG_DIR}/containers.inspect.size.json"
    : >"${inspect_json}"
    timeout "${DU_TIMEOUT}" xargs -r -n 50 docker inspect --size <"${ids_file}" >>"${inspect_json}" 2>"${LOG_DIR}/container_inspect.err" || true
    python3 - "${inspect_json}" "${TOP_N}" <<'PY'
import json, sys
path, top_n = sys.argv[1], int(sys.argv[2])
text = open(path, "r", encoding="utf-8", errors="replace").read().strip()
if not text:
    print("  no inspect data")
    raise SystemExit
decoder = json.JSONDecoder()
idx = 0
items = []
while idx < len(text):
    while idx < len(text) and text[idx].isspace():
        idx += 1
    if idx >= len(text):
        break
    obj, end = decoder.raw_decode(text, idx)
    idx = end
    if isinstance(obj, list):
        items.extend(obj)
rows = []
for c in items:
    state = c.get("State") or {}
    gd = c.get("GraphDriver", {}).get("Data", {}) or {}
    name = str(c.get("Name") or "").lstrip("/")
    rows.append((
        int(c.get("SizeRw") or 0),
        int(c.get("SizeRootFs") or 0),
        name,
        state.get("Status"),
        str(c.get("Image") or "")[:20],
        gd.get("UpperDir") or gd.get("MergedDir") or "",
    ))
rows.sort(reverse=True)
for size_rw, size_root, name, status, image, upper in rows[:top_n]:
    print(f"  rw={size_rw/1024/1024/1024:.2f}G rootfs={size_root/1024/1024/1024:.2f}G status={status} image={image} name={name} upper={upper}")
if len(rows) > top_n:
    print(f"  ... suppressed {len(rows)-top_n} more containers")
PY
}

large_image_report() {
    log "Images larger than ${IMAGE_LARGE_GB}G virtual size (not unique disk usage):"
    local ids_file="${LOG_DIR}/image_ids.txt"
    docker_t image ls -q --no-trunc | sort -u >"${ids_file}" 2>/dev/null || true
    if [[ ! -s "${ids_file}" ]]; then
        log "  no images"
        return 0
    fi
    local inspect_json="${LOG_DIR}/images.inspect.json"
    : >"${inspect_json}"
    timeout "${DU_TIMEOUT}" xargs -r -n 50 docker image inspect <"${ids_file}" >>"${inspect_json}" 2>"${LOG_DIR}/image_inspect.err" || true
    python3 - "${inspect_json}" "${IMAGE_LARGE_GB}" "${TOP_N}" <<'PY'
import json, sys
path, threshold_gb, top_n = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
text = open(path, "r", encoding="utf-8", errors="replace").read().strip()
decoder = json.JSONDecoder()
idx = 0
items = []
while idx < len(text):
    while idx < len(text) and text[idx].isspace():
        idx += 1
    if idx >= len(text):
        break
    obj, end = decoder.raw_decode(text, idx)
    idx = end
    if isinstance(obj, list):
        items.extend(obj)
rows = []
for img in items:
    size = int(img.get("Size") or 0)
    if size >= threshold_gb * 1024**3:
        tags = img.get("RepoTags") or ["<none>:<none>"]
        rows.append((size, img.get("Created") or "", str(img.get("Id") or "")[:20], ",".join(tags)))
rows.sort(reverse=True)
print(f"  count={len(rows)} threshold={threshold_gb:.1f}G")
if rows:
    print("  NOTE: image size is virtual size; shared base layers are counted repeatedly by Docker.")
for size, created, iid, tags in rows[:top_n]:
    print(f"  size={size/1024/1024/1024:.2f}G created={created} id={iid} tags={tags}")
if len(rows) > top_n:
    print(f"  ... suppressed {len(rows)-top_n} more images")
PY
}

docker_df_report() {
    if [[ "${RUN_DOCKER_DF}" != "1" ]]; then
        log "Skipping docker system df -v. Set RUN_DOCKER_DF=1 to enable; it can be slow."
        return 0
    fi
    log "docker system df -v (timeout=${DU_TIMEOUT}s):"
    timeout "${DU_TIMEOUT}" docker system df -v 2>"${LOG_DIR}/docker_system_df.err" \
        | tee "${LOG_DIR}/docker_system_df.txt" || {
            log "WARN: docker system df -v timed out or failed"
            tail -30 "${LOG_DIR}/docker_system_df.err" 2>/dev/null || true
        }
}

layer_metadata_report() {
    log "overlay2 metadata reference counts:"
    local overlay="${DOCKER_DATA_ROOT}/overlay2"
    [[ -d "${overlay}" ]] || return 0
    local layerdb="${DOCKER_DATA_ROOT}/image/overlay2/layerdb"
    local image_cache_file="${LOG_DIR}/image_layer_cache_ids.txt"
    local mount_file="${LOG_DIR}/container_mount_ids.txt"
    local overlay_names_file="${LOG_DIR}/overlay2_dir_names.txt"
    local known_overlay_file="${LOG_DIR}/overlay2_known_dir_names.txt"
    local unknown_overlay_file="${LOG_DIR}/overlay2_unknown_dir_names.txt"
    local cache_file_count mount_file_count
    cache_file_count="$(find "${layerdb}" -name cache-id -type f 2>/dev/null | wc -l || echo 0)"
    mount_file_count="$(find "${layerdb}/mounts" -name mount-id -type f 2>/dev/null | wc -l || echo 0)"
    find "${layerdb}" -name cache-id -type f -print0 2>/dev/null \
        | xargs -0 -r awk 'NF {print $0}' 2>/dev/null \
        | sort -u >"${image_cache_file}" || true
    find "${layerdb}/mounts" -name mount-id -type f -print0 2>/dev/null \
        | xargs -0 -r awk 'NF {print $0}' 2>/dev/null \
        | sort -u >"${mount_file}" || true
    find "${overlay}" -mindepth 1 -maxdepth 1 -type d ! -name l -printf '%f\n' 2>/dev/null \
        | sort -u >"${overlay_names_file}" || true
    cat "${image_cache_file}" "${mount_file}" 2>/dev/null | sort -u >"${known_overlay_file}" || true
    comm -23 "${overlay_names_file}" "${known_overlay_file}" >"${unknown_overlay_file}" 2>/dev/null || true
    printf '  cache-id files=%s image_layer_cache_ids=%s mount-id files=%s container_mount_ids=%s overlay2_dirs=%s unmatched_overlay2_dirs=%s\n' \
        "${cache_file_count}" \
        "$(wc -l <"${image_cache_file}" 2>/dev/null || echo 0)" \
        "${mount_file_count}" \
        "$(wc -l <"${mount_file}" 2>/dev/null || echo 0)" \
        "$(wc -l <"${overlay_names_file}" 2>/dev/null || echo 0)" \
        "$(wc -l <"${unknown_overlay_file}" 2>/dev/null || echo 0)"

    for overlay_report in "${LOG_DIR}/overlay2_top_bytes.tsv" "${LOG_DIR}/overlay2_sample_bytes.tsv"; do
        [[ -s "${overlay_report}" ]] || continue
        log "Classifying $(basename "${overlay_report}") dirs:"
        python3 - "${overlay_report}" "${image_cache_file}" "${mount_file}" <<'PY'
import os, sys
top, image_cache, mounts = sys.argv[1:4]
image_ids = set(open(image_cache, "r", encoding="utf-8", errors="replace").read().split())
mount_ids = set(open(mounts, "r", encoding="utf-8", errors="replace").read().split())
for line in open(top, "r", encoding="utf-8", errors="replace"):
    parts = line.strip().split(None, 1)
    if len(parts) != 2:
        continue
    size, path = parts
    layer = os.path.basename(path)
    if layer in image_ids:
        kind = "image-layer"
    elif layer in mount_ids:
        kind = "container-writable-layer"
    else:
        kind = "unknown-buildkit-or-orphan"
    print(f"  {int(size)/1024/1024/1024:.2f}G {kind} {layer}")
PY
    done

    local cache_count mount_count image_count overlay_count
    cache_count="$(wc -l <"${image_cache_file}" 2>/dev/null || echo 0)"
    mount_count="$(wc -l <"${mount_file}" 2>/dev/null || echo 0)"
    image_count="$(docker_t image ls -q 2>/dev/null | sort -u | wc -l || echo 0)"
    overlay_count="$(find "${overlay}" -mindepth 1 -maxdepth 1 -type d ! -name l 2>/dev/null | wc -l || echo 0)"
    if [[ "${image_count}" -gt 0 && "${cache_count}" -le 1 && "${overlay_count}" -gt 100 ]]; then
        log "WARN: Docker reports images but layerdb cache-id count is unexpectedly tiny."
        log "      This can indicate metadata layout mismatch/corruption or a stale overlay2 tree; verify before any manual overlay2 cleanup."
        log "      Suggested check: find ${layerdb} -maxdepth 4 -type f | head -50"
    fi
    if [[ "${mount_count}" -gt 0 && "$(docker_t ps -aq 2>/dev/null | wc -l || echo 0)" -eq 0 ]]; then
        log "WARN: layerdb has container mount ids but Docker reports zero containers; this may be stale metadata."
    fi
}

container_log_report() {
    local containers_dir="${DOCKER_DATA_ROOT}/containers"
    [[ -d "${containers_dir}" ]] || return 0
    log "Largest Docker JSON logs:"
    find "${containers_dir}" -name '*-json.log' -type f -printf '%s %p\n' 2>/dev/null \
        | sort -n \
        | tail -20 \
        | awk '{printf "  %.2fG %s\n", $1/1024/1024/1024, $2}' \
        | tee "${LOG_DIR}/container_json_logs_top.txt" || true
}

repair_conservative() {
    log "Conservative repair: no tagged image deletion."
    log "Repair config: APPLY=${APPLY} CONTAINER_UNTIL=${CONTAINER_UNTIL} BUILDER_CACHE_UNTIL=${BUILDER_CACHE_UNTIL} PRUNE_VOLUMES=${PRUNE_VOLUMES} TRUNCATE_LOGS=${TRUNCATE_LOGS}"
    run_repair docker container prune -f --filter "until=${CONTAINER_UNTIL}"
    run_repair docker network prune -f
    if [[ "${PRUNE_VOLUMES}" == "1" ]]; then
        run_repair docker volume prune -f
    else
        log "Skipping volume prune. Set PRUNE_VOLUMES=1 if unused volumes are disposable."
    fi
    if [[ "${BUILDER_CACHE_UNTIL}" =~ ^([Aa][Ll][Ll]|0|0h)$ ]]; then
        run_repair docker builder prune -af
    else
        run_repair docker builder prune -af --filter "until=${BUILDER_CACHE_UNTIL}"
    fi
    run_repair docker image prune -f

    if [[ "${TRUNCATE_LOGS}" == "1" ]]; then
        log "Truncating container JSON logs larger than ${LOG_TRUNCATE_THRESHOLD_MB}MB."
        if [[ "${APPLY}" != "1" ]]; then
            find "${DOCKER_DATA_ROOT}/containers" -name '*-json.log' -type f -size +"${LOG_TRUNCATE_THRESHOLD_MB}"M -print 2>/dev/null \
                | sed 's/^/DRY_RUN would truncate log: /' || true
        else
            find "${DOCKER_DATA_ROOT}/containers" -name '*-json.log' -type f -size +"${LOG_TRUNCATE_THRESHOLD_MB}"M -print -exec truncate -s 0 {} \; 2>/dev/null || true
        fi
    fi
}

main() {
    acquire_lock
    print_header
    preflight
    disk_report
    docker_summary
    top_level_du
    overlay_dir_count
    overlay_top_dirs
    overlay_sample_dirs
    layer_metadata_report
    container_size_report
    container_log_report
    large_image_report
    docker_df_report

    case "${MODE}" in
        diagnose)
            log "Diagnosis complete. No repair executed. Set MODE=repair APPLY=1 for conservative repair."
            ;;
        repair)
            repair_conservative
            log "Post-repair snapshot:"
            disk_report
            docker_summary
            ;;
        *)
            die "Unsupported MODE=${MODE}; use diagnose or repair"
            ;;
    esac
    log "Done. Report directory: ${LOG_DIR}"
}

main "$@"
