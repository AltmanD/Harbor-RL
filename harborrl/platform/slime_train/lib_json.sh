# shellcheck shell=bash
# Helpers for embedding environment values in Ray runtime-env JSON.

append_env_passthrough() {
  local var="${1:?environment variable name required}"
  local value="${!var:-}"
  [[ -z "${value}" ]] && return 0

  local encoded
  if ! encoded=$(python3 -c 'import json, sys; print(json.dumps(sys.argv[1]))' "${value}"); then
    echo "[ERROR] cannot JSON encode ${var}" >&2
    return 1
  fi
  EXTRA_ENV_PASSTHROUGH_JSON+=$'\n    '"\"${var}\": ${encoded},"
}
