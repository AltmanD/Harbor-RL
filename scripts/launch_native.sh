#!/usr/bin/env bash
# Bootstrap pinned backends when needed, then run the native launch pipeline.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/launch_native.sh --config PATH [--backends PATH]
                                  [--doctor|--dry-run]

Options:
  --config PATH     Private native training YAML.
  --backends PATH   Bootstrap prefix. Existing harborrl-backend.env is reused.
  --doctor          Run preflight checks without starting training.
  --dry-run         Print the resolved launch plan without side effects.
  -h, --help        Show this help.
USAGE
}

config=""
backends=""
mode="train"

while (($#)); do
  case "$1" in
    --config)
      (($# >= 2)) || { usage >&2; exit 2; }
      config=$2
      shift 2
      ;;
    --backends)
      (($# >= 2)) || { usage >&2; exit 2; }
      backends=$2
      shift 2
      ;;
    --doctor)
      mode="doctor"
      shift
      ;;
    --dry-run)
      mode="dry-run"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$config" ]]; then
  echo "--config is required" >&2
  usage >&2
  exit 2
fi

config=$(realpath -e "$config")
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

if [[ -n "$backends" ]]; then
  backends=$(realpath -m "$backends")
  backend_env="$backends/harborrl-backend.env"
  if [[ ! -f "$backend_env" ]]; then
    echo "[launch-native] bootstrapping pinned backends in $backends"
    bash "$root/scripts/bootstrap_backends.sh" "$backends"
  fi
  set -a
  # shellcheck disable=SC1090
  source "$backend_env"
  set +a
fi

if [[ "$mode" != "dry-run" ]]; then
  for variable in SLIME_DIR MEGATRON_DIR SGLANG_IMAGE; do
    if [[ -z "${!variable:-}" ]]; then
      echo "missing $variable; pass --backends or source harborrl-backend.env" >&2
      exit 1
    fi
  done
fi

cd "$root"
python_bin="${HARBORRL_PYTHON:-python3}"

case "$mode" in
  doctor)
    exec "$python_bin" -m harborrl.cli doctor --config "$config"
    ;;
  dry-run)
    exec "$python_bin" -m harborrl.cli train --config "$config" --dry-run
    ;;
  train)
    exec "$python_bin" -m harborrl.cli train --config "$config"
    ;;
esac
