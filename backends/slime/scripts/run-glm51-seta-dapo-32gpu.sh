#!/usr/bin/env bash
# Tracked profile wrapper for the eight-node/four-GPU-per-node GLM-5.1 smoke.
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." >/dev/null 2>&1 && pwd)"
export GLM_GPU_PROFILE=32
export NUM_ROLLOUT="${NUM_ROLLOUT:-4}"
exec bash "${RUNS_ROOT:-${ROOT}/runs}/rjob/run_glm51_nonasync_smoke.sh" "$@"
