#!/usr/bin/env bash
# Run from a configured GPU training environment with a verified static catalog.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${ROLLOUT_PROMPT_DATA:?Set the verified Harbor catalog JSONL}"
: "${HARBOR_LOGPROB_SOURCE:?Set the audited serving package/version identifier}"
: "${HARBOR_LOGPROB_SEMANTICS:?Set the audited serving logprob semantics (raw_model)}"
[[ "${HARBOR_LOGPROB_SEMANTICS}" == raw_model ]] || { echo 'Only raw_model logprobs are supported'; exit 1; }
export DATASET=harbor_terminal ALGO=grpo
export SLIME_ENTRYPOINT="${ROOT}/backends/slime/train.py"
export RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs}"
export HARBOR_IR_ROOT="${HARBOR_IR_ROOT:-${RUNS_ROOT}/trajectories}"
export HARBOR_LOGPROB_SOURCE HARBOR_LOGPROB_SEMANTICS
export EXPLORE_INTRINSIC=0 EXPLORE_AGENT57_LITE=0 DAPO_OVERLONG_BUFFER_ENABLE=0
exec bash "${ROOT}/harborrl/platform/slime_train.sh" "$@"
