#!/usr/bin/env bash
set -euo pipefail

SLIME_REPO="${SLIME_REPO:-https://github.com/THUDM/slime.git}"
MEGATRON_REPO="${MEGATRON_REPO:-https://github.com/NVIDIA/Megatron-LM.git}"
SGLANG_IMAGE="${SGLANG_IMAGE:-lmsysorg/sglang:v0.5.15.post1-cu129}"
SLIME_COMMIT=3778dbf6d1a533ab478ecf5ddaa11449a47752b2
MEGATRON_COMMIT=1dcf0dafa884ad52ffb243625717a3471643e087
PREFIX="${1:-${BACKEND_ROOT:-$PWD/.backends}}"

need() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "missing required command: $1" >&2
    exit 1
  }
}

clone_pinned() {
  local repo="$1" path="$2" commit="$3"
  if [[ ! -d "$path/.git" ]]; then
    git clone "$repo" "$path"
  fi
  git -C "$path" fetch origin "$commit" --force
  git -C "$path" checkout --detach "$commit"
  [[ "$(git -C "$path" rev-parse HEAD)" == "$commit" ]]
}

need git
need docker
mkdir -p "$PREFIX"
clone_pinned "$SLIME_REPO" "$PREFIX/slime" "$SLIME_COMMIT"
clone_pinned "$MEGATRON_REPO" "$PREFIX/Megatron-LM" "$MEGATRON_COMMIT"
docker pull "$SGLANG_IMAGE"

cat >"$PREFIX/harborrl-backend.env" <<ENV
SLIME_DIR=$PREFIX/slime
MEGATRON_DIR=$PREFIX/Megatron-LM
SGLANG_IMAGE=$SGLANG_IMAGE
ENV

echo "Bootstrap complete. Source $PREFIX/harborrl-backend.env before doctor."
