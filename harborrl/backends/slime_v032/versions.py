"""Version gates for the Slime v0.3.2 backend contract."""
from __future__ import annotations

import json
import os
from pathlib import Path

BACKEND_CONTRACT = "slime-v0.3.2-native-v1"
SLIME_TAG = "v0.3.2"
SLIME_VERSION = "0.3.2"
SLIME_COMMIT = "3778dbf6d1a533ab478ecf5ddaa11449a47752b2"
MEGATRON_COMMIT = "1dcf0dafa884ad52ffb243625717a3471643e087"
SGLANG_IMAGE_TAG = "v0.5.15.post1-cu129"

CONFIG_CONTRACTS = frozenset({BACKEND_CONTRACT})


def expected_versions():
    return {
        "backend_contract": BACKEND_CONTRACT,
        "slime_version": SLIME_VERSION,
        "slime_commit": SLIME_COMMIT,
        "megatron_commit": MEGATRON_COMMIT,
        "sglang_image": SGLANG_IMAGE_TAG,
    }


def validate_backend_versions(
    *,
    slime_version=None,
    slime_commit=None,
    megatron_commit=None,
    sglang_image=None,
):
    """Fail closed unless the runtime matches the locked official combination."""
    expected = expected_versions()
    supplied = {
        "slime_version": (slime_version, lambda value: str(value).strip().lower().lstrip("v")),
        "slime_commit": (slime_commit, lambda value: str(value).strip().lower()),
        "megatron_commit": (megatron_commit, lambda value: str(value).strip().lower()),
        "sglang_image": (sglang_image, lambda value: str(value).strip()),
    }
    observed = {}
    for key, (value, normalize) in supplied.items():
        if value is None:
            raise ValueError(f"{key} is required for backend contract {BACKEND_CONTRACT}")
        normalized = normalize(value)
        if normalized != expected[key]:
            raise ValueError(
                f"{key} {normalized!r} does not match the locked {expected[key]!r}"
            )
        observed[key] = str(value).strip()
    return observed


def runtime_contract_id(config_contract):
    """Map the config contract to the launcher's runtime selector."""
    value = str(config_contract).strip()
    if value == BACKEND_CONTRACT:
        return "slime-v032"
    raise ValueError(f"unsupported training backend contract: {config_contract!r}")


def read_locked_policy_version(run_dir):
    """Read the policy-pool version that must be published before generation."""
    path = Path(run_dir) / "policy-pool.json"
    if not path.is_file():
        raise ValueError(f"native rollout requires a locked policy pool: {path}")
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid policy pool record: {path}") from exc
    version = payload.get("weight_version") if isinstance(payload, dict) else None
    if not isinstance(version, str) or not version.strip():
        raise ValueError(f"native rollout requires a versioned policy pool: {path}")
    return version


def lock_policy_version(args):
    """Lock the weight version before any native generation request."""
    run_dir = os.environ.get("RUN_DIR")
    if not isinstance(run_dir, str) or not run_dir.strip():
        raise ValueError("native rollout requires RUN_DIR")
    return read_locked_policy_version(run_dir)
