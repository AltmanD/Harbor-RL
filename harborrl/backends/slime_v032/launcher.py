"""Hook assembly and dry-run support for the Slime v0.3.2 adapter."""
from __future__ import annotations

import sys

from harborrl.backends.slime_v032.versions import BACKEND_CONTRACT

HOOK_PATHS = {
    "rollout": "harborrl.backends.slime_v032.rollout.generate_rollout",
    "converter": "harborrl.backends.slime_v032.converter.convert_samples_to_train_data",
    "advantage": "harborrl.backends.slime_v032.advantage.compute_advantages_and_returns",
    "loss": "harborrl.backends.slime_v032.loss.loss_function",
    "postprocess": "harborrl.backends.slime_v032.postprocess.rollout_data_postprocess",
}


def hook_arguments(contract=BACKEND_CONTRACT):
    if str(contract) != BACKEND_CONTRACT:
        raise ValueError(f"unsupported backend contract: {contract!r}")
    return [
        "--rollout-function-path", HOOK_PATHS["rollout"],
        "--custom-convert-samples-to-train-data-path", HOOK_PATHS["converter"],
        "--custom-advantage-function-path", HOOK_PATHS["advantage"],
        "--loss-type", "custom_loss",
        "--custom-loss-function-path", HOOK_PATHS["loss"],
        "--rollout-data-postprocess-path", HOOK_PATHS["postprocess"],
    ]


def validate_hook_assignment(assignment, *, contract=BACKEND_CONTRACT):
    if str(contract) != BACKEND_CONTRACT:
        raise ValueError(f"unsupported backend contract: {contract!r}")
    observed = {key: value for key, value in assignment.items() if key in HOOK_PATHS}
    if observed != HOOK_PATHS:
        raise ValueError(f"Slime hook assignment must exactly match {BACKEND_CONTRACT}")
    return observed


def hook_modules():
    return {name: path.rsplit(".", 1)[0] for name, path in HOOK_PATHS.items()}


def dry_run_command(entrypoint, base_args=(), *, contract=BACKEND_CONTRACT, python=sys.executable):
    return [
        str(python), "-u", str(entrypoint),
        *[str(arg) for arg in base_args],
        *hook_arguments(contract),
    ]
