"""Static hygiene guard: pyflakes must report no undefined names in the
rollout path.  Introduced after a refactor moved code between modules and
dropped imports that only failed at runtime inside the cluster job
(RayTaskError(NameError)); compileall and the unit suite cannot see those."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

pyflakes_available = (
    shutil.which("pyflakes") is not None
    or subprocess.run(
        [sys.executable, "-m", "pyflakes", "--version"],
        capture_output=True,
    ).returncode
    == 0
)


@pytest.mark.skipif(not pyflakes_available, reason="pyflakes not installed")
def test_no_undefined_names_in_rollout_path():
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pyflakes",
            str(REPO_ROOT / "harborrl" / "rollout"),
            str(REPO_ROOT / "harborrl" / "environments"),
            str(REPO_ROOT / "harborrl" / "misc"),
        ],
        capture_output=True,
        text=True,
    )
    undefined = [
        line
        for line in (proc.stdout + proc.stderr).splitlines()
        if "undefined name" in line
    ]
    assert not undefined, "undefined names found:\n" + "\n".join(undefined)


def test_slime_hook_modules_importable():
    """Every module path slime_train.sh hands to slime via --custom-*-path
    must import cleanly; a stale cross-module import only blows up at runtime
    otherwise (see the r2 validation ImportError)."""
    import importlib

    sys.path.insert(0, str(REPO_ROOT / "backends" / "slime"))
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    for module, func in (
        ("harborrl.rollout.entrypoint", "generate"),
        ("harborrl.rollout.generate_steps", "_run_turn_loop"),
        ("harborrl.misc.rollout_log", "rollout_log"),
        ("harborrl.misc.rollout_log", "eval_rollout_log"),
        ("harborrl.algorithms.dive_po.rewards.dual_stream", "post_process_rewards"),
    ):
        assert callable(getattr(importlib.import_module(module), func)), module
