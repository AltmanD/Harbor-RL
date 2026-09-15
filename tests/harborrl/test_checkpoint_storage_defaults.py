from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PERSIST_ROOT = "runs/.persistent"


def test_persistent_checkpoint_and_wandb_defaults_are_portable():
    paths = (ROOT / "harborrl/platform/paths.py").read_text(encoding="utf-8")
    run_dir = (ROOT / "harborrl/platform/slime_train/lib_run_dir.sh").read_text(encoding="utf-8")

    assert PERSIST_ROOT in paths
    assert '${RUNS_ROOT}/.persistent' in run_dir
    assert '${LIGHTRL_PERSIST_ROOT}/checkpoints' in run_dir
    assert '${LIGHTRL_PERSIST_ROOT}/wandb/${RUN_ID}' in run_dir
    assert "/mnt/shared-storage" not in paths
    assert "/mnt/shared-storage" not in run_dir


def test_wandb_is_offline_enabled_and_does_not_put_key_on_cli():
    args = (ROOT / "harborrl/platform/slime_train/lib_args.sh").read_text(encoding="utf-8")

    assert 'WANDB_MODE="${WANDB_MODE:-offline}"' in args
    assert 'WANDB_ENABLE="${WANDB_ENABLE:-1}"' in args
    assert "unset WANDB_API_KEY WANDB_KEY" in args
    assert "--wandb-key" not in args


def test_shell_cleanup_cannot_blindly_remove_checkpoint_directories():
    launcher = (ROOT / "harborrl/platform/slime_train/lib_launch.sh").read_text(encoding="utf-8")

    assert 'rm -rf "${CKPT_DIRS[$i]}"' not in launcher
    assert "checkpoint_utils.py" in launcher
