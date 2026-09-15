"""Centralized path management for a single run.

Runs are grouped by lifecycle under ``runs/``:

* ``training/<run-id>`` for model training;
* ``evaluation/<run-id>`` for benchmark/evaluation jobs;
* ``testing/<run-id>`` for smoke/validation jobs;
* ``testing/debug/<run-id>`` for debug jobs.

Set ``RUN_CATEGORY`` to override the default (``training``). An explicit
``RUN_DIR`` always wins, which keeps resume commands and external integrations
backwards compatible.

Usage in shell script:
    RUN_ID=$(python3 -m harborrl.platform.paths init --runs-root ./runs --ckpt-root /mnt/.../ckpt)
    # prints JSON with all paths to stdout

Usage in Python (rollout/entrypoint.py etc.):
    from harborrl.platform.paths import RunPaths
    rp = RunPaths.from_env()  # reads RUN_DIR env var; None when unset
    traj_dir = rp.trajectories_dir
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_PERSIST_ROOT = os.getenv("LIGHTRL_PERSIST_ROOT", "runs/.persistent")
DEFAULT_CKPT_ROOT = f"{DEFAULT_PERSIST_ROOT}/checkpoints"

RUN_CATEGORIES = ("training", "evaluation", "testing", "testing/debug")
_CATEGORY_ALIASES = {
    "train": "training",
    "training": "training",
    "eval": "evaluation",
    "evaluate": "evaluation",
    "evaluation": "evaluation",
    "test": "testing",
    "testing": "testing",
    "debug": "testing/debug",
    "testing/debug": "testing/debug",
}


def normalize_run_category(category: str | None = None, *, debug: bool = False) -> str:
    """Return a safe canonical run category.

    ``debug=True`` intentionally routes to ``testing/debug`` so the repository
    no longer needs a top-level ``runs/debug`` directory.
    """
    raw = str(category or os.getenv("RUN_CATEGORY", "training")).strip().strip("/").lower()
    canonical = _CATEGORY_ALIASES.get(raw)
    if canonical is None:
        choices = ", ".join(RUN_CATEGORIES)
        raise ValueError(f"unknown run category {category!r}; expected one of: {choices}")
    return "testing/debug" if debug else canonical


def resolve_run_dir(
    run_id: str,
    runs_root: str | Path = "runs",
    *,
    category: str | None = None,
    debug: bool = False,
) -> Path:
    """Resolve the canonical directory for ``run_id``."""
    return Path(runs_root) / normalize_run_category(category, debug=debug) / run_id


class RunPaths:
    def __init__(
        self,
        run_id: str,
        runs_root: Path,
        ckpt_root: Path,
        *,
        category: str | None = None,
        run_dir: Path | None = None,
    ):
        self.run_id = run_id
        # ``run_dir`` is used by ``from_env`` and is deliberately authoritative
        # for resumed/externally managed runs. New runs use the categorized
        # layout by default.
        self.category = normalize_run_category(category) if run_dir is None else (category or "")
        self.run_dir = Path(run_dir) if run_dir is not None else resolve_run_dir(
            run_id, runs_root, category=self.category
        )
        self.config_dir = self.run_dir / "config"
        self.logs_dir = self.run_dir / "logs"
        self.trajectories_dir = self.run_dir / "trajectories"
        self.metrics_dir = self.run_dir / "metrics"
        self.wandb_dir = Path(os.getenv("WANDB_DIR", str(self.metrics_dir / "wandb")))
        self.analysis_dir = self.metrics_dir / "analysis"
        self.environment_outputs_dir = self.run_dir / "environment_outputs"
        self.ckpt_real = ckpt_root / run_id
        self.ckpt_link = self.run_dir / "ckpt"
        self.meta_file = self.run_dir / "meta.json"

    @classmethod
    def from_env(cls) -> "RunPaths | None":
        run_dir = os.getenv("RUN_DIR", "")
        if not run_dir:
            return None
        run_dir = Path(run_dir)
        run_id = run_dir.name
        runs_root = run_dir.parent
        default_ckpt_root = (
            Path(os.getenv("RUNS_ROOT", "runs")) / ".persistent" / "checkpoints"
        )
        ckpt_root = Path(os.getenv("CKPT_ROOT", str(default_ckpt_root)))
        return cls(run_id, runs_root, ckpt_root, run_dir=run_dir)

    def create_all(self) -> None:
        for d in [
            self.config_dir, self.logs_dir, self.trajectories_dir,
            self.metrics_dir, self.wandb_dir, self.analysis_dir,
            self.environment_outputs_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)
        if os.getenv("MAX_CKPT_KEEP", "0") != "0":
            self.ckpt_real.mkdir(parents=True, exist_ok=True)
            if not self.ckpt_link.exists():
                try:
                    self.ckpt_link.symlink_to(self.ckpt_real)
                except OSError:
                    pass

    def write_meta(self, extra: dict[str, Any] | None = None) -> Path:
        meta = {
            "run_id": self.run_id,
            "category": self.category or None,
            "start_time": datetime.now(timezone.utc).isoformat(),
            "hostname": os.uname().nodename,
            "git_commit": _git_commit(),
            "git_branch": _git_branch(),
            "command": " ".join(sys.argv),
            "paths": {
                "run_dir": str(self.run_dir),
                "logs": str(self.logs_dir),
                "trajectories": str(self.trajectories_dir),
                "metrics": str(self.metrics_dir),
                "wandb": str(self.wandb_dir),
                "environment_outputs": str(self.environment_outputs_dir),
                "ckpt_real": str(self.ckpt_real),
            },
        }
        if extra:
            meta.update(extra)
        self.meta_file.write_text(json.dumps(meta, indent=2, default=str))
        return self.meta_file

    def print_summary(self) -> None:
        print("=" * 60)
        print(f"  Run ID:        {self.run_id}")
        print(f"  Run dir:       {self.run_dir}")
        print(f"  Logs:          {self.logs_dir}")
        print(f"  Trajectories:  {self.trajectories_dir}")
        print(f"  Metrics/wandb: {self.wandb_dir}")
        print(f"  Ckpt (real):   {self.ckpt_real}")
        print(f"  Ckpt (link):   {self.ckpt_link}")
        print(f"  Meta:          {self.meta_file}")
        print("=" * 60)

    def to_env_dict(self) -> dict[str, str]:
        return {
            "RUN_ID": self.run_id,
            "RUN_DIR": str(self.run_dir),
            "RUN_CATEGORY": self.category,
            "RUN_LOG_DIR": str(self.logs_dir),
            "TERMINAL_SAVE_TRAJ_DIR": str(self.trajectories_dir),
            "WANDB_DIR": str(self.wandb_dir),
            "TBENCH_OUTPUT_ROOT": str(self.environment_outputs_dir),
            "SAVE_CKPT": str(self.ckpt_real),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_env_dict(), indent=2)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def _git_branch() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def generate_run_id(model: str = "qwen3-8b", method: str = "grpo") -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{model}_{method}"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    init_p = sub.add_parser("init")
    init_p.add_argument("--runs-root", default=os.getenv("RUNS_ROOT", "./runs"))
    init_p.add_argument(
        "--ckpt-root",
        default="",
        help="checkpoint root (default: <runs-root>/.persistent/checkpoints)",
    )

    init_p.add_argument("--model", default="qwen3-8b")
    init_p.add_argument("--method", default="grpo")
    init_p.add_argument("--run-id", default="")
    init_p.add_argument(
        "--run-dir",
        default="",
        help="explicit run directory; overrides the categorized default",
    )
    init_p.add_argument(
        "--category",
        "--run-category",
        dest="category",
        default=None,
        choices=tuple(_CATEGORY_ALIASES),
        help="lifecycle directory under runs/ (default: RUN_CATEGORY or training)",
    )
    init_p.add_argument(
        "--debug",
        action="store_true",
        help="route the run to testing/debug/<run-id>",
    )

    args = parser.parse_args()
    if args.cmd == "init":
        run_id = args.run_id or generate_run_id(args.model, args.method)
        category = args.category or os.getenv("RUN_CATEGORY", "training")
        runs_root = Path(args.runs_root)
        ckpt_root = Path(args.ckpt_root or (runs_root / ".persistent" / "checkpoints"))
        rp = RunPaths(
            run_id,
            runs_root,
            ckpt_root,
            category=normalize_run_category(category, debug=args.debug),
            run_dir=Path(args.run_dir) if args.run_dir else None,
        )
        rp.create_all()
        rp.write_meta()
        rp.print_summary()
        print(rp.to_json())
    else:
        parser.print_help()
