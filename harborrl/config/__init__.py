"""Native schema-2 configuration loading."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

__all__ = ["ROOT", "load_config", "launch_plan"]


def load_config(path, overrides=None):
    from .native import load_config as _load_config

    return _load_config(path, overrides)


def launch_plan(config):
    from .native import launch_plan as _launch_plan

    return _launch_plan(config)
