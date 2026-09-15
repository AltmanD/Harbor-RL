"""Configuration loading for the evaluation tool layer.

YAML files are loaded with PyYAML (a declared project dependency, imported
lazily here), support ``${VAR}`` environment expansion, and can be overridden
from the CLI with repeated ``--set a.b=c`` options.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand_env(text: str) -> str:
    """Expand ``${VAR}`` / ``${VAR:-default}`` / ``$VAR`` references."""

    def repl(match: re.Match) -> str:
        name, default = match.group(1), match.group(2)
        if name in os.environ:
            return os.environ[name]
        if default is not None:
            return default
        return match.group(0)

    return os.path.expandvars(_ENV_PATTERN.sub(repl, text))


def repo_root() -> Path:
    return _REPO_ROOT


def default_run_dir(category: str, run_id: str) -> Path:
    """Return a categorized repository-local run directory.

    ``RUNS_ROOT`` lets cluster jobs keep the same layout on shared storage;
    otherwise the repository's ``runs/`` directory is used.
    """
    from harborrl.platform.paths import resolve_run_dir

    runs_root = os.getenv("RUNS_ROOT", str(_REPO_ROOT / "runs"))
    return resolve_run_dir(run_id, runs_root, category=category)


def ensure_repo_on_path() -> None:
    """Make ``harborrl`` importable when running as a loose script."""
    import sys

    try:
        import harborrl  # noqa: F401
        return
    except ImportError:
        pass
    root = str(_REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    """Load a YAML config, expand ``${VAR}``, then apply ``a.b=c`` overrides."""
    import yaml  # lazy: optional dependency, declared in pyproject.toml

    text = _expand_env(Path(path).read_text(encoding="utf-8"))
    data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config root must be a mapping: {path}")
    for override in overrides or []:
        apply_override(data, override)
    return data


def apply_override(config: dict, override: str) -> None:
    """Apply one ``a.b=c`` override; ``c`` is parsed as a YAML scalar."""
    import yaml

    if "=" not in override:
        raise ValueError(f"invalid --set override (expected a.b=c): {override!r}")
    key, _, raw_value = override.partition("=")
    value = yaml.safe_load(raw_value)
    parts = [part for part in key.split(".") if part]
    if not parts:
        raise ValueError(f"invalid --set override (empty key): {override!r}")
    node: dict = config
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = child
    node[parts[-1]] = value


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into a copy of ``base``."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def build_specs(config: dict) -> tuple[Any, Any]:
    """Turn a loaded config dict into ``(EvalRunSpec, ServingSpec)``."""
    ensure_repo_on_path()
    from harborrl.harnesses.eval.base import EvalRunSpec, ServingSpec

    serving_cfg = dict(config.get("serving") or {})
    serving = ServingSpec(
        mode=str(serving_cfg.get("mode", "external")),
        api_base=str(serving_cfg.get("api_base", "")),
        model_path=str(serving_cfg.get("model_path", "")),
        model_name=str(serving_cfg.get("model_name", "")),
        port=int(serving_cfg.get("port", 30000)),
        gpu_ids=[int(g) for g in serving_cfg.get("gpu_ids") or []],
        tp_size=int(serving_cfg.get("tp_size", 1)),
        mem_fraction=float(serving_cfg.get("mem_fraction", 0.70)),
        extra_args=[str(a) for a in serving_cfg.get("extra_args") or []],
        health_timeout_s=float(serving_cfg.get("health_timeout_s", 900.0)),
    )

    run_cfg = dict(config.get("run") or {})
    dataset_cfg = dict(config.get("dataset") or {})
    task_names = dataset_cfg.get("task_names")
    job_name = str(config["job_name"])
    output_dir = config.get("output_dir")
    if output_dir is None or not str(output_dir).strip():
        output_dir = str(default_run_dir("evaluation", job_name))
    spec = EvalRunSpec(
        harness=str(config.get("harness", "terminus-2")),
        job_name=job_name,
        dataset_path=str(dataset_cfg["path"]),
        task_names=[str(t) for t in task_names] if task_names else None,
        output_dir=str(output_dir),
        n_attempts=int(run_cfg.get("n_attempts", 1)),
        concurrency=int(run_cfg.get("concurrency", 4)),
        max_retries=int(run_cfg.get("max_retries", 1)),
        timeout_multiplier=float(run_cfg.get("timeout_multiplier", 1.0)),
        max_input_tokens=int(run_cfg.get("max_input_tokens", 8192)),
        max_output_tokens=int(run_cfg.get("max_output_tokens", 8192)),
        environment={str(k): str(v) for k, v in dict(config.get("environment") or {}).items()},
        serving=serving,
        extra=dict(config.get("extra") or {}),
    )
    return spec, serving
