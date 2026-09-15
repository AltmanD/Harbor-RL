"""Shared Tau2 dependency adapters and task normalization."""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path
from typing import Any


def _install_deepdiff_stub() -> None:
    if "deepdiff" in sys.modules or importlib.util.find_spec("deepdiff") is not None:
        return

    module = types.ModuleType("deepdiff")

    class DeepDiff(dict):
        def __init__(self, left: Any, right: Any, *args: Any, **kwargs: Any) -> None:
            super().__init__()
            if left != right:
                self["values_changed"] = {
                    "root": {"old_value": left, "new_value": right}
                }

    module.DeepDiff = DeepDiff
    sys.modules["deepdiff"] = module


def _install_addict_stub() -> None:
    if "addict" in sys.modules or importlib.util.find_spec("addict") is not None:
        return

    module = types.ModuleType("addict")

    class Dict(dict):
        def __getattr__(self, key: str) -> Any:
            try:
                value = self[key]
            except KeyError as exc:
                raise AttributeError(key) from exc
            if isinstance(value, dict) and not isinstance(value, Dict):
                value = Dict(value)
                self[key] = value
            return value

        def __setattr__(self, key: str, value: Any) -> None:
            self[key] = value

        def update(self, other: Any = None, **kwargs: Any) -> None:
            items = dict(other or {})
            items.update(kwargs)
            for key, value in items.items():
                if key in self and isinstance(self[key], dict) and isinstance(value, dict):
                    nested = self[key]
                    if not isinstance(nested, Dict):
                        nested = Dict(nested)
                    nested.update(value)
                    self[key] = nested
                else:
                    self[key] = Dict(value) if isinstance(value, dict) else value

        def to_dict(self) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in self.items():
                if isinstance(value, Dict):
                    result[key] = value.to_dict()
                elif isinstance(value, dict):
                    result[key] = Dict(value).to_dict()
                else:
                    result[key] = value
            return result

    module.Dict = Dict
    sys.modules["addict"] = module


def _install_toml_stub() -> None:
    if "toml" in sys.modules or importlib.util.find_spec("toml") is not None:
        return

    import tomllib

    module = types.ModuleType("toml")

    def load(fp: Any) -> Any:
        if hasattr(fp, "read"):
            return tomllib.loads(fp.read())
        return tomllib.loads(Path(fp).read_text(encoding="utf-8"))

    def loads(text: str) -> Any:
        return tomllib.loads(text)

    module.load = load
    module.loads = loads
    sys.modules["toml"] = module


def ensure_tau2_importable(root: Path) -> None:
    _install_deepdiff_stub()
    _install_addict_stub()
    _install_toml_stub()

    src_dir = root / "src"
    if not src_dir.exists():
        raise FileNotFoundError(f"tau2 src dir not found: {src_dir}")
    src_dir_str = str(src_dir)
    if src_dir_str not in sys.path:
        sys.path.insert(0, src_dir_str)
    os.environ.setdefault("TAU2_DATA_DIR", str(root / "data"))


def _structured_instruction_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()

    lines: list[str] = []
    for label, attr in (
        ("Domain", "domain"),
        ("Reason", "reason_for_call"),
        ("Known info", "known_info"),
        ("Unknown info", "unknown_info"),
        ("Task instructions", "task_instructions"),
    ):
        raw = getattr(value, attr, None)
        if raw:
            lines.append(f"{label}: {raw}")
    return "\n".join(lines).strip()


def task_instruction(task: Any) -> str:
    ticket = getattr(task, "ticket", None)
    if ticket:
        return str(ticket).strip()

    user_scenario = getattr(task, "user_scenario", None)
    if user_scenario is not None:
        instructions = getattr(user_scenario, "instructions", None)
        structured = _structured_instruction_text(instructions)
        if structured:
            return structured
        if instructions is not None:
            return str(instructions).strip()

    description = getattr(task, "description", None)
    if description is not None:
        for attr in ("notes", "purpose"):
            raw = getattr(description, attr, None)
            if raw:
                return str(raw).strip()

    return str(getattr(task, "id", "unknown")).strip()


__all__ = ["ensure_tau2_importable", "task_instruction"]
