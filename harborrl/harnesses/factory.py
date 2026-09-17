from __future__ import annotations

import inspect
from importlib import import_module
from typing import Any

from harborrl.harnesses.identity import aliases_for, get_harness_descriptor

_HARNESS_ALIASES = aliases_for("train")
_HARNESS_TARGETS = {
    descriptor.canonical_name: tuple(descriptor.train_target.rsplit(":", 1))
    for descriptor in (get_harness_descriptor(name) for name in _HARNESS_ALIASES.values())
    if descriptor.train_target is not None
}
_HARNESS_DISPLAY_NAMES = {
    descriptor.canonical_name: descriptor.display_name
    for descriptor in (get_harness_descriptor(name) for name in _HARNESS_ALIASES.values())
}


def normalize_harness_name(value: str | None) -> str:
    requested = value or "camel_agent"
    try:
        return get_harness_descriptor(requested, capability="train").canonical_name
    except ValueError as exc:
        raise ValueError(
            f"Unsupported harness option: {value!r}. Available: camel-agent, claude-code"
        ) from exc


def display_harness_name(value: str | None) -> str:
    """Resolve any accepted harness alias to its canonical display name."""
    canonical = normalize_harness_name(value)
    return _HARNESS_DISPLAY_NAMES.get(canonical, canonical.replace("_", "-"))


def create_harness(name: str, **kwargs: Any) -> Any:
    """Instantiate a supported harness while keeping optional imports lazy."""
    module_name, class_name = _HARNESS_TARGETS[normalize_harness_name(name)]
    factory = getattr(import_module(module_name), class_name)
    signature = inspect.signature(factory)
    descriptor = get_harness_descriptor(name, capability="train")
    accepted = {key: value for key, value in kwargs.items()
                if key not in descriptor.ignored_train_arguments}
    try:
        signature.bind(**accepted)
    except TypeError as exc:
        raise ValueError(f"Invalid arguments for {descriptor.canonical_name}: {exc}") from exc
    return factory(**accepted)
