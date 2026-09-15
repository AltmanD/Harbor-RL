"""Evaluation harness adapters (distinct from the training rollout harnesses).

Lazy registry mirroring :mod:`harborrl.harnesses.factory`; importing this
package never pulls heavy optional dependencies.
"""

from __future__ import annotations

from importlib import import_module

from harborrl.harnesses.identity import aliases_for, get_harness_descriptor

_EVAL_HARNESS_ALIASES = aliases_for("eval")
_EVAL_HARNESS_TARGETS = {
    descriptor.canonical_name: tuple(descriptor.eval_target.rsplit(":", 1))
    for descriptor in (get_harness_descriptor(name) for name in _EVAL_HARNESS_ALIASES.values())
    if descriptor.eval_target is not None
}


def normalize_eval_harness_name(value: str | None) -> str:
    requested = value or "terminus2"
    try:
        return get_harness_descriptor(requested, capability="eval").canonical_name
    except ValueError as exc:
        raise ValueError(
            f"Unsupported eval harness option: {value!r}. "
            "Available: terminus-2, claude-code, camel-agent"
        ) from exc


def create_eval_harness(name: str | None):
    """Instantiate an eval harness adapter, importing its module lazily."""
    module_name, class_name = _EVAL_HARNESS_TARGETS[normalize_eval_harness_name(name)]
    return getattr(import_module(module_name), class_name)()


__all__ = ["create_eval_harness", "normalize_eval_harness_name"]
