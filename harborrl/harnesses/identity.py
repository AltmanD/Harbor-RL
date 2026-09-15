"""Shared harness identity and capability metadata.

Training harnesses and offline evaluation adapters intentionally remain
separate implementations.  This module only owns the stable names exposed to
users, so aliases cannot drift between the two registries.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HarnessDescriptor:
    """Public identity for one harness family.

    Targets use ``module:attribute`` notation and are kept as strings so
    importing the registry never imports optional runtime dependencies.
    """

    canonical_name: str
    display_name: str
    aliases: tuple[str, ...]
    train_target: str | None = None
    eval_target: str | None = None

    @property
    def capabilities(self) -> frozenset[str]:
        return frozenset(
            capability
            for capability, target in (("train", self.train_target), ("eval", self.eval_target))
            if target is not None
        )


HARNESS_DESCRIPTORS: tuple[HarnessDescriptor, ...] = (
    HarnessDescriptor(
        canonical_name="camel_agent",
        display_name="camel-agent",
        aliases=("camel_agent", "camel", "camel-agent", "camelagent"),
        train_target="harborrl.harnesses.camel.agent:CamelAgent",
        eval_target="harborrl.harnesses.eval.camel_agent:CamelAgentEvalHarness",
    ),
    HarnessDescriptor(
        canonical_name="claude_code_cli",
        display_name="claude-code",
        aliases=(
            "claude_code_cli",
            "claude",
            "claude-code",
            "claude_code",
            "claude-code-cli",
            "claude-code-harness",
        ),
        train_target="harborrl.harnesses.claude_code.agent:ClaudeCodeAgent",
        eval_target="harborrl.harnesses.eval.claude_code_cli:ClaudeCodeCliEvalHarness",
    ),
    HarnessDescriptor(
        canonical_name="terminus2",
        display_name="terminus-2",
        aliases=("terminus2", "terminus-2", "terminus_2"),
        eval_target="harborrl.harnesses.eval.terminus2:Terminus2EvalHarness",
    ),
)

_BY_ALIAS = {
    alias: descriptor
    for descriptor in HARNESS_DESCRIPTORS
    for alias in descriptor.aliases
}
_BY_CANONICAL = {descriptor.canonical_name: descriptor for descriptor in HARNESS_DESCRIPTORS}


def get_harness_descriptor(value: str | None, *, capability: str | None = None) -> HarnessDescriptor:
    """Resolve an alias and optionally require ``train`` or ``eval`` support."""

    requested = value or ("camel_agent" if capability == "train" else "terminus2")
    descriptor = _BY_ALIAS.get(requested) or _BY_CANONICAL.get(requested)
    if descriptor is None:
        raise ValueError(f"Unknown harness name: {value!r}")
    if capability is not None and capability not in descriptor.capabilities:
        raise ValueError(f"Harness {value!r} does not support {capability}")
    return descriptor


def aliases_for(capability: str) -> dict[str, str]:
    """Return the alias map for a capability (useful for CLI registries)."""

    return {
        alias: descriptor.canonical_name
        for descriptor in HARNESS_DESCRIPTORS
        if capability in descriptor.capabilities
        for alias in descriptor.aliases
    }


__all__ = ["HARNESS_DESCRIPTORS", "HarnessDescriptor", "aliases_for", "get_harness_descriptor"]
