"""Camel-Agent harness with an optional-dependency-safe public export."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .agent import CamelAgent

__all__ = ["CamelAgent"]


def __getattr__(name: str) -> Any:
    if name == "CamelAgent":
        from .agent import CamelAgent

        return CamelAgent
    raise AttributeError(name)
