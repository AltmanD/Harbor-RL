"""Canonical trajectory IR v1. No training/runtime dependencies."""

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Trajectory:
    schema_version: str = "1"
    task: dict[str, Any] = field(default_factory=dict)
    trajectory_id: str = ""
    attempt_id: str = ""
    group_id: str = ""
    execution_backend: str = "interactive"
    policy: dict[str, Any] = field(default_factory=dict)
    turns: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    evaluation: dict[str, Any] = field(default_factory=dict)
    execution_status: str = "incomplete"
    termination_reason: str = ""
    readiness: str = "EVAL_ONLY"
    derived: dict[str, Any] = field(default_factory=dict)
    integrity_errors: list[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        return cls(**value)
