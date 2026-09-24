"""Backend-neutral schemas for Native IR v2 training export.

The schemas are immutable and dependency free on purpose: Slime, Torch, and
Megatron types must stay inside a backend adapter so the same CPU-testable
contract can serve future training backends.
"""
from __future__ import annotations

from dataclasses import dataclass

SCHEMA_VERSION = "harborrl-training-batch-v1"
REDUCTION = "trajectory_weighted_clipped_pg_v1"


def _as_tuple(value, label):
    if isinstance(value, (str, bytes)) or not hasattr(value, "__iter__"):
        raise ValueError(f"{label} must be a sequence")
    return tuple(value)


@dataclass(frozen=True)
class TokenSpan:
    """One policy turn expanded from a Native trajectory."""

    turn_index: int
    tokens: tuple[int, ...]
    prompt_length: int
    response_length: int
    loss_mask: tuple[bool, ...]
    old_logprobs: tuple[float, ...]
    policy_version: str

    def __post_init__(self):
        object.__setattr__(self, "tokens", _as_tuple(self.tokens, "tokens"))
        object.__setattr__(self, "loss_mask", _as_tuple(self.loss_mask, "loss_mask"))
        object.__setattr__(self, "old_logprobs", _as_tuple(self.old_logprobs, "old_logprobs"))


@dataclass(frozen=True)
class TrajectoryTrainingUnit:
    """A complete trajectory with its group-normalized training semantics."""

    trajectory_id: str
    rollout_id: int
    group_key: str
    reward: float
    advantage: float
    token_count: int
    turns: tuple[TokenSpan, ...]

    def __post_init__(self):
        object.__setattr__(self, "turns", _as_tuple(self.turns, "turns"))


@dataclass(frozen=True)
class TrainingGroup:
    """A complete prompt group of exactly ``group_size`` trajectories."""

    group_key: str
    group_size: int
    reward_mean: float
    reward_std: float
    trajectories: tuple[TrajectoryTrainingUnit, ...]

    def __post_init__(self):
        object.__setattr__(self, "trajectories", _as_tuple(self.trajectories, "trajectories"))


@dataclass(frozen=True)
class TrainingBatch:
    """The only object a training backend adapter is allowed to consume."""

    schema_version: str
    groups: tuple[TrainingGroup, ...]
    reduction: str
    policy_versions: frozenset[str]

    def __post_init__(self):
        object.__setattr__(self, "groups", _as_tuple(self.groups, "groups"))
        object.__setattr__(self, "policy_versions", frozenset(self.policy_versions))
