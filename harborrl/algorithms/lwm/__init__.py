"""Agentic-RL facing lazy API for the optional latent world model.

The implementation lives under ``slime.world_model`` because the offline
trainer and Megatron hook share that runtime.  Keeping this facade tiny avoids
importing torch/transformers during ordinary rollout startup.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "attach_terminal_world_model_metadata",
    "is_world_model_enabled",
    "world_model_records_from_samples",
    "TrajectoryReplayBuffer",
]


def __getattr__(name: str) -> Any:
    if name in {"attach_terminal_world_model_metadata", "is_world_model_enabled"}:
        from slime.world_model.metadata import attach_terminal_world_model_metadata, is_world_model_enabled

        return {
            "attach_terminal_world_model_metadata": attach_terminal_world_model_metadata,
            "is_world_model_enabled": is_world_model_enabled,
        }[name]
    if name in {"world_model_records_from_samples", "TrajectoryReplayBuffer"}:
        from slime.world_model.replay_buffer import TrajectoryReplayBuffer, world_model_records_from_samples

        return {
            "world_model_records_from_samples": world_model_records_from_samples,
            "TrajectoryReplayBuffer": TrajectoryReplayBuffer,
        }[name]
    raise AttributeError(name)
