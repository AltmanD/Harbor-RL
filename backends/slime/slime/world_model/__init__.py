"""Optional JEPA-style latent world model utilities for text agents."""

from .metadata import attach_terminal_world_model_metadata, is_world_model_enabled

__all__ = [
    "SIGReg",
    "TextLatentWorldModel",
    "TextLatentWorldModelConfig",
    "TerminalTransition",
    "TrajectoryReplayBuffer",
    "MPCPlan",
    "plan_one_step",
    "plan_stream",
    "attach_terminal_world_model_metadata",
    "is_world_model_enabled",
]


def __getattr__(name):
    if name in {"SIGReg", "TextLatentWorldModel", "TextLatentWorldModelConfig"}:
        from .modules import SIGReg, TextLatentWorldModel, TextLatentWorldModelConfig

        values = {
            "SIGReg": SIGReg,
            "TextLatentWorldModel": TextLatentWorldModel,
            "TextLatentWorldModelConfig": TextLatentWorldModelConfig,
        }
        return values[name]
    if name == "TerminalTransition":
        from .seta_dataset import TerminalTransition

        return TerminalTransition
    if name == "TrajectoryReplayBuffer":
        from .replay_buffer import TrajectoryReplayBuffer

        return TrajectoryReplayBuffer
    if name in {"MPCPlan", "plan_one_step"}:
        from .mpc import MPCPlan, plan_one_step

        return {"MPCPlan": MPCPlan, "plan_one_step": plan_one_step}[name]
    if name == "plan_stream":
        from .stream_latent import plan_stream

        return plan_stream
    raise AttributeError(name)
