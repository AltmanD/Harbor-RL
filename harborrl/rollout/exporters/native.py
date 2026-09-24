"""Trajectory-aware GRPO export, independent of Slime/Torch imports.

These records must enter a backend that honors token_weights and advantages.
They must NOT be passed through the interactive reward/whitening builder.
"""
import math
from harborrl.trajectories.native import Identity, finite, require_ready


def export_group(trajectories, group_size, *, epsilon=1e-6, allow_synthetic=False):
    if type(group_size) is not int or group_size < 2 or len(trajectories) != group_size:
        raise ValueError("a complete group of at least two trajectories is required")
    if not finite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be positive and finite")
    for ir in trajectories:
        require_ready(ir, allow_synthetic=allow_synthetic)
    ids = [Identity(**ir["identity"]) for ir in trajectories]
    if len({i.group_key for i in ids}) != 1:
        raise ValueError("mixed group, policy, task or profiles")
    for key in ("slot_id", "attempt_id", "trajectory_id"):
        if len({getattr(i, key) for i in ids}) != group_size:
            raise ValueError(f"duplicate {key}")
    rewards = [ir["evaluation"]["training_reward"] for ir in trajectories]
    mean = math.fsum(rewards) / group_size
    std = math.sqrt(math.fsum((r - mean) ** 2 for r in rewards) / group_size)
    if not finite(mean) or not finite(std):
        raise ValueError("nonfinite group statistics")
    advantages = [0.0 if std == 0 else (r - mean) / (std + epsilon) for r in rewards]
    records = []
    for ir, ident, advantage in zip(trajectories, ids, advantages):
        count = sum(len(t["output_ids"]) for t in ir["turns"])
        for index, turn in enumerate(ir["turns"]):
            n = len(turn["output_ids"])
            records.append({"identity": ident.to_dict(), "turn_index": index,
                            "tokens": turn["input_ids"] + turn["output_ids"],
                            "prompt_length": len(turn["input_ids"]), "response_length": n,
                            "loss_mask": [0] * len(turn["input_ids"]) + [1] * n,
                            "old_logprobs": turn["logprobs"], "advantages": [advantage] * n,
                            "token_weights": [1.0 / (group_size * count)] * n,
                            "training_reward": ir["evaluation"]["training_reward"],
                            "padding": False})
    return {"export_version": "native-grpo-v2", "group_key": ids[0].group_key,
            "trajectory_count": group_size, "mean": mean, "std": std,
            "advantages": advantages, "records": records,
            "synthetic": allow_synthetic,
            "reduction": "sum_token_weighted_loss; no secondary whitening"}
