"""
DAPO-style Dynamic Sampling: group-quality admission filter for buffer + rollout queue.

Reference: DAPO (ByteDance, 2025) §4.1 "Dynamic Sampling".

Core idea
---------
In GRPO, all `n_samples_per_prompt` responses for a prompt share the same
prompt-level reward distribution. If every response in the group obtains the
same reward (std=0), the resulting group-relative advantages are all zero,
producing no gradient. Admitting such groups wastes downstream compute and,
worse, biases the advantage normalization on the next batch.

This module supplies:
  - `compute_group_quality(group)`  → diagnostic stats
  - `should_admit_group(quality, args)` → boolean gate, gated by
    `--enable-dynamic-sampling`
  - `select_admissible_groups(groups, args)` → applies the gate over a list,
    returns (admitted, rejected, stats).

All defaults are no-ops unless `--enable-dynamic-sampling` is set, so this
module has zero impact on baseline behavior.
"""

from __future__ import annotations

import math
from typing import List, Tuple

from slime.utils.types import Sample


def _group_rewards(group):
    rewards = []
    for s in group:
        r = getattr(s, "reward", None)
        if r is None and s.metadata:
            r = s.metadata.get("reward", None)
        if r is None:
            continue
        try:
            rewards.append(float(r))
        except Exception:
            continue
    return rewards


def compute_group_quality(group):
    """Return summary stats for a single sample group."""
    if not group:
        return {
            "n": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0,
            "n_correct": 0, "all_same": True, "has_signal": False,
        }
    rewards = _group_rewards(group)
    n = len(rewards)
    if n == 0:
        return {
            "n": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0,
            "n_correct": 0, "all_same": True, "has_signal": False,
        }
    mu = sum(rewards) / n
    var = sum((r - mu) ** 2 for r in rewards) / n
    std = math.sqrt(max(var, 0.0))
    n_correct = sum(1 for r in rewards if r > 0.5)
    all_same = std < 1e-12
    return {
        "n": n,
        "mean": mu,
        "std": std,
        "min": min(rewards),
        "max": max(rewards),
        "n_correct": n_correct,
        "all_same": all_same,
        "has_signal": not all_same,
    }


def should_admit_group(quality, args):
    """
    Decide whether a group should enter the buffer.

    Returns True if the group passes admission, False if it should be rejected.

    Gate rules (only active when `enable_dynamic_sampling`):
      1. `std >= dynamic_sample_min_std`  (reject zero-variance groups)
      2. `n >= dynamic_sample_min_correct_lo` correct
         AND `n <= dynamic_sample_min_correct_hi` correct (optional, both default off)
      3. Always admit if dynamic sampling is disabled.
    """
    if not getattr(args, "enable_dynamic_sampling", False):
        return True

    min_std = float(getattr(args, "dynamic_sample_min_std", 1e-4))
    if quality.get("std", 0.0) < min_std:
        return False

    lo = getattr(args, "dynamic_sample_min_correct_lo", None)
    hi = getattr(args, "dynamic_sample_min_correct_hi", None)
    n_correct = quality.get("n_correct", 0)
    if lo is not None and n_correct < int(lo):
        return False
    if hi is not None and n_correct > int(hi):
        return False

    return True


def select_admissible_groups(groups, args):
    """
    Filter a list of groups (list[list[Sample]]) by admission rule.

    Returns:
        admitted:   list of admitted groups (preserves order)
        rejected:   list of rejected groups
        stats:      dict with admit/reject counts + reasons (for wandb logging)
    """
    if not groups:
        return [], [], {
            "n_input": 0, "n_admitted": 0, "n_rejected": 0,
            "n_rejected_zero_std": 0, "n_rejected_correct_bound": 0,
            "admit_rate": 0.0,
        }
    if not getattr(args, "enable_dynamic_sampling", False):
        # no-op: keep behavior identical when feature is off
        return list(groups), [], {
            "n_input": len(groups), "n_admitted": len(groups), "n_rejected": 0,
            "n_rejected_zero_std": 0, "n_rejected_correct_bound": 0,
            "admit_rate": 1.0,
        }

    min_std = float(getattr(args, "dynamic_sample_min_std", 1e-4))
    lo = getattr(args, "dynamic_sample_min_correct_lo", None)
    hi = getattr(args, "dynamic_sample_min_correct_hi", None)

    admitted = []
    rejected = []
    n_rej_std = 0
    n_rej_corr = 0
    for g in groups:
        q = compute_group_quality(g)
        ok_std = q["std"] >= min_std
        ok_corr_lo = (lo is None) or (q["n_correct"] >= int(lo))
        ok_corr_hi = (hi is None) or (q["n_correct"] <= int(hi))
        if not ok_std:
            rejected.append(g)
            n_rej_std += 1
            continue
        if not (ok_corr_lo and ok_corr_hi):
            rejected.append(g)
            n_rej_corr += 1
            continue
        # tag the group with quality so downstream (PER, monitoring) can use it
        head = g[0]
        if head.metadata is None:
            head.metadata = {}
        head.metadata["group_quality_std"] = q["std"]
        head.metadata["group_quality_mean"] = q["mean"]
        head.metadata["group_quality_n_correct"] = q["n_correct"]
        admitted.append(g)

    n = len(groups)
    stats = {
        "n_input": n,
        "n_admitted": len(admitted),
        "n_rejected": len(rejected),
        "n_rejected_zero_std": n_rej_std,
        "n_rejected_correct_bound": n_rej_corr,
        "admit_rate": float(len(admitted)) / float(n) if n > 0 else 0.0,
    }
    return admitted, rejected, stats
