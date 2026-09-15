"""
Prioritized Experience Replay (PER) for Off-Policy GRPO Buffer.

Implements proportional PER (Schaul et al., 2016) adapted to group-based GRPO:
- Priority p_i = (|reward_dev_i| + eps)^alpha  (per group; safe default source = reward deviation)
- Sample probability P(i) = p_i / sum(p_j)
- Importance-sampling correction w_i = (N * P(i))^(-beta) / max_w  (normalized to <=1)
- beta linearly anneals from beta_start (=0.4) to beta_end (=1.0)
  over `per_beta_anneal_steps` rollouts (or via current_policy_version).

Design notes:
- Subclasses `BaseSamplingStrategy` so it plugs into the existing factory in
  `slime.utils.buffer_sampling_strategies.get_sampling_strategy`.
- Uses an O(N) re-derivation each call instead of a persistent SumTree:
  buffer cap is typically <=4096 for 7B-20B runs, far cheaper than rollout.
- IS weights are attached to `sample.metadata['per_is_weight']` of every
  sample inside the sampled group; `rollout.py` carries them as
  `per_is_weights`, and `loss.py` expands them to per-token loss weights.
- Priority is cached in `group[0].metadata['per_priority']` on first encounter;
  call `update_per_priorities_from_advantages()` explicitly if a caller wants to
  refresh priorities after fresh advantage estimates are available.

All behavior is gated by --buffer-sampling-strategy=per (no impact when off).
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np

from slime.utils.types import Sample
from slime.utils.buffer_sampling_strategies import BaseSamplingStrategy


def _safe_mean(values):
    vs = [v for v in values if v is not None]
    if not vs:
        return 0.0
    return float(sum(vs)) / float(len(vs))


def _extract_group_priority_value(group, source):
    """Return scalar priority signal for a group, or None if unavailable."""
    if source == "advantage":
        advs = []
        for s in group:
            md = s.metadata or {}
            a = md.get("advantage_abs_mean", None)
            if a is None:
                a = md.get("advantage", None)
            if a is None:
                continue
            try:
                a = float(np.abs(np.asarray(a)).mean())
            except Exception:
                continue
            advs.append(a)
        if not advs:
            return _extract_group_priority_value(group, "reward_dev")
        return _safe_mean(advs)

    if source == "reward":
        rs = []
        for s in group:
            r = getattr(s, "reward", None)
            if r is None and s.metadata:
                r = s.metadata.get("reward", None)
            if r is None:
                continue
            try:
                rs.append(float(r))
            except Exception:
                continue
        if not rs:
            return None
        return float(abs(_safe_mean(rs)))

    if source == "reward_dev":
        rs = []
        for s in group:
            r = getattr(s, "reward", None)
            if r is None and s.metadata:
                r = s.metadata.get("reward", None)
            if r is None:
                continue
            try:
                rs.append(float(r))
            except Exception:
                continue
        if len(rs) < 2:
            return None
        mu = sum(rs) / len(rs)
        return float(sum(abs(r - mu) for r in rs) / len(rs))

    raise ValueError(f"Unknown PER priority source: {source}")


class PrioritizedReplayStrategy(BaseSamplingStrategy):
    """
    Proportional Prioritized Experience Replay sampling strategy.

    Differs from the existing `PrioritySamplingStrategy` (soft rank-by-reward
    without IS correction) by implementing the full Schaul-2016 PER formulation:
    power-law priority, probabilistic sampling, importance-sampling correction
    with beta annealing.

    Configuration (all via args):
        --buffer-sampling-strategy per       enable this strategy
        --per-alpha                  0.6     priority exponent
        --per-beta-start             0.4     starting IS-correction beta
        --per-beta-end               1.0     final IS-correction beta
        --per-beta-anneal-steps      1000    rollout count for beta to reach 1.0
        --per-priority-source        reward_dev advantage|reward|reward_dev
        --per-priority-eps           1e-3    additive epsilon for zero-prio guard
        --per-min-priority           1e-6    floor for new samples
        --per-max-priority           1e3     ceiling clip
    """

    def __init__(self, args, current_policy_version):
        super().__init__(args, current_policy_version)
        self.alpha = float(getattr(args, "per_alpha", 0.6))
        self.beta_start = float(getattr(args, "per_beta_start", 0.4))
        self.beta_end = float(getattr(args, "per_beta_end", 1.0))
        self.beta_anneal_steps = max(1, int(getattr(args, "per_beta_anneal_steps", 1000)))
        self.priority_source = str(getattr(args, "per_priority_source", "reward_dev"))
        self.priority_eps = float(getattr(args, "per_priority_eps", 1e-3))
        self.priority_floor = float(getattr(args, "per_min_priority", 1e-6))
        self.priority_ceil = float(getattr(args, "per_max_priority", 1e3))

        self.last_beta = self.beta_start
        self.last_priority_max = 0.0
        self.last_priority_mean = 0.0
        self.last_is_weight_mean = 1.0
        self.last_is_weight_min = 1.0
        # Keep sampling deterministic per run and independent of process-wide
        # numpy RNG consumers in rollout/evaluation code.
        self._rng = np.random.default_rng(int(getattr(args, "seed", 42)))

    def get_name(self):
        return f"per_{self.priority_source}"

    def _group_priority(self, group):
        if not group:
            return self.priority_floor
        head = group[0]
        if head.metadata is None:
            head.metadata = {}
        cached = head.metadata.get("per_priority", None)
        if cached is not None:
            try:
                return float(cached)
            except Exception:
                pass
        raw = _extract_group_priority_value(group, self.priority_source)
        if raw is None:
            # New / unscored sample -> max known priority (Schaul §3.3)
            new_p = max(self.priority_floor, self.last_priority_max)
        else:
            new_p = max(self.priority_floor, float(raw))
        new_p = min(self.priority_ceil, new_p)
        head.metadata["per_priority"] = new_p
        return new_p

    def _current_beta(self):
        progress = min(1.0, max(0.0, self.current_policy_version / self.beta_anneal_steps))
        beta = self.beta_start + (self.beta_end - self.beta_start) * progress
        self.last_beta = beta
        return beta

    def sample(self, buffer, num_samples):
        if len(buffer) == 0 or num_samples == 0:
            return []

        valid_groups, stale_groups = self.filter_by_staleness(buffer)
        if stale_groups:
            self.remove_from_buffer(buffer, stale_groups)

        reusable = []
        for g in valid_groups:
            head = g[0]
            rc = head.metadata.get("buffer_reuse_count", 0) if head.metadata else 0
            if self.max_reuse_count <= 0 or rc < self.max_reuse_count:
                reusable.append(g)
        if not reusable:
            return []

        priorities = np.asarray(
            [self._group_priority(g) for g in reusable], dtype=np.float64
        )
        priorities = priorities + self.priority_eps
        p_alpha = np.power(priorities, self.alpha)
        total = float(p_alpha.sum())
        if not math.isfinite(total) or total <= 0:
            probs = np.full(len(reusable), 1.0 / len(reusable), dtype=np.float64)
        else:
            probs = p_alpha / total

        k = int(min(num_samples, len(reusable)))
        try:
            idxs = self._rng.choice(len(reusable), size=k, replace=False, p=probs)
        except ValueError:
            idxs = self._rng.choice(len(reusable), size=k, replace=True, p=probs)

        beta = self._current_beta()
        N = len(reusable)
        sample_probs = np.maximum(probs[idxs], 1e-12)
        is_weights = np.power(N * sample_probs, -beta)
        w_max = float(is_weights.max())
        if w_max > 0:
            is_weights = is_weights / w_max
        is_weights = np.clip(is_weights, 0.0, 1.0)

        sampled = []
        for idx, w in zip(idxs.tolist(), is_weights.tolist()):
            group = reusable[idx]
            for s in group:
                if s.metadata is None:
                    s.metadata = {}
                s.metadata["per_is_weight"] = float(w)
                s.metadata["per_sample_prob"] = float(probs[idx])
            sampled.append(group)

        self.total_sampled += len(sampled)
        self.last_priority_max = float(priorities.max())
        self.last_priority_mean = float(priorities.mean())
        self.last_is_weight_mean = float(np.mean(is_weights))
        self.last_is_weight_min = float(np.min(is_weights))

        if self.remove_on_sample and sampled:
            self.remove_from_buffer(buffer, sampled)
        elif sampled:
            self.increment_reuse_count(sampled)

        return sampled

    def get_statistics(self):
        base = super().get_statistics()
        base.update({
            "per_alpha": self.alpha,
            "per_beta": self.last_beta,
            "per_beta_start": self.beta_start,
            "per_beta_end": self.beta_end,
            "per_priority_source": self.priority_source,
            "per_priority_max": self.last_priority_max,
            "per_priority_mean": self.last_priority_mean,
            "per_is_weight_mean": self.last_is_weight_mean,
            "per_is_weight_min": self.last_is_weight_min,
        })
        return base


def update_per_priorities_from_advantages(
    samples,
    priority_source="reward_dev",
    priority_floor=1e-6,
    priority_ceil=1e3,
):
    """Refresh metadata['per_priority'] after fresh advantage computation."""
    if not samples:
        return
    by_group = {}
    for s in samples:
        key = getattr(s, "group_index", None)
        by_group.setdefault(key, []).append(s)
    for key, group in by_group.items():
        raw = _extract_group_priority_value(group, priority_source)
        if raw is None:
            continue
        new_p = max(priority_floor, min(priority_ceil, float(raw)))
        for s in group:
            if s.metadata is None:
                s.metadata = {}
            s.metadata["per_priority"] = new_p
