"""
SPEAR Self-Imitation Learning (SIL) buffer.

Stores high-score trajectories for replay training. Only admits trajectories
whose reward >= score_threshold, optionally restricted to those with positive
advantage (enable_trajectory_posadv).

Advantage re-estimation modes at sample time:
    weight_decay = -1.0  (default)
        A_new = R - baseline  (baseline = historical group-reward p50)
    weight_decay in [0, 1]
        A_new = weight_decay * A_stored

Ref: "Learn the Ropes, Then Trust the Wins: Self-imitation with Progressive
     Exploration" (Qin et al., 2026)
     https://github.com/TencentYoutuResearch/SPEAR
"""

import math
import random
from collections import deque
from typing import Any, Dict, Iterable, List, Optional

__all__ = ["SILBuffer", "normalize_sil_loss_mask"]


def normalize_sil_loss_mask(raw_mask: Any, response_length: int) -> List[int]:
    """Return a binary response mask cropped/padded to ``response_length``."""
    response_length = max(int(response_length), 0)
    if raw_mask is None:
        mask = [1] * response_length
    else:
        try:
            mask = list(raw_mask)
        except TypeError:
            mask = [raw_mask]
    mask = mask[:response_length]
    if len(mask) < response_length:
        mask.extend([1] * (response_length - len(mask)))
    return [1 if float(v.item() if hasattr(v, "item") else v) != 0.0 else 0 for v in mask]


class SILBuffer:
    """Fixed-capacity FIFO buffer for SPEAR self-imitation replay.

    Each stored entry is a dict with at minimum the following keys:
        tokens          list[int]              full sequence (prompt+response)
        response_length int                    number of response tokens
        loss_mask       list[float/int]        per-response-token loss mask
        reward          float                  scalar reward
        advantage       float                  advantage at collection time
        rollout_log_probs  Optional[tensor]    behavior policy log probs
        policy_version  int                    behavior policy version
        step_collected  int                    global step at admission
    """

    def __init__(
        self,
        buffer_size: int = 2048,
        score_threshold: float = 1.0,
        posadv_only: bool = False,
        weight_decay: float = -1.0,
        baseline_buffer_size: int = 10240,
        tolerate_steps: int = 10,
        seed: int = 42,
    ) -> None:
        if not (weight_decay == -1.0 or 0.0 <= weight_decay <= 1.0):
            raise ValueError(
                f"weight_decay must be -1.0 (p50 recompute) or in [0,1] (decay), got {weight_decay}"
            )
        if int(buffer_size) <= 0:
            raise ValueError(f"buffer_size must be positive, got {buffer_size}")
        if int(baseline_buffer_size) <= 0:
            raise ValueError(f"baseline_buffer_size must be positive, got {baseline_buffer_size}")
        if int(tolerate_steps) < 0:
            raise ValueError(f"tolerate_steps must be non-negative, got {tolerate_steps}")
        self.buffer_size = int(buffer_size)
        self.score_threshold = float(score_threshold)
        self.posadv_only = posadv_only
        self.weight_decay = weight_decay
        self.baseline_buffer_size = int(baseline_buffer_size)
        # SPEAR discards trajectories that are too old to be useful.  Keep the
        # default conservative and cap it to the same ten-step window used by
        # the reference implementation.
        self.tolerate_steps = min(int(tolerate_steps), 10)
        self.seed = int(seed)
        self._rng = random.Random(self.seed)
        self._buf: deque = deque(maxlen=self.buffer_size)
        # Each item is (rollout_step, group reward means).  The reference SPEAR
        # implementation uses the p50 of this history as the SIL baseline.
        self._reward_history: deque = deque()
        self.total_admitted: int = 0
        self.total_rejected: int = 0

    def _trim_old(self, current_step: int) -> None:
        if self.tolerate_steps < 0:
            return
        cutoff = int(current_step) - self.tolerate_steps
        self._buf = deque(
            (entry for entry in self._buf if int(entry.get("step_collected", 0)) >= cutoff),
            maxlen=self.buffer_size,
        )

    def _trim_reward_history(self) -> None:
        count = sum(len(values) for _, values in self._reward_history)
        while count > self.baseline_buffer_size and self._reward_history:
            _, values = self._reward_history.popleft()
            count -= len(values)

    def observe_rewards(self, rewards: Iterable[float], current_step: int) -> None:
        """Record per-group reward means for the moving SPEAR baseline."""
        values = []
        for reward in rewards:
            try:
                value = float(reward)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                values.append(value)
        if values:
            self._reward_history.append((int(current_step), values))
            self._trim_reward_history()

    def baseline_reward(self) -> Optional[float]:
        """Return the p50 historical group reward used by SPEAR SIL."""
        values = [value for _, rewards in self._reward_history for value in rewards]
        if not values:
            return None
        values.sort()
        middle = len(values) // 2
        if len(values) % 2:
            return float(values[middle])
        return float((values[middle - 1] + values[middle]) / 2.0)

    def push(
        self,
        entries: List[Dict[str, Any]],
        current_step: int,
        group_rewards: Optional[Iterable[float]] = None,
    ) -> None:
        """Attempt to admit trajectory dicts into the buffer."""
        if group_rewards is not None:
            self.observe_rewards(group_rewards, current_step)
        self._trim_old(current_step)
        for entry in entries:
            try:
                reward = float(entry.get("reward", 0.0))
                advantage = float(entry.get("advantage", reward))
            except (TypeError, ValueError):
                self.total_rejected += 1
                continue
            if not math.isfinite(reward) or not math.isfinite(advantage):
                self.total_rejected += 1
                continue
            if reward < self.score_threshold:
                self.total_rejected += 1
                continue
            if self.posadv_only and advantage <= 0.0:
                self.total_rejected += 1
                continue
            record = dict(entry)
            record["step_collected"] = current_step
            record["advantage"] = advantage
            self._buf.append(record)
            self.total_admitted += 1

    def sample(
        self,
        n: int,
        current_step: int,
        baseline_reward: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Sample up to n entries with re-estimated advantages."""
        if len(self._buf) == 0 or n <= 0:
            return []
        self._trim_old(current_step)
        if len(self._buf) == 0:
            return []
        # Recalibration is itself an admission gate in SPEAR: when the
        # historical p50 moves above a stored reward, that trajectory must not
        # become a negative SIL target.  Shuffle the eligible pool first so a
        # small request does not systematically favour FIFO entries.
        candidates = list(self._buf)
        result = []
        if baseline_reward is None:
            baseline_reward = self.baseline_reward()
        self._rng.shuffle(candidates)
        for entry in candidates:
            e = dict(entry)
            if self.weight_decay == -1.0:
                if baseline_reward is not None:
                    e["advantage"] = e["reward"] - baseline_reward
            else:
                # The reference implementation applies this coefficient once
                # to the replay loss; it is not an additional age decay.
                e["advantage"] = self.weight_decay * e["advantage"]
            if self.posadv_only and e["advantage"] <= 0.0:
                continue
            result.append(e)
            if len(result) >= int(n):
                break
        return result

    def __len__(self) -> int:
        return len(self._buf)

    def stats(self) -> Dict[str, float]:
        total = self.total_admitted + self.total_rejected
        return {
            "sil_buffer_size": float(len(self._buf)),
            "sil_buffer_capacity": float(self.buffer_size),
            "sil_total_admitted": float(self.total_admitted),
            "sil_total_rejected": float(self.total_rejected),
            "sil_admit_rate": float(self.total_admitted) / max(total, 1),
        }

    def state_dict(self) -> Dict[str, Any]:
        """Return a CPU/torch-save-friendly snapshot of the buffer."""
        return {
            "schema_version": "lightrl_spear_sil_v1",
            "buffer_size": self.buffer_size,
            "score_threshold": self.score_threshold,
            "posadv_only": self.posadv_only,
            "weight_decay": self.weight_decay,
            "baseline_buffer_size": self.baseline_buffer_size,
            "tolerate_steps": self.tolerate_steps,
            "seed": self.seed,
            # Persist the local sampler state so resume produces the same
            # trajectory order instead of silently restarting its RNG stream.
            "rng_state": self._rng.getstate(),
            "records": list(self._buf),
            "reward_history": list(self._reward_history),
            "total_admitted": self.total_admitted,
            "total_rejected": self.total_rejected,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self._buf.clear()
        for entry in state.get("records") or []:
            if isinstance(entry, dict):
                self._buf.append(dict(entry))
        self._reward_history.clear()
        for item in state.get("reward_history") or []:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                try:
                    step = int(item[0])
                    rewards = [float(value) for value in item[1] if math.isfinite(float(value))]
                except (TypeError, ValueError):
                    continue
                if rewards:
                    self._reward_history.append((step, rewards))
        self._trim_reward_history()
        self.total_admitted = int(state.get("total_admitted", len(self._buf)))
        self.total_rejected = int(state.get("total_rejected", 0))
        rng_state = state.get("rng_state")
        if rng_state is not None:
            try:
                def _tuple_tree(value):
                    if isinstance(value, list):
                        return tuple(_tuple_tree(item) for item in value)
                    if isinstance(value, tuple):
                        return tuple(_tuple_tree(item) for item in value)
                    return value

                self._rng.setstate(_tuple_tree(rng_state))
            except (TypeError, ValueError):
                # Snapshots written before rng_state was introduced remain
                # loadable and fall back to the configured seed.
                self._rng.seed(self.seed)
