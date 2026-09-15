from __future__ import annotations

from collections import deque
import random
from pathlib import Path
from typing import Any, Iterable

import torch

from .metadata import stable_hash


class TrajectoryReplayBuffer:
    """Fixed-capacity replay buffer for DAPO-collected world-model records.

    The public contract is ``push(entries, current_step)`` /
    ``sample(n, current_step, baseline_reward)``.  Unlike success-filtered
    self-imitation (SIL) buffers, this buffer admits both success and failure
    transitions by default because latent dynamics need the full outcome
    distribution; pass ``score_threshold`` to opt into quality gating.
    """

    def __init__(
        self,
        buffer_size: int = 2048,
        *,
        score_threshold: float | None = None,
        seed: int = 42,
    ) -> None:
        if buffer_size <= 0:
            raise ValueError(f"buffer_size must be positive, got {buffer_size}")
        self.buffer_size = int(buffer_size)
        self.score_threshold = score_threshold
        self.seed = int(seed)
        self._rng = random.Random(self.seed)
        self._records: deque[dict[str, Any]] = deque(maxlen=self.buffer_size)
        self._ids: set[str] = set()
        self.total_admitted = 0
        self.total_rejected = 0
        self.total_sampled = 0

    @staticmethod
    def _record_id(record: dict[str, Any]) -> str:
        return str(
            record.get("transition_id")
            or stable_hash(
                {
                    "uid": record.get("uid") or record.get("trajectory_id"),
                    "turn_idx": record.get("turn_idx"),
                    "context_hash": record.get("context_hash"),
                    "action_hash": record.get("action_hash") or record.get("action_text"),
                }
            )
        )

    def push(self, entries: Iterable[dict[str, Any] | Any], current_step: int = 0) -> None:
        for entry in entries:
            if hasattr(entry, "to_dict"):
                record = dict(entry.to_dict())
            elif isinstance(entry, dict):
                record = dict(entry)
            else:
                raise TypeError(f"Replay entries must be dict-like, got {type(entry).__name__}")
            reward = record.get("reward_score", record.get("reward"))
            if self.score_threshold is not None and (reward is None or float(reward) < self.score_threshold):
                self.total_rejected += 1
                continue
            record_id = self._record_id(record)
            if record_id in self._ids:
                self.total_rejected += 1
                continue
            if len(self._records) == self.buffer_size:
                evicted = self._records[0]
                self._ids.discard(self._record_id(evicted))
            record["transition_id"] = record_id
            record["step_collected"] = int(current_step)
            self._records.append(record)
            self._ids.add(record_id)
            self.total_admitted += 1

    def sample(
        self,
        n: int,
        current_step: int = 0,
        baseline_reward: float | None = None,
    ) -> list[dict[str, Any]]:
        del current_step, baseline_reward
        if n <= 0 or not self._records:
            return []
        sampled = self._rng.sample(list(self._records), min(int(n), len(self._records)))
        self.total_sampled += len(sampled)
        return [dict(record) for record in sampled]

    def records(self) -> list[dict[str, Any]]:
        return [dict(record) for record in self._records]

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "openclaw_terminal_wm_replay_v1",
            "buffer_size": self.buffer_size,
            "score_threshold": self.score_threshold,
            "seed": self.seed,
            "records": self.records(),
            "total_admitted": self.total_admitted,
            "total_rejected": self.total_rejected,
            "total_sampled": self.total_sampled,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._records.clear()
        self._ids.clear()
        for record in state.get("records") or []:
            if not isinstance(record, dict):
                continue
            record = dict(record)
            record_id = self._record_id(record)
            record["transition_id"] = record_id
            if record_id in self._ids:
                continue
            if len(self._records) == self.buffer_size:
                evicted = self._records[0]
                self._ids.discard(self._record_id(evicted))
            self._records.append(record)
            self._ids.add(record_id)
        self.total_admitted = int(state.get("total_admitted", len(self._records)))
        self.total_rejected = int(state.get("total_rejected", 0))
        self.total_sampled = int(state.get("total_sampled", 0))

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), output)

    @classmethod
    def load(cls, path: str | Path) -> "TrajectoryReplayBuffer":
        state = torch.load(Path(path), map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "world_model_replay" in state:
            state = state["world_model_replay"]
        if not isinstance(state, dict):
            raise TypeError(f"Expected replay state dict, got {type(state).__name__}")
        buffer = cls(
            buffer_size=int(state.get("buffer_size", max(1, len(state.get("records") or [])))),
            score_threshold=state.get("score_threshold"),
            seed=int(state.get("seed", 42)),
        )
        buffer.load_state_dict(state)
        return buffer

    def stats(self) -> dict[str, float]:
        attempted = self.total_admitted + self.total_rejected
        return {
            "wm_replay_size": float(len(self)),
            "wm_replay_capacity": float(self.buffer_size),
            "wm_replay_total_admitted": float(self.total_admitted),
            "wm_replay_total_rejected": float(self.total_rejected),
            "wm_replay_total_sampled": float(self.total_sampled),
            "wm_replay_admit_rate": float(self.total_admitted) / max(attempted, 1),
        }

    def __len__(self) -> int:
        return len(self._records)


def world_model_records_from_samples(samples: Iterable[Any]) -> list[dict[str, Any]]:
    """Extract attached world-model metadata from flat or grouped samples."""

    records: list[dict[str, Any]] = []
    for sample in samples:
        if isinstance(sample, (list, tuple)):
            records.extend(world_model_records_from_samples(sample))
            continue
        train_metadata = getattr(sample, "train_metadata", None)
        metadata = getattr(sample, "metadata", None)
        train_metadata = train_metadata if isinstance(train_metadata, dict) else {}
        metadata = metadata if isinstance(metadata, dict) else {}
        record = train_metadata.get("world_model") or metadata.get("world_model")
        if isinstance(record, dict):
            records.append(dict(record))
    return records
