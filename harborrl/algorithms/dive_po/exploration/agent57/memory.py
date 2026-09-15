from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
import os
from typing import Any

import numpy as np


from harborrl.env import (
    env_bool as _env_bool,
    env_float as _env_float,
    env_int as _env_int,
    env_optional_int as _env_optional_int,
)


def _canonical_state(state: Any) -> str:
    if isinstance(state, np.ndarray):
        value: Any = state.tolist()
    elif isinstance(state, bytes):
        return state.decode("utf-8", errors="replace")
    else:
        value = state
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        return repr(value)


def _state_digest(state: Any, *, n: int = 20) -> str:
    text = _canonical_state(state)
    return hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()[:n]


def _state_tokens(value: Any, *, prefix: str = "") -> list[str]:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, dict):
        tokens: list[str] = []
        for key in sorted(value.keys(), key=str):
            key_text = str(key)
            child_prefix = f"{prefix}{key_text}="
            tokens.append(f"{prefix}{key_text}")
            tokens.extend(_state_tokens(value.get(key), prefix=child_prefix))
        return tokens
    if isinstance(value, (list, tuple)):
        tokens = []
        for idx, item in enumerate(value):
            tokens.extend(_state_tokens(item, prefix=f"{prefix}{idx}:"))
        return tokens
    text = str(value)
    if not text:
        return [f"{prefix}<empty>"]
    parts = text.split()
    if not parts:
        parts = [text]
    return [f"{prefix}{part}" for part in parts]


def _as_numeric_vector(state: Any, *, fallback_dim: int = 128) -> np.ndarray:
    try:
        arr = np.asarray(state, dtype=np.float64)
        if arr.size > 0 and np.all(np.isfinite(arr)):
            return arr.reshape(-1)
    except (TypeError, ValueError):
        pass

    vec = np.zeros(max(1, int(fallback_dim)), dtype=np.float64)
    tokens = _state_tokens(state)
    if not tokens:
        text = _canonical_state(state)
        tokens = text.split() or [text or "<empty>"]
    for token in tokens:
        digest = hashlib.md5(token.encode("utf-8", errors="ignore")).digest()
        idx = int.from_bytes(digest[:4], "little") % vec.size
        sign = 1.0 if digest[4] & 1 else -1.0
        vec[idx] += sign
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def _fit_vector_dim(vector: np.ndarray, dim: int) -> np.ndarray:
    if vector.size == dim:
        return vector.astype(np.float64, copy=False)
    if vector.size > dim:
        return vector[:dim].astype(np.float64, copy=False)
    out = np.zeros(dim, dtype=np.float64)
    out[: vector.size] = vector
    return out


class EpisodicMemoryBackend(ABC):
    """Pluggable per-episode novelty memory used by Agent57-style exploration."""

    @abstractmethod
    def add(self, state: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def compute_novelty(self, state: Any) -> float:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def load_state_dict(self, state: dict[str, Any]) -> None:
        raise NotImplementedError


@dataclass(frozen=True)
class CountBasedEpisodicMemoryConfig:
    capacity: int = 4096
    decay: float = 1.0
    clear_on_reset: bool = True


class CountBasedEpisodicMemory(EpisodicMemoryBackend):
    """Episode memory using a stable hash key and 1/sqrt(count + 1) novelty."""

    def __init__(
        self,
        config: CountBasedEpisodicMemoryConfig | None = None,
    ) -> None:
        self.config = config or CountBasedEpisodicMemoryConfig()
        self.capacity = max(0, int(self.config.capacity))
        self.decay = min(1.0, max(0.0, float(self.config.decay)))
        self._counts: OrderedDict[str, float] = OrderedDict()
        self._adds = 0

    def add(self, state: Any) -> None:
        self._apply_decay()
        key = _state_digest(state)
        if key in self._counts:
            self._counts.move_to_end(key)
        elif self.capacity > 0:
            while len(self._counts) >= self.capacity:
                self._counts.popitem(last=False)
        self._counts[key] = float(self._counts.get(key, 0.0)) + 1.0
        self._adds += 1

    def compute_novelty(self, state: Any) -> float:
        count = max(0.0, float(self._counts.get(_state_digest(state), 0.0)))
        return 1.0 / math.sqrt(count + 1.0)

    def reset(self) -> None:
        if self.config.clear_on_reset:
            self._counts.clear()
        else:
            self._apply_decay(force=True)

    def state_dict(self) -> dict[str, Any]:
        return {
            "backend": "count",
            "config": {
                "capacity": self.capacity,
                "decay": self.decay,
                "clear_on_reset": bool(self.config.clear_on_reset),
            },
            "counts": list(self._counts.items()),
            "adds": self._adds,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        config = state.get("config")
        if not isinstance(config, dict):
            config = {}
        self.capacity = max(0, int(config.get("capacity", self.capacity)))
        self.decay = min(1.0, max(0.0, float(config.get("decay", self.decay))))
        counts = state.get("counts", [])
        self._counts.clear()
        for item in counts:
            try:
                key, value = item
            except (TypeError, ValueError):
                continue
            try:
                count = float(value)
            except (TypeError, ValueError):
                continue
            if count <= 0.0:
                continue
            self._counts[str(key)] = count
            if self.capacity > 0:
                while len(self._counts) > self.capacity:
                    self._counts.popitem(last=False)
        try:
            self._adds = int(state.get("adds", 0))
        except (TypeError, ValueError):
            self._adds = 0

    def _apply_decay(self, *, force: bool = False) -> None:
        if self.decay >= 1.0 and not force:
            return
        if self.decay <= 0.0:
            self._counts.clear()
            return
        for key in list(self._counts.keys()):
            next_count = float(self._counts[key]) * self.decay
            if next_count <= 1e-12:
                del self._counts[key]
            else:
                self._counts[key] = next_count


@dataclass(frozen=True)
class SimHashKNNEpisodicMemoryConfig:
    hash_bits: int = 64
    bucket_capacity: int = 256
    k: int = 5
    distance_metric: str = "cosine"
    vector_dim: int = 128
    random_seed: int | None = None
    epsilon: float = 1e-8
    multi_probe_radius: int = 1
    novelty_floor: float = 0.05


class SimHashKNNEpisodicMemory(EpisodicMemoryBackend):
    """SimHash buckets with in-bucket KNN novelty over compact embeddings."""

    def __init__(
        self,
        config: SimHashKNNEpisodicMemoryConfig | None = None,
    ) -> None:
        self.config = config or SimHashKNNEpisodicMemoryConfig()
        self.hash_bits = max(1, int(self.config.hash_bits))
        self.bucket_capacity = max(1, int(self.config.bucket_capacity))
        self.k = max(1, int(self.config.k))
        self.distance_metric = self._normalize_distance(self.config.distance_metric)
        self.vector_dim = max(1, int(self.config.vector_dim))
        self.epsilon = max(1e-12, float(self.config.epsilon))
        self.multi_probe_radius = max(0, int(self.config.multi_probe_radius))
        self.novelty_floor = min(1.0, max(0.0, float(self.config.novelty_floor)))
        self._rng = np.random.default_rng(self.config.random_seed)
        self._hyperplanes: np.ndarray | None = None
        self._buckets: OrderedDict[str, list[np.ndarray]] = OrderedDict()
        self._last_query_stats: dict[str, Any] = {}

    def add(self, state: Any) -> None:
        vector = self._vector(state)
        key = self._bucket_key(vector)
        bucket = self._buckets.setdefault(key, [])
        bucket.append(vector.copy())
        if len(bucket) > self.bucket_capacity:
            del bucket[: len(bucket) - self.bucket_capacity]
        self._buckets.move_to_end(key)

    def compute_novelty(self, state: Any) -> float:
        vector = self._vector(state)
        bucket_keys = self._probe_bucket_keys(vector)
        candidates: list[np.ndarray] = []
        for key in bucket_keys:
            candidates.extend(self._buckets.get(key, []))
        if not candidates:
            self._last_query_stats = {
                "empty_bucket": 1.0,
                "exact_repeat": 0.0,
                "candidate_count": 0,
                "probe_count": len(bucket_keys),
            }
            return 1.0
        distances = sorted(self._distance(vector, candidate) for candidate in candidates)
        nearest = distances[: min(self.k, len(distances))]
        if not nearest:
            self._last_query_stats = {
                "empty_bucket": 1.0,
                "exact_repeat": 0.0,
                "candidate_count": len(candidates),
                "probe_count": len(bucket_keys),
            }
            return 1.0
        mean_dist = sum(nearest) / len(nearest)
        if not math.isfinite(mean_dist):
            self._last_query_stats = {
                "empty_bucket": 0.0,
                "exact_repeat": 0.0,
                "candidate_count": len(candidates),
                "probe_count": len(bucket_keys),
            }
            return 1.0
        exact_repeat = any(distance <= self.epsilon for distance in nearest)
        novelty = max(0.0, min(1.0, mean_dist / (mean_dist + 1.0)))
        self._last_query_stats = {
            "empty_bucket": 0.0,
            "exact_repeat": 1.0 if exact_repeat else 0.0,
            "candidate_count": len(candidates),
            "probe_count": len(bucket_keys),
        }
        return max(self.novelty_floor, novelty)

    def last_query_stats(self) -> dict[str, Any]:
        return dict(self._last_query_stats)

    def reset(self) -> None:
        self._buckets.clear()

    def state_dict(self) -> dict[str, Any]:
        return {
            "backend": "simhash_knn",
            "config": {
                "hash_bits": self.hash_bits,
                "bucket_capacity": self.bucket_capacity,
                "k": self.k,
                "distance_metric": self.distance_metric,
                "vector_dim": self.vector_dim,
                "epsilon": self.epsilon,
                "multi_probe_radius": self.multi_probe_radius,
                "novelty_floor": self.novelty_floor,
            },
            "hyperplanes": (
                self._hyperplanes.tolist() if self._hyperplanes is not None else None
            ),
            "buckets": {
                key: [vector.tolist() for vector in vectors]
                for key, vectors in self._buckets.items()
            },
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        config = state.get("config")
        if not isinstance(config, dict):
            config = {}
        self.hash_bits = max(1, int(config.get("hash_bits", self.hash_bits)))
        self.bucket_capacity = max(
            1, int(config.get("bucket_capacity", self.bucket_capacity))
        )
        self.k = max(1, int(config.get("k", self.k)))
        self.distance_metric = self._normalize_distance(
            str(config.get("distance_metric", self.distance_metric))
        )
        self.vector_dim = max(1, int(config.get("vector_dim", self.vector_dim)))
        self.epsilon = max(1e-12, float(config.get("epsilon", self.epsilon)))
        self.multi_probe_radius = max(
            0,
            int(config.get("multi_probe_radius", self.multi_probe_radius)),
        )
        self.novelty_floor = min(
            1.0,
            max(0.0, float(config.get("novelty_floor", self.novelty_floor))),
        )

        raw_planes = state.get("hyperplanes") if isinstance(state, dict) else None
        if raw_planes is None:
            self._hyperplanes = None
        else:
            planes = np.asarray(raw_planes, dtype=np.float64)
            if planes.ndim != 2 or planes.shape[0] != self.hash_bits:
                self._hyperplanes = None
            else:
                self._hyperplanes = planes
                self.vector_dim = int(planes.shape[1])

        self._buckets.clear()
        raw_buckets = state.get("buckets", {}) if isinstance(state, dict) else {}
        if isinstance(raw_buckets, dict):
            for key, vectors in raw_buckets.items():
                if not isinstance(vectors, list):
                    continue
                bucket: list[np.ndarray] = []
                for vector in vectors[-self.bucket_capacity :]:
                    try:
                        arr = _fit_vector_dim(
                            np.asarray(vector, dtype=np.float64),
                            self.vector_dim,
                        )
                    except (TypeError, ValueError):
                        continue
                    if np.all(np.isfinite(arr)):
                        bucket.append(arr)
                if bucket:
                    self._buckets[str(key)] = bucket

    def _vector(self, state: Any) -> np.ndarray:
        vector = _as_numeric_vector(state, fallback_dim=self.vector_dim)
        if self._hyperplanes is None:
            dim = vector.size if vector.size > 0 else self.vector_dim
            self.vector_dim = max(1, int(dim))
            self._hyperplanes = self._rng.normal(size=(self.hash_bits, self.vector_dim))
        return _fit_vector_dim(vector, self.vector_dim)

    def _bucket_key(self, vector: np.ndarray) -> str:
        if self._hyperplanes is None:
            self._vector(vector)
        assert self._hyperplanes is not None
        bits = (self._hyperplanes @ vector) >= 0.0
        return "".join("1" if bit else "0" for bit in bits.tolist())

    def _probe_bucket_keys(self, vector: np.ndarray) -> list[str]:
        key = self._bucket_key(vector)
        radius = min(max(0, self.multi_probe_radius), 2)
        if radius <= 0:
            return [key]
        keys = [key]
        bit_count = len(key)
        chars = list(key)
        for distance in range(1, radius + 1):
            for indices in itertools.combinations(range(bit_count), distance):
                probe = chars.copy()
                for idx in indices:
                    probe[idx] = "0" if probe[idx] == "1" else "1"
                keys.append("".join(probe))
        return keys

    def _distance(self, left: np.ndarray, right: np.ndarray) -> float:
        if self.distance_metric == "l2":
            return float(np.linalg.norm(left - right))
        if self.distance_metric == "hamming":
            return float(self._hamming_distance(left, right))
        left_norm = float(np.linalg.norm(left))
        right_norm = float(np.linalg.norm(right))
        if left_norm <= self.epsilon or right_norm <= self.epsilon:
            return 1.0
        cosine = float(np.dot(left, right) / (left_norm * right_norm))
        cosine = max(-1.0, min(1.0, cosine))
        return (1.0 - cosine) / 2.0

    def _hamming_distance(self, left: np.ndarray, right: np.ndarray) -> float:
        assert self._hyperplanes is not None
        left_bits = (self._hyperplanes @ left) >= 0.0
        right_bits = (self._hyperplanes @ right) >= 0.0
        return float(np.count_nonzero(left_bits != right_bits)) / float(self.hash_bits)

    @staticmethod
    def _normalize_distance(value: str) -> str:
        text = (value or "cosine").strip().lower()
        return text if text in {"cosine", "l2", "hamming"} else "cosine"


def resolve_episodic_backend_name(name: str | None) -> str:
    text = (name or "legacy").strip().lower()
    if text in {"", "default", "legacy", "signature"}:
        return "legacy"
    if text in {"knn", "simhash", "simhash-knn", "simhash_knn"}:
        return "simhash_knn"
    if text in {"count", "count_based", "count-based"}:
        return "count"
    return "legacy"


def create_episodic_memory_backend(
    name: str | None = None,
) -> EpisodicMemoryBackend | None:
    """Create an episodic backend from explicit name or environment.

    `None` means preserve terminal-rl's current signature novelty path.
    """
    backend = resolve_episodic_backend_name(
        name
        or os.getenv("EXPLORE_AGENT57_EPISODIC_BACKEND")
        or os.getenv("EPISODIC_MEMORY_BACKEND")
    )
    if backend == "legacy":
        return None
    if backend == "count":
        return CountBasedEpisodicMemory(
            CountBasedEpisodicMemoryConfig(
                capacity=max(
                    0,
                    _env_int(
                        "EXPLORE_AGENT57_EPISODIC_CAPACITY",
                        _env_int("EPISODIC_MEMORY_CAPACITY", 4096),
                    ),
                ),
                decay=min(
                    1.0,
                    max(
                        0.0,
                        _env_float(
                            "EXPLORE_AGENT57_EPISODIC_COUNT_DECAY",
                            _env_float("EPISODIC_MEMORY_COUNT_DECAY", 1.0),
                        ),
                    ),
                ),
                clear_on_reset=_env_bool(
                    "EXPLORE_AGENT57_EPISODIC_CLEAR_ON_RESET",
                    _env_bool("EPISODIC_MEMORY_CLEAR_ON_RESET", True),
                ),
            )
        )
    return SimHashKNNEpisodicMemory(
        SimHashKNNEpisodicMemoryConfig(
            hash_bits=max(
                1,
                _env_int(
                    "EXPLORE_AGENT57_EPISODIC_SIMHASH_BITS",
                    _env_int("EPISODIC_MEMORY_SIMHASH_BITS", 64),
                ),
            ),
            bucket_capacity=max(
                1,
                _env_int(
                    "EXPLORE_AGENT57_EPISODIC_BUCKET_CAPACITY",
                    _env_int("EPISODIC_MEMORY_BUCKET_CAPACITY", 256),
                ),
            ),
            k=max(
                1,
                _env_int(
                    "EXPLORE_AGENT57_EPISODIC_K",
                    _env_int("EPISODIC_MEMORY_K", 5),
                ),
            ),
            distance_metric=(
                os.getenv("EXPLORE_AGENT57_EPISODIC_DISTANCE")
                or os.getenv("EPISODIC_MEMORY_DISTANCE")
                or "cosine"
            ),
            vector_dim=max(
                1,
                _env_int(
                    "EXPLORE_AGENT57_EPISODIC_VECTOR_DIM",
                    _env_int("EPISODIC_MEMORY_VECTOR_DIM", 128),
                ),
            ),
            random_seed=(
                _env_optional_int("EXPLORE_AGENT57_EPISODIC_RANDOM_SEED")
                if os.getenv("EXPLORE_AGENT57_EPISODIC_RANDOM_SEED") is not None
                else _env_optional_int("EPISODIC_MEMORY_RANDOM_SEED")
            ),
            multi_probe_radius=max(
                0,
                _env_int(
                    "EXPLORE_AGENT57_EPISODIC_MULTI_PROBE_RADIUS",
                    _env_int("EPISODIC_MEMORY_MULTI_PROBE_RADIUS", 1),
                ),
            ),
            novelty_floor=min(
                1.0,
                max(
                    0.0,
                    _env_float(
                        "EXPLORE_AGENT57_EPISODIC_NOVELTY_FLOOR",
                        _env_float("EPISODIC_MEMORY_NOVELTY_FLOOR", 0.05),
                    ),
                ),
            ),
        )
    )


__all__ = [
    "CountBasedEpisodicMemory",
    "CountBasedEpisodicMemoryConfig",
    "EpisodicMemoryBackend",
    "SimHashKNNEpisodicMemory",
    "SimHashKNNEpisodicMemoryConfig",
    "create_episodic_memory_backend",
    "resolve_episodic_backend_name",
]
