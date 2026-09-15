from __future__ import annotations

import math
import sys
from pathlib import Path


TERMINAL_RL_DIR = Path(__file__).resolve().parents[2] / "harborrl"
if str(TERMINAL_RL_DIR) not in sys.path:
    sys.path.insert(0, str(TERMINAL_RL_DIR))

from harborrl.algorithms.dive_po.exploration.agent57.memory import (  # noqa: E402
    CountBasedEpisodicMemory,
    CountBasedEpisodicMemoryConfig,
    SimHashKNNEpisodicMemory,
    SimHashKNNEpisodicMemoryConfig,
    _as_numeric_vector,
    create_episodic_memory_backend,
    resolve_episodic_backend_name,
)


def test_count_backend_empty_repeat_overflow_and_reset():
    memory = CountBasedEpisodicMemory(
        CountBasedEpisodicMemoryConfig(capacity=2, decay=1.0)
    )

    assert memory.compute_novelty("alpha") == 1.0
    memory.add("alpha")
    assert math.isclose(memory.compute_novelty("alpha"), 1.0 / math.sqrt(2.0))

    memory.add("beta")
    memory.add("gamma")

    assert memory.compute_novelty("alpha") == 1.0
    assert memory.compute_novelty("beta") < 1.0
    assert memory.compute_novelty("gamma") < 1.0

    memory.reset()
    assert memory.compute_novelty("gamma") == 1.0


def test_count_backend_decay_without_clear_on_reset():
    memory = CountBasedEpisodicMemory(
        CountBasedEpisodicMemoryConfig(capacity=4, decay=0.5, clear_on_reset=False)
    )

    memory.add({"state": "x"})
    before = memory.compute_novelty({"state": "x"})
    memory.reset()
    after = memory.compute_novelty({"state": "x"})

    assert before < after < 1.0


def test_count_backend_state_dict_roundtrip():
    memory = CountBasedEpisodicMemory(CountBasedEpisodicMemoryConfig(capacity=8))
    memory.add("alpha")
    memory.add("alpha")
    memory.add("beta")

    restored = CountBasedEpisodicMemory()
    restored.load_state_dict(memory.state_dict())

    assert restored.compute_novelty("alpha") == memory.compute_novelty("alpha")
    assert restored.compute_novelty("beta") == memory.compute_novelty("beta")


def test_backends_ignore_malformed_state_dict_entries():
    count = CountBasedEpisodicMemory()
    count.load_state_dict({"counts": ["not-a-pair", ("ok", 2.0)]})
    assert count.compute_novelty({"new": "state"}) == 1.0

    simhash = SimHashKNNEpisodicMemory()
    simhash.load_state_dict({"buckets": {"1010": "bad"}})
    assert simhash.compute_novelty("") == 1.0
    simhash.add("")
    assert simhash.compute_novelty("") == simhash.novelty_floor


def test_simhash_knn_empty_repeat_capacity_and_roundtrip():
    memory = SimHashKNNEpisodicMemory(
        SimHashKNNEpisodicMemoryConfig(
            hash_bits=8,
            bucket_capacity=2,
            k=2,
            distance_metric="cosine",
            random_seed=7,
            novelty_floor=0.05,
        )
    )
    vector = [1.0, 0.0, 0.0, 0.0]

    assert memory.compute_novelty(vector) == 1.0
    memory.add(vector)
    duplicate = memory.compute_novelty(vector)
    assert duplicate == 0.05

    memory.add(vector)
    memory.add(vector)
    assert all(len(bucket) <= 2 for bucket in memory.state_dict()["buckets"].values())

    restored = SimHashKNNEpisodicMemory()
    restored.load_state_dict(memory.state_dict())

    assert restored.compute_novelty(vector) == duplicate


def test_simhash_knn_records_query_stats_and_floor():
    memory = SimHashKNNEpisodicMemory(
        SimHashKNNEpisodicMemoryConfig(
            hash_bits=8,
            bucket_capacity=8,
            k=1,
            random_seed=13,
            multi_probe_radius=1,
            novelty_floor=0.07,
        )
    )
    state = {"signature": "shell|service|atd|start", "obs": "success"}

    assert memory.compute_novelty(state) == 1.0
    assert memory.last_query_stats()["empty_bucket"] == 1.0
    memory.add(state)

    assert memory.compute_novelty(state) == 0.07
    stats = memory.last_query_stats()
    assert stats["exact_repeat"] == 1.0
    assert stats["probe_count"] == 9


def test_simhash_fallback_vector_tokenizes_structured_state():
    vector = _as_numeric_vector(
        {
            "tool": "shell",
            "signature": "shell|pytest",
            "observation": "test_pass:lenS",
            "exit": "exit0",
        },
        fallback_dim=64,
    )

    assert int((vector != 0).sum()) > 1
    assert math.isclose(float((vector * vector).sum()), 1.0)


def test_simhash_knn_supports_l2_and_string_states():
    memory = SimHashKNNEpisodicMemory(
        SimHashKNNEpisodicMemoryConfig(
            hash_bits=4,
            bucket_capacity=4,
            k=1,
            distance_metric="l2",
            vector_dim=16,
            random_seed=11,
        )
    )

    memory.add("run pytest and inspect failure")

    assert 0.0 <= memory.compute_novelty("run pytest and inspect failure") <= 1.0
    assert memory.state_dict()["config"]["distance_metric"] == "l2"


def test_episodic_backend_factory_env_aliases(monkeypatch):
    monkeypatch.delenv("EXPLORE_AGENT57_EPISODIC_BACKEND", raising=False)
    monkeypatch.delenv("EPISODIC_MEMORY_BACKEND", raising=False)
    assert create_episodic_memory_backend() is None

    monkeypatch.setenv("EPISODIC_MEMORY_BACKEND", "count")
    assert isinstance(create_episodic_memory_backend(), CountBasedEpisodicMemory)

    monkeypatch.setenv("EXPLORE_AGENT57_EPISODIC_BACKEND", "knn")
    monkeypatch.setenv("EXPLORE_AGENT57_EPISODIC_MULTI_PROBE_RADIUS", "2")
    monkeypatch.setenv("EXPLORE_AGENT57_EPISODIC_NOVELTY_FLOOR", "0.11")
    assert resolve_episodic_backend_name("knn") == "simhash_knn"
    backend = create_episodic_memory_backend()
    assert isinstance(backend, SimHashKNNEpisodicMemory)
    assert backend.multi_probe_radius == 2
    assert backend.novelty_floor == 0.11
