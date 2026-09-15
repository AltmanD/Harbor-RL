from types import SimpleNamespace

from slime.world_model.replay_buffer import TrajectoryReplayBuffer, world_model_records_from_samples


def test_trajectory_replay_buffer_roundtrip(tmp_path):
    buffer = TrajectoryReplayBuffer(buffer_size=2, seed=7)
    buffer.push(
        [
            {"uid": "a", "turn_idx": 0, "action_text": "one", "reward_score": 0.0},
            {"uid": "b", "turn_idx": 0, "action_text": "two", "reward_score": 1.0},
            {"uid": "c", "turn_idx": 0, "action_text": "three", "reward_score": -1.0},
        ],
        current_step=3,
    )
    path = tmp_path / "replay.pt"
    buffer.save(path)
    loaded = TrajectoryReplayBuffer.load(path)

    assert len(loaded) == 2
    assert {row["uid"] for row in loaded.records()} == {"b", "c"}
    assert len(loaded.sample(10, current_step=4)) == 2


def test_world_model_records_from_grouped_samples():
    sample = SimpleNamespace(
        train_metadata={"world_model": {"uid": "x", "turn_idx": 0}},
        metadata={},
    )

    assert world_model_records_from_samples([[sample]]) == [{"uid": "x", "turn_idx": 0}]
