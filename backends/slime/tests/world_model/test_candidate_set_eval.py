from slime.world_model.candidate_set_eval import _group_records


def test_group_records_drops_missing_reward_candidates():
    records = [
        {"context_hash": "ctx", "reward_score": 1.0},
        {"context_hash": "ctx", "reward_score": None},
        {"context_hash": "ctx", "reward_score": -1.0},
    ]

    groups = _group_records(
        records,
        group_key="context_hash",
        min_candidates=2,
        max_candidates=8,
        require_reward_variation=True,
    )

    assert groups == [[0, 2]]
