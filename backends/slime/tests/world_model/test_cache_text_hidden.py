import json
import sys

import pytest
import torch

from slime.world_model.cache_text_hidden import _hash_encode, main, _record_action_text, _record_state_text, _record_target_text


def test_hash_encode_is_deterministic():
    first = _hash_encode(["state", "action"], hidden_dim=8)
    second = _hash_encode(["state", "action"], hidden_dim=8)
    assert first.shape == (2, 8)
    assert torch.allclose(first, second)
    assert torch.allclose(first.norm(dim=-1), torch.ones(2))


def test_record_text_extraction_prefers_context_text():
    record = {
        "context_text": json.dumps([{"role": "user", "content": "hi"}]),
        "context_hash": "hash",
        "action_text": "do thing",
        "next_observation_text": "result",
    }
    assert "hi" in _record_state_text(record)
    assert _record_action_text(record) == "do thing"
    assert _record_target_text(record) == "result"


def test_record_text_extraction_falls_back_for_old_records():
    record = {
        "context_hash": "abc",
        "task_name": "42",
        "task_path": "seta_env/42",
        "action_hash": "act",
        "next_observation_hash": "obs",
    }
    assert "abc" in _record_state_text(record)
    assert _record_action_text(record) == "act"
    assert _record_target_text(record) == "obs"


def test_cache_text_hidden_writes_metadata_and_reward_mask(tmp_path, monkeypatch):
    records = [
        {
            "context_text": "state one",
            "action_text": "act one",
            "next_observation_text": "obs one",
            "uid": "u1",
            "reward_score": 1.5,
        },
        {
            "context_text": "state two",
            "action_text": "act two",
            "next_observation_text": "obs two",
            "uid": "u2",
            "reward_score": None,
        },
    ]
    input_path = tmp_path / "records.jsonl"
    output_path = tmp_path / "cached_hidden.pt"
    input_path.write_text("\n".join(json.dumps(row) for row in records) + "\n", encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cache_text_hidden",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--encoder",
            "hash",
            "--hidden-dim",
            "8",
        ],
    )
    main()

    payload = torch.load(output_path, map_location="cpu", weights_only=False)
    assert payload["state_hidden"].shape == (2, 8)
    assert payload["record_count"] == 2
    assert payload["metadata"]["schema_version"] == "openclaw_text_jepa_hidden_cache_v1"
    assert payload["record_metadata"][0]["uid"] == "u1"
    assert payload["reward"].tolist() == [1.5, 0.0]
    assert payload["reward_mask"].tolist() == [True, False]


def test_cache_text_hidden_rejects_empty_input(tmp_path, monkeypatch):
    input_path = tmp_path / "records.jsonl"
    output_path = tmp_path / "cached_hidden.pt"
    input_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cache_text_hidden",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--encoder",
            "hash",
            "--hidden-dim",
            "8",
        ],
    )
    with pytest.raises(ValueError, match="No world-model records"):
        main()
