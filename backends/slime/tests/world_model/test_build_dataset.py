import json

import torch

from slime.world_model.build_dataset import extract_world_model_records, summarize_world_model_records


def test_extract_world_model_records_backfills_legacy_context_text(tmp_path):
    payload = {
        "samples": [
            {
                "status": "completed",
                "prompt": [{"role": "user", "content": "state text"}],
                "metadata": {},
                "train_metadata": {
                    "world_model": {
                        "schema": "openclaw_text_jepa_world_model_v1",
                        "context_hash": "abc",
                        "action_text": "act",
                        "next_observation_text": "obs",
                    }
                },
            }
        ]
    }
    path = tmp_path / "rollout.pt"
    torch.save(payload, path)

    records = extract_world_model_records(path)

    assert len(records) == 1
    assert json.loads(records[0]["context_text"])[0]["content"] == "state text"
    assert records[0]["context_text_source"] == "sample.prompt"


def test_extract_world_model_records_can_repair_prefix_truncated_context(tmp_path):
    bad_context = "shared system prompt " * 100
    payload = {
        "samples": [
            {
                "status": "completed",
                "prompt": [{"role": "user", "content": "task-specific state"}],
                "metadata": {},
                "train_metadata": {
                    "world_model": {
                        "schema": "openclaw_text_jepa_world_model_v1",
                        "context_hash": "abc",
                        "context_text": bad_context[:128],
                        "context_text_source": "context_messages",
                        "action_text": "act",
                        "next_observation_text": "obs",
                    }
                },
            }
        ]
    }
    path = tmp_path / "rollout.pt"
    torch.save(payload, path)

    records = extract_world_model_records(path, context_max_chars=128, context_source="sample_prompt")

    assert len(records) == 1
    assert "task-specific state" in records[0]["context_text"]
    assert "shared system prompt" not in records[0]["context_text"]
    assert records[0]["context_text_source"] == "sample.prompt"


def test_extract_world_model_records_filters_and_summary(tmp_path):
    payload = {
        "samples": [
            {
                "status": "completed",
                "prompt": "state one",
                "metadata": {},
                "train_metadata": {
                    "world_model": {
                        "schema": "openclaw_text_jepa_world_model_v1",
                        "status": "completed",
                        "uid": "u1",
                        "task_name": "task-a",
                        "context_hash": "ctx1",
                        "action_text": "act",
                        "next_observation_text": "tool result",
                        "has_tool_result": True,
                        "reward_score": 1.0,
                    }
                },
            },
            {
                "status": "failed",
                "prompt": "state two",
                "metadata": {
                    "world_model": {
                        "schema": "openclaw_text_jepa_world_model_v1",
                        "status": "failed",
                        "uid": "u2",
                        "task_name": "task-b",
                        "context_hash": "ctx2",
                        "action_text": "bad act",
                        "next_observation_text": json.dumps({"eval_reason": "eval_timeout"}),
                        "has_tool_result": False,
                        "reward_score": -2.0,
                    }
                },
                "train_metadata": {},
            },
        ]
    }
    path = tmp_path / "rollout.pt"
    torch.save(payload, path)

    all_records = extract_world_model_records(path)
    filtered = extract_world_model_records(
        path,
        statuses={"completed"},
        exclude_eval_reasons={"eval_timeout"},
        require_tool_result=True,
    )
    summary = summarize_world_model_records(filtered, input_record_count=len(all_records), filter_args={"bucket": "clean"})

    assert len(all_records) == 2
    assert len(filtered) == 1
    assert filtered[0]["uid"] == "u1"
    assert summary["record_count"] == 1
    assert summary["input_record_count"] == 2
    assert summary["dropped_record_count"] == 1
    assert summary["filter_args"]["bucket"] == "clean"
