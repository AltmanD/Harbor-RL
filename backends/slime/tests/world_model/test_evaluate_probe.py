import json

import pytest
import torch

from slime.world_model.evaluate_probe import evaluate_probe
from slime.world_model.modules import TextLatentWorldModel, TextLatentWorldModelConfig


def _write_checkpoint(path, *, hidden_dim=4, latent_dim=3):
    config = TextLatentWorldModelConfig(
        state_hidden_dim=hidden_dim,
        action_hidden_dim=hidden_dim,
        target_hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        sigreg_num_proj=4,
    )
    model = TextLatentWorldModel(config)
    torch.save({"config": config.__dict__, "state_dict": model.state_dict()}, path)


def test_evaluate_probe_old_artifacts_single_sample_marks_shuffle_unavailable(tmp_path):
    torch.manual_seed(0)
    ckpt_path = tmp_path / "probe.pt"
    cache_path = tmp_path / "cache.pt"
    out_path = tmp_path / "eval.json"
    _write_checkpoint(ckpt_path)
    torch.save(
        {
            "state_hidden": torch.randn(1, 4),
            "action_hidden": torch.randn(1, 4),
            "target_hidden": torch.randn(1, 4),
        },
        cache_path,
    )

    summary = evaluate_probe(
        checkpoint=ckpt_path,
        cache=cache_path,
        output=out_path,
        device_name="cpu",
        bootstrap_samples=0,
    )
    payload = json.loads(out_path.read_text(encoding="utf-8"))

    assert summary["record_count"] == 1
    assert payload["metrics"]["shuffled_action_available"] is False
    assert payload["metrics"]["shuffled_action_reason"] == "fewer_than_two_samples"
    assert payload["metrics"]["shuffle_gap_mse_shuffled_minus_real"] is None
    assert payload["metrics"]["pred_mse_real"] is not None


def test_evaluate_probe_uses_reward_mask_and_constant_reward_reason(tmp_path):
    torch.manual_seed(0)
    ckpt_path = tmp_path / "probe.pt"
    cache_path = tmp_path / "cache.pt"
    out_path = tmp_path / "eval.json"
    _write_checkpoint(ckpt_path)
    torch.save(
        {
            "state_hidden": torch.randn(3, 4),
            "action_hidden": torch.randn(3, 4),
            "target_hidden": torch.randn(3, 4),
            "reward": torch.tensor([1.0, 1.0, 99.0]),
            "reward_mask": torch.tensor([True, True, False]),
            "record_metadata": [
                {"uid": "u1", "task_name": "task", "status": "completed", "has_tool_result": True},
                {"uid": "u2", "task_name": "task", "status": "completed", "has_tool_result": False},
                {"uid": "u3", "task_name": "task", "status": "failed", "has_tool_result": False},
            ],
        },
        cache_path,
    )

    summary = evaluate_probe(
        checkpoint=ckpt_path,
        cache=cache_path,
        output=out_path,
        device_name="cpu",
        bootstrap_samples=8,
    )
    metrics = summary["metrics"]

    assert summary["counts"]["n_reward_masked"] == 2
    assert metrics["shuffled_action_available"] is True
    assert metrics["shuffle_gap_mse_shuffled_minus_real"] is not None
    assert metrics["action_delta"] is not None
    assert metrics["value_reward"]["reward_mask_count"] == 2
    assert metrics["value_reward"]["spearman"] is None
    assert metrics["value_reward"]["reason"] == "constant_input"
    assert summary["record_metadata_summary"]["status_hist"]["completed"] == 2


def test_evaluate_probe_rejects_empty_cache(tmp_path):
    ckpt_path = tmp_path / "probe.pt"
    cache_path = tmp_path / "cache.pt"
    out_path = tmp_path / "eval.json"
    _write_checkpoint(ckpt_path)
    torch.save(
        {
            "state_hidden": torch.empty(0, 4),
            "action_hidden": torch.empty(0, 4),
            "target_hidden": torch.empty(0, 4),
        },
        cache_path,
    )

    with pytest.raises(ValueError, match="No cached world-model records"):
        evaluate_probe(checkpoint=ckpt_path, cache=cache_path, output=out_path, device_name="cpu")
