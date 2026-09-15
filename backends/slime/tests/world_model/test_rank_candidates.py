import torch

from slime.world_model.rank_candidates import _select_scores


def test_rank_candidates_auto_prefers_value_over_target_error():
    out = {
        "pred_latent": torch.tensor([[0.0, 0.0], [10.0, 10.0]]),
        "target_latent": torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
        "value": torch.tensor([0.1, 0.9]),
        "uncertainty": torch.tensor([10.0, 0.1]),
    }

    scores, source = _select_scores(out, score_mode="auto")

    assert source == "value"
    assert torch.equal(scores, out["value"])


def test_rank_candidates_pred_error_requires_explicit_mode():
    out = {
        "pred_latent": torch.tensor([[0.0, 0.0], [10.0, 10.0]]),
        "target_latent": torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
        "value": torch.tensor([0.1, 0.9]),
        "uncertainty": torch.tensor([10.0, 0.1]),
    }

    scores, source = _select_scores(out, score_mode="pred_error")

    assert source == "negative_pred_error"
    assert scores[0] > scores[1]
