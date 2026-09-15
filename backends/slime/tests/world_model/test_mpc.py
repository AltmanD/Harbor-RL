import pytest
import torch

from slime.world_model.mpc import plan_one_step
from slime.world_model.modules import TextLatentWorldModel, TextLatentWorldModelConfig


def test_one_step_mpc_broadcasts_one_state_to_candidates():
    config = TextLatentWorldModelConfig(
        state_hidden_dim=4,
        action_hidden_dim=4,
        target_hidden_dim=4,
        latent_dim=4,
        predictor_num_heads=2,
        sigreg_num_proj=4,
        uncertainty_head=False,
    )
    model = TextLatentWorldModel(config)
    plan = plan_one_step(
        model,
        state_hidden=torch.randn(4),
        candidate_action_hidden=torch.randn(3, 4),
    )
    assert plan.scores.shape == (3,)
    assert 0 <= plan.selected_index < 3


def test_mpc_requires_value_head():
    config = TextLatentWorldModelConfig(
        state_hidden_dim=2,
        action_hidden_dim=2,
        target_hidden_dim=2,
        latent_dim=2,
        predictor_num_heads=1,
        value_head=False,
        uncertainty_head=False,
        sigreg_num_proj=2,
    )
    model = TextLatentWorldModel(config)
    with pytest.raises(ValueError, match="value_head"):
        plan_one_step(model, state_hidden=torch.randn(2), candidate_action_hidden=torch.randn(2, 2))
