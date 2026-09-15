"""Small latent-space planners kept independent from the policy trainer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from .modules import TextLatentWorldModel


@dataclass(frozen=True)
class MPCPlan:
    """One-step model-predictive choice for a fixed state."""

    selected_index: int
    scores: torch.Tensor
    values: torch.Tensor
    uncertainties: torch.Tensor | None


@torch.no_grad()
def plan_one_step(
    model: TextLatentWorldModel,
    *,
    state_hidden: torch.Tensor,
    candidate_action_hidden: torch.Tensor,
    uncertainty_coef: float = 0.0,
) -> MPCPlan:
    """Score several candidate actions from one common state.

    The state is broadcast to every candidate before the AdaLN predictor.  A
    candidate-specific state would silently turn this into ordinary ranking,
    so shape checks are intentionally strict.
    """

    if model.value_head is None:
        raise ValueError("latent MPC requires a checkpoint trained with value_head")
    state = state_hidden.float()
    actions = candidate_action_hidden.float()
    if state.dim() == 1:
        state = state.unsqueeze(0)
    if state.dim() != 2 or state.size(0) != 1:
        raise ValueError(f"state_hidden must have shape (D,) or (1,D), got {tuple(state.shape)}")
    if actions.dim() != 2 or actions.size(0) == 0:
        raise ValueError(f"candidate_action_hidden must have shape (N,D), got {tuple(actions.shape)}")
    state = state.expand(actions.size(0), -1)
    out = model(state_hidden=state, action_hidden=actions)
    values = out["value"]
    if values is None:
        raise ValueError("model did not return value predictions")
    uncertainty = out["uncertainty"]
    if uncertainty is None:
        scores = values
    else:
        scores = values - float(uncertainty_coef) * uncertainty
    selected = int(torch.argmax(scores).item())
    return MPCPlan(
        selected_index=selected,
        scores=scores.detach().cpu(),
        values=values.detach().cpu(),
        uncertainties=None if uncertainty is None else uncertainty.detach().cpu(),
    )


def candidate_tensor(payload: dict[str, torch.Tensor], indices: Iterable[int] | None = None) -> torch.Tensor:
    """Extract candidate action hiddens from a hidden-cache payload.

    Args:
        payload: Loaded ``hidden_cache.pt``-style dict holding an
            ``action_hidden`` tensor of shape ``(N, D)``.
        indices: Optional row subset, preserving the given order; ``None``
            selects every row.

    Returns:
        Float32 ``(M, D)`` candidate tensor, where ``M`` is ``N`` or
        ``len(indices)``.

    Raises:
        KeyError: If the payload has no ``action_hidden`` tensor.
    """

    actions = payload.get("action_hidden")
    if actions is None:
        raise KeyError("input cache is missing action_hidden")
    if indices is None:
        return actions.float()
    index = torch.tensor(list(indices), dtype=torch.long)
    return actions.index_select(0, index).float()
