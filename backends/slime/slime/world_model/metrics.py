from __future__ import annotations

import torch
import torch.nn.functional as F


def effective_rank(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Entropy-based effective rank for collapse diagnostics."""
    if x.numel() == 0:
        return torch.zeros((), device=x.device)
    if x.dim() > 2:
        x = x.flatten(0, -2)
    if x.size(0) < 2:
        return torch.ones((), device=x.device)
    x = x.float()
    x = x - x.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(x)
    probs = singular_values / singular_values.sum().clamp_min(eps)
    entropy = -(probs * probs.clamp_min(eps).log()).sum()
    return entropy.exp()


def mean_cosine_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = F.normalize(x.float(), dim=-1)
    y = F.normalize(y.float(), dim=-1)
    return 1.0 - (x * y).sum(dim=-1).mean()


def action_delta(pred: torch.Tensor, shuffled_pred: torch.Tensor) -> torch.Tensor:
    """Mean latent movement caused by replacing the action latent."""
    return (pred.float() - shuffled_pred.float()).pow(2).mean(dim=-1).sqrt().mean()
