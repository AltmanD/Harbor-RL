"""
TOPR: Trajectory / Token-level OPR importance correction.

Computes a sequence-level importance-sampling weight:

    log_w_seq_i = mean_{t in response_i}( log pi_prox(t) - log pi_behav(t) )
    w_seq_i    = exp( clip(log_w_seq_i, -logw_cap, +logw_cap) )
    w_seq_i    = clip(w_seq_i, w_min, w_max)

In long agentic trajectories (>2k tokens) the token-level IS weight
π_prox/π_behav accumulates numerical drift across tokens and across staleness
steps. TOPR collapses the IS correction to one scalar per response, yielding a
much tighter empirical distribution while preserving the unbiased expectation
in expectation over response tokens.

This module is self-contained (no other slime imports) and intentionally
torch-only so it can run under context-parallel (the per-response mean is taken
over locally-available response tokens; the consumer is responsible for any
all-reduce when CP > 1).

References:
- TOPR (arXiv 2025); DAPO §3.3 sequence-level IS; AReaL appendix B.
"""

from __future__ import annotations

from typing import List, Tuple

import torch


def _per_sample_log_ratio_mean(
    proximal_log_probs_list: List[torch.Tensor],
    behavior_log_probs_list: List[torch.Tensor],
    loss_masks_list: List[torch.Tensor],
) -> torch.Tensor:
    """
    Return a 1D tensor of per-sample mean log-ratio over response tokens.

    Each entry corresponds to one sample (response). The mean uses only tokens
    where loss_mask is non-zero (i.e. response tokens, excluding prompt/pad).
    Samples whose mask sums to zero get log_ratio_mean = 0 (=> w_seq = 1).
    """
    assert len(proximal_log_probs_list) == len(behavior_log_probs_list) == len(loss_masks_list), (
        f"shape mismatch: prox={len(proximal_log_probs_list)} "
        f"behav={len(behavior_log_probs_list)} masks={len(loss_masks_list)}"
    )
    if not proximal_log_probs_list:
        return torch.zeros(0)

    dtype = torch.float32
    means = []
    for p, b, m in zip(proximal_log_probs_list, behavior_log_probs_list, loss_masks_list):
        device = p.device
        p = p.to(device=device, dtype=dtype)
        b = b.to(device=device, dtype=dtype)
        mask = m.to(dtype=dtype, device=device)
        diff = (p - b) * mask
        denom = mask.sum().clamp_min(1.0)
        means.append(diff.sum() / denom)
    return torch.stack(means)


def compute_topr_seq_weights(
    proximal_log_probs_list: List[torch.Tensor],
    behavior_log_probs_list: List[torch.Tensor],
    loss_masks_list: List[torch.Tensor],
    logw_cap: float = 2.0,
    w_min: float = 0.0,
    w_max: float = 5.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute TOPR sequence-level IS weights.

    Returns:
        w_seq_per_sample:   1D tensor shape [N_samples], the scalar weight per sample
        w_seq_token_concat: 1D tensor shape [sum(len_i)] aligned to the post-cat
            token layout used by `decoupled_policy_loss_function`, broadcasting
            each sample's scalar across its response tokens.
    """
    log_ratio = _per_sample_log_ratio_mean(
        proximal_log_probs_list, behavior_log_probs_list, loss_masks_list
    )
    log_ratio = torch.clamp(log_ratio, min=-float(logw_cap), max=float(logw_cap))
    w_seq = torch.exp(log_ratio)
    if w_min is not None:
        w_seq = torch.clamp(w_seq, min=float(w_min))
    if w_max is not None:
        w_seq = torch.clamp(w_seq, max=float(w_max))

    # Broadcast each sequence scalar to the concatenated token layout.
    token_pieces = []
    for w, m in zip(w_seq, loss_masks_list):
        token_pieces.append(w.expand(int(m.numel())))
    w_seq_token = (
        torch.cat(token_pieces, dim=0)
        if token_pieces
        else torch.zeros(0, dtype=w_seq.dtype, device=w_seq.device)
    )
    return w_seq, w_seq_token


def blend_token_and_seq_weights(
    w_token: torch.Tensor,
    w_seq_token: torch.Tensor,
    blend_lambda: float,
) -> torch.Tensor:
    """
    Geometric blend between token-level and sequence-level IS weights:
        w = w_token^(1 - lambda) * w_seq_token^(lambda)

    blend_lambda=1.0 (default) → pure TOPR sequence-level.
    blend_lambda=0.0           → original token-level (no-op).
    """
    if blend_lambda >= 1.0:
        return w_seq_token
    if blend_lambda <= 0.0:
        return w_token
    return torch.pow(w_token, 1.0 - blend_lambda) * torch.pow(w_seq_token, blend_lambda)


def compute_dual_clip_metrics(
    log_ratio_intermediate: torch.Tensor,
    eps_low: float,
    eps_high: float,
    loss_mask: torch.Tensor,
) -> dict:
    """
    Diagnostic metrics for the dual-clip region (DAPO's asymmetric clipping).

    log_ratio_intermediate = log(π_prox) - log(π_θ); the PPO ratio
    used by `compute_decoupled_policy_loss` is exp(-log_ratio_intermediate).

    Returns counts and rates of tokens hitting each clip side, plus the masked
    fraction of tokens overall (for sanity).
    """
    if log_ratio_intermediate.numel() == 0:
        z = torch.zeros((), device=log_ratio_intermediate.device)
        return {
            "dual_clip_low_rate": z,
            "dual_clip_high_rate": z,
            "dual_clip_low_count": z,
            "dual_clip_high_count": z,
        }
    ratio = torch.exp(-log_ratio_intermediate)
    mask = loss_mask.to(dtype=ratio.dtype, device=ratio.device)
    denom = mask.sum().clamp_min(1.0)
    low_mask = (ratio < (1.0 - eps_low)).to(ratio.dtype) * mask
    high_mask = (ratio > (1.0 + eps_high)).to(ratio.dtype) * mask
    return {
        "dual_clip_low_rate": low_mask.sum() / denom,
        "dual_clip_high_rate": high_mask.sum() / denom,
        "dual_clip_low_count": low_mask.sum().detach(),
        "dual_clip_high_count": high_mask.sum().detach(),
    }
