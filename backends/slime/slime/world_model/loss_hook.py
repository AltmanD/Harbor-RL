from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F

from slime.utils.misc import load_function


def _as_latent_batch(value: Any, *, device: torch.device) -> torch.Tensor | None:
    """Convert optional per-sample latent payloads to a dense ``(B, D)`` tensor."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        # Missing entries are not meaningful supervision.  Require a dense
        # payload instead of silently shifting targets between samples.
        if any(item is None for item in value):
            return None
        value = torch.stack(
            [item if isinstance(item, torch.Tensor) else torch.as_tensor(item) for item in value]
        )
    elif not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    value = value.to(device=device, dtype=torch.float32)
    if value.dim() == 1:
        value = value.unsqueeze(0)
    elif value.dim() > 2:
        value = value.flatten(start_dim=1, end_dim=-2).mean(dim=1)
    return value


def _response_pooled_logits(logits: torch.Tensor, args: Any, batch: Mapping[str, Any]) -> torch.Tensor:
    """Pool policy logits over each response span while retaining gradients.

    Megatron emits ``[1, T, V]`` for packed ``thd`` and ``[B, T, V]`` for
    ``bshd``.  This helper mirrors the response alignment in ``get_responses``
    without importing the backend loss module (which would create a cycle).
    Context-parallel batches are already local slices at this point; pooling
    the local response tokens remains a valid, deterministic auxiliary signal.
    """
    if logits.dim() != 3:
        raise ValueError(f"Expected policy logits with shape [1,T,V] or [B,T,V], got {tuple(logits.shape)}")
    flat = logits.squeeze(0) if str(getattr(args, "qkv_format", "thd")) == "thd" else logits.reshape(-1, logits.size(-1))
    totals = list(batch.get("total_lengths") or [])
    responses = list(batch.get("response_lengths") or [])
    max_lens = list(batch.get("max_seq_lens") or [])
    if not totals or len(totals) != len(responses):
        return flat.mean(dim=0, keepdim=True)
    rows: list[torch.Tensor] = []
    end = 0
    qkv_format = str(getattr(args, "qkv_format", "thd"))
    for i, (total, response) in enumerate(zip(totals, responses, strict=False)):
        total = int(total)
        response = int(response)
        if response <= 0:
            rows.append(flat.sum(dim=0) * 0.0)
            continue
        if qkv_format == "bshd":
            max_len = int(max_lens[i] if i < len(max_lens) else total)
            end = i * max_len + total
            start = end - response
        else:
            end += total
            start = end - response
        # get_responses predicts token t from logits at t-1, hence the same
        # one-position shift used here.  Clamp malformed spans to avoid an
        # auxiliary hook taking down an otherwise valid training run.
        lo, hi = max(0, start - 1), min(flat.size(0), end - 1)
        rows.append(flat[lo:hi].mean(dim=0) if hi > lo else flat.sum(dim=0) * 0.0)
    return torch.stack(rows, dim=0)


def _hash_project(features: torch.Tensor, latent_dim: int) -> torch.Tensor:
    """Project vocabulary features with a stateless signed hash.

    A dense ``V x D`` matrix is prohibitively expensive for an 8B policy.  A
    signed bucket projection has the same differentiable contract while using
    only ``O(V + D)`` temporary storage and no trainable parameters.  The
    target latent is learned by the separate latent WM, so this projection is
    intentionally fixed and cannot become a shortcut around the policy.
    """
    latent_dim = max(1, int(latent_dim))
    vocab = features.size(-1)
    indices = torch.arange(vocab, device=features.device, dtype=torch.int64)
    buckets = torch.remainder(indices * 1103515245 + 12345, latent_dim)
    signs = torch.where(
        torch.remainder(indices * 214013 + 2531011, 2) == 0,
        torch.ones_like(indices, dtype=features.dtype),
        -torch.ones_like(indices, dtype=features.dtype),
    )
    projected = features.new_zeros((features.size(0), latent_dim))
    projected.scatter_add_(1, buckets.unsqueeze(0).expand(features.size(0), -1), features * signs)
    return projected / max(vocab, 1) ** 0.5


def _as_scalar_tensor(value: Any, *, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        value = value.to(device=device)
        if value.numel() != 1:
            value = value.float().mean()
        return value.reshape(())
    return torch.tensor(float(value), dtype=torch.float32, device=device)


def _as_sample_mask(value: Any, *, batch_size: int, device: torch.device) -> torch.Tensor | None:
    """Normalize an optional per-sample target mask to shape ``(B,)``."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        value = [
            item.detach().float().reshape(-1).mean() if isinstance(item, torch.Tensor) else item
            for item in value
        ]
    try:
        mask = torch.as_tensor(value, device=device, dtype=torch.float32)
    except (TypeError, ValueError):
        return None
    if mask.numel() == 1:
        mask = mask.reshape(1).expand(batch_size)
    elif mask.numel() == batch_size:
        mask = mask.reshape(batch_size)
    elif mask.numel() % batch_size == 0:
        mask = mask.reshape(batch_size, -1).mean(dim=1)
    else:
        return None
    return mask.clamp(0.0, 1.0)


def default_world_model_loss_hook(
    args: Any,
    batch: dict[str, Any],
    logits: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute an optional policy-level latent WM loss.

    ``wm_pred_latents`` remains the explicit adapter escape hatch.  When
    ``--world-model-backprop-to-llm`` is set and only environment targets are
    present, response-pooled policy logits are projected into the same latent
    space with a fixed signed hash.  Since the projection is applied directly
    to ``logits`` (rather than ``logits.detach()``), the auxiliary loss creates
    a real policy gradient.  A collector that can expose true transformer
    hidden states can pass them as ``wm_pred_latents`` and bypass this
    vocabulary projection.
    """
    device = logits.device
    metadata = batch.get("wm_metadata") or []
    pred = batch.get("wm_pred_latents")
    target = batch.get("wm_target_latents")
    aux_loss = logits.sum() * 0.0
    available = 0.0
    projection = str(getattr(args, "world_model_policy_projection", "hash"))
    if target is not None and pred is None and bool(getattr(args, "world_model_backprop_to_llm", False)):
        pooled = _response_pooled_logits(logits, args, batch).float()
        latent_dim = int(getattr(args, "world_model_latent_dim", 128) or 128)
        pred = pooled.mean(dim=-1, keepdim=True).expand(-1, latent_dim) if projection == "mean" else _hash_project(pooled, latent_dim)
    pred = _as_latent_batch(pred, device=device)
    target = _as_latent_batch(target, device=device)
    if pred is not None and target is not None:
        if pred.size(0) != target.size(0):
            raise ValueError(f"world-model latent batch mismatch: pred={tuple(pred.shape)} target={tuple(target.shape)}")
        if pred.size(-1) != target.size(-1):
            raise ValueError(f"world-model latent width mismatch: pred={tuple(pred.shape)} target={tuple(target.shape)}")
        per_sample_loss = F.mse_loss(pred, target.detach(), reduction="none").mean(dim=-1)
        target_mask = _as_sample_mask(
            batch.get("wm_target_mask"), batch_size=pred.size(0), device=device
        )
        if target_mask is None:
            target_mask = torch.ones_like(per_sample_loss)
        aux_loss = (per_sample_loss * target_mask).sum() / target_mask.sum().clamp_min(1.0)
        available = 1.0
    metrics = {
        "wm/loss": aux_loss.detach(),
        "wm/metadata_count": torch.tensor(float(len(metadata)), device=device),
        "wm/latent_available": torch.tensor(available, device=device),
        "wm/policy_gradient_path": torch.tensor(
            1.0 if available and pred is not None and pred.requires_grad else 0.0,
            device=device,
        ),
        "wm/target_mask_fraction": torch.tensor(
            float(target_mask.mean().item()) if pred is not None and target is not None else 0.0,
            device=device,
        ),
    }
    return aux_loss, metrics


def apply_world_model_loss(
    *,
    args: Any,
    batch: dict[str, Any],
    logits: torch.Tensor,
    loss: torch.Tensor,
    reported_loss: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, Any]]:
    enabled = bool(getattr(args, "world_model_enable", False))
    coef = float(getattr(args, "world_model_loss_coef", 0.0) or 0.0)
    if not enabled or coef == 0.0:
        return loss, reported_loss

    hook_path = getattr(args, "world_model_loss_hook_path", None)
    hook = load_function(hook_path) if hook_path else default_world_model_loss_hook
    hook_result = hook(args, batch, logits)
    if isinstance(hook_result, Mapping):
        aux_loss = hook_result.get("loss", logits.sum() * 0.0)
        metrics = dict(hook_result.get("metrics", {}))
    else:
        aux_loss, metrics = hook_result

    aux_loss = _as_scalar_tensor(aux_loss, device=logits.device)
    loss = loss + coef * aux_loss
    reported_loss["wm/loss"] = aux_loss.detach()
    reported_loss["wm/loss_coef"] = torch.tensor(coef, dtype=torch.float32, device=logits.device)
    for name, value in dict(metrics or {}).items():
        key = name if str(name).startswith("wm/") else f"wm/{name}"
        reported_loss[key] = _as_scalar_tensor(value, device=logits.device).detach()
    reported_loss["loss"] = loss.clone().detach()
    return loss, reported_loss
