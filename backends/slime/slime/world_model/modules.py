from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .metrics import action_delta, effective_rank, mean_cosine_distance


def _masked_mean(x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if x.dim() == 2:
        return x
    if x.dim() != 3:
        raise ValueError(f"Expected hidden tensor with 2 or 3 dims, got {tuple(x.shape)}")
    if mask is None:
        return x.mean(dim=1)
    mask = mask.to(device=x.device, dtype=x.dtype).unsqueeze(-1)
    return (x * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


class SIGReg(nn.Module):
    """Sketch Isotropic Gaussian Regularizer adapted from local le-wm/module.py."""

    def __init__(self, knots: int = 17, num_proj: int = 1024) -> None:
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        """Isotropy statistic over random projections; 0 for batches < 2 rows."""

        if proj.dim() == 2:
            proj = proj.unsqueeze(0)
        if proj.size(-2) < 2:
            return proj.sum() * 0.0
        a = torch.randn(proj.size(-1), self.num_proj, device=proj.device, dtype=proj.dtype)
        a = a / a.norm(p=2, dim=0, keepdim=True).clamp_min(1e-6)
        x_t = (proj @ a).unsqueeze(-1) * self.t.to(device=proj.device, dtype=proj.dtype)
        phi = self.phi.to(device=proj.device, dtype=proj.dtype)
        weights = self.weights.to(device=proj.device, dtype=proj.dtype)
        err = (x_t.cos().mean(-3) - phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ weights) * proj.size(-2)
        return statistic.mean()


class StableProjector(nn.Module):
    """Normalize and project LLM hidden states into a controlled latent space."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int | None = None,
        *,
        clip_value: float = 30.0,
        output_norm: bool = True,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or max(input_dim, latent_dim)
        self.clip_value = float(clip_value)
        self.output_norm = output_norm
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float().clamp(min=-self.clip_value, max=self.clip_value)
        z = self.net(x)
        if self.output_norm:
            z = F.normalize(z, dim=-1)
        return z


class ActionConditionedPredictor(nn.Module):
    """Lightweight action-conditioned MLP without token or feature concat.

    The action is converted to FiLM/AdaLN parameters and only modulates the
    state stream.  Keeping this small predictor as an ablation avoids making
    the ``mlp`` option a loophole around the same action-conditioning contract
    used by the transformer predictor.
    """

    def __init__(self, latent_dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or latent_dim * 4
        self.state_norm = nn.LayerNorm(latent_dim, elementwise_affine=False)
        self.action_to_adaln = nn.Sequential(nn.SiLU(), nn.Linear(latent_dim, latent_dim * 2))
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, state_latent: torch.Tensor, action_latent: torch.Tensor) -> torch.Tensor:
        shift, scale = self.action_to_adaln(action_latent).chunk(2, dim=-1)
        pred = self.net(_modulate(self.state_norm(state_latent), shift, scale))
        return F.normalize(pred, dim=-1)


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale) + shift


class ActionAdaLNBlock(nn.Module):
    """Transformer block whose normalization and residual gates are action-conditioned.

    Observation/state latents are the only self-attention tokens.  The action
    latent is mapped to AdaLN shift/scale/gate parameters and is never appended
    to the token sequence.
    """

    def __init__(self, latent_dim: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        if latent_dim % num_heads != 0:
            raise ValueError(f"latent_dim={latent_dim} must be divisible by num_heads={num_heads}")
        self.norm1 = nn.LayerNorm(latent_dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(latent_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(latent_dim, elementwise_affine=False, eps=1e-6)
        mlp_dim = int(latent_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, mlp_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_dim, latent_dim),
        )
        self.action_to_adaln = nn.Sequential(nn.SiLU(), nn.Linear(latent_dim, latent_dim * 6))

    def forward(
        self,
        state_tokens: torch.Tensor,
        action_condition: torch.Tensor,
        *,
        causal_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = self.action_to_adaln(
            action_condition
        ).chunk(6, dim=-1)
        attn_input = _modulate(self.norm1(state_tokens), shift_attn, scale_attn)
        attn_out, _ = self.attn(
            attn_input,
            attn_input,
            attn_input,
            attn_mask=causal_mask,
            need_weights=False,
        )
        state_tokens = state_tokens + gate_attn * attn_out
        mlp_input = _modulate(self.norm2(state_tokens), shift_mlp, scale_mlp)
        return state_tokens + gate_mlp * self.mlp(mlp_input)


class ActionConditionedTransformerPredictor(nn.Module):
    """LeWM-style latent predictor with action AdaLN conditioning.

    Inputs may be independent transitions ``(B, D)`` or turn sequences
    ``(B, T, D)``.  In both cases, only state latents enter self-attention.
    """

    def __init__(
        self,
        latent_dim: int,
        *,
        depth: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        max_turns: int = 64,
    ) -> None:
        super().__init__()
        self.max_turns = int(max_turns)
        self.position = nn.Parameter(torch.zeros(1, self.max_turns, latent_dim))
        nn.init.normal_(self.position, std=0.02)
        self.blocks = nn.ModuleList(
            [ActionAdaLNBlock(latent_dim, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)]
        )
        self.final_norm = nn.LayerNorm(latent_dim, elementwise_affine=False, eps=1e-6)
        self.final_adaln = nn.Sequential(nn.SiLU(), nn.Linear(latent_dim, latent_dim * 2))
        self.output = nn.Linear(latent_dim, latent_dim)

    def forward(self, state_latent: torch.Tensor, action_latent: torch.Tensor) -> torch.Tensor:
        """Predict per-turn next-state latents; turn self-attention is causal."""

        squeeze_turn = state_latent.dim() == 2
        if squeeze_turn:
            state_latent = state_latent.unsqueeze(1)
        if action_latent.dim() == 2:
            action_latent = action_latent.unsqueeze(1)
        if state_latent.dim() != 3 or action_latent.dim() != 3:
            raise ValueError(
                "state_latent and action_latent must have shape (B,D) or (B,T,D); "
                f"got {tuple(state_latent.shape)} and {tuple(action_latent.shape)}"
            )
        if state_latent.shape != action_latent.shape:
            raise ValueError(
                f"state/action latent shapes must match, got {tuple(state_latent.shape)} and "
                f"{tuple(action_latent.shape)}"
            )
        turns = state_latent.size(1)
        if turns > self.max_turns:
            raise ValueError(f"turn sequence length {turns} exceeds max_turns={self.max_turns}")

        x = state_latent + self.position[:, :turns].to(dtype=state_latent.dtype)
        causal_mask = None
        if turns > 1:
            causal_mask = torch.triu(
                torch.ones(turns, turns, device=x.device, dtype=torch.bool),
                diagonal=1,
            )
        for block in self.blocks:
            x = block(x, action_latent, causal_mask=causal_mask)
        shift, scale = self.final_adaln(action_latent).chunk(2, dim=-1)
        pred = self.output(_modulate(self.final_norm(x), shift, scale))
        pred = F.normalize(pred, dim=-1)
        return pred.squeeze(1) if squeeze_turn else pred


@dataclass
class TextLatentWorldModelConfig:
    state_hidden_dim: int
    action_hidden_dim: int
    target_hidden_dim: int
    latent_dim: int = 1024
    adapter_dim: int | None = None
    projector_hidden_dim: int | None = None
    predictor_hidden_dim: int | None = None
    predictor_type: str = "adaln"
    predictor_depth: int = 2
    predictor_num_heads: int = 1
    predictor_mlp_ratio: float = 4.0
    predictor_max_turns: int = 64
    clip_value: float = 30.0
    sigreg_num_proj: int = 1024
    action_contrast_margin: float = 0.05
    value_head: bool = True
    uncertainty_head: bool = True
    # Offline LeWM follows the upstream joint-embedding objective by default.
    # Online callers can enable stop-gradient/EMA targets explicitly.
    stop_grad_target: bool = False


class TextLatentWorldModel(nn.Module):
    """JEPA-style text world model for terminal-agent replay data.

    The module treats policy/frozen-encoder hidden states as raw material only.
    All branches pass through explicit projectors before entering the shared
    latent space, matching the hidden-to-belief-latent design in the survey.
    """

    def __init__(self, config: TextLatentWorldModelConfig) -> None:
        super().__init__()
        self.config = config
        adapter_dim = config.adapter_dim or config.latent_dim
        self.state_adapter = StableProjector(
            config.state_hidden_dim,
            adapter_dim,
            config.projector_hidden_dim,
            clip_value=config.clip_value,
            output_norm=False,
        )
        self.action_projector = StableProjector(
            config.action_hidden_dim,
            config.latent_dim,
            config.projector_hidden_dim,
            clip_value=config.clip_value,
        )
        self.target_adapter = StableProjector(
            config.target_hidden_dim,
            adapter_dim,
            config.projector_hidden_dim,
            clip_value=config.clip_value,
            output_norm=False,
        )
        self.shared_projector = StableProjector(
            adapter_dim,
            config.latent_dim,
            config.projector_hidden_dim,
            clip_value=config.clip_value,
        )
        if config.predictor_type == "mlp":
            self.predictor = ActionConditionedPredictor(config.latent_dim, config.predictor_hidden_dim)
        elif config.predictor_type == "adaln":
            self.predictor = ActionConditionedTransformerPredictor(
                config.latent_dim,
                depth=config.predictor_depth,
                num_heads=config.predictor_num_heads,
                mlp_ratio=config.predictor_mlp_ratio,
                max_turns=config.predictor_max_turns,
            )
        else:
            raise ValueError(f"Unknown predictor_type={config.predictor_type!r}; expected 'adaln' or 'mlp'")
        # Candidate quality must depend on the predicted consequence, not only
        # on the current state, so value/uncertainty heads consume pred_latent.
        self.value_head = nn.Linear(config.latent_dim, 1) if config.value_head else None
        self.uncertainty_head = nn.Linear(config.latent_dim, 1) if config.uncertainty_head else None
        self.sigreg = SIGReg(num_proj=config.sigreg_num_proj)

    def forward(
        self,
        *,
        state_hidden: torch.Tensor,
        action_hidden: torch.Tensor,
        target_hidden: torch.Tensor | None = None,
        next_state_hidden: torch.Tensor | None = None,
        state_mask: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
        target_mask: torch.Tensor | None = None,
        next_state_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        """Project raw hiddens into the shared latent space and predict.

        Hidden inputs may be pre-pooled ``(B, D)`` or token-level ``(B, T, D)``
        rows reduced with their matching mask.  ``value`` and ``uncertainty``
        are read from the *predicted* latent, so candidate quality reflects
        the modeled consequence rather than the current state alone.

        Args:
            state_hidden: Current-state hiddens from the policy encoder.
            action_hidden: Action hiddens with the same batch size.
            target_hidden: Optional environment-feedback hiddens (JEPA target).
            next_state_hidden: Optional successor-state hiddens for alignment.
            state_mask: Token mask for ``state_hidden`` when 3-D.
            action_mask: Token mask for ``action_hidden`` when 3-D.
            target_mask: Token mask for ``target_hidden`` when 3-D.
            next_state_mask: Token mask for ``next_state_hidden`` when 3-D.

        Returns:
            Dict with ``state_latent``, ``action_latent``, ``pred_latent``;
            ``target_latent``/``next_state_latent`` are None when the matching
            input is absent, and ``value``/``uncertainty`` are None when the
            respective head is disabled.
        """

        state_feat = _masked_mean(state_hidden, state_mask)
        action_feat = _masked_mean(action_hidden, action_mask)
        state_latent = self.shared_projector(self.state_adapter(state_feat))
        action_latent = self.action_projector(action_feat)
        pred_latent = self.predictor(state_latent, action_latent)
        target_latent = None
        if target_hidden is not None:
            target_feat = _masked_mean(target_hidden, target_mask)
            target_latent = self.shared_projector(self.target_adapter(target_feat))

        next_state_latent = None
        if next_state_hidden is not None:
            next_state_feat = _masked_mean(next_state_hidden, next_state_mask)
            next_state_latent = self.shared_projector(self.state_adapter(next_state_feat))

        value = self.value_head(pred_latent).squeeze(-1) if self.value_head is not None else None
        uncertainty = None
        if self.uncertainty_head is not None:
            uncertainty = F.softplus(self.uncertainty_head(pred_latent).squeeze(-1))

        return {
            "state_latent": state_latent,
            "action_latent": action_latent,
            "pred_latent": pred_latent,
            "target_latent": target_latent,
            "next_state_latent": next_state_latent,
            "value": value,
            "uncertainty": uncertainty,
        }

    def compute_loss(
        self,
        *,
        state_hidden: torch.Tensor,
        action_hidden: torch.Tensor,
        target_hidden: torch.Tensor,
        next_state_hidden: torch.Tensor | None = None,
        has_next: torch.Tensor | None = None,
        reward: torch.Tensor | None = None,
        reward_mask: torch.Tensor | None = None,
        pred_loss_type: str = "mse",
        sigreg_coef: float = 0.1,
        action_contrast_coef: float = 0.1,
        alignment_coef: float = 0.1,
        value_coef: float = 0.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Combine the latent prediction objective with gated auxiliary losses.

        Term gating:

        - prediction loss: always on; ``pred_loss_type`` selects mse /
          smooth_l1 / cosine, and the target branch is detached when
          ``config.stop_grad_target``.
        - SIGReg: always computed on the state latents, scaled by
          ``sigreg_coef``.
        - action contrast: only with batch size > 1 and
          ``action_contrast_coef != 0``; roll-shifted actions act as negatives.
        - alignment: only when ``next_state_hidden`` is given and
          ``alignment_coef != 0``; ``has_next`` restricts it to real
          successors when provided.
        - value: only when ``reward`` is given, the value head exists, and
          ``value_coef != 0``; ``reward_mask`` limits the regression to rows
          with observed rewards.

        Args:
            state_hidden: Per-transition state hiddens, ``(B, D)`` or ``(B, T, D)``.
            action_hidden: Action hiddens matching the batch.
            target_hidden: Environment-feedback hiddens (prediction target).
            next_state_hidden: Optional successor-state hiddens.
            has_next: Optional bool mask of rows with a real successor.
            reward: Optional return targets for the value head.
            reward_mask: Optional mask marking rows with observed rewards.
            pred_loss_type: ``mse`` | ``smooth_l1`` | ``cosine``.
            sigreg_coef: Weight of the SIGReg manifold regularizer.
            action_contrast_coef: Weight of the shuffled-action contrast term.
            alignment_coef: Weight of the next-state alignment term.
            value_coef: Weight of the value regression term; 0 disables it.

        Returns:
            ``(total_loss, metrics)``; metrics are detached ``wm/``-keyed term
            values plus diagnostics (effective rank, action delta, value mask
            count) for guardrail monitoring.
        """

        out = self(
            state_hidden=state_hidden,
            action_hidden=action_hidden,
            target_hidden=target_hidden,
            next_state_hidden=next_state_hidden,
        )
        pred = out["pred_latent"]
        target = out["target_latent"]
        pred_target = target.detach() if self.config.stop_grad_target else target
        if pred_loss_type == "cosine":
            pred_loss = mean_cosine_distance(pred, pred_target)
        elif pred_loss_type == "smooth_l1":
            pred_loss = F.smooth_l1_loss(pred, pred_target)
        else:
            pred_loss = F.mse_loss(pred, pred_target)

        # SIGReg regularizes the learned state manifold.  The feedback branch
        # is the stop-gradient anchor and is not regularized here.
        sigreg_loss = self.sigreg(out["state_latent"])
        contrast_loss = pred_loss * 0.0
        delta = pred_loss * 0.0
        if action_hidden.size(0) > 1 and action_contrast_coef != 0.0:
            shuffled_action = torch.roll(out["action_latent"], shifts=1, dims=0)
            shuffled_pred = self.predictor(out["state_latent"], shuffled_action)
            pos_dist = (pred - pred_target).pow(2).mean(dim=-1)
            neg_dist = (shuffled_pred - pred_target).pow(2).mean(dim=-1)
            margin = float(self.config.action_contrast_margin)
            contrast_loss = F.relu(margin + pos_dist - neg_dist).mean()
            delta = action_delta(pred, shuffled_pred)

        alignment_loss = pred_loss * 0.0
        next_state_latent = out["next_state_latent"]
        if next_state_latent is not None and alignment_coef != 0.0:
            align_target = target.detach()
            if has_next is None:
                alignment_loss = F.mse_loss(next_state_latent, align_target)
            else:
                mask = has_next.to(device=pred.device, dtype=torch.bool).view(-1)
                if mask.any():
                    alignment_loss = F.mse_loss(next_state_latent[mask], align_target[mask])

        value_loss = pred_loss * 0.0
        if reward is not None and out["value"] is not None and value_coef != 0.0:
            reward = reward.float().view_as(out["value"])
            if reward_mask is not None:
                mask = reward_mask.to(device=out["value"].device, dtype=torch.bool).view_as(out["value"])
                if mask.any():
                    value_loss = F.smooth_l1_loss(out["value"][mask], reward[mask])
            else:
                value_loss = F.smooth_l1_loss(out["value"], reward)

        loss = (
            pred_loss
            + sigreg_coef * sigreg_loss
            + action_contrast_coef * contrast_loss
            + alignment_coef * alignment_loss
            + value_coef * value_loss
        )
        metrics = {
            "wm/pred_loss": pred_loss.detach(),
            "wm/sigreg_loss": sigreg_loss.detach(),
            "wm/action_contrast_loss": contrast_loss.detach(),
            "wm/alignment_loss": alignment_loss.detach(),
            "wm/value_loss": value_loss.detach(),
            "wm/value_mask_count": (
                reward_mask.to(device=pred_loss.device, dtype=torch.float32).sum().detach()
                if reward_mask is not None
                else torch.as_tensor(0.0 if reward is None else reward.numel(), device=pred_loss.device)
            ),
            "wm/effective_rank": effective_rank(out["state_latent"]).detach(),
            "wm/action_delta": delta.detach(),
        }
        return loss, metrics
