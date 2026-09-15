"""In-process joint LWM training with an EMA target provider.

This is the single-process variant of the online-learner architecture: instead
of a separate learner job polling replay snapshots, the LWM updates inside the
rollout-side process of the training run itself, on the same rollout batch that
the policy is about to learn from.  The world-model target provider then serves
target latents from an **EMA copy** of the LWM target branch
(``target_adapter`` + ``shared_projector``), so the policy auxiliary loss always
regresses toward a slow-moving target while the online LWM keeps learning —
the BYOL/JEMA-style moving-target guard adapted to the joint setting.

Activation is env-gated from the provider entry point
(``target_provider.frozen_wm_target_provider``): when ``LWM_ONLINE_JOINT=1`` the
provider routes here.  Knobs (all optional):

- ``LWM_JOINT_BATCH_SIZE`` (16), ``LWM_JOINT_REPLAY_RATIO`` (0.5),
  ``LWM_JOINT_BUFFER_SIZE`` (4096): replay-mixed LWM updates per rollout batch,
  ``ceil(new / batch_size)`` steps, matching the equal-compute rule.
- ``LWM_JOINT_LR`` (1e-4), ``LWM_JOINT_EMA_DECAY`` (0.99),
  ``LWM_JOINT_SEED`` (42).
- ``LWM_TARGET_CHECKPOINT``: warm-start the online LWM from a pooled/prior
  checkpoint (recommended); otherwise it starts from scratch.
- ``LWM_JOINT_OUTPUT_DIR``: where ``latent_world_model_joint.pt`` and
  ``replay_buffer_joint.pt`` are persisted after each update (default: unset →
  no persistence).
- Encoder reuse: ``LWM_TARGET_*`` envs (model path / device / encode batch) are
  shared with the provider; the same frozen encoder serves LWM training inputs
  and target emission, so joint mode costs one 8B load per worker process.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import torch

from .replay_buffer import TrajectoryReplayBuffer
from .seta_dataset import TerminalTransition, transition_from_world_model_record
from .stream_latent import _run_batches, compose_arm_batches

_state: dict[str, Any] = {}


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


class JointLWMTrainer:
    """Process-resident LWM trainer + EMA target branch.

    One instance per worker process (see ``get_trainer``).  ``update`` ingests
    the current rollout batch, trains the LWM on replay-mixed hidden batches,
    refreshes the EMA target branch, and returns one detached target latent per
    ingested sample (None where the sample carried no observation).
    """

    def __init__(self, args: Any) -> None:
        self.args = args
        self.latent_dim = int(getattr(args, "world_model_latent_dim", 128) or 128)
        self.batch_size = _env_int("LWM_JOINT_BATCH_SIZE", 16)
        self.replay_ratio = _env_float("LWM_JOINT_REPLAY_RATIO", 0.5)
        self.seed = _env_int("LWM_JOINT_SEED", 42)
        self.ema_decay = _env_float("LWM_JOINT_EMA_DECAY", 0.99)
        self.output_dir = os.environ.get("LWM_JOINT_OUTPUT_DIR") or None
        torch.manual_seed(self.seed)

        encoder, model, device = _load_or_build_model(args)
        self.encoder, self.model, self.device = encoder, model, device
        self.ema_target_adapter = copy.deepcopy(model.target_adapter).eval()
        self.ema_shared_projector = copy.deepcopy(model.shared_projector).eval()
        for module in (self.ema_target_adapter, self.ema_shared_projector):
            for param in module.parameters():
                param.requires_grad_(False)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=_env_float("LWM_JOINT_LR", 1e-4))
        self.buffer = TrajectoryReplayBuffer(_env_int("LWM_JOINT_BUFFER_SIZE", 4096), seed=self.seed)
        self.seen_ids: set[str] = set()
        self.transitions: list[TerminalTransition] = []
        self.hidden_rows: dict[str, list[torch.Tensor]] = {}
        self.total_steps = 0
        self.total_calls = 0

    def _ema_update(self) -> None:
        with torch.no_grad():
            for ema, live in (
                (self.ema_target_adapter, self.model.target_adapter),
                (self.ema_shared_projector, self.model.shared_projector),
            ):
                for ema_param, live_param in zip(ema.parameters(), live.parameters(), strict=True):
                    ema_param.mul_(self.ema_decay).add_(live_param.detach(), alpha=1.0 - self.ema_decay)

    def _persist(self) -> None:
        if not self.output_dir:
            return
        out = Path(self.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": "openclaw_terminal_latent_wm_joint_v1",
                "config": self.model.config.__dict__,
                "state_dict": self.model.state_dict(),
                "ema_target_adapter": self.ema_target_adapter.state_dict(),
                "ema_shared_projector": self.ema_shared_projector.state_dict(),
                "total_steps": self.total_steps,
                "replay_stats": self.buffer.stats(),
            },
            out / "latent_world_model_joint.pt",
        )
        self.buffer.save(out / "replay_buffer_joint.pt")

    def _records_to_transitions(self, samples: list[Any]) -> list[TerminalTransition]:
        transitions: list[TerminalTransition] = []
        for index, sample in enumerate(samples):
            metadata = getattr(sample, "metadata", None)
            record = metadata.get("world_model") if isinstance(metadata, dict) else None
            if not isinstance(record, dict):
                continue
            try:
                transition = transition_from_world_model_record(record, source_path=f"joint:{index}")
            except (TypeError, ValueError, KeyError):
                continue
            if transition.action_text and transition.feedback_text:
                transitions.append(transition)
        return transitions

    def _encode_fresh(self, fresh: list[TerminalTransition]) -> None:
        encode_batch = _env_int("LWM_TARGET_ENCODE_BATCH", 4)
        batched: dict[str, list[torch.Tensor]] = {}
        for start in range(0, len(fresh), encode_batch):
            encoded = self.encoder(fresh[start : start + encode_batch])
            for key, value in encoded.items():
                batched.setdefault(key, []).append(value.detach().cpu())
        for key, value in batched.items():
            self.hidden_rows.setdefault(key, []).extend(value)

    def _cached_hidden(self) -> dict[str, torch.Tensor] | None:
        if not self.hidden_rows:
            return None
        return {key: torch.cat(values, dim=0) for key, values in self.hidden_rows.items()}

    @torch.no_grad()
    def _ema_targets(self, texts: list[str]) -> list[list[float]]:
        latents: list[list[float]] = []
        encode_batch = _env_int("LWM_TARGET_ENCODE_BATCH", 4)
        for start in range(0, len(texts), encode_batch):
            rows = [self.encoder._target_ids(text) for text in texts[start : start + encode_batch]]
            hidden, mask = self.encoder._forward_hidden(rows, require_grad=False)
            pooled = (hidden * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)
            latent = self.ema_shared_projector(self.ema_target_adapter(pooled))
            latents.extend(latent.float().cpu().tolist())
        return latents

    def update(self, samples: list[Any]) -> dict[int, list[float]]:
        """Ingest one rollout batch; return ``{sample_index: ema_target_latent}``."""

        self.total_calls += 1
        transitions = self._records_to_transitions(samples)
        fresh_indices: list[int] = []
        for transition in transitions:
            if transition.transition_id in self.seen_ids:
                continue
            self.seen_ids.add(transition.transition_id)
            self.transitions.append(transition)
            fresh_indices.append(len(self.transitions) - 1)

        if fresh_indices:
            fresh = [self.transitions[index] for index in fresh_indices]
            self._encode_fresh(fresh)
            self.buffer.push(fresh, current_step=self.total_calls)
            id_to_index = {row.transition_id: i for i, row in enumerate(self.transitions)}
            batches = compose_arm_batches(
                arm="replay",
                chunk_indices=fresh_indices,
                buffer=self.buffer,
                id_to_index=id_to_index,
                batch_size=self.batch_size,
                replay_ratio=self.replay_ratio,
                seed=self.seed + self.total_calls,
                current_step=self.total_calls,
            )
            _run_batches(
                model=self.model,
                transitions=self.transitions,
                batches=batches,
                cached_hidden=self._cached_hidden(),
                policy_encoder=None,
                optimizer=self.optimizer,
                device=self.device,
                sigreg_coef=_env_float("LWM_JOINT_SIGREG_COEF", 0.09),
                action_contrast_coef=_env_float("LWM_JOINT_ACTION_CONTRAST_COEF", 0.1),
                alignment_coef=_env_float("LWM_JOINT_ALIGNMENT_COEF", 0.1),
                value_coef=0.0,
                reward_targets=None,
                gradient_clip=1.0,
            )
            self.total_steps += len(batches)
            self._ema_update()
            self._persist()

        texts: dict[int, str] = {}
        for index, sample in enumerate(samples):
            metadata = getattr(sample, "metadata", None)
            record = metadata.get("world_model") if isinstance(metadata, dict) else None
            if isinstance(record, dict):
                text = str(record.get("next_observation_text") or "").strip()
                if text:
                    texts[index] = text
        if not texts:
            return {}
        ordered = sorted(texts)
        values = self._ema_targets([texts[index] for index in ordered])
        return dict(zip(ordered, values, strict=True))


def _load_or_build_model(args: Any):
    """Warm-start from LWM_TARGET_CHECKPOINT (required for config parity)."""

    from .target_provider import _load_hf_stack  # noqa: PLC0415

    if not os.environ.get("LWM_TARGET_CHECKPOINT"):
        raise RuntimeError(
            "LWM_ONLINE_JOINT=1 requires LWM_TARGET_CHECKPOINT for a config-compatible "
            "warm start (cold-start wiring is intentionally not implicit)"
        )
    encoder, model, device = _load_hf_stack(args)
    # The provider loader freezes the model; joint mode trains it, so re-enable.
    for param in model.parameters():
        param.requires_grad_(True)
    model.train()
    return encoder, model, device


def get_trainer(args: Any) -> JointLWMTrainer:
    if "joint_trainer" not in _state:
        _state["joint_trainer"] = JointLWMTrainer(args)
    return _state["joint_trainer"]


def joint_update_and_targets(*, args: Any, samples: list[Any]) -> dict[int, list[float]]:
    """Provider-facing entry: train the in-process LWM, serve EMA targets."""

    return get_trainer(args).update(samples)
