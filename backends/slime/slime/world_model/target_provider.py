"""Frozen latent-WM target provider for the policy-level auxiliary loss.

Loaded through ``--world-model-target-provider-path
slime.world_model.target_provider:frozen_wm_target_provider``.  For every
rollout sample the provider encodes the recorded environment observation
(``metadata["world_model"]["next_observation_text"]``) with a frozen policy
encoder plus the trained LWM target branch (``target_adapter`` +
``shared_projector``) and writes one detached target latent per sample into
``metadata["world_model"]["target_latents"]`` (with ``target_mask``).

Runtime configuration is intentionally env-based so the rollout-side call
site (``attach_terminal_world_model_metadata``) needs no signature change:

- ``LWM_TARGET_PROVIDER_ENCODER``: ``hf`` (default, frozen LWM target
  branch), ``raw`` (frozen encoder hidden, L2-normalized, no learned head —
  the zero-training ablation; pair with ``--world-model-latent-dim
  <encoder hidden size>`` so the policy-side projection width matches), or
  ``hash`` (plumbing smoke: deterministic pseudo-targets, no semantics).
- ``LWM_ONLINE_JOINT``: when set to ``1``, routing goes to
  ``joint_online.joint_update_and_targets`` instead — the LWM trains inside
  this same process on each rollout batch and targets come from its EMA
  target branch (see ``joint_online.py``).
- ``LWM_TARGET_CHECKPOINT``: trained LWM checkpoint with ``config`` +
  ``state_dict`` (required in ``hf`` mode; warm start in joint mode).
- ``LWM_TARGET_ENCODER_MODEL``: local HF policy checkpoint used as the frozen
  text encoder (default: site Qwen3-8B path).
- ``LWM_TARGET_DEVICE``: ``auto`` (default), ``cuda`` or ``cpu``.
- ``LWM_TARGET_ENCODE_BATCH``: texts per encode forward (default 4).
"""

from __future__ import annotations

import os
from typing import Any

import torch

from .hidden_encoder import PolicyHiddenEncoder, hash_hidden_batch
from .modules import TextLatentWorldModel, TextLatentWorldModelConfig
from .seta_dataset import TerminalTransition

_DEFAULT_ENCODER_MODEL = "/mnt/shared-storage-user/puyuan/code/slime/Qwen3-8B"

_state: dict[str, Any] = {}


def _device() -> torch.device:
    pref = os.environ.get("LWM_TARGET_DEVICE", "auto")
    if pref == "auto":
        pref = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(pref)


def _latent_dim(args: Any) -> int:
    return int(getattr(args, "world_model_latent_dim", 128) or 128)


def _load_hf_stack(args: Any) -> tuple[PolicyHiddenEncoder, TextLatentWorldModel, torch.device]:
    """Load the frozen encoder and trained LWM once per worker process."""

    if "hf_stack" in _state:
        return _state["hf_stack"]
    checkpoint_path = os.environ.get("LWM_TARGET_CHECKPOINT")
    if not checkpoint_path:
        raise RuntimeError(
            "LWM_TARGET_CHECKPOINT is required for the frozen latent-WM target "
            "provider in hf mode (a train_latent/stream_latent checkpoint with "
            "config + state_dict)"
        )
    device = _device()
    model_path = os.environ.get("LWM_TARGET_ENCODER_MODEL", _DEFAULT_ENCODER_MODEL)
    encoder = PolicyHiddenEncoder.from_pretrained(
        model_path,
        device=str(device),
        dtype="bfloat16" if device.type == "cuda" else "float32",
        local_files_only=True,
        backprop_to_llm=False,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = TextLatentWorldModelConfig(**checkpoint["config"])
    model = TextLatentWorldModel(config)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    _state["hf_stack"] = (encoder, model, device)
    return _state["hf_stack"]


def _sample_observation_text(sample: Any) -> str | None:
    metadata = getattr(sample, "metadata", None)
    if not isinstance(metadata, dict):
        return None
    record = metadata.get("world_model")
    if not isinstance(record, dict):
        return None
    text = record.get("next_observation_text") or record.get("feedback_text")
    text = str(text).strip() if text else ""
    return text or None


def _hash_targets(texts: list[str], latent_dim: int) -> list[list[float]]:
    """Deterministic pseudo-targets for plumbing tests (no semantic content)."""

    rows = hash_hidden_batch(
        [
            TerminalTransition(
                trajectory_id="provider-hash",
                task_name=None,
                data_source=None,
                turn_idx=index,
                context_messages=[],
                action_text="",
                feedback_text=text,
                next_context_messages=None,
                done=False,
                reward=None,
                status=None,
                source_path="",
            )
            for index, text in enumerate(texts)
        ],
        latent_dim,
    )
    return rows["target_hidden"].tolist()


@torch.no_grad()
def _hf_targets(
    texts: list[str],
    *,
    encoder: PolicyHiddenEncoder,
    model: TextLatentWorldModel,
    device: torch.device,
    encode_batch: int,
) -> list[list[float]]:
    latents: list[list[float]] = []
    for start in range(0, len(texts), encode_batch):
        chunk = texts[start : start + encode_batch]
        rows = [encoder._target_ids(text) for text in chunk]
        hidden, mask = encoder._forward_hidden(rows, require_grad=False)
        pooled = (hidden * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)
        latent = model.shared_projector(model.target_adapter(pooled))
        latents.extend(latent.float().cpu().tolist())
    return latents


def _load_raw_encoder() -> tuple[PolicyHiddenEncoder, torch.device]:
    """Load only the frozen encoder (raw mode needs no LWM checkpoint)."""

    if "raw_encoder" in _state:
        return _state["raw_encoder"]
    device = _device()
    model_path = os.environ.get("LWM_TARGET_ENCODER_MODEL", _DEFAULT_ENCODER_MODEL)
    encoder = PolicyHiddenEncoder.from_pretrained(
        model_path,
        device=str(device),
        dtype="bfloat16" if device.type == "cuda" else "float32",
        local_files_only=True,
        backprop_to_llm=False,
    )
    _state["raw_encoder"] = (encoder, device)
    return _state["raw_encoder"]


@torch.no_grad()
def _raw_targets(
    texts: list[str],
    *,
    encoder: PolicyHiddenEncoder,
    encode_batch: int,
) -> list[list[float]]:
    """Frozen-encoder pooled hidden, L2-normalized — the no-learned-head target."""

    latents: list[list[float]] = []
    for start in range(0, len(texts), encode_batch):
        rows = [encoder._target_ids(text) for text in texts[start : start + encode_batch]]
        hidden, mask = encoder._forward_hidden(rows, require_grad=False)
        pooled = (hidden * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)
        latents.extend(torch.nn.functional.normalize(pooled, dim=-1).float().cpu().tolist())
    return latents


def frozen_wm_target_provider(
    *,
    args: Any,
    samples: list[Any],
    turn_records: list[dict[str, Any]] | None = None,
    task_meta: dict[str, Any] | None = None,
    run_ctx: Any = None,
) -> None:
    """Attach frozen LWM target latents to each sample (mutation contract).

    Samples without a recorded observation keep ``target_mask=0`` and a None
    latent so the loss hook's per-sample mask can skip them.  Returns None
    (the provider mutates metadata directly to avoid copying tensors through
    the caller).
    """

    del turn_records, task_meta, run_ctx
    latent_dim = _latent_dim(args)
    texts = [_sample_observation_text(sample) for sample in samples]
    present = {index: text for index, text in enumerate(texts) if text is not None}

    latents: dict[int, list[float]] = {}
    if os.environ.get("LWM_ONLINE_JOINT") == "1":
        from .joint_online import joint_update_and_targets  # noqa: PLC0415

        # Joint mode ingests the whole batch (training happens even when some
        # samples lack an observation text) and returns EMA targets per index.
        latents = joint_update_and_targets(args=args, samples=samples)
    elif present:
        ordered = [present[index] for index in sorted(present)]
        mode = os.environ.get("LWM_TARGET_PROVIDER_ENCODER", "hf")
        if mode == "hash":
            values = _hash_targets(ordered, latent_dim)
        elif mode == "raw":
            encoder, _device_ = _load_raw_encoder()
            values = _raw_targets(
                ordered,
                encoder=encoder,
                encode_batch=int(os.environ.get("LWM_TARGET_ENCODE_BATCH", "4")),
            )
        else:
            encoder, model, device = _load_hf_stack(args)
            values = _hf_targets(
                ordered,
                encoder=encoder,
                model=model,
                device=device,
                encode_batch=int(os.environ.get("LWM_TARGET_ENCODE_BATCH", "4")),
            )
        for index, value in zip(sorted(present), values, strict=True):
            latents[index] = value

    for index, sample in enumerate(samples):
        metadata = sample.metadata if isinstance(getattr(sample, "metadata", None), dict) else {}
        sample.metadata = metadata
        record = metadata.setdefault("world_model", {})
        target = latents.get(index)
        record["target_latents"] = target
        record["target_mask"] = 1.0 if target is not None else 0.0
        train_metadata = dict(sample.train_metadata or {})
        train_metadata["world_model"] = record
        sample.train_metadata = train_metadata
    return None
