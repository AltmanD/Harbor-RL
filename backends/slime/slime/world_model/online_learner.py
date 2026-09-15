"""Online latent world-model learner fed by rollout-side replay snapshots.

During online DAPO with ``--world-model-use-dapo-replay-buffer`` the
RolloutManager persists ``world_model_replay_<rollout_id>.pt`` snapshots under
``<save>/rollout/`` (see ``slime/slime/ray/rollout.py``).  This learner polls
that directory and trains the LWM incrementally as new transitions arrive —
the online counterpart of ``stream_latent.py``'s chunked A/B, and the piece
that lets the replay buffer actually feed LWM gradients during RL training.

Per newly observed snapshot:

1. newly admitted transitions (by ``transition_id``) are encoded with the
   frozen policy encoder (or re-encoded per batch under ``--backprop-to-llm``);
2. the LWM takes ``ceil(new/batch_size)`` update steps whose batches mix the
   fresh transitions with ``replay_ratio`` samples from the learner's own
   FIFO buffer (same equal-compute-per-fresh-sample rule as the offline
   streaming protocol);
3. held-out metrics are evaluated on a fixed probe set (``--val-input``) and
   appended to ``metrics.jsonl``.

The probe set comes from completed evaluation runs, not from the training
stream, so it also measures how the representation holds up under the policy
drift of the ongoing RL run.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

import torch

from .hidden_encoder import PolicyHiddenEncoder, hash_hidden_batch
from .modules import TextLatentWorldModel, TextLatentWorldModelConfig
from .replay_buffer import TrajectoryReplayBuffer
from .seta_dataset import TerminalTransition, load_terminal_transitions, transition_from_world_model_record
from .stream_latent import _full_pass_batches, _run_batches, compose_arm_batches
from .train_latent import _device

SNAPSHOT_GLOB = "world_model_replay*.pt"
DONE_MARKER = "world_model_replay.DONE"


def _load_snapshot_transitions(path: Path) -> list[TerminalTransition]:
    buffer = TrajectoryReplayBuffer.load(path)
    transitions: list[TerminalTransition] = []
    for record in buffer.records():
        try:
            transition = transition_from_world_model_record(record, source_path=str(path))
        except (TypeError, ValueError, KeyError):
            continue
        if transition.action_text and transition.feedback_text:
            transitions.append(transition)
    return transitions


def run_online(args: argparse.Namespace) -> dict[str, Any]:
    """Run the online LWM loop over rollout-side replay snapshots.

    Polls ``args.snapshot_dir`` for ``world_model_replay*.pt`` snapshots
    written by the RolloutManager.  Newly admitted transitions (deduplicated
    by ``transition_id``) are encoded incrementally — once, into a growing
    local cache, when the policy is frozen — then mixed with samples from the
    learner's own FIFO buffer for ``ceil(new/batch_size)`` updates per
    snapshot, matching the equal-compute rule of the offline streaming
    protocol.  After each snapshot the model is evaluated on a fixed held-out
    probe set (``--val-input``) and a row is appended to ``metrics.jsonl``.
    The loop exits on the ``world_model_replay.DONE`` marker, ``--once``,
    ``--max-snapshots``, idle timeout, or wall-clock timeout, then writes
    ``latent_world_model_online.pt``, ``replay_buffer_online.pt``, and
    ``online_summary.json``.

    Args:
        args: Parsed CLI namespace from ``_build_parser``.

    Returns:
        The summary dict also persisted as ``online_summary.json``.

    Raises:
        ValueError: On incompatible encoder/backprop flags, or when the
            held-out probe input yields no transitions.
    """

    if args.encoder == "hf-policy" and not args.hf_model:
        raise ValueError("--hf-model is required when --encoder hf-policy")
    if args.encoder == "hash" and args.backprop_to_llm:
        raise ValueError("--backprop-to-llm requires --encoder hf-policy")

    snapshot_dir = Path(args.snapshot_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _device(args.device)

    # Fixed held-out probe set (never part of the training stream).
    val_transitions = load_terminal_transitions(
        args.val_input,
        max_trajectories=args.val_max_trajectories,
        max_transitions=args.val_max_transitions,
        data_source=args.val_data_source,
    )
    if not val_transitions:
        raise ValueError(f"No held-out transitions found in {args.val_input}")

    policy_encoder: PolicyHiddenEncoder | None = None
    if args.encoder == "hf-policy":
        policy_encoder = PolicyHiddenEncoder.from_pretrained(
            args.hf_model,
            device=str(device),
            dtype=args.hf_dtype,
            local_files_only=args.hf_local_files_only,
            hidden_layer=args.hidden_layer,
            action_pool=args.action_pool,
            max_context_tokens=args.max_context_tokens,
            max_action_tokens=args.max_action_tokens,
            max_feedback_tokens=args.max_feedback_tokens,
            backprop_to_llm=args.backprop_to_llm,
        )
        hidden_dim = policy_encoder.hidden_size
    else:
        hidden_dim = args.hash_hidden_dim

    def encode(rows: list[TerminalTransition]) -> dict[str, torch.Tensor]:
        """Encode a batch of transitions into one concatenated hidden dict.

        The hash path delegates to ``hash_hidden_batch``; the HF path runs the
        policy encoder in ``--encode-batch-size`` chunks and concatenates the
        detached CPU tensors, so the result can be cached or saved directly.

        Args:
            rows: Transitions to encode.

        Returns:
            Hidden dict with the same keys as ``PolicyHiddenEncoder.forward``.
        """

        if args.encoder == "hash":
            return hash_hidden_batch(rows, args.hash_hidden_dim)
        assert policy_encoder is not None
        batched: dict[str, list[torch.Tensor]] = {}
        for start in range(0, len(rows), args.encode_batch_size):
            encoded = policy_encoder(rows[start : start + args.encode_batch_size])
            for key, value in encoded.items():
                batched.setdefault(key, []).append(value.detach().cpu())
        return {key: torch.cat(values, dim=0) for key, values in batched.items()}

    val_hidden = encode(val_transitions)
    torch.save(
        {
            **val_hidden,
            "record_metadata": [row.to_dict() for row in val_transitions],
            "encoder": args.encoder,
            "hf_model": args.hf_model,
        },
        output_dir / "val_hidden_cache.pt",
    )

    config = TextLatentWorldModelConfig(
        state_hidden_dim=hidden_dim,
        action_hidden_dim=hidden_dim,
        target_hidden_dim=hidden_dim,
        latent_dim=args.latent_dim,
        adapter_dim=args.adapter_dim,
        predictor_type=args.predictor_type,
        predictor_depth=args.predictor_depth,
        predictor_num_heads=args.predictor_num_heads,
        predictor_mlp_ratio=args.predictor_mlp_ratio,
        value_head=args.value_coef != 0.0,
        uncertainty_head=False,
        stop_grad_target=args.stop_grad_target,
    )
    torch.manual_seed(args.seed)
    model = TextLatentWorldModel(config).to(device)
    parameter_groups: list[dict[str, Any]] = [{"params": model.parameters(), "lr": args.lr}]
    if args.backprop_to_llm:
        assert policy_encoder is not None
        parameter_groups.append({"params": policy_encoder.model.parameters(), "lr": args.llm_lr})
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=args.weight_decay)
    buffer = TrajectoryReplayBuffer(args.replay_buffer_size, seed=args.seed)

    metrics_path = output_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    seen_ids: set[str] = set()
    # The learner owns a growing local store of (transition, hidden) pairs so
    # replayed samples never need re-encoding when the policy is frozen.
    local_transitions: list[TerminalTransition] = []
    local_hidden: dict[str, list[torch.Tensor]] = {}

    processed_snapshots: set[str] = set()
    start_time = time.perf_counter()
    last_progress = start_time
    total_steps = 0

    while True:
        snapshot_paths = sorted(snapshot_dir.glob(SNAPSHOT_GLOB))
        new_rows = 0
        for path in snapshot_paths:
            if path.name in processed_snapshots:
                continue
            try:
                transitions = _load_snapshot_transitions(path)
            except Exception as exc:  # noqa: BLE001 - snapshot may be mid-write
                print(f"[lwm-online] skip unreadable snapshot {path.name}: {exc}")
                continue
            processed_snapshots.add(path.name)
            fresh_indices: list[int] = []
            for transition in transitions:
                transition_id = transition.transition_id
                if transition_id in seen_ids:
                    continue
                seen_ids.add(transition_id)
                local_transitions.append(transition)
                fresh_indices.append(len(local_transitions) - 1)
            if not fresh_indices:
                continue
            if args.backprop_to_llm:
                cached = None
            else:
                encoded = encode([local_transitions[index] for index in fresh_indices])
                for key, value in encoded.items():
                    local_hidden.setdefault(key, []).append(value)
                cached = {key: torch.cat(values, dim=0) for key, values in local_hidden.items()}
            buffer.push([local_transitions[index] for index in fresh_indices], current_step=len(processed_snapshots))
            id_to_local = {row.transition_id: index for index, row in enumerate(local_transitions)}
            batches = compose_arm_batches(
                arm="replay",
                chunk_indices=fresh_indices,
                buffer=buffer,
                id_to_index=id_to_local,
                batch_size=args.batch_size,
                replay_ratio=args.replay_ratio,
                seed=args.seed + len(processed_snapshots),
                current_step=len(processed_snapshots),
            )
            train_loss, train_metrics = _run_batches(
                model=model,
                transitions=local_transitions,
                batches=batches,
                cached_hidden=cached,
                policy_encoder=policy_encoder if args.backprop_to_llm else None,
                optimizer=optimizer,
                device=device,
                sigreg_coef=args.sigreg_coef,
                action_contrast_coef=args.action_contrast_coef,
                alignment_coef=args.alignment_coef,
                value_coef=args.value_coef,
                reward_targets=None,
                gradient_clip=args.gradient_clip,
            )
            val_loss, val_metrics = _run_batches(
                model=model,
                transitions=val_transitions,
                batches=_full_pass_batches(list(range(len(val_transitions))), args.batch_size),
                cached_hidden=val_hidden,
                policy_encoder=None,
                optimizer=None,
                device=device,
                sigreg_coef=args.sigreg_coef,
                action_contrast_coef=args.action_contrast_coef,
                alignment_coef=args.alignment_coef,
                value_coef=args.value_coef,
                reward_targets=None,
                gradient_clip=0.0,
            )
            new_rows += len(fresh_indices)
            total_steps += len(batches)
            last_progress = time.perf_counter()
            row = {
                "snapshot": path.name,
                "snapshots_seen": len(processed_snapshots),
                "new_transitions": len(fresh_indices),
                "total_transitions": len(local_transitions),
                "cumulative_steps": total_steps,
                "seconds_since_start": round(last_progress - start_time, 3),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "replay_stats": buffer.stats(),
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            print(json.dumps({k: v for k, v in row.items() if not k.endswith("metrics")}, sort_keys=True))

        if (snapshot_dir / DONE_MARKER).is_file():
            print("[lwm-online] DONE marker observed; finishing")
            break
        if args.once:
            break
        if args.max_snapshots and len(processed_snapshots) >= args.max_snapshots:
            break
        idle_for = time.perf_counter() - last_progress
        if args.idle_timeout_sec and idle_for > args.idle_timeout_sec and processed_snapshots:
            print(f"[lwm-online] idle for {idle_for:.0f}s; finishing")
            break
        if args.timeout_sec and time.perf_counter() - start_time > args.timeout_sec:
            print("[lwm-online] wall-clock timeout; finishing")
            break
        time.sleep(args.poll_interval_sec)

    checkpoint = {
        "schema_version": "openclaw_terminal_latent_wm_online_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config.__dict__,
        "state_dict": model.state_dict(),
        "runtime": vars(args),
        "transition_count": len(local_transitions),
        "snapshots_seen": sorted(processed_snapshots),
        "replay_stats": buffer.stats(),
    }
    torch.save(checkpoint, output_dir / "latent_world_model_online.pt")
    buffer.save(output_dir / "replay_buffer_online.pt")
    summary = {
        "schema_version": "lwm_online_summary_v1",
        "encoder": args.encoder,
        "backprop_to_llm": args.backprop_to_llm,
        "snapshot_dir": str(snapshot_dir),
        "snapshots_seen": len(processed_snapshots),
        "transition_count": len(local_transitions),
        "cumulative_steps": total_steps,
        "val_transition_count": len(val_transitions),
        "replay_stats": buffer.stats(),
        "wall_seconds": round(time.perf_counter() - start_time, 3),
    }
    (output_dir / "online_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"saved online world-model outputs to {output_dir}")
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", required=True, help="Directory with world_model_replay_*.pt snapshots.")
    parser.add_argument("--val-input", required=True, help="Held-out tb2.1/SETA trajectory source.")
    parser.add_argument("--val-data-source", default="auto")
    parser.add_argument("--val-max-trajectories", type=int, default=32)
    parser.add_argument("--val-max-transitions", type=int, default=512)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--encoder", choices=["hash", "hf-policy"], default="hash")
    parser.add_argument("--hash-hidden-dim", type=int, default=256)
    parser.add_argument("--hf-model", default=None)
    parser.add_argument("--hf-local-files-only", action="store_true")
    parser.add_argument("--hf-dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hidden-layer", type=int, default=-1)
    parser.add_argument("--action-pool", choices=["mean", "last"], default="mean")
    parser.add_argument("--max-context-tokens", type=int, default=1536)
    parser.add_argument("--max-action-tokens", type=int, default=512)
    parser.add_argument("--max-feedback-tokens", type=int, default=512)
    parser.add_argument("--encode-batch-size", type=int, default=8)
    parser.add_argument(
        "--backprop-to-llm",
        "--world-model-backprop-to-llm",
        dest="backprop_to_llm",
        action="store_true",
        default=False,
    )
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--adapter-dim", type=int, default=None)
    parser.add_argument("--predictor-type", choices=["adaln", "mlp"], default="adaln")
    parser.add_argument("--predictor-depth", type=int, default=2)
    parser.add_argument("--predictor-num-heads", type=int, default=4)
    parser.add_argument("--predictor-mlp-ratio", type=float, default=4.0)
    parser.add_argument("--stop-grad-target", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--llm-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--sigreg-coef", type=float, default=0.09)
    parser.add_argument("--action-contrast-coef", type=float, default=0.1)
    parser.add_argument("--alignment-coef", type=float, default=0.1)
    parser.add_argument("--value-coef", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--replay-buffer-size", type=int, default=4096)
    parser.add_argument("--replay-ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--poll-interval-sec", type=float, default=60.0)
    parser.add_argument("--idle-timeout-sec", type=float, default=3600.0)
    parser.add_argument("--timeout-sec", type=float, default=0.0)
    parser.add_argument("--max-snapshots", type=int, default=0)
    parser.add_argument("--once", action="store_true", help="Process current snapshots once and exit.")
    return parser


def main() -> None:
    """CLI entry: parse args, validate flag combinations, run the polling loop."""

    args = _build_parser().parse_args()
    if args.latent_dim % args.predictor_num_heads != 0 and args.predictor_type == "adaln":
        raise ValueError("--latent-dim must be divisible by --predictor-num-heads")
    run_online(args)


if __name__ == "__main__":
    main()
