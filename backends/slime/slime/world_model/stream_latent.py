"""Streaming (online-style) A/B trainer for the latent world model.

The offline phase protocol in ``train_latent.py`` trains on the full pooled
train split, so enabling the replay buffer there only re-samples the same
distribution and cannot show a sample-efficiency gain.  This module instead
simulates rollout arrival on tb2.1 trajectories: the train split is ordered
into trajectory-contiguous chunks ("rollout steps") and every arm performs
the SAME number of optimizer updates per chunk with the SAME batch size;
only the batch composition differs.

- ``noreplay`` ("before"): batches contain only the current chunk's fresh
  transitions.
- ``replay`` ("after"): the chunk is first pushed into a FIFO
  ``TrajectoryReplayBuffer`` (admitting both successes and failures, since
  latent dynamics need the full outcome distribution), then each batch mixes
  ``replay_ratio`` samples from the buffer with fresh chunk transitions.

Held-out metrics on a fixed trajectory-grouped validation split are recorded
after every chunk, so training efficiency is read as held-out prediction
quality versus cumulative fresh transitions / gradient steps, plus
steps-to-threshold.  Both arms re-seed ``torch.manual_seed(args.seed)``
before model construction, so they start from identical weights; the
treatment is data composition only.

Design notes borrowed from ECHO/SPEAR (see docs/algorithms/):
- ECHO contributes the "environment feedback is free supervision" framing;
  ECHO itself is strictly on-policy and has no replay buffer.
- SPEAR contributes replay discipline: bounded FIFO, quality/staleness
  awareness, and a warm-up on the replay weight.  We keep admit-all (a world
  model needs failures too), implement capacity-based staleness, and offer
  ``--replay-warmup-chunks`` (linear ramp of the effective replay ratio).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Sequence

import torch

from .hidden_encoder import PolicyHiddenEncoder, hash_hidden_batch
from .modules import TextLatentWorldModel, TextLatentWorldModelConfig
from .replay_buffer import TrajectoryReplayBuffer
from .seta_dataset import (
    TerminalTransition,
    build_data_manifest,
    load_terminal_transitions,
)
from .train_latent import (
    _cache_hidden,
    _device,
    _discounted_returns,
    _select_hidden,
    _split_indices,
    _write_predictions,
)

REPLAY_ARMS = ("noreplay", "replay")


def plan_stream(
    transitions: Sequence[TerminalTransition],
    train_indices: Sequence[int],
    num_chunks: int,
    *,
    seed: int = 42,
    shuffle_trajectories: bool = True,
) -> list[list[int]]:
    """Cut the train split into trajectory-contiguous stream chunks.

    Trajectories stay intact (a chunk never splits one trajectory's turns),
    turns stay in ``turn_idx`` order, and chunk sizes are balanced by
    transition count with a greedy fill.  Trajectory order is shuffled with
    ``seed`` by default so the arrival order does not follow source-directory
    layout; both arms consume the exact same chunk plan.
    """

    if num_chunks < 1:
        raise ValueError(f"num_chunks must be >= 1, got {num_chunks}")
    groups: dict[str, list[tuple[int, int]]] = {}
    for index in train_indices:
        row = transitions[index]
        groups.setdefault(row.trajectory_id, []).append((row.turn_idx, index))
    ordered_groups = [
        [index for _, index in sorted(items, key=lambda item: item[0])]
        for items in groups.values()
    ]
    if shuffle_trajectories:
        random.Random(seed).shuffle(ordered_groups)
    else:
        first_index = {tuple(group): min(group) for group in ordered_groups}
        ordered_groups.sort(key=lambda group: first_index[tuple(group)])

    total = sum(len(group) for group in ordered_groups)
    if total == 0:
        return []
    num_chunks = min(num_chunks, len(ordered_groups))
    target = math.ceil(total / num_chunks)
    chunks: list[list[int]] = []
    current: list[int] = []
    for group in ordered_groups:
        if current and len(current) >= target and len(chunks) < num_chunks - 1:
            chunks.append(current)
            current = []
        current.extend(group)
    if current:
        chunks.append(current)
    return chunks


def compose_arm_batches(
    *,
    arm: str,
    chunk_indices: Sequence[int],
    buffer: TrajectoryReplayBuffer | None,
    id_to_index: dict[str, int],
    batch_size: int,
    replay_ratio: float,
    seed: int,
    current_step: int,
) -> list[list[int]]:
    """Compose one chunk's training batches for one arm.

    Both arms produce ``ceil(len(chunk)/batch_size)`` batches of
    ``batch_size`` indices (fresh indices are cycled when a chunk does not
    fill the trailing batch), so per-chunk gradient-step counts and sample
    counts match exactly; only the replay/fresh composition differs.
    """

    if arm not in REPLAY_ARMS:
        raise ValueError(f"unknown stream arm: {arm!r}")
    fresh = list(chunk_indices)
    if not fresh:
        return []
    random.Random(seed).shuffle(fresh)
    steps = max(1, math.ceil(len(fresh) / batch_size))
    batches: list[list[int]] = []
    cursor = 0
    for _ in range(steps):
        n_replay = 0
        if arm == "replay" and buffer is not None and len(buffer) > 0 and replay_ratio > 0:
            n_replay = min(batch_size - 1, max(1, int(round(batch_size * replay_ratio))))
        n_fresh = min(batch_size - n_replay, len(fresh))
        fresh_take = [fresh[(cursor + offset) % len(fresh)] for offset in range(n_fresh)]
        cursor += n_fresh
        replay_take: list[int] = []
        if n_replay > 0 and buffer is not None:
            rows = buffer.sample(n_replay, current_step=current_step)
            replay_take = [
                id_to_index[transition_id]
                for row in rows
                if (transition_id := str(row.get("transition_id"))) in id_to_index
            ]
        batch = fresh_take + replay_take
        if not batch:
            batch = [fresh[0]]
        batches.append(batch)
    return batches


def _run_batches(
    *,
    model: TextLatentWorldModel,
    transitions: Sequence[TerminalTransition],
    batches: Sequence[Sequence[int]],
    cached_hidden: dict[str, torch.Tensor] | None,
    policy_encoder: PolicyHiddenEncoder | None,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    sigreg_coef: float,
    action_contrast_coef: float,
    alignment_coef: float,
    value_coef: float,
    reward_targets: torch.Tensor | None,
    gradient_clip: float,
) -> tuple[float, dict[str, float]]:
    """One pass over pre-composed batches; mirrors ``train_latent._run_epoch``."""

    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    total_loss = 0.0
    total_count = 0
    grad_context = torch.enable_grad() if training else torch.no_grad()
    with grad_context:
        for batch_indices in batches:
            batch_indices = list(batch_indices)
            batch_transitions = [transitions[index] for index in batch_indices]
            if cached_hidden is not None:
                hidden = _select_hidden(cached_hidden, batch_indices, device)
            else:
                if policy_encoder is None:
                    raise RuntimeError("End-to-end streaming requires a policy hidden encoder")
                hidden = policy_encoder(batch_transitions)
            if reward_targets is None:
                rewards = torch.tensor(
                    [0.0 if row.reward is None else float(row.reward) for row in batch_transitions],
                    dtype=torch.float32,
                    device=device,
                )
            else:
                rewards = reward_targets.index_select(
                    0, torch.tensor(batch_indices, dtype=torch.long, device=reward_targets.device)
                ).to(device=device)
            reward_mask = torch.tensor(
                [row.reward is not None for row in batch_transitions],
                dtype=torch.bool,
                device=device,
            )
            loss, metrics = model.compute_loss(
                state_hidden=hidden["state_hidden"],
                action_hidden=hidden["action_hidden"],
                target_hidden=hidden["target_hidden"],
                next_state_hidden=hidden["next_state_hidden"],
                has_next=hidden["has_next"],
                reward=rewards,
                reward_mask=reward_mask,
                sigreg_coef=sigreg_coef,
                action_contrast_coef=action_contrast_coef,
                alignment_coef=alignment_coef,
                value_coef=value_coef,
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()
            count = len(batch_indices)
            total_loss += float(loss.detach().cpu()) * count
            total_count += count
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach().cpu()) * count
    averaged = {key: value / max(total_count, 1) for key, value in totals.items()}
    return total_loss / max(total_count, 1), averaged


def _full_pass_batches(indices: Sequence[int], batch_size: int) -> list[list[int]]:
    return [list(indices)[start : start + batch_size] for start in range(0, len(indices), batch_size)]


def run_stream(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the streaming A/B protocol and write artifacts to output_dir."""

    if args.encoder == "hf-policy" and not args.hf_model:
        raise ValueError("--hf-model is required when --encoder hf-policy")
    if args.encoder == "hash" and args.backprop_to_llm:
        raise ValueError("--backprop-to-llm requires --encoder hf-policy")
    if not 0.0 <= args.replay_ratio <= 1.0:
        raise ValueError("--replay-ratio must be between 0 and 1")
    if args.backprop_to_llm and len(args.stream_arms) > 1:
        # A backprop arm owns and updates its private backbone copy; two such
        # copies in one process double 8B memory and silently diverge feature
        # spaces.  Run backprop arms as separate single-arm jobs instead.
        raise ValueError("--backprop-to-llm streaming runs must use a single --stream-arms entry")

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    transitions = load_terminal_transitions(
        args.input,
        max_trajectories=args.max_trajectories,
        max_transitions=args.max_transitions,
        require_tool_feedback=args.require_tool_feedback,
        data_source=args.data_source,
        supplement_inputs=args.supplement_input,
        min_turns=args.min_turns,
        include_terminal=not args.exclude_terminal,
    )
    if not transitions:
        raise ValueError(f"No valid terminal transitions found in {args.input}")

    data_manifest = build_data_manifest(transitions, requested_source=args.data_source)
    (output_dir / "data_manifest.json").write_text(
        json.dumps(data_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    device = _device(args.device)
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

    cached_hidden = None
    if not args.backprop_to_llm:
        # Frozen policy: encode once and share the cache across arms.  Besides
        # halving the dominant cost, this guarantees both arms see bit-identical
        # features, so the comparison isolates batch composition.
        cached_hidden = _cache_hidden(
            transitions,
            encoder_kind=args.encoder,
            hash_hidden_dim=args.hash_hidden_dim,
            policy_encoder=policy_encoder,
            batch_size=args.encode_batch_size,
        )
        torch.save(
            {
                **cached_hidden,
                "record_metadata": [row.to_dict() for row in transitions],
                "encoder": args.encoder,
                "hf_model": args.hf_model,
            },
            output_dir / "hidden_cache.pt",
        )
        if policy_encoder is not None:
            del policy_encoder
            policy_encoder = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    train_indices, val_indices = _split_indices(
        len(transitions),
        args.val_ratio,
        args.seed,
        [row.trajectory_id for row in transitions],
    )
    reward_targets = _discounted_returns(transitions, args.gamma)
    id_to_index = {row.transition_id: index for index, row in enumerate(transitions)}
    chunks = plan_stream(
        transitions,
        train_indices,
        args.stream_chunks,
        seed=args.seed,
        shuffle_trajectories=not args.stream_keep_source_order,
    )
    if not chunks:
        raise ValueError("Stream planning produced no chunks; check --val-ratio/--stream-chunks")

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

    metrics_path = output_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()
    arm_results: dict[str, Any] = {}
    for arm in args.stream_arms:
        if arm not in REPLAY_ARMS:
            raise ValueError(f"unknown stream arm: {arm!r}")
        # Identical init across arms: re-seed before every model construction.
        torch.manual_seed(args.seed)
        model = TextLatentWorldModel(config).to(device)
        parameter_groups: list[dict[str, Any]] = [{"params": model.parameters(), "lr": args.lr}]
        if args.backprop_to_llm:
            assert policy_encoder is not None
            parameter_groups.append({"params": policy_encoder.model.parameters(), "lr": args.llm_lr})
        optimizer = torch.optim.AdamW(parameter_groups, weight_decay=args.weight_decay)
        buffer: TrajectoryReplayBuffer | None = None
        if arm == "replay":
            buffer = TrajectoryReplayBuffer(args.replay_buffer_size, seed=args.seed)

        history: list[dict[str, Any]] = []
        cumulative_fresh = 0
        cumulative_steps = 0
        start_time = time.perf_counter()
        for chunk_idx, chunk in enumerate(chunks):
            if buffer is not None:
                buffer.push([transitions[index] for index in chunk], current_step=chunk_idx)
            ratio = args.replay_ratio
            if args.replay_warmup_chunks > 0:
                ratio = ratio * min(1.0, (chunk_idx + 1) / args.replay_warmup_chunks)
            batches: list[list[int]] = []
            for epoch in range(args.stream_epochs_per_chunk):
                batches.extend(
                    compose_arm_batches(
                        arm=arm,
                        chunk_indices=chunk,
                        buffer=buffer,
                        id_to_index=id_to_index,
                        batch_size=args.batch_size,
                        replay_ratio=ratio,
                        seed=args.seed + chunk_idx * 1000 + epoch,
                        current_step=chunk_idx,
                    )
                )
            train_loss, train_metrics = _run_batches(
                model=model,
                transitions=transitions,
                batches=batches,
                cached_hidden=cached_hidden,
                policy_encoder=policy_encoder,
                optimizer=optimizer,
                device=device,
                sigreg_coef=args.sigreg_coef,
                action_contrast_coef=args.action_contrast_coef,
                alignment_coef=args.alignment_coef,
                value_coef=args.value_coef,
                reward_targets=reward_targets,
                gradient_clip=args.gradient_clip,
            )
            val_loss = None
            val_metrics: dict[str, float] = {}
            if val_indices:
                val_loss, val_metrics = _run_batches(
                    model=model,
                    transitions=transitions,
                    batches=_full_pass_batches(val_indices, args.batch_size),
                    cached_hidden=cached_hidden,
                    policy_encoder=policy_encoder,
                    optimizer=None,
                    device=device,
                    sigreg_coef=args.sigreg_coef,
                    action_contrast_coef=args.action_contrast_coef,
                    alignment_coef=args.alignment_coef,
                    value_coef=args.value_coef,
                    reward_targets=reward_targets,
                    gradient_clip=0.0,
                )
            cumulative_fresh += len(chunk)
            cumulative_steps += len(batches)
            row = {
                "arm": arm,
                "chunk": chunk_idx + 1,
                "num_chunks": len(chunks),
                "chunk_transitions": len(chunk),
                "cumulative_fresh": cumulative_fresh,
                "cumulative_steps": cumulative_steps,
                "replay_ratio_effective": ratio if arm == "replay" else 0.0,
                "seconds_since_arm_start": round(time.perf_counter() - start_time, 3),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "replay_stats": buffer.stats() if buffer is not None else None,
            }
            history.append(row)
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            print(json.dumps(row, sort_keys=True))

        checkpoint = {
            "schema_version": "openclaw_terminal_latent_wm_stream_v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config": config.__dict__,
            "state_dict": model.state_dict(),
            "runtime": vars(args),
            "arm": arm,
            "record_count": len(transitions),
            "train_count": len(train_indices),
            "val_count": len(val_indices),
            "history": history,
            "replay_stats": buffer.stats() if buffer is not None else None,
            "data_manifest": data_manifest,
        }
        torch.save(checkpoint, output_dir / f"latent_world_model_{arm}.pt")
        if buffer is not None:
            buffer.save(output_dir / f"replay_buffer_{arm}.pt")
        _write_predictions(
            path=output_dir / f"predictions_{arm}.jsonl",
            model=model,
            transitions=transitions,
            cached_hidden=cached_hidden,
            policy_encoder=policy_encoder,
            batch_size=args.batch_size,
            device=device,
        )
        arm_results[arm] = {
            "final": history[-1],
            "replay_stats": buffer.stats() if buffer is not None else None,
            "checkpoint": f"latent_world_model_{arm}.pt",
        }
        del model, optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = {
        "schema_version": "lwm_stream_summary_v1",
        "encoder": args.encoder,
        "backprop_to_llm": args.backprop_to_llm,
        "record_count": len(transitions),
        "train_count": len(train_indices),
        "val_count": len(val_indices),
        "num_chunks": len(chunks),
        "chunk_sizes": [len(chunk) for chunk in chunks],
        "stream_arms": list(args.stream_arms),
        "replay_ratio": args.replay_ratio,
        "replay_warmup_chunks": args.replay_warmup_chunks,
        "replay_buffer_size": args.replay_buffer_size,
        "gamma": args.gamma,
        "seed": args.seed,
        "data_manifest": data_manifest,
        "arms": arm_results,
    }
    (output_dir / "stream_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"saved streaming world-model outputs to {output_dir}")
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Streaming A/B training for the latent world model on tb2.1/SETA transitions."
    )
    parser.add_argument("--input", required=True, help="tb2.1/SETA directory, records JSONL, or replay .pt.")
    parser.add_argument(
        "--supplement-input",
        action="append",
        default=[],
        help="Optional lower-priority data root/file; repeat to add multiple supplements.",
    )
    parser.add_argument(
        "--data-source",
        choices=["auto", "tb21", "seta", "records", "replay", "mixed"],
        default="auto",
        help="Input parser and priority. auto/mixed prefer tb2.1 trajectory.json.",
    )
    parser.add_argument("--min-turns", type=int, default=1)
    parser.add_argument("--exclude-terminal", action="store_true")
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
    parser.add_argument(
        "--backprop-to-llm",
        "--world-model-backprop-to-llm",
        dest="backprop_to_llm",
        action="store_true",
        default=False,
        help="Allow latent losses to update the policy LLM backbone. Default: false.",
    )
    parser.add_argument("--save-updated-llm", action="store_true")
    parser.add_argument("--max-trajectories", type=int, default=None)
    parser.add_argument("--max-transitions", type=int, default=None)
    parser.add_argument("--require-tool-feedback", action="store_true")
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--adapter-dim", type=int, default=None)
    parser.add_argument("--predictor-type", choices=["adaln", "mlp"], default="adaln")
    parser.add_argument("--predictor-depth", type=int, default=2)
    parser.add_argument("--predictor-num-heads", type=int, default=4)
    parser.add_argument("--predictor-mlp-ratio", type=float, default=4.0)
    parser.add_argument("--stop-grad-target", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--encode-batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--llm-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--sigreg-coef", type=float, default=0.09)
    parser.add_argument("--action-contrast-coef", type=float, default=0.1)
    parser.add_argument("--alignment-coef", type=float, default=0.1)
    parser.add_argument("--value-coef", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount for offline return targets.")
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--stream-chunks",
        type=int,
        default=8,
        help="Number of trajectory-contiguous stream chunks (simulated rollout steps).",
    )
    parser.add_argument(
        "--stream-arms",
        type=lambda value: [item.strip() for item in str(value).split(",") if item.strip()],
        default=["noreplay", "replay"],
        help="Comma-separated arms to run in one process (default: noreplay,replay).",
    )
    parser.add_argument(
        "--stream-epochs-per-chunk",
        type=int,
        default=1,
        help="Passes over each chunk per arm; multiplies gradient steps equally for both arms.",
    )
    parser.add_argument(
        "--stream-keep-source-order",
        action="store_true",
        help="Order the stream by loader discovery order instead of seed-shuffled trajectories.",
    )
    parser.add_argument("--replay-buffer-size", type=int, default=4096)
    parser.add_argument(
        "--replay-ratio",
        type=float,
        default=0.5,
        help="Per-batch fraction sampled from the replay buffer in the replay arm.",
    )
    parser.add_argument(
        "--replay-warmup-chunks",
        type=int,
        default=0,
        help="Linear ramp of the effective replay ratio 0->target over the first N chunks (SPEAR-style warm-up).",
    )
    return parser


def main() -> None:
    """Parse CLI args, then train each stream arm and write the run summary.

    Flag validation happens here so config errors surface before any encoding
    work; ``run_stream`` then loads transitions, builds both arms from
    identical seeds, trains them chunk by chunk, and writes ``metrics.jsonl``,
    per-arm checkpoints, and ``stream_summary.json``.
    """

    args = _build_parser().parse_args()
    if args.latent_dim % args.predictor_num_heads != 0 and args.predictor_type == "adaln":
        raise ValueError("--latent-dim must be divisible by --predictor-num-heads")
    if args.stream_epochs_per_chunk < 1:
        raise ValueError("--stream-epochs-per-chunk must be >= 1")
    if args.replay_warmup_chunks < 0:
        raise ValueError("--replay-warmup-chunks must be non-negative")
    run_stream(args)


if __name__ == "__main__":
    main()
