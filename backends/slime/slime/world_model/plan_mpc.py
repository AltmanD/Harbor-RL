"""CLI for verified one-step latent MPC over cached action candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .mpc import candidate_tensor, plan_one_step
from .modules import TextLatentWorldModel, TextLatentWorldModelConfig


def main() -> None:
    """Run one-step latent MPC for one state against cached candidate actions.

    Loads a ``TextLatentWorldModel`` checkpoint and a ``hidden_cache.pt``
    payload, scores the chosen state's candidate actions by predicted value
    (optionally uncertainty-penalized), and writes the ranked plan as JSON to
    ``--output``.  The command is fail-closed: a checkpoint trained without a
    value head raises ``ValueError`` before any output is written, and
    out-of-range ``--state-index``/``--candidate-indices`` raise ``IndexError``.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="hidden_cache.pt containing state/action hidden tensors")
    parser.add_argument("--state-index", type=int, default=0)
    parser.add_argument("--candidate-indices", default=None, help="Comma-separated action rows; default: all rows")
    parser.add_argument("--uncertainty-coef", type=float, default=0.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = TextLatentWorldModelConfig(**checkpoint["config"])
    model = TextLatentWorldModel(config)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    payload = torch.load(args.input, map_location="cpu", weights_only=False)
    state_hidden = payload["state_hidden"].float()
    if not 0 <= args.state_index < state_hidden.size(0):
        raise IndexError(f"--state-index {args.state_index} is outside [0, {state_hidden.size(0)})")
    indices = None
    if args.candidate_indices:
        indices = [int(item.strip()) for item in args.candidate_indices.split(",") if item.strip()]
        if not indices or any(index < 0 or index >= payload["action_hidden"].size(0) for index in indices):
            raise IndexError("--candidate-indices contains an out-of-range or empty index")
    plan = plan_one_step(
        model,
        state_hidden=state_hidden[args.state_index],
        candidate_action_hidden=candidate_tensor(payload, indices),
        uncertainty_coef=args.uncertainty_coef,
    )
    selected_global = plan.selected_index if indices is None else indices[plan.selected_index]
    rows = []
    for local_index, score in enumerate(plan.scores.tolist()):
        global_index = local_index if indices is None else indices[local_index]
        row = {
            "rank": local_index,
            "candidate_index": int(global_index),
            "score": float(score),
            "value": float(plan.values[local_index].item()),
            "uncertainty": None
            if plan.uncertainties is None
            else float(plan.uncertainties[local_index].item()),
        }
        rows.append(row)
    rows.sort(key=lambda row: row["score"], reverse=True)
    for rank, row in enumerate(rows):
        row["rank"] = rank
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "schema_version": "lwm_latent_mpc_v1",
                "state_index": args.state_index,
                "selected_candidate_index": int(selected_global),
                "uncertainty_coef": args.uncertainty_coef,
                "candidates": rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"selected candidate {selected_global}; wrote {len(rows)} scores to {output}")


if __name__ == "__main__":
    main()
