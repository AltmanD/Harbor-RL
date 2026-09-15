from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

import torch

from .evaluate_probe import _spearman
from .modules import TextLatentWorldModel, TextLatentWorldModelConfig


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        value = value.detach().float().cpu().item()
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _json_sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_sanitize(item) for item in value]
    if isinstance(value, tuple):
        return [_json_sanitize(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _load_checkpoint(path: Path, device: torch.device) -> tuple[TextLatentWorldModel, dict[str, Any]]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = TextLatentWorldModelConfig(**ckpt["config"])
    model = TextLatentWorldModel(config).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt.get("metadata", {})


def _load_cache(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict cache payload in {path}, got {type(payload).__name__}")
    for key in ["state_hidden", "action_hidden", "target_hidden"]:
        if key not in payload:
            raise KeyError(f"Missing {key} in {path}")
    count = int(payload["state_hidden"].shape[0])
    for key in ["action_hidden", "target_hidden"]:
        if int(payload[key].shape[0]) != count:
            raise ValueError(f"Inconsistent {key} length: expected {count}, got {int(payload[key].shape[0])}")
    return payload


def _rankdata(values: list[float]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32)


def _group_records(
    records: list[dict[str, Any]],
    *,
    group_key: str,
    min_candidates: int,
    max_candidates: int,
    require_reward_variation: bool,
) -> list[list[int]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(records):
        key = row.get(group_key)
        if key is not None:
            grouped[str(key)].append(idx)

    groups: list[list[int]] = []
    for indices in grouped.values():
        valid_indices = [idx for idx in indices if _float(records[idx].get("reward_score")) is not None]
        if len(valid_indices) < min_candidates:
            continue
        if max_candidates > 0:
            valid_indices = valid_indices[:max_candidates]
        if len(valid_indices) < min_candidates:
            continue
        if require_reward_variation:
            rewards = {
                _float(records[idx].get("reward_score"))
                for idx in valid_indices
                if _float(records[idx].get("reward_score")) is not None
            }
            if len(rewards) < 2:
                continue
        groups.append(valid_indices)
    return groups


def evaluate_candidate_sets(
    *,
    checkpoint: Path,
    cache: Path,
    records_path: Path,
    output: Path,
    groups_output: Path | None = None,
    group_key: str = "context_hash",
    min_candidates: int = 2,
    max_candidates: int = 8,
    require_reward_variation: bool = True,
    device_name: str = "auto",
    uncertainty_coef: float = 0.0,
) -> dict[str, Any]:
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    records = _read_jsonl(records_path)
    payload = _load_cache(cache)
    count = int(payload["state_hidden"].shape[0])
    if len(records) != count:
        raise ValueError(f"records/cache length mismatch: records={len(records)} cache={count}")

    groups = _group_records(
        records,
        group_key=group_key,
        min_candidates=min_candidates,
        max_candidates=max_candidates,
        require_reward_variation=require_reward_variation,
    )

    model, checkpoint_metadata = _load_checkpoint(checkpoint, device)
    tensors = {
        "state_hidden": payload["state_hidden"].float().to(device),
        "action_hidden": payload["action_hidden"].float().to(device),
        "target_hidden": payload["target_hidden"].float().to(device),
    }
    with torch.no_grad():
        out = model(**tensors)
        if out["value"] is None:
            raise ValueError("candidate-set eval requires a checkpoint with value_head enabled")
        value = out["value"].detach().float().cpu()
        uncertainty = out["uncertainty"].detach().float().cpu() if out["uncertainty"] is not None else torch.zeros_like(value)
        score = value - float(uncertainty_coef) * uncertainty

    group_rows: list[dict[str, Any]] = []
    top1_rewards: list[float] = []
    random_rewards: list[float] = []
    oracle_rewards: list[float] = []
    regrets: list[float] = []
    hit_oracle: list[float] = []
    spearman_values: list[float] = []

    for group_id, indices in enumerate(groups):
        rewards = torch.tensor([float(records[idx].get("reward_score")) for idx in indices], dtype=torch.float32)
        scores = score[indices]
        values = value[indices]
        uncertainties = uncertainty[indices]
        order = torch.argsort(scores, descending=True)
        best_idx = int(order[0].item())
        oracle_idx = int(torch.argmax(rewards).item())
        top_reward = float(rewards[best_idx].item())
        oracle_reward = float(rewards[oracle_idx].item())
        random_reward = float(rewards.mean().item())
        regret = oracle_reward - top_reward
        corr, corr_reason = _spearman(scores, rewards)
        if corr is not None:
            spearman_values.append(float(corr))
        top1_rewards.append(top_reward)
        random_rewards.append(random_reward)
        oracle_rewards.append(oracle_reward)
        regrets.append(regret)
        hit_oracle.append(1.0 if top_reward == oracle_reward else 0.0)
        first_record = records[indices[0]]
        group_rows.append(
            {
                "group_id": group_id,
                "group_key": group_key,
                "group_value": first_record.get(group_key),
                "candidate_count": len(indices),
                "task_name": first_record.get("task_name"),
                "task_path": first_record.get("task_path"),
                "selected_local_index": best_idx,
                "selected_record_index": indices[best_idx],
                "selected_reward": top_reward,
                "oracle_local_index": oracle_idx,
                "oracle_record_index": indices[oracle_idx],
                "oracle_reward": oracle_reward,
                "random_expected_reward": random_reward,
                "oracle_regret": regret,
                "hit_oracle": top_reward == oracle_reward,
                "spearman_score_reward": corr,
                "spearman_reason": corr_reason,
                "candidates": [
                    {
                        "record_index": int(idx),
                        "rank": int((order == local_idx).nonzero(as_tuple=False)[0].item()),
                        "score": float(scores[local_idx].item()),
                        "value": float(values[local_idx].item()),
                        "uncertainty": float(uncertainties[local_idx].item()),
                        "reward": float(rewards[local_idx].item()),
                        "uid": records[idx].get("uid"),
                        "sample_index": records[idx].get("sample_index"),
                        "turn_idx": records[idx].get("turn_idx"),
                        "done": records[idx].get("done"),
                        "has_tool_result": records[idx].get("has_tool_result"),
                        "action_hash": records[idx].get("action_hash"),
                    }
                    for local_idx, idx in enumerate(indices)
                ],
            }
        )

    def mean(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    candidate_count_hist = Counter(len(group) for group in groups)
    summary = {
        "schema_version": "openclaw_text_jepa_u2_candidate_set_eval_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint),
        "cache": str(cache),
        "records": str(records_path),
        "groups_output": str(groups_output) if groups_output is not None else None,
        "device": str(device),
        "group_key": group_key,
        "min_candidates": int(min_candidates),
        "max_candidates": int(max_candidates),
        "require_reward_variation": bool(require_reward_variation),
        "uncertainty_coef": float(uncertainty_coef),
        "record_count": len(records),
        "candidate_group_count": len(groups),
        "candidate_record_count": sum(len(group) for group in groups),
        "candidate_count_hist": {str(key): int(value) for key, value in candidate_count_hist.items()},
        "candidate_group_rows": len(group_rows),
        "metrics": {
            "wm_top1_reward_mean": mean(top1_rewards),
            "random_expected_reward_mean": mean(random_rewards),
            "oracle_reward_mean": mean(oracle_rewards),
            "wm_minus_random_reward": (
                mean(top1_rewards) - mean(random_rewards)
                if top1_rewards and random_rewards
                else None
            ),
            "oracle_regret_mean": mean(regrets),
            "hit_oracle_rate": mean(hit_oracle),
            "group_spearman_mean": mean(spearman_values),
            "group_spearman_count": len(spearman_values),
        },
        "checkpoint_metadata": checkpoint_metadata,
        "cache_metadata": payload.get("metadata", {}),
        "notes": [
            "This is an offline candidate-set evaluation over already executed candidates.",
            "Ranking scores use only state/action features via the value head; target_hidden is used only as observed-label context in the trained checkpoint/cache, not as the selection score.",
            "For production U2, candidate actions must be generated before execution and evaluated against real execution labels.",
        ],
    }
    summary = _json_sanitize(summary)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    if groups_output is not None:
        groups_output.parent.mkdir(parents=True, exist_ok=True)
        with groups_output.open("w", encoding="utf-8") as fh:
            for row in group_rows:
                fh.write(json.dumps(_json_sanitize(row), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate U2-style candidate sets for a text latent world model.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--records", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--groups-output", default=None)
    parser.add_argument("--group-key", default="context_hash")
    parser.add_argument("--min-candidates", type=int, default=2)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--allow-constant-reward-groups", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--uncertainty-coef", type=float, default=0.0)
    args = parser.parse_args()

    summary = evaluate_candidate_sets(
        checkpoint=Path(args.checkpoint),
        cache=Path(args.cache),
        records_path=Path(args.records),
        output=Path(args.output),
        groups_output=Path(args.groups_output) if args.groups_output else None,
        group_key=args.group_key,
        min_candidates=args.min_candidates,
        max_candidates=args.max_candidates,
        require_reward_variation=not args.allow_constant_reward_groups,
        device_name=args.device,
        uncertainty_coef=args.uncertainty_coef,
    )
    metrics = summary["metrics"]
    print(
        "wrote candidate-set eval to "
        f"{args.output} "
        f"groups_output={args.groups_output} "
        f"(groups={summary['candidate_group_count']} "
        f"wm_top1={metrics['wm_top1_reward_mean']} "
        f"random={metrics['random_expected_reward_mean']} "
        f"oracle={metrics['oracle_reward_mean']})"
    )


if __name__ == "__main__":
    main()
