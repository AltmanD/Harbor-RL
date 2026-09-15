from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _nested(payload: dict[str, Any], path: list[str], default: Any = None) -> Any:
    cur: Any = payload
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _load_tensor_stats(cache_path: Path) -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on runtime env.
        return {"available": False, "reason": f"torch_import_failed:{exc}"}

    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    stats: dict[str, Any] = {"available": True}
    for key in ["state_hidden", "action_hidden", "target_hidden"]:
        tensor = payload[key].float()
        row: dict[str, Any] = {
            "shape": [int(dim) for dim in tensor.shape],
            "finite": bool(torch.isfinite(tensor).all()),
            "std": float(tensor.std(unbiased=False)),
        }
        if tensor.shape[0] > 1:
            row["pairwise_l2_mean"] = float(torch.pdist(tensor).mean())
            row["feature_var_mean"] = float(tensor.var(dim=0, unbiased=False).mean())
        else:
            row["pairwise_l2_mean"] = 0.0
            row["feature_var_mean"] = 0.0
        stats[key] = row
    return stats


def _record_threshold(bucket: str, args: argparse.Namespace) -> int:
    if bucket == "full":
        return int(args.min_full_records)
    if bucket == "clean":
        return int(args.min_clean_records)
    if bucket == "tool_only":
        return int(args.min_tool_records)
    return int(args.min_records)


def _summarize_bucket(root: Path, bucket: str, args: argparse.Namespace) -> dict[str, Any]:
    bucket_dir = root / bucket
    paths = {
        "records": bucket_dir / "records.jsonl",
        "records_summary": bucket_dir / "records_summary.json",
        "cache": bucket_dir / "cached_hidden.pt",
        "checkpoint": bucket_dir / "probe.pt",
        "eval_summary": bucket_dir / "eval_summary.json",
        "rankings": bucket_dir / "rankings.jsonl",
        "config": bucket_dir / "stage_a_config.json",
        "log": bucket_dir / "logs" / "stage_a.log",
    }
    missing = [name for name, path in paths.items() if not path.exists() or path.stat().st_size == 0]

    records_summary = _read_json(paths["records_summary"]) if paths["records_summary"].exists() else {}
    eval_summary = _read_json(paths["eval_summary"]) if paths["eval_summary"].exists() else {}
    config = _read_json(paths["config"]) if paths["config"].exists() else {}
    metrics = eval_summary.get("metrics") or {}
    latents = metrics.get("latents") or {}

    tensor_stats = _load_tensor_stats(paths["cache"]) if paths["cache"].exists() else {"available": False, "reason": "missing_cache"}
    rankings_count = None
    if paths["rankings"].exists():
        rankings_count = sum(1 for line in paths["rankings"].open("r", encoding="utf-8") if line.strip())

    record_count = int(records_summary.get("record_count") or eval_summary.get("record_count") or 0)
    min_records = _record_threshold(bucket, args)
    context_unique = int(records_summary.get("context_text_unique_count") or 0)
    context_truncated_ratio = _float(records_summary.get("context_truncated_ratio"))
    state_hidden_pairwise = _float(_nested(tensor_stats, ["state_hidden", "pairwise_l2_mean"]))
    state_hidden_finite = bool(_nested(tensor_stats, ["state_hidden", "finite"], False))
    state_latent_rank = _float(_nested(latents, ["state", "effective_rank"]))
    state_latent_var = _float(_nested(latents, ["state", "variance_mean"]))
    shuffle_gap = _float(metrics.get("shuffle_gap_mse_shuffled_minus_real"))
    shuffle_ratio = _float(metrics.get("shuffle_gap_ratio_mse"))
    shuffle_positive_fraction = _float(metrics.get("shuffle_gap_positive_fraction"))
    zero_gap = _float(metrics.get("zero_action_gap_mse_zero_minus_real"))
    action_delta = _float(metrics.get("action_delta"))
    value_coef = _float((config or {}).get("value_coef"))
    value_spearman = _float(_nested(metrics, ["value_reward", "spearman"]))
    uncertainty_spearman = _float(_nested(metrics, ["uncertainty_error", "spearman_uncertainty_vs_pred_mse"]))

    checks = {
        "artifacts_ok": not missing,
        "record_count_ok": record_count >= min_records,
        "context_unique_ok": context_unique >= min(int(args.min_context_unique), max(2, record_count)),
        "context_not_fully_truncated": context_truncated_ratio is None or context_truncated_ratio < float(args.max_context_truncated_ratio),
        "state_hidden_ok": state_hidden_finite and state_hidden_pairwise is not None and state_hidden_pairwise > float(args.min_state_hidden_pairwise_l2),
        "state_latent_rank_ok": state_latent_rank is not None and state_latent_rank >= float(args.min_state_latent_rank),
        "state_latent_var_ok": state_latent_var is not None and state_latent_var > float(args.min_state_latent_var),
        "shuffle_gap_ok": shuffle_gap is not None and shuffle_gap > float(args.min_shuffle_gap),
        "zero_action_gap_ok": zero_gap is not None and zero_gap > float(args.min_zero_action_gap),
        "action_delta_ok": action_delta is not None and action_delta > float(args.min_action_delta),
        "rankings_ok": rankings_count == record_count if rankings_count is not None else False,
    }
    failed = [name for name, ok in checks.items() if not ok]

    return {
        "bucket": bucket,
        "out_dir": str(bucket_dir),
        "missing_artifacts": missing,
        "record_count": record_count,
        "min_records": min_records,
        "rankings": rankings_count,
        "context_text_unique_count": context_unique,
        "context_truncated_ratio": context_truncated_ratio,
        "state_hidden_pairwise_l2_mean": state_hidden_pairwise,
        "state_hidden_feature_var_mean": _float(_nested(tensor_stats, ["state_hidden", "feature_var_mean"])),
        "state_latent_effective_rank": state_latent_rank,
        "state_latent_variance_mean": state_latent_var,
        "pred_mse_real": _float(metrics.get("pred_mse_real")),
        "shuffle_gap_mse_shuffled_minus_real": shuffle_gap,
        "shuffle_gap_ratio_mse": shuffle_ratio,
        "shuffle_gap_positive_fraction": shuffle_positive_fraction,
        "zero_action_gap_mse_zero_minus_real": zero_gap,
        "action_delta": action_delta,
        "value_coef": value_coef,
        "value_reward_spearman": value_spearman,
        "uncertainty_pred_mse_spearman": uncertainty_spearman,
        "checks": checks,
        "passed": not failed,
        "failed_checks": failed,
    }


def summarize_stage_a(root: Path, args: argparse.Namespace) -> dict[str, Any]:
    buckets = [item.strip() for item in args.buckets.split(",") if item.strip()]
    rows = [_summarize_bucket(root, bucket, args) for bucket in buckets]
    failed = [row for row in rows if not row["passed"]]
    return {
        "schema_version": "openclaw_text_jepa_stage_a_gate_summary_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": str(root),
        "passed": not failed,
        "failed_buckets": [row["bucket"] for row in failed],
        "buckets": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize and gate OpenClaw text JEPA Stage-A evaluation outputs.")
    parser.add_argument("--input", required=True, help="Stage-A output directory.")
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    parser.add_argument("--buckets", default="full,clean,tool_only")
    parser.add_argument("--min-records", type=int, default=128)
    parser.add_argument("--min-full-records", type=int, default=128)
    parser.add_argument("--min-clean-records", type=int, default=128)
    parser.add_argument("--min-tool-records", type=int, default=128)
    parser.add_argument("--min-context-unique", type=int, default=8)
    parser.add_argument("--max-context-truncated-ratio", type=float, default=0.5)
    parser.add_argument("--min-state-hidden-pairwise-l2", type=float, default=1e-4)
    parser.add_argument("--min-state-latent-rank", type=float, default=3.0)
    parser.add_argument("--min-state-latent-var", type=float, default=1e-9)
    parser.add_argument("--min-shuffle-gap", type=float, default=0.0)
    parser.add_argument("--min-zero-action-gap", type=float, default=0.0)
    parser.add_argument("--min-action-delta", type=float, default=0.0)
    parser.add_argument("--no-fail", action="store_true", help="Always exit 0 after writing the report.")
    args = parser.parse_args()

    summary = summarize_stage_a(Path(args.input), args)
    text = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text, end="")

    if not summary["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
