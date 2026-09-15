from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import torch

from slime.utils.types import Sample


def _iter_sample_dicts(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix in {".pt", ".pth"}:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        samples = payload.get("samples", payload) if isinstance(payload, dict) else payload
        for sample in samples:
            yield sample.to_dict() if hasattr(sample, "to_dict") else dict(sample)
        return

    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _parse_multi_values(values: list[str] | None) -> set[str]:
    parsed: set[str] = set()
    for value in values or []:
        for part in str(value).split(","):
            part = part.strip().lower()
            if part:
                parsed.add(part)
    return parsed


def _truncate_text(value: Any, max_chars: int, *, strategy: str = "head") -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    if strategy == "tail":
        return text[-max_chars:]
    if strategy == "head_tail":
        marker = "\n[openclaw_truncated_middle]\n"
        if max_chars <= len(marker) + 2:
            return text[:max_chars]
        keep = max_chars - len(marker)
        head_chars = max(1, keep // 4)
        tail_chars = keep - head_chars
        return text[:head_chars] + marker + text[-tail_chars:]
    return text[:max_chars]


def _sample_prompt(item: dict[str, Any]) -> Any:
    prompt = item.get("prompt")
    if prompt is None:
        metadata = item.get("metadata")
        if isinstance(metadata, dict):
            prompt = metadata.get("instruction")
    return prompt


def _looks_prefix_truncated(record: dict[str, Any], *, context_max_chars: int) -> bool:
    text = str(record.get("context_text") or "")
    if not text:
        return True
    if context_max_chars > 0 and len(text) >= context_max_chars:
        return True
    source = str(record.get("context_text_source") or "")
    return source == "sample.prompt" and "[openclaw_truncated_middle]" not in text and len(text) >= context_max_chars


def _enrich_legacy_record(
    record: dict[str, Any],
    item: dict[str, Any],
    *,
    context_max_chars: int,
    context_source: str,
    context_truncation: str,
) -> dict[str, Any]:
    prompt = _sample_prompt(item)
    should_replace = False
    if context_source == "world_model":
        should_replace = not bool(record.get("context_text"))
    elif context_source == "sample_prompt":
        should_replace = prompt is not None
    elif context_source == "sample_prompt_if_missing":
        should_replace = not bool(record.get("context_text")) and prompt is not None
    elif context_source == "auto":
        should_replace = prompt is not None and _looks_prefix_truncated(record, context_max_chars=context_max_chars)
    else:
        raise ValueError(f"unknown context_source: {context_source}")

    if not should_replace:
        return record
    if prompt is None:
        return record
    enriched = dict(record)
    enriched["context_text"] = _truncate_text(prompt, context_max_chars, strategy=context_truncation)
    enriched["context_text_source"] = "sample.prompt"
    enriched["context_text_truncation"] = context_truncation
    return enriched


def _eval_reason(record: dict[str, Any]) -> str | None:
    for key in ("eval_reason", "reason"):
        value = record.get(key)
        if value:
            return str(value)
    text = record.get("next_observation_text")
    if not text:
        return None
    try:
        payload = json.loads(str(text))
    except Exception:
        return None
    if isinstance(payload, dict):
        value = payload.get("eval_reason") or payload.get("reason") or payload.get("message")
        return None if value is None else str(value)
    return None


def _observation_source(record: dict[str, Any]) -> str:
    explicit = record.get("observation_source")
    if explicit:
        return str(explicit)
    if bool(record.get("has_tool_result")):
        return "tool_result"
    text = record.get("next_observation_text")
    if text:
        try:
            payload = json.loads(str(text))
        except Exception:
            return "unknown"
        if isinstance(payload, dict) and (
            "status" in payload or "score" in payload or "raw_score" in payload or "eval_reason" in payload
        ):
            return "eval_summary"
    return "unknown"


def _record_matches(
    record: dict[str, Any],
    *,
    statuses: set[str] | None,
    exclude_eval_reasons: set[str] | None,
    require_tool_result: bool,
) -> bool:
    if statuses:
        status = str(record.get("status", "")).lower()
        if status not in statuses:
            return False
    if exclude_eval_reasons:
        reason = _eval_reason(record)
        if reason is not None and reason.lower() in exclude_eval_reasons:
            return False
    if require_tool_result and not bool(record.get("has_tool_result")):
        return False
    return True


def extract_world_model_records(
    path: Path,
    *,
    context_max_chars: int = 4096,
    context_source: str = "world_model",
    context_truncation: str = "head_tail",
    statuses: set[str] | None = None,
    exclude_eval_reasons: set[str] | None = None,
    require_tool_result: bool = False,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in _iter_sample_dicts(path):
        if "status" in item:
            try:
                sample = Sample.from_dict(item)
            except Exception:
                sample = item
        else:
            sample = item
        metadata = sample.metadata if isinstance(sample, Sample) else item.get("metadata", {})
        train_metadata = sample.train_metadata if isinstance(sample, Sample) else item.get("train_metadata", None)
        wm = None
        if isinstance(train_metadata, dict):
            wm = train_metadata.get("world_model")
        if wm is None and isinstance(metadata, dict):
            wm = metadata.get("world_model")
        if isinstance(wm, dict):
            record = _enrich_legacy_record(
                wm,
                item,
                context_max_chars=context_max_chars,
                context_source=context_source,
                context_truncation=context_truncation,
            )
            if _record_matches(
                record,
                statuses=statuses,
                exclude_eval_reasons=exclude_eval_reasons,
                require_tool_result=require_tool_result,
            ):
                records.append(record)
    return records


def _numeric_stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
    }


def _length_stats(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "min": None, "max": None, "p95": None}
    ordered = sorted(values)
    p95_idx = min(len(ordered) - 1, int(round((len(ordered) - 1) * 0.95)))
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
        "p95": ordered[p95_idx],
    }


def summarize_world_model_records(
    records: list[dict[str, Any]],
    *,
    context_max_chars: int = 4096,
    filter_args: dict[str, Any] | None = None,
    input_record_count: int | None = None,
) -> dict[str, Any]:
    def hist(values: Iterable[Any]) -> dict[str, int]:
        return {str(key): int(value) for key, value in Counter(values).items()}

    uid_values = [record.get("uid") for record in records if record.get("uid") is not None]
    task_values = [record.get("task_name") or record.get("task_path") for record in records if record.get("task_name") or record.get("task_path")]
    sample_values = [record.get("sample_index") for record in records if record.get("sample_index") is not None]
    rollout_values = [record.get("rollout_id") for record in records if record.get("rollout_id") is not None]
    context_hashes = [record.get("context_hash") for record in records if record.get("context_hash")]
    context_text_hashes = [
        hashlib.blake2b(str(record.get("context_text") or "").encode("utf-8"), digest_size=8).hexdigest()
        for record in records
        if record.get("context_text") is not None
    ]
    next_hashes = [record.get("next_observation_hash") for record in records if record.get("next_observation_hash")]
    reward_scores = [float(record["reward_score"]) for record in records if record.get("reward_score") is not None]
    reward_raw = [float(record["reward_raw_score"]) for record in records if record.get("reward_raw_score") is not None]
    context_lens = [len(str(record.get("context_text") or "")) for record in records]
    action_lens = [len(str(record.get("action_text") or "")) for record in records]
    target_lens = [len(str(record.get("next_observation_text") or "")) for record in records]

    return {
        "record_count": len(records),
        "input_record_count": input_record_count,
        "dropped_record_count": None if input_record_count is None else max(0, input_record_count - len(records)),
        "uid_count": len(set(uid_values)),
        "task_count": len(set(task_values)),
        "sample_count": len(set(sample_values)),
        "rollout_count": len(set(rollout_values)),
        "filter_args": filter_args or {},
        "status_hist": hist(record.get("status") for record in records),
        "done_hist": hist(record.get("done") for record in records),
        "turn_idx_hist": hist(record.get("turn_idx") for record in records),
        "num_turns_hist": hist(record.get("num_turns") for record in records),
        "eval_reason_hist": hist(_eval_reason(record) for record in records),
        "observation_source_hist": hist(_observation_source(record) for record in records),
        "has_tool_result_hist": hist(bool(record.get("has_tool_result")) for record in records),
        "records_per_uid": hist(uid_values),
        "records_per_task": hist(task_values),
        "records_per_context_hash": hist(context_hashes),
        "duplicate_context_hash_count": sum(count > 1 for count in Counter(context_hashes).values()),
        "context_text_unique_count": len(set(context_text_hashes)),
        "duplicate_context_text_count": sum(count > 1 for count in Counter(context_text_hashes).values()),
        "duplicate_next_observation_hash_count": sum(count > 1 for count in Counter(next_hashes).values()),
        "reward_score": _numeric_stats(reward_scores),
        "reward_raw_score": _numeric_stats(reward_raw),
        "zero_reward_raw_ratio": (
            sum(value == 0.0 for value in reward_raw) / len(reward_raw) if reward_raw else None
        ),
        "context_length": _length_stats(context_lens),
        "action_length": _length_stats(action_lens),
        "target_length": _length_stats(target_lens),
        "context_truncated_ratio": (
            sum(length >= context_max_chars for length in context_lens) / len(context_lens) if context_lens else None
        ),
        "action_truncated_ratio": (
            sum(length >= context_max_chars for length in action_lens) / len(action_lens) if action_lens else None
        ),
        "target_truncated_ratio": (
            sum(length >= context_max_chars for length in target_lens) / len(target_lens) if target_lens else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract OpenClaw world-model metadata records from rollout samples.")
    parser.add_argument("--input", required=True, help="Debug rollout .pt/.pth or JSONL sample file.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--context-max-chars", type=int, default=4096, help="Max prompt chars to backfill for legacy records.")
    parser.add_argument(
        "--context-source",
        choices=["world_model", "sample_prompt", "sample_prompt_if_missing", "auto"],
        default="world_model",
        help=(
            "Which state text to write into context_text. "
            "Use sample_prompt to repair old records whose world_model context was prefix-truncated."
        ),
    )
    parser.add_argument(
        "--context-truncation",
        choices=["head", "tail", "head_tail"],
        default="head_tail",
        help="Truncation strategy when backfilling or repairing context_text.",
    )
    parser.add_argument("--status", action="append", default=[], help="Keep status values; comma-separated and repeatable.")
    parser.add_argument("--exclude-eval-reason", action="append", default=[], help="Drop eval reasons; comma-separated and repeatable.")
    parser.add_argument("--require-tool-result", action="store_true", help="Keep only records with tool observations.")
    parser.add_argument("--summary-output", default=None, help="Optional JSON summary path for the kept records.")
    args = parser.parse_args()

    statuses = _parse_multi_values(args.status)
    exclude_eval_reasons = _parse_multi_values(args.exclude_eval_reason)
    all_records = extract_world_model_records(
        Path(args.input),
        context_max_chars=args.context_max_chars,
        context_source=args.context_source,
        context_truncation=args.context_truncation,
    )
    records = [
        record
        for record in all_records
        if _record_matches(
            record,
            statuses=statuses or None,
            exclude_eval_reasons=exclude_eval_reasons or None,
            require_tool_result=args.require_tool_result,
        )
    ]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    if args.summary_output:
        summary = summarize_world_model_records(
            records,
            context_max_chars=args.context_max_chars,
            input_record_count=len(all_records),
            filter_args={
                "status": sorted(statuses),
                "exclude_eval_reason": sorted(exclude_eval_reasons),
                "require_tool_result": bool(args.require_tool_result),
            },
        )
        summary_path = Path(args.summary_output)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {len(records)} world-model records to {out}")


if __name__ == "__main__":
    main()
