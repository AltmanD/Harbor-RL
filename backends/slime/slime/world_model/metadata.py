from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from slime.utils.misc import load_function

logger = logging.getLogger(__name__)


def is_world_model_enabled(args: Any) -> bool:
    raw = getattr(args, "world_model_enable", False)
    if isinstance(raw, str):
        raw = raw.strip().lower() in {"1", "true", "yes", "on"}
    return bool(raw)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return str(value)


def stable_hash(value: Any, *, digest_size: int = 16) -> str:
    payload = json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=digest_size).hexdigest()


def _truncate(text: Any, max_chars: int, *, strategy: str = "head") -> str:
    text = "" if text is None else str(text)
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


def _tool_action_text(call: dict[str, Any], max_chars: int) -> str:
    name = str(call.get("tool_name") or call.get("name") or "tool")
    args = call.get("args")
    if args is None:
        args = call.get("arguments")
    args_text = json.dumps(_jsonable(args), ensure_ascii=False, sort_keys=True, default=str)
    return _truncate(f"{name}({args_text})", max_chars, strategy="head_tail")


def _render_tool_result(value: Any) -> str:
    """Render one raw tool result without terminal/prompt decoration."""
    if isinstance(value, dict):
        fields = [
            f"{key}={value[key]}"
            for key in ("stdout", "stderr", "exit_code")
            if value.get(key) is not None
        ]
        if fields:
            return "\n".join(fields)
        return json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True, default=str)
    return str(value)


def _extract_action_text(turn: dict[str, Any], max_chars: int) -> str:
    parts: list[str] = []
    assistant = str(turn.get("assistant_output") or "").strip()
    if assistant:
        parts.append(assistant)
    for call in turn.get("tool_calls") or []:
        if isinstance(call, dict):
            parts.append(_tool_action_text(call, max_chars))
    return _truncate("\n".join(parts), max_chars, strategy="head_tail")


def _extract_context_text(turn: dict[str, Any], *, task_meta: dict[str, Any], max_chars: int) -> str:
    context_messages = turn.get("context_messages") or []
    payload = {
        "task_name": task_meta.get("task_name"),
        "task_path": task_meta.get("task_path"),
        "data_source": task_meta.get("data_source"),
        "turn_idx": turn.get("turn_idx"),
        "context_messages": context_messages,
    }
    return _truncate(
        json.dumps(_jsonable(payload), ensure_ascii=False, sort_keys=True, default=str),
        max_chars,
        strategy="head_tail",
    )


def _extract_observation_text(
    turn: dict[str, Any],
    *,
    status: Any,
    reward: dict[str, Any],
    eval_details: dict[str, Any] | None,
    eval_error: str | None,
    max_chars: int,
) -> str:
    observations: list[str] = []
    for call in turn.get("tool_calls") or []:
        if isinstance(call, dict) and call.get("result") is not None:
            observations.append(_render_tool_result(call.get("result")))
    if not observations:
        reason = None
        if isinstance(eval_details, dict):
            reason = eval_details.get("reason") or eval_details.get("message")
        observations.append(
            json.dumps(
                {
                    "status": getattr(status, "value", str(status)),
                    "score": reward.get("score"),
                    "raw_score": reward.get("raw_score"),
                    "eval_reason": reason,
                    "eval_error": eval_error,
                },
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
        )
    return _truncate("\n".join(observations), max_chars, strategy="head_tail")


def build_terminal_world_model_record(
    *,
    sample: Any,
    turn: dict[str, Any],
    task_meta: dict[str, Any],
    run_ctx: Any,
    status: Any,
    eval_details: dict[str, Any] | None,
    eval_error: str | None,
    max_chars: int,
    next_turn: dict[str, Any] | None = None,
) -> dict[str, Any]:
    reward = sample.reward if isinstance(getattr(sample, "reward", None), dict) else {}
    context_text = _extract_context_text(turn, task_meta=task_meta, max_chars=max_chars)
    action_text = _extract_action_text(turn, max_chars)
    next_observation_text = _extract_observation_text(
        turn,
        status=status,
        reward=reward,
        eval_details=eval_details,
        eval_error=eval_error,
        max_chars=max_chars,
    )
    context_messages = turn.get("context_messages") or []
    response_length = int(getattr(sample, "response_length", 0) or 0)
    token_count = len(getattr(sample, "tokens", []) or [])
    sample_metadata = getattr(sample, "metadata", {}) if isinstance(getattr(sample, "metadata", {}), dict) else {}
    turn_idx = int(turn.get("turn_idx", sample_metadata.get("turn_idx", 0) or 0))
    num_turns = sample_metadata.get("num_turns")
    status_value = getattr(status, "value", str(status))
    abnormal_done = status_value in {"truncated", "aborted", "failed"}
    final_turn = num_turns is not None and turn_idx >= int(num_turns) - 1
    next_context_text = None
    if next_turn is not None:
        next_context_text = _extract_context_text(next_turn, task_meta=task_meta, max_chars=max_chars)
    return {
        "schema": "openclaw_terminal_latent_world_model_v2",
        "hidden_source": "policy_prompt_action_spans",
        "task_name": task_meta.get("task_name"),
        "task_path": task_meta.get("task_path"),
        "data_source": task_meta.get("data_source"),
        "uid": getattr(run_ctx, "uid", None),
        "group_index": getattr(run_ctx, "group_index", None),
        "sample_index": getattr(run_ctx, "sample_index", None),
        "rollout_id": getattr(run_ctx, "rollout_id", None),
        "train_step": getattr(run_ctx, "train_step", None),
        "turn_idx": turn_idx,
        "num_turns": num_turns,
        "status": status_value,
        "done": bool(final_turn or (num_turns is None and abnormal_done)),
        "context_hash": stable_hash(context_messages),
        "action_hash": stable_hash(action_text),
        "next_observation_hash": stable_hash(next_observation_text),
        "context_token_len": max(0, token_count - response_length),
        "action_token_len": response_length,
        "context_text": context_text,
        "context_text_source": "context_messages.head_tail",
        "context_text_truncation": "head_tail",
        "next_context_text": next_context_text,
        "next_context_hash": stable_hash(next_turn.get("context_messages") or []) if next_turn else None,
        "action_text": action_text,
        "next_observation_text": next_observation_text,
        "reward_score": reward.get("score"),
        "reward_base_score": reward.get("base_score"),
        "reward_raw_score": reward.get("raw_score"),
        "has_tool_result": any(
            isinstance(call, dict) and bool(call.get("result"))
            for call in (turn.get("tool_calls") or [])
        ),
    }


def attach_terminal_world_model_metadata(
    *,
    args: Any,
    samples: list[Any],
    turn_records: list[dict[str, Any]],
    task_meta: dict[str, Any],
    run_ctx: Any,
    status: Any,
    eval_details: dict[str, Any] | None = None,
    eval_error: str | None = None,
) -> None:
    if not is_world_model_enabled(args) or not samples:
        return
    max_chars = int(getattr(args, "world_model_metadata_max_chars", 4096) or 4096)
    turn_by_idx = {int(turn.get("turn_idx", i)): turn for i, turn in enumerate(turn_records or [])}
    for sample in samples:
        metadata = sample.metadata if isinstance(getattr(sample, "metadata", None), dict) else {}
        sample.metadata = metadata
        turn_idx = int(metadata.get("turn_idx", 0) or 0)
        turn = turn_by_idx.get(turn_idx)
        if turn is None:
            continue
        record = build_terminal_world_model_record(
            sample=sample,
            turn=turn,
            task_meta=task_meta,
            run_ctx=run_ctx,
            status=status,
            eval_details=eval_details,
            eval_error=eval_error,
            max_chars=max_chars,
            next_turn=turn_by_idx.get(turn_idx + 1),
        )
        sample.metadata["world_model"] = record
        train_metadata = dict(sample.train_metadata or {})
        train_metadata["world_model"] = record
        sample.train_metadata = train_metadata

    provider_path = getattr(args, "world_model_target_provider_path", None)
    if not provider_path:
        return
    provider = load_function(provider_path)
    provided = provider(
        args=args,
        samples=samples,
        turn_records=turn_records,
        task_meta=task_meta,
        run_ctx=run_ctx,
    )
    if provided is None:
        # A provider may mutate sample.metadata directly.  This is useful when
        # it keeps a resident frozen encoder and avoids copying large tensors.
        return
    if not isinstance(provided, (list, tuple)) or len(provided) != len(samples):
        raise ValueError(
            "world-model target provider must return one latent payload per sample "
            "or mutate sample.metadata and return None"
        )
    for sample, target in zip(samples, provided, strict=True):
        world_model = sample.metadata.setdefault("world_model", {})
        if target is not None:
            world_model["target_latents"] = target
        train_metadata = dict(sample.train_metadata or {})
        train_metadata["world_model"] = world_model
        sample.train_metadata = train_metadata
    logger.info("Attached frozen latent WM targets with provider=%s", provider_path)
