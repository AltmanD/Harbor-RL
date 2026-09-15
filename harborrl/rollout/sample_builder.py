from __future__ import annotations

import json
import os
from copy import deepcopy
from typing import Any, Dict, List

from slime.utils.types import Sample

from harborrl.types import Interaction, TaskSpec
from harborrl.algorithms.dive_po.rewards.shared import _sync_reward_aliases
from harborrl.environments.registry import (
    safety_split_applies as _safety_split_applies,
)
from harborrl.rollout.trajectory_store import _optional_int


def _extract_task_meta(sample: Sample) -> Dict[str, Any]:
    if isinstance(sample.prompt, dict):
        return sample.prompt

    metadata = sample.metadata or {}
    task_meta = metadata.get("task_meta") if isinstance(metadata, dict) else None
    if isinstance(task_meta, dict):
        return task_meta

    if isinstance(metadata, dict):
        return metadata

    return {}


def _make_task_spec(meta: Dict[str, Any]) -> TaskSpec:
    return TaskSpec(
        task_name=meta.get("task_name", "unknown"),
        task_path=meta.get("task_path", ""),
        instruction=meta.get("instruction", ""),
    )


def _last_eval_details(env_client: Any) -> dict[str, Any] | None:
    details = getattr(env_client, "last_evaluate_details", None)
    if isinstance(details, dict):
        return deepcopy(details)
    nested = getattr(env_client, "_env", None)
    details = getattr(nested, "_last_eval", None)
    if isinstance(details, dict):
        return deepcopy(details)
    return None


def _safety_split_from_meta(task_meta: dict[str, Any]) -> str:
    data_source = str(task_meta.get("data_source") or "")
    if not _safety_split_applies(data_source):
        return "agentic"
    raw = task_meta.get("fulfillable")
    try:
        fulfillable = int(raw)
    except (TypeError, ValueError):
        if str(task_meta.get("agentharm_task_type") or "").lower() == "benign":
            fulfillable = 1
        else:
            fulfillable = 0
    return "benign_should_comply" if fulfillable == 1 else "harmful_should_refuse"


from harborrl.env import env_bool as _env_flag


def _dapo_overlong_cfg(args) -> dict[str, Any] | None:
    if os.getenv("ALGO", "grpo").strip().lower() != "dapo":
        return None
    if not _env_flag("DAPO_OVERLONG_BUFFER_ENABLE", default=True):
        return None

    max_resp_len = _optional_int(os.getenv("DAPO_MAX_RESPONSE_LEN"))
    if max_resp_len is None:
        max_resp_len = _optional_int(getattr(args, "rollout_max_response_len", None))
    buffer_len = _optional_int(os.getenv("DAPO_OVERLONG_BUFFER_LEN", "4096"))
    try:
        penalty_factor = float(os.getenv("DAPO_OVERLONG_PENALTY_FACTOR", "1.0"))
    except ValueError:
        penalty_factor = 1.0

    if max_resp_len is None or max_resp_len <= 0 or buffer_len is None or buffer_len <= 0:
        return None
    buffer_len = min(buffer_len, max_resp_len)
    return {
        "max_resp_len": int(max_resp_len),
        "buffer_len": int(buffer_len),
        "penalty_factor": float(penalty_factor),
        "expected_len": int(max_resp_len - buffer_len),
    }


def _dapo_overlong_reward(response_length: int, cfg: dict[str, Any] | None) -> float:
    if not cfg:
        return 0.0
    exceed_len = int(response_length) - int(cfg["expected_len"])
    return min(-exceed_len / float(cfg["buffer_len"]) * float(cfg["penalty_factor"]), 0.0)


def _build_samples(
    interactions: List[Interaction],
    base_sample: Sample,
    outcome: float,
    status: Sample.Status,
    prm_turn_scores: dict[int, float] | None = None,
    prm_coef: float = 1.0,
    discount: float = 1.0,
    encourage: bool = False,
    outcome_is_score: bool = False,
    penalize_short_response: bool = True,
    dapo_overlong_cfg: dict[str, Any] | None = None,
) -> List[Sample]:
    """Create one Sample per interaction with discounted reward."""
    num_turns = len(interactions)
    samples: List[Sample] = []

    accuracy = float(outcome)
    raw_score = accuracy + (accuracy == 1.0) * int(encourage)
    if outcome_is_score:
        base_outcome = accuracy
        raw_score = accuracy
    else:
        base_outcome = 2.0 * accuracy - 1.0

    for interaction in interactions:
        turn_idx = interaction.turn_idx
        s = deepcopy(base_sample)
        s.tokens = interaction.input_ids + interaction.output_token_ids
        s.response_length = len(interaction.output_token_ids)
        s.loss_mask = [1] * s.response_length
        s.rollout_log_probs = list(interaction.output_token_logprobs)
        s.response = interaction.output_text
        s.status = status

        s.metadata.update(
            {
                "turn_idx": turn_idx,
                "num_turns": num_turns,
                "finish_reason": interaction.finish_reason,
                "latency_ms": interaction.latency_ms,
            }
        )

        steps_from_end = num_turns - 1 - turn_idx
        discounted_base = base_outcome * (discount**steps_from_end)

        prm = 0.0
        if prm_turn_scores is not None:
            prm = prm_turn_scores.get(turn_idx, 0.0)
            final = discounted_base + prm_coef * prm
        else:
            final = discounted_base

        # Penalize empty/trivial outputs to prevent mode collapse.
        # If total response is too short, override score to -1.0.
        min_response_tokens = 10
        if (
            penalize_short_response
            and s.response_length < min_response_tokens
            and num_turns == 1
        ):
            final = -1.0

        dapo_overlong_reward = _dapo_overlong_reward(s.response_length, dapo_overlong_cfg)
        final += dapo_overlong_reward

        if prm_turn_scores is not None:
            s.metadata["step_wise"] = {
                "step_scores": [prm],
                "step_scores_with_outcome": [final],
                "step_indices": [turn_idx],
                "step_token_spans": [[0, s.response_length]],
            }

        s.reward = {
            "accuracy": accuracy,
            "raw_score": raw_score,
            "base_score": discounted_base,
            "score": final,
        }
        if outcome_is_score:
            s.reward["outcome_is_score"] = True
        if dapo_overlong_cfg is not None:
            s.reward["dapo_overlong_reward"] = dapo_overlong_reward
            s.reward["dapo_overlong"] = dapo_overlong_reward < 0.0
            s.reward["dapo_overlong_expected_len"] = dapo_overlong_cfg["expected_len"]
            s.reward["dapo_overlong_buffer_len"] = dapo_overlong_cfg["buffer_len"]

        if prm_turn_scores is not None:
            s.reward["prm_turn_score"] = prm
        _sync_reward_aliases(s.reward)
        samples.append(s)

    return samples


def _mark_non_trainable_samples(samples: List[Sample]) -> None:
    for sample in samples:
        if sample.status in {Sample.Status.ABORTED, Sample.Status.FAILED}:
            if sample.reward is None:
                sample.reward = {"score": 0.0}
            _sync_reward_aliases(sample.reward)
            sample.remove_sample = True


def _build_agent_safetybench_eval_payload(
    *,
    task_meta: Dict[str, Any],
    turn_records: List[Dict[str, Any]],
    final_response: Any,
    interactions: List[Interaction],
    status: Sample.Status | str | None = None,
    parse_error_count: int = 0,
) -> Dict[str, Any]:
    messages: list[dict[str, Any]] = []
    if not isinstance(task_meta.get("dialog"), list):
        messages.append(
            {"role": "user", "content": str(task_meta.get("instruction", ""))}
        )

    for turn in turn_records:
        tool_calls = [
            call for call in turn.get("tool_calls", []) if isinstance(call, dict)
        ]
        if tool_calls:
            assistant_msg = {"role": "assistant", "tool_calls": []}
            for idx, call in enumerate(tool_calls):
                tool_call_id = str(
                    call.get("tool_call_id")
                    or f"call_{turn.get('turn_idx', 0)}_{idx}"
                )
                assistant_msg["tool_calls"].append(
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": str(call.get("tool_name", "")),
                            "arguments": json.dumps(
                                call.get("args") or {}, ensure_ascii=False
                            ),
                        },
                    }
                )
            messages.append(assistant_msg)
            for idx, call in enumerate(tool_calls):
                tool_call_id = str(
                    call.get("tool_call_id")
                    or f"call_{turn.get('turn_idx', 0)}_{idx}"
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": str(call.get("tool_name", "")),
                        "content": str(call.get("result", "")),
                    }
                )
            continue

        assistant_output = str(turn.get("assistant_output") or "").strip()
        if assistant_output:
            messages.append({"role": "assistant", "content": assistant_output})

    final_text = ""
    if final_response is not None:
        final_text = str(getattr(final_response, "msg", "") or "")
    if not final_text:
        for turn in reversed(turn_records):
            final_text = str(turn.get("assistant_output") or "").strip()
            if final_text:
                break

    if isinstance(status, Sample.Status):
        status_value = status.value
    else:
        status_value = str(status or "")

    return {
        "messages": messages,
        "turn_records": turn_records,
        "final_response": final_text,
        "status": status_value,
        "parse_error_count": int(parse_error_count or 0),
        "n_output_tokens": sum(
            len(it.output_token_ids or []) for it in interactions
        ),
    }
