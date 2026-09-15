import math
from .schema import Trajectory


def finite(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def validate(ir: Trajectory) -> list[str]:
    errors = []
    for key in ("task", "evaluation", "policy", "derived"):
        if not isinstance(getattr(ir, key), dict):
            errors.append(f"{key}.type")
    if not isinstance(ir.turns, list):
        errors.append("turns.type")
    if errors:
        return errors
    if ir.schema_version != "1":
        errors.append("schema_version")
    for key in ("dataset", "task", "source_digest"):
        if not isinstance(ir.task.get(key), str) or not ir.task[key]:
            errors.append(f"task.{key}")
    for key in ("trajectory_id", "attempt_id", "group_id"):
        if not isinstance(getattr(ir, key), str) or not getattr(ir, key):
            errors.append(key)
    if ir.execution_status not in ("completed", "truncated"):
        errors.append("execution_status")
    reward = ir.evaluation.get("raw_reward")
    if not finite(reward) or not 0 <= reward <= 1:
        errors.append("raw_reward")
    if ir.evaluation.get("error") or ir.evaluation.get("exception_stage"):
        errors.append("evaluation_error")
    for key in ("weight_version", "tokenizer", "chat_template", "logprob_source"):
        if not isinstance(ir.policy.get(key), str) or not ir.policy[key]:
            errors.append(f"policy.{key}")
    if ir.policy.get("logprob_semantics") != "raw_model":
        errors.append("policy.logprob_semantics")
    if not ir.turns:
        errors.append("empty_turns")
    for i, turn in enumerate(ir.turns):
        prefix = f"turns.{i}"
        if not isinstance(turn, dict):
            errors.append(prefix + ".type")
            continue
        if turn.get("turn_idx") != i:
            errors.append(prefix + ".order")
        for key in ("input_ids", "output_token_ids"):
            ids = turn.get(key)
            if (
                not isinstance(ids, list)
                or not ids
                or any(type(x) is not int or x < 0 for x in ids)
            ):
                errors.append(prefix + "." + key)
        logps = turn.get("output_token_logprobs")
        ids = turn.get("output_token_ids")
        if (
            not isinstance(logps, list)
            or not isinstance(ids, list)
            or len(logps) != len(ids)
            or any(not finite(p) or p > 0 for p in logps)
        ):
            errors.append(prefix + ".logprobs")
        meta = turn.get("generation_meta")
        if not isinstance(meta, dict):
            errors.append(prefix + ".generation_meta")
            continue
        if (
            str(meta.get("weight_version", ""))
            != str(ir.policy.get("weight_version", ""))
            or "weight_version" not in meta
        ):
            errors.append(prefix + ".weight_version")
    return errors


def require_rl_ready(ir, *, weight_version, group_id):
    errors = validate(ir)
    if ir.readiness != "RL_READY":
        errors.append("readiness")
    if not isinstance(ir.policy, dict) or str(ir.policy.get("weight_version")) != str(
        weight_version
    ):
        errors.append("current_weight_version")
    if ir.group_id != str(group_id):
        errors.append("current_group")
    if errors:
        raise ValueError("IR is not RL ready: " + ", ".join(errors))
