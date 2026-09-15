from __future__ import annotations

from typing import Any, List


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


from harborrl.env import env_flag as _env_enabled


def _format_float(value: Any, width: int = 8) -> str:
    parsed = _to_float(value)
    if parsed is None:
        return "-".rjust(width)
    return f"{parsed:.4f}".rjust(width)


def _format_dataset_table(rows: List[dict[str, Any]]) -> str:
    if not rows:
        return ""
    header = (
        "dataset                 n  ratio train  reward    pass resp_len  comp trunc fail abort"
    )
    line = "-" * len(header)
    body = []
    for row in rows:
        body.append(
            f"{str(row['dataset'])[:22]:22} "
            f"{int(row['count']):4d} "
            f"{row['ratio']:6.2%} "
            f"{int(row['trainable']):5d} "
            f"{_format_float(row['reward_mean'])} "
            f"{_format_float(row['acc_mean'])} "
            f"{_format_float(row['response_mean'])} "
            f"{int(row['completed']):5d} "
            f"{int(row['truncated']):5d} "
            f"{int(row['failed']):4d} "
            f"{int(row['aborted']):5d}"
        )
    return "\n".join([header, line, *body])


def _format_reward_breakdown_table(rows: List[dict[str, Any]]) -> str:
    if not rows:
        return ""
    header = (
        "dataset                 n train task_reward total_reward score_bonus adv_intr penalty episodic lifelong signal trust trustT arms"
    )
    line = "-" * len(header)
    body = []
    for row in rows:
        body.append(
            f"{str(row['dataset'])[:22]:22} "
            f"{int(row['count']):4d} "
            f"{int(row['trainable']):5d} "
            f"{_format_float(row.get('base_reward_mean'), width=11)} "
            f"{_format_float(row.get('reward_mean'), width=12)} "
            f"{_format_float(row.get('exploration_reward_score_mean'), width=10)} "
            f"{_format_float(row.get('adv_intrinsic_reward_mean'), width=8)} "
            f"{_format_float(row.get('adv_penalty_mean'), width=7)} "
            f"{_format_float(row.get('intrinsic_episodic_reward_mean'), width=8)} "
            f"{_format_float(row.get('intrinsic_lifelong_reward_mean'), width=8)} "
            f"{_format_float(row.get('intrinsic_signal_mean'), width=6)} "
            f"{_format_float(row.get('agent57_trust_mean'), width=5)} "
            f"{_format_float(row.get('agent57_trust_truncated_mean'), width=6)} "
            f"{_format_float(row.get('agent57_arm_count'), width=4)}"
        )
    return "\n".join([header, line, *body])


def _format_agent57_table(rows: List[dict[str, Any]]) -> str:
    if not _env_enabled("TERMINAL_AGENT57_VERBOSE_METRICS", "0"):
        return ""
    agent_rows = [
        row for row in rows
        if row.get("explore_agent57_lifelong_bonus_mean") is not None
        or row.get("agent57_arm_count") is not None
    ]
    if not agent_rows:
        return ""
    header = (
        "dataset                 n arms top_arm a57_raw a57_bonus eligible stateerr suppressed"
    )
    line = "-" * len(header)
    body = []
    for row in agent_rows:
        suppressed = str(row.get("agent57_top_suppressed_reason") or "-")
        suppressed_ratio = _to_float(row.get("agent57_top_suppressed_ratio"))
        if suppressed != "-" and suppressed_ratio is not None:
            suppressed = f"{suppressed[:18]}:{suppressed_ratio:.0%}"
        top_arm = row.get("agent57_top_arm")
        top_arm_ratio = _to_float(row.get("agent57_top_arm_ratio"))
        top_arm_text = "-"
        if top_arm is not None:
            try:
                top_arm_text = str(int(float(top_arm)))
            except (TypeError, ValueError):
                top_arm_text = str(top_arm)
            if top_arm_ratio is not None:
                top_arm_text = f"{top_arm_text}:{top_arm_ratio:.0%}"
        arm_count = row.get("agent57_arm_count")
        try:
            arm_count_text = str(int(float(arm_count))) if arm_count is not None else "-"
        except (TypeError, ValueError):
            arm_count_text = str(arm_count)
        body.append(
            f"{str(row['dataset'])[:22]:22} "
            f"{int(row['count']):4d} "
            f"{arm_count_text[:4]:4} "
            f"{top_arm_text[:7]:7} "
            f"{_format_float(row.get('explore_agent57_lifelong_raw_mean'), width=7)} "
            f"{_format_float(row.get('explore_agent57_lifelong_bonus_mean'), width=9)} "
            f"{_format_float(row.get('agent57_lifelong_eligible_rate'), width=8)} "
            f"{_format_float(row.get('agent57_lifelong_state_error_rate'), width=8)} "
            f"{suppressed}"
        )
    return "\n".join([header, line, *body])


def _format_split_table(rows: List[dict[str, Any]]) -> str:
    if not rows:
        return ""
    header = (
        "dataset                 split                    n  ratio train  rew_mean     pass "
        "refuse   tools   empty top_reason"
    )
    line = "-" * len(header)
    body = []
    for row in rows:
        top_reason = str(row.get("top_reason") or "-")
        top_ratio = _to_float(row.get("top_reason_ratio"))
        if top_ratio is not None and top_reason != "-":
            top_reason = f"{top_reason[:24]}:{top_ratio:.0%}"
        body.append(
            f"{str(row['dataset'])[:22]:22} "
            f"{str(row['split'])[:24]:24} "
            f"{int(row['count']):4d} "
            f"{row['ratio']:6.2%} "
            f"{int(row['trainable']):5d} "
            f"{_format_float(row['reward_mean'])} "
            f"{_format_float(row['acc_mean'])} "
            f"{_format_float(row['verbal_refused_rate'])} "
            f"{_format_float(row['attempted_tool_rate'])} "
            f"{_format_float(row['empty_response_rate'])} "
            f"{top_reason}"
        )
    return "\n".join([header, line, *body])
