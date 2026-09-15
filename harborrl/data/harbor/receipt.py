"""Strict scalar receipt parsing: a missing/invalid score is an infra failure."""

import math


def parse_reward(text: str) -> dict:
    try:
        reward = float(text.strip())
    except (TypeError, ValueError) as exc:
        raise ValueError("Harbor reward.txt is missing or not a scalar") from exc
    if not math.isfinite(reward) or not 0 <= reward <= 1:
        raise ValueError("Harbor reward must be finite and in [0, 1]")
    return {
        "raw_reward": reward,
        "score": reward,
        "profile": "harbor_reward_txt_v1",
        "exception_stage": None,
    }


class HarborVerifierError(RuntimeError):
    def __init__(self, details):
        self.details = dict(details)
        super().__init__(str(details.get("error") or details.get("exception_stage")))
