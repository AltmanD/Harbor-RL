from __future__ import annotations

from typing import Any

SKIP_TRAIN_KEY = "__slime_skip_train__"


def make_skip_train_result(
    *,
    rollout_id: int,
    reason: str,
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        SKIP_TRAIN_KEY: True,
        "rollout_id": int(rollout_id),
        "reason": str(reason),
        "metrics": metrics or {},
    }


def is_skip_train_result(value: Any) -> bool:
    return isinstance(value, dict) and bool(value.get(SKIP_TRAIN_KEY))
