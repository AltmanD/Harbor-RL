from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)
STRUCTURED_LOG_PREFIX = "LIGHTRL_METRIC_JSON"


def structured_metrics_path() -> Path | None:
    configured = os.getenv("TERMINAL_METRICS_JSONL")
    if configured:
        return Path(configured)
    run_dir = os.getenv("RUN_DIR")
    if run_dir:
        return Path(run_dir) / "logs" / "metrics.jsonl"
    return None


def write_structured_metrics(records: list[dict[str, Any]]) -> None:
    enabled = os.getenv("TERMINAL_STRUCTURED_METRICS", "1").strip().lower()
    if enabled in {"0", "false", "no", "off"} or not records:
        return

    lines = [
        json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        for record in records
    ]
    for text in lines:
        logger.info("%s %s", STRUCTURED_LOG_PREFIX, text)

    path = structured_metrics_path()
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            for text in lines:
                # One write per JSONL record minimizes interleaving when the
                # rollout manager and actor rank append to the shared file.
                handle.write(text + "\n")
    except Exception as exc:
        logger.warning("Failed to write structured rollout metrics to %s: %s", path, exc)
