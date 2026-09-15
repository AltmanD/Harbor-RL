#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
AGENTIC_RL_DIR = REPO_ROOT / "harborrl"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harborrl.data.convert_agentharm_to_dataset import convert_record  # noqa: E402
from harborrl.environments.agentharm.runtime import AgentHarmEnv  # noqa: E402


FILES = {
    "harmful": ("harmful_test_public.jsonl", "test_public"),
    "benign": ("harmless_benign_test_public.jsonl", "test_public"),
    "chat": ("chat_test_public.jsonl", "test_public"),
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _dummy_args(schema: dict[str, Any]) -> dict[str, Any]:
    args: dict[str, Any] = {}
    props = schema.get("function", {}).get("parameters", {}).get("properties", {})
    for name, prop in props.items():
        typ = prop.get("type", "string")
        if typ == "integer":
            args[name] = 0
        elif typ == "number":
            args[name] = 0.0
        elif typ == "boolean":
            args[name] = False
        elif typ == "array":
            args[name] = []
        elif typ == "object":
            args[name] = {}
        else:
            args[name] = ""
    return args


async def run(
    *,
    task_type: str,
    sample_id: int,
    raw_dir: Path,
    root: Path,
    reward_mode: str,
) -> dict[str, Any]:
    os.environ["AGENTHARM_REWARD"] = reward_mode
    filename, split = FILES[task_type]
    rows = _read_jsonl(raw_dir / filename)
    example = rows[sample_id]
    record = convert_record(
        example,
        task_type=task_type,
        split=split,
        source_file=filename,
    )
    task_meta = record["metadata"]

    with tempfile.TemporaryDirectory() as tmp:
        env = AgentHarmEnv(root=str(root))
        user_msg, tool_schemas = await env.reset(
            task_meta=task_meta,
            task_spec=SimpleNamespace(task_name=task_meta["task_name"]),
            run_ctx=SimpleNamespace(uid="agentharm-smoke", log_dir=Path(tmp)),
        )

        tool_result = None
        turn_records = []
        if tool_schemas:
            first_tool = tool_schemas[0]["function"]["name"]
            args = _dummy_args(tool_schemas[0])
            tool_result = await env.exec_tool(first_tool, args)
            turn_records.append(
                {
                    "turn_idx": 0,
                    "assistant_output": "",
                    "tool_calls": [
                        {
                            "tool_call_id": "call_smoke",
                            "tool_name": first_tool,
                            "args": args,
                            "result": tool_result,
                        }
                    ],
                }
            )

        score_after_tool = await env.evaluate(
            {
                "turn_records": turn_records,
                "final_response": "I completed what I could with the available tools.",
                "status": "completed",
            }
        )
        refusal_score = await env.evaluate(
            {
                "turn_records": [],
                "final_response": "I cannot safely help with that request.",
                "status": "completed",
            }
        )
        last_eval = env._last_eval
        await env.close()

    return {
        "task_type": task_type,
        "sample_id": sample_id,
        "task_name": task_meta["task_name"],
        "user_msg_preview": user_msg[:160],
        "n_tools": len(tool_schemas),
        "first_tool": tool_schemas[0]["function"]["name"] if tool_schemas else None,
        "first_tool_result_preview": (
            tool_result[:200] if isinstance(tool_result, str) else tool_result
        ),
        "score_after_tool": score_after_tool,
        "refusal_score": refusal_score,
        "reward_mode": reward_mode,
        "last_eval": last_eval,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-type", choices=sorted(FILES), default="harmful")
    parser.add_argument("--sample-id", type=int, default=0)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("benchmarks/datasets/agentharm"),
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("../inspect_evals/src/inspect_evals/agentharm"),
    )
    parser.add_argument("--reward-mode", default="dense_rule")
    args = parser.parse_args()
    print(
        json.dumps(
            asyncio.run(
                run(
                    task_type=args.task_type,
                    sample_id=args.sample_id,
                    raw_dir=args.raw_dir,
                    root=args.root,
                    reward_mode=args.reward_mode,
                )
            ),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
