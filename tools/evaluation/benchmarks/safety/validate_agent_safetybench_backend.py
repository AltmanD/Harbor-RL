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


REPO_ROOT = Path(__file__).resolve().parents[4]
AGENTIC_RL_DIR = REPO_ROOT / "harborrl"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harborrl.environments.agent_safetybench.runtime import AgentSafetyBenchEnv  # noqa: E402


async def run(sample_id: int, root: Path, reward_mode: str) -> dict:
    os.environ["SAFETY_BENCH_REWARD"] = reward_mode
    data_path = root / "data" / "released_data.json"
    data = json.loads(data_path.read_text(encoding="utf-8"))
    example = data[sample_id]
    task_meta = {
        "task_name": str(example["id"]),
        "task_path": f"agent_safetybench/{example['id']}",
        "instruction": example["instruction"],
        "data_source": "agent_safetybench",
        "fulfillable": example["fulfillable"],
        "failure_modes": example["failure_modes"],
        "risks": example["risks"],
        "environments": example["environments"],
    }
    if "dialog" in example:
        task_meta["dialog"] = example["dialog"]

    with tempfile.TemporaryDirectory() as tmp:
        env = AgentSafetyBenchEnv(root=str(root))
        user_msg, tool_schemas = await env.reset(
            task_meta=task_meta,
            task_spec=SimpleNamespace(task_name=str(example["id"])),
            run_ctx=SimpleNamespace(uid="smoke", log_dir=Path(tmp)),
        )

        tool_result = None
        turn_records = []
        if tool_schemas:
            first_tool = tool_schemas[0]["function"]["name"]
            tool_result = await env.exec_tool(first_tool, {})
            turn_records.append(
                {
                    "turn_idx": 0,
                    "assistant_output": "",
                    "tool_calls": [
                        {
                            "tool_call_id": "call_smoke",
                            "tool_name": first_tool,
                            "args": {},
                            "result": tool_result,
                        }
                    ],
                }
            )

        score_after_tool = await env.evaluate(
            {
                "turn_records": turn_records,
                "final_response": "I cannot safely help with that request.",
                "status": "completed",
                "messages": [
                    {"role": "user", "content": example["instruction"]},
                    {
                        "role": "assistant",
                        "content": "I cannot safely help with that request.",
                    },
                ],
            }
        )
        refusal_score = await env.evaluate(
            {
                "turn_records": [],
                "final_response": "I cannot safely help with that request.",
                "status": "completed",
                "messages": [
                    {"role": "user", "content": example["instruction"]},
                    {
                        "role": "assistant",
                        "content": "I cannot safely help with that request.",
                    },
                ],
            }
        )
        last_eval = env._last_eval
        await env.close()

    return {
        "sample_id": sample_id,
        "user_msg_preview": user_msg[:120],
        "n_tools": len(tool_schemas),
        "first_tool_result": tool_result,
        "score_after_tool": score_after_tool,
        "refusal_score": refusal_score,
        "reward_mode": reward_mode,
        "last_eval": last_eval,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-id", type=int, default=0)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("../Agent-SafetyBench"),
    )
    parser.add_argument("--reward-mode", default="rule")
    args = parser.parse_args()
    print(
        json.dumps(
            asyncio.run(run(args.sample_id, args.root, args.reward_mode)),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
