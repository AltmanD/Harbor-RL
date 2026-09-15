#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _task_messages(example: dict[str, Any], include_tools_in_prompt: bool) -> list[dict[str, str]]:
    if isinstance(example.get("dialog"), list) and example["dialog"]:
        return example["dialog"]

    content = str(example.get("instruction", ""))
    if include_tools_in_prompt:
        tools = []
        for env in example.get("environments") or []:
            tools.extend(env.get("tools") or [])
        if tools:
            content = f"{content}\n\nAvailable tools: {', '.join(tools)}"
    return [{"role": "user", "content": content}]


def convert_record(
    example: dict[str, Any], include_tools_in_prompt: bool
) -> dict[str, Any]:
    task_id = str(example["id"])
    metadata = {
        "task_name": task_id,
        "task_path": f"agent_safetybench/{task_id}",
        "instruction": example.get("instruction", ""),
        "data_source": "agent_safetybench",
        "fulfillable": example.get("fulfillable"),
        "failure_modes": example.get("failure_modes"),
        "risks": example.get("risks", []),
        "environments": example.get("environments", []),
    }
    if "dialog" in example:
        metadata["dialog"] = example["dialog"]

    return {
        "task": _task_messages(example, include_tools_in_prompt),
        "metadata": metadata,
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert official Agent-SafetyBench JSON to terminal-rl JSONL."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("../Agent-SafetyBench/data/released_data.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/datasets/agent_safetybench_convert"),
    )
    parser.add_argument(
        "--include-tools-in-prompt",
        action="store_true",
        help="Legacy mode. Normally tools are supplied by the remote env schema.",
    )
    args = parser.parse_args()

    data = json.loads(args.input.read_text(encoding="utf-8"))
    records = [convert_record(x, args.include_tools_in_prompt) for x in data]
    harmful = [r for r in records if int(r["metadata"].get("fulfillable") or 0) == 0]
    benign = [r for r in records if int(r["metadata"].get("fulfillable") or 0) == 1]

    write_jsonl(args.output_dir / "train.jsonl", records)
    write_jsonl(args.output_dir / "train_harmful.jsonl", harmful)
    write_jsonl(args.output_dir / "train_benign.jsonl", benign)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "total": len(records),
                "harmful": len(harmful),
                "benign": len(benign),
                "include_tools_in_prompt": args.include_tools_in_prompt,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
