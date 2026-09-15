#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from harborrl.data.tau2_support import ensure_tau2_importable, task_instruction


def convert_task(task: Any, *, domain: str, task_split: str | None, policy_type: str) -> dict[str, Any]:
    instruction = task_instruction(task)
    metadata = {
        "task_name": f"tau2_{domain}_{task.id}",
        "task_path": f"tau2/{domain}/{task.id}",
        "instruction": instruction,
        "data_source": "tau2",
        "tau2_domain": domain,
        "tau2_task_id": str(task.id),
        "tau2_task_split": task_split or "",
        "tau2_policy_type": policy_type,
        "tau2_ticket": str(getattr(task, "ticket", "") or ""),
        "tau2_has_ticket": bool(getattr(task, "ticket", None)),
        "tau2_solo_mode": 1,
    }
    return {
        "task": [{"role": "user", "content": instruction}],
        "metadata": metadata,
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert tau2 solo-compatible tasks to terminal-rl JSONL."
    )
    parser.add_argument("--tau2-root", type=Path, default=Path("tau2-bench"))
    parser.add_argument("--domain", choices=["mock", "telecom"], default="telecom")
    parser.add_argument("--task-split", default="train")
    parser.add_argument(
        "--task-id",
        action="append",
        dest="task_ids",
        default=None,
        help="Optional specific task id; may be repeated",
    )
    parser.add_argument("--num-tasks", type=int, default=None)
    parser.add_argument(
        "--policy-type",
        choices=["manual", "workflow"],
        default="manual",
        help="Only meaningful for telecom.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/datasets/tau2_telecom_train_solo"),
    )
    args = parser.parse_args()

    ensure_tau2_importable(args.tau2_root.resolve())
    from tau2.runner.helpers import get_tasks

    tasks = get_tasks(
        task_set_name=args.domain,
        task_split_name=args.task_split or None,
        task_ids=args.task_ids,
        num_tasks=args.num_tasks,
    )
    if not tasks:
        raise ValueError(
            f"No tau2 tasks loaded for domain={args.domain} split={args.task_split}"
        )

    records = [
        convert_task(
            task,
            domain=args.domain,
            task_split=args.task_split or None,
            policy_type=args.policy_type,
        )
        for task in tasks
    ]

    write_jsonl(args.output_dir / "train.jsonl", records)
    write_jsonl(args.output_dir / "val.jsonl", [])

    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "domain": args.domain,
                "task_split": args.task_split,
                "policy_type": args.policy_type,
                "count": len(records),
                "sample_task_ids": [
                    record["metadata"]["tau2_task_id"] for record in records[:20]
                ],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
