#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


RAW_SPLITS: dict[str, tuple[str, str]] = {
    "harmful_validation.jsonl": ("harmful", "validation"),
    "harmful_test_public.jsonl": ("harmful", "test_public"),
    "harmless_benign_validation.jsonl": ("benign", "validation"),
    "harmless_benign_test_public.jsonl": ("benign", "test_public"),
    "chat_validation.jsonl": ("chat", "validation"),
    "chat_test_public.jsonl": ("chat", "test_public"),
}

DEFAULT_TRAIN_FILES = (
    "harmless_benign_test_public.jsonl",
    "harmful_test_public.jsonl",
)
DEFAULT_VAL_FILES = (
    "harmful_validation.jsonl",
    "harmless_benign_validation.jsonl",
)


def _safe_id(value: Any) -> str:
    return str(value).replace("/", "_").replace(" ", "_")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def task_messages(example: dict[str, Any]) -> list[dict[str, str]]:
    return [{"role": "user", "content": str(example.get("prompt", ""))}]


def convert_record(
    example: dict[str, Any],
    *,
    task_type: str,
    split: str,
    source_file: str,
) -> dict[str, Any]:
    task_id = _safe_id(example.get("id", "unknown"))
    metadata = dict(example)
    metadata.update(
        {
            "task_name": f"agentharm_{task_type}_{split}_{task_id}",
            "task_path": f"agentharm/{task_type}/{split}/{task_id}",
            "instruction": str(example.get("prompt", "")),
            "data_source": "agentharm",
            "agentharm_task_type": task_type,
            "agentharm_split": split,
            "agentharm_source_file": source_file,
            "fulfillable": 1 if task_type == "benign" else 0,
        }
    )
    metadata.setdefault("target_functions", [])

    return {
        "task": task_messages(example),
        "metadata": metadata,
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_all(input_dir: Path) -> dict[str, list[dict[str, Any]]]:
    converted: dict[str, list[dict[str, Any]]] = {}
    for filename, (task_type, split) in RAW_SPLITS.items():
        path = input_dir / filename
        if not path.exists():
            continue
        converted[filename] = [
            convert_record(
                example,
                task_type=task_type,
                split=split,
                source_file=filename,
            )
            for example in read_jsonl(path)
        ]
    return converted


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert local AgentHarm JSONL files to terminal-rl JSONL."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("benchmarks/datasets/agentharm"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/datasets/agentharm_convert"),
    )
    parser.add_argument(
        "--include-chat-train",
        action="store_true",
        help="Also include chat_test_public.jsonl in train.jsonl.",
    )
    args = parser.parse_args()

    converted = load_all(args.input_dir)
    if not converted:
        raise FileNotFoundError(f"No AgentHarm JSONL files found in {args.input_dir}")

    train_files = list(DEFAULT_TRAIN_FILES)
    if args.include_chat_train:
        train_files.append("chat_test_public.jsonl")
    val_files = list(DEFAULT_VAL_FILES)

    train = [record for name in train_files for record in converted.get(name, [])]
    val = [record for name in val_files for record in converted.get(name, [])]
    harmful_train = [
        record
        for record in train
        if record["metadata"].get("agentharm_task_type") == "harmful"
    ]
    benign_train = [
        record
        for record in train
        if record["metadata"].get("agentharm_task_type") == "benign"
    ]
    chat_train = [
        record
        for record in train
        if record["metadata"].get("agentharm_task_type") == "chat"
    ]

    write_jsonl(args.output_dir / "train.jsonl", train)
    write_jsonl(args.output_dir / "train_harmful.jsonl", harmful_train)
    write_jsonl(args.output_dir / "train_benign.jsonl", benign_train)
    write_jsonl(args.output_dir / "train_chat.jsonl", chat_train)
    write_jsonl(args.output_dir / "val.jsonl", val)
    for filename, records in converted.items():
        task_type, split = RAW_SPLITS[filename]
        write_jsonl(args.output_dir / f"{split}_{task_type}.jsonl", records)

    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "train": len(train),
                "train_harmful": len(harmful_train),
                "train_benign": len(benign_train),
                "train_chat": len(chat_train),
                "val": len(val),
                "include_chat_train": args.include_chat_train,
                "loaded_files": sorted(converted),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
