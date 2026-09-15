#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_source(raw: str) -> tuple[Path, float]:
    if ":" not in raw:
        raise ValueError(f"source must be PATH:RATIO, got {raw!r}")
    path_raw, ratio_raw = raw.rsplit(":", 1)
    ratio = float(ratio_raw)
    if ratio <= 0:
        raise ValueError(f"source ratio must be positive, got {raw!r}")
    return Path(path_raw), ratio


def allocate_counts(
    lengths: list[int], ratios: list[float], total: int | None, oversample: bool
) -> list[int]:
    ratio_sum = sum(ratios)
    if total is None:
        if oversample:
            total = sum(lengths)
        else:
            total = int(min(length / (ratio / ratio_sum) for length, ratio in zip(lengths, ratios)))

    counts = [int(total * ratio / ratio_sum) for ratio in ratios]
    while sum(counts) < total:
        remainders = [
            total * ratio / ratio_sum - count for ratio, count in zip(ratios, counts)
        ]
        idx = max(range(len(remainders)), key=remainders.__getitem__)
        counts[idx] += 1

    if not oversample:
        for count, length in zip(counts, lengths):
            if count > length:
                raise ValueError(
                    f"requested {count} samples from source with only {length}; "
                    "lower --total or pass --oversample"
                )
    return counts


def allocate_counts_all_visible(
    lengths: list[int], ratios: list[float], total: int | None
) -> list[int]:
    """Allocate ratio counts without hiding any source record.

    The rollout data source consumes one static JSONL sequentially, so ratio mixing
    has to be represented by a finite epoch manifest. This mode includes every
    original sample at least once and repeats smaller sources when needed to match
    the requested ratios.
    """

    ratio_sum = sum(ratios)
    min_total_for_visibility = sum(lengths)
    if total is None:
        total = max(
            math.ceil(length / (ratio / ratio_sum))
            for length, ratio in zip(lengths, ratios)
        )
    elif total < min_total_for_visibility:
        raise ValueError(
            f"--total={total} is too small for all_visible mode; "
            f"need at least {min_total_for_visibility} to include all source records"
        )

    raw_counts = [total * ratio / ratio_sum for ratio in ratios]
    counts = [
        max(length, int(math.floor(raw)))
        for length, raw in zip(lengths, raw_counts)
    ]

    if sum(counts) > total:
        raise ValueError(
            f"--total={total} cannot satisfy both all_visible source coverage and "
            "the requested ratios; increase --total or omit it"
        )

    while sum(counts) < total:
        deficits = [raw - count for raw, count in zip(raw_counts, counts)]
        idx = max(range(len(deficits)), key=lambda i: (deficits[i], ratios[i]))
        if deficits[idx] <= 0:
            idx = max(range(len(ratios)), key=ratios.__getitem__)
        counts[idx] += 1

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deterministically mix JSONL datasets by source ratios."
    )
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="Input as PATH:RATIO. Pass multiple times.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total", type=int, default=None)
    parser.add_argument("--oversample", action="store_true")
    parser.add_argument(
        "--mode",
        choices=("ratio_cap", "all_visible"),
        default="ratio_cap",
        help=(
            "ratio_cap preserves the legacy behavior and may subsample large "
            "sources. all_visible includes every source record at least once and "
            "duplicates smaller sources to approximate ratios in a static epoch."
        ),
    )
    args = parser.parse_args()

    parsed_sources = [parse_source(x) for x in args.source]
    rng = random.Random(args.seed)
    datasets = [read_jsonl(path) for path, _ratio in parsed_sources]
    ratios = [ratio for _path, ratio in parsed_sources]
    lengths = [len(dataset) for dataset in datasets]
    if args.mode == "all_visible":
        counts = allocate_counts_all_visible(lengths, ratios, args.total)
    else:
        counts = allocate_counts(lengths, ratios, args.total, args.oversample)

    mixed: list[dict[str, Any]] = []
    summary = []
    for (path, ratio), dataset, count in zip(parsed_sources, datasets, counts):
        if args.mode == "all_visible" and count >= len(dataset):
            selected = list(dataset)
            selected.extend(rng.choice(dataset) for _ in range(count - len(dataset)))
        elif args.oversample and count > len(dataset):
            selected = [rng.choice(dataset) for _ in range(count)]
        else:
            selected = rng.sample(dataset, count)
        mixed.extend(selected)
        summary.append(
            {
                "path": str(path),
                "available": len(dataset),
                "selected": count,
                "unique_visible": min(count, len(dataset)),
                "duplicates": max(0, count - len(dataset)),
                "ratio": ratio,
            }
        )

    rng.shuffle(mixed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for record in mixed:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "output": str(args.output),
                "total": len(mixed),
                "seed": args.seed,
                "mode": args.mode,
                "sources": summary,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
