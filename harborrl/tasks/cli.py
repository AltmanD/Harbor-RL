"""CPU-only inspection of locally fetched native Harbor tasks."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from uuid import uuid4

from harborrl.data.harbor.native_inspector import inspect_native


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    probe = sub.add_parser("inspect", help="inspect local tasks without Docker or model dependencies")
    probe.add_argument("--tasks-dir", type=Path, required=True)
    probe.add_argument("--dataset", required=True)
    probe.add_argument("--revision", required=True, help="source revision recorded with the content digest")
    probe.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "inspect":
        if not args.tasks_dir.is_dir():
            parser.error("tasks-dir must exist")
        tasks = sorted(path.parent for path in args.tasks_dir.rglob("task.toml"))
        if not tasks:
            parser.error("no task.toml found")
        rows = []
        for task in tasks:
            result = inspect_native(task)
            rows.append({
                **result,
                "dataset": args.dataset,
                "task": task.name,
                "revision": args.revision,
                "execution_backend": "harbor_job",
                "stage": "inspect",
                "evaluable": False,
                "trainable": False,
                "later_stages": "not_attempted",
            })
        args.run_dir.mkdir(parents=True, exist_ok=True)
        # Each inspection attempt is immutable; a new result never overwrites evidence.
        output = args.run_dir / f"{uuid4().hex}.jsonl"
        output.write_text("".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ))

    rows = [
        json.loads(line)
        for path in sorted(args.run_dir.glob("*.jsonl"))
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    counts = Counter(row["status"] for row in rows)
    summary = {
        "attempts": len(rows),
        "status_counts": dict(counts),
        "scope": "local native format inspection only; no execution or training evidence",
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    (args.run_dir / "report.md").write_text(
        "# Harbor native task inspection\n\n"
        "This report only records static task format checks. Execution, reward, and "
        "training are not attempted.\n\n"
        + "\n".join(
            f"- {name}: {count} inspection(s)" for name, count in sorted(counts.items())
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
