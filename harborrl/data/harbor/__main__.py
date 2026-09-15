import argparse
import json
from pathlib import Path
from .inspector import inspect
from .materializer import materialize


def main():
    parser = argparse.ArgumentParser(
        description="Inspect/materialize pinned local Harbor tasks"
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--tasks-dir", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument(
        "--receipts",
        type=Path,
        help="JSON mapping task names to TerminalEnv probe receipts",
    )
    parser.add_argument("--output-env-dir", type=Path)
    parser.add_argument("--output-jsonl", type=Path)
    args = parser.parse_args()
    if bool(args.output_env_dir) != bool(args.output_jsonl):
        parser.error("both output paths are required for materialization")
    receipts = json.loads(args.receipts.read_text()) if args.receipts else {}
    reports, rows = [], []
    for source in sorted(args.tasks_dir.iterdir()):
        if not source.is_dir():
            continue
        receipt = receipts.get(source.name)
        report = inspect(source, args.dataset, receipt)
        reports.append(report.to_dict())
        if args.output_env_dir and report.status == "SUPPORTED":
            rows.append(materialize(source, args.dataset, args.output_env_dir, receipt))
    payload = {
        "scanned": len(reports),
        "supported": sum(r["status"] == "SUPPORTED" for r in reports),
        "execution_verified": sum(r["status"] == "SUPPORTED" for r in reports),
        "tasks": reports,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2) + "\n")
    if args.output_jsonl:
        if not rows:
            raise SystemExit(
                "no verified tasks; inspect the report and run probes first"
            )
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.output_jsonl.open("x") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if k != "tasks"}))


if __name__ == "__main__":
    main()
