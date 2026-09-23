"""Public entry point for the native HarborRL training pipeline."""
from __future__ import annotations

import argparse
import json
import sys


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "hub":
        from harborrl.tasks.cli import main as hub_main

        return hub_main(argv[1:])
    parser = argparse.ArgumentParser(prog="harborrl", description=__doc__)
    parser.add_argument("command", choices=["train", "doctor"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    from harborrl.config import launch_plan, load_config

    try:
        plan = launch_plan(load_config(args.config, args.set))
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    from harborrl.platform.native_train import dispatch

    return dispatch(plan, doctor_only=args.command == "doctor")


if __name__ == "__main__":
    raise SystemExit(main())
