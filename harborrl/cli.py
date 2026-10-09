"""Public entry point for the native HarborRL training pipeline."""
from __future__ import annotations

import argparse
import json
import sys


ALGORITHM_CONTRACTS = {"grpo": "slime-v0.3.2-native-v1"}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "hub":
        from harborrl.tasks.cli import main as hub_main

        return hub_main(argv[1:])
    parser = argparse.ArgumentParser(
        prog="harborrl", description=__doc__, allow_abbrev=False
    )
    parser.add_argument("-model", help="Slime model preset to select")
    parser.add_argument("-harness", help="agent harness to select")
    parser.add_argument("-task", help="task ID to select from the locked catalog")
    parser.add_argument("-alg", help="training algorithm to select")
    parser.add_argument("command", choices=["train", "doctor"])
    parser.add_argument("config_path", nargs="?", help="YAML configuration file")
    parser.add_argument("--config", dest="config_option", help="YAML configuration file")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    from harborrl.config import launch_plan, load_config

    if args.config_path and args.config_option:
        parser.error(
            "specify the configuration file either positionally or with --config, not both"
        )
    dimensions = (args.model, args.harness, args.task, args.alg)
    config_path = args.config_option or args.config_path
    if config_path is None and any(value is not None for value in dimensions):
        config_path = "config.yaml"
    if config_path is None:
        parser.error("a configuration file is required")

    overrides = list(args.set)
    if args.model is not None:
        overrides.append(f"model.args_file={args.model}")
    if args.harness is not None:
        overrides.append(f"harness.name={args.harness.replace('-', '_')}")
    if args.alg is not None:
        contract = ALGORITHM_CONTRACTS.get(args.alg)
        if contract is None:
            parser.error(
                f"unsupported algorithm: {args.alg}; "
                f"supported algorithms: {', '.join(sorted(ALGORITHM_CONTRACTS))}"
            )
        overrides.append(f"training.backend_contract={contract}")

    try:
        plan = launch_plan(load_config(
            config_path,
            overrides,
            task_ids=None if args.task is None else [args.task],
        ))
    except (TypeError, ValueError, OSError) as exc:
        parser.error(str(exc))
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    from harborrl.platform.native_train import dispatch

    return dispatch(plan, doctor_only=args.command == "doctor")


if __name__ == "__main__":
    raise SystemExit(main())
