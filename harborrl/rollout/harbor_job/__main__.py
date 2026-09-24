"""Offline native development tools. Live training remains separately gated."""
import argparse
import json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("inspect", help="inspect an original task without Docker or GPU")
    check.add_argument("task")
    check.add_argument("--expected-digest")
    smoke = sub.add_parser("offline-smoke", help="synthetic HTTP/session/reward/IR/group check")
    smoke.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "inspect":
        from harborrl.data.harbor.native_inspector import inspect_native
        result = inspect_native(args.task, expected_digest=args.expected_digest)
        print(json.dumps(result, indent=2))
        return int(result["status"] in ("INVALID", "UNSUPPORTED"))
    from .offline import smoke as run_smoke
    print(json.dumps(run_smoke(args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
