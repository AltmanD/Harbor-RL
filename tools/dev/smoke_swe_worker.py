#!/usr/bin/env python3
"""Smoke-test a SWE-smith or SWE-Verified terminal-rl Docker worker."""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

import smoke_swesmith_worker as smith


def _load_row(path: Path, index: int) -> dict:
    row_index = 0
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            if row_index == index:
                return json.loads(line)
            row_index += 1
    raise IndexError(f"{path} does not contain row index {index}")


def _run_swesmith(argv: list[str]) -> int:
    forwarded = [argv[0]]
    skip_next = False
    for index, value in enumerate(argv[1:], 1):
        if skip_next:
            skip_next = False
            continue
        if value == "--suite":
            skip_next = True
            continue
        if value.startswith("--suite="):
            continue
        forwarded.append(value)
    original = sys.argv
    try:
        sys.argv = forwarded
        return smith.main()
    finally:
        sys.argv = original


def _run_sweverified(args: argparse.Namespace) -> int:
    dataset = Path(args.dataset)
    row = _load_row(dataset, args.index)
    meta = row.get("metadata") or {}

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    from harborrl.data.convert_sweverified import (
        DATASET_NAME,
        DATASET_REVISION,
        SWEBENCH_COMMIT,
        SWEBENCH_VERSION,
        TASK_FORMAT_VERSION,
        expected_task_path,
    )

    instance_id = str(meta.get("swe_instance_id") or "")
    expected = {
        "data_source": "sweverified",
        "source_dataset": DATASET_NAME,
        "source_revision": DATASET_REVISION,
        "swebench_harness_version": SWEBENCH_VERSION,
        "swebench_harness_commit": SWEBENCH_COMMIT,
        "task_format_version": TASK_FORMAT_VERSION,
        "task_path": expected_task_path(instance_id),
    }
    mismatches = {
        key: (value, meta.get(key))
        for key, value in expected.items()
        if meta.get(key) != value
    }
    if mismatches:
        raise SystemExit(
            f"[ERROR] selected SWE-Verified row is stale/untrusted: {mismatches}"
        )

    code, body = smith._get_json(args.worker_url, "/healthz", timeout=10)
    print(f"[smoke] healthz HTTP {code}: {json.dumps(body, ensure_ascii=False)[:500]}")
    if code >= 400 or not body.get("ok"):
        return 2

    task_key = f"{meta['task_name']}:{meta['task_path']}"
    request_id = f"sweverified-smoke-{uuid.uuid4().hex[:12]}"
    lease_id = ""
    result_code = 0
    try:
        code, body = smith._post_json(
            args.worker_url,
            "/allocate",
            {"task_key": task_key, "request_id": request_id},
            timeout=60,
        )
        print(f"[smoke] allocate HTTP {code}: {json.dumps(body, ensure_ascii=False)[:1000]}")
        if code >= 400 or not body.get("ok"):
            raise smith.SmokeFailure(3)
        lease_id = str(body["lease_id"])

        reset_payload = {
            "lease_id": lease_id,
            "task_meta": meta,
            "run_ctx": {
                "uid": request_id,
                "group_index": 0,
                "sample_index": args.index,
            },
            "task_timeouts": {
                "ensure_image": args.ensure_image_timeout,
                "reset_session": args.reset_session_timeout,
                "eval": args.eval_timeout,
                "close_session": 90,
            },
        }
        code, body = smith._post_json(
            args.worker_url,
            "/reset",
            reset_payload,
            timeout=(
                args.build_queue_timeout
                + args.ensure_image_timeout
                + args.reset_session_timeout
                + 600
            ),
        )
        print(f"[smoke] reset HTTP {code}: {json.dumps(body, ensure_ascii=False)[:1500]}")
        if code >= 400 or not body.get("ok"):
            raise smith.SmokeFailure(4)
        user_msg = str(body.get("user_msg") or "")
        if "/testbed" not in user_msg or "Do not clone" not in user_msg:
            print("[ERROR] worker did not pin the /testbed workspace", file=sys.stderr)
            raise smith.SmokeFailure(5)

        code, body = smith._post_json(
            args.worker_url,
            "/exec_tool",
            {
                "lease_id": lease_id,
                "tool_call": {
                    "name": "shell_exec",
                    "arguments": {
                        "id": "sweverified-smoke",
                        "command": (
                            "test \"$(pwd)\" = /testbed && "
                            "git rev-parse --is-inside-work-tree && "
                            "printf '\\n# terminal-rl smoke\\n' > "
                            ".terminal_rl_smoke_probe"
                        ),
                    },
                },
            },
            timeout=120,
        )
        print(f"[smoke] workspace HTTP {code}: {json.dumps(body, ensure_ascii=False)[:1000]}")
        if code >= 400 or not body.get("ok"):
            raise smith.SmokeFailure(6)

        code, body = smith._post_json(
            args.worker_url,
            "/evaluate",
            {
                "lease_id": lease_id,
                "trajectory": {"swebench_defer_grading": True},
            },
            timeout=args.eval_timeout + 30,
        )
        print(f"[smoke] export HTTP {code}: {json.dumps(body, ensure_ascii=False)[:1500]}")
        details = body.get("details") if isinstance(body, dict) else None
        if (
            code >= 400
            or not body.get("ok")
            or float(body.get("score", -1)) != 0.0
            or not isinstance(details, dict)
            or details.get("grader") != "swebench_prediction_export"
            or details.get("grading_deferred") is not True
            or ".terminal_rl_smoke_probe" not in str(details.get("model_patch") or "")
        ):
            print("[ERROR] deferred prediction export is invalid", file=sys.stderr)
            raise smith.SmokeFailure(7)
    except smith.SmokeFailure as exc:
        result_code = exc.exit_code
    finally:
        if lease_id:
            try:
                code, body = smith._post_json(
                    args.worker_url,
                    "/close",
                    {"lease_id": lease_id},
                    timeout=180,
                )
                print(f"[smoke] close HTTP {code}: {json.dumps(body, ensure_ascii=False)[:1000]}")
                if code >= 400 or not body.get("ok") or body.get("found") is not True:
                    result_code = result_code or 8
                else:
                    closed, failure = smith._wait_for_close(
                        args.worker_url, lease_id, args.close_wait_timeout
                    )
                    if not closed:
                        print(
                            f"[ERROR] close did not finish: {failure}",
                            file=sys.stderr,
                        )
                        result_code = result_code or 8
            except Exception as exc:
                print(f"[ERROR] close failed: {exc}", file=sys.stderr)
                result_code = result_code or 8
    if result_code:
        return result_code
    print("[smoke] SWE-Verified prediction worker API path OK")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("swesmith", "sweverified"), required=True)
    parser.add_argument("--worker-url", required=True)
    parser.add_argument("--dataset")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--ensure-image-timeout", type=float, default=3600.0)
    parser.add_argument("--build-queue-timeout", type=float, default=1800.0)
    parser.add_argument("--reset-session-timeout", type=float, default=900.0)
    parser.add_argument("--eval-timeout", type=float, default=300.0)
    parser.add_argument("--close-wait-timeout", type=float, default=240.0)
    args, _unknown = parser.parse_known_args()

    if args.suite == "swesmith":
        return _run_swesmith(sys.argv)
    if not args.dataset:
        args.dataset = str(
            Path(__file__).resolve().parents[2]
            / "dataset"
            / "sweverified_convert"
            / "test.jsonl"
        )
    started = time.monotonic()
    result = _run_sweverified(args)
    print(f"[smoke] elapsed={time.monotonic() - started:.1f}s")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
