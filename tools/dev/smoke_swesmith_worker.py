#!/usr/bin/env python3
"""Smoke-test a SWE-smith terminal-rl Docker worker.

This script exercises the same worker API used during rollout:
allocate -> reset -> evaluate -> close. It is intentionally small and does not
start model training.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
AGENTIC_RL_DIR = REPO_ROOT / "harborrl"
sys.path.insert(0, str(REPO_ROOT))
from harborrl.data.convert_swesmith import (  # noqa: E402
    OFFICIAL_TEST_COMMANDS,
    TASK_FORMAT_VERSION,
    expected_swesmith_task_path,
    infer_test_runner,
)


_NO_PROXY_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class SmokeFailure(Exception):
    def __init__(self, exit_code: int):
        super().__init__(f"SWE-smith smoke failed with exit code {exit_code}")
        self.exit_code = exit_code


def _post_json(base_url: str, path: str, payload: dict, timeout: float) -> tuple[int, dict]:
    url = base_url.rstrip("/") + path
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with _NO_PROXY_OPENER.open(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return resp.status, json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            body = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            body = {"raw": raw}
        return exc.code, body


def _get_json(base_url: str, path: str, timeout: float) -> tuple[int, dict]:
    url = base_url.rstrip("/") + path
    try:
        with _NO_PROXY_OPENER.open(url, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return resp.status, json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            body = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            body = {"raw": raw}
        return exc.code, body


def _load_row(path: Path, index: int) -> dict:
    row_index = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            if row_index == index:
                return json.loads(line)
            row_index += 1
    raise IndexError(f"{path} does not contain row index {index}")


def _status_contains_lease(body: dict, lease_id: str) -> bool:
    pool = body.get("pool") if isinstance(body, dict) else None
    if not isinstance(pool, dict):
        return True
    tasks = pool.get("tasks")
    if isinstance(tasks, dict):
        for task in tasks.values():
            runs = task.get("runs") if isinstance(task, dict) else None
            if isinstance(runs, dict) and lease_id in runs:
                return True
    for key in ("pending_close_labels", "pending_force_cleanup_labels"):
        labels = pool.get(key)
        if isinstance(labels, list) and any(lease_id in str(label) for label in labels):
            return True
    return False


def _status_close_failure(body: dict, lease_id: str) -> dict | None:
    pool = body.get("pool") if isinstance(body, dict) else None
    failures = pool.get("recent_close_failures") if isinstance(pool, dict) else None
    if not isinstance(failures, list):
        return None
    for failure in failures:
        if isinstance(failure, dict) and str(failure.get("lease_id")) == lease_id:
            return failure
    return None


def _wait_for_close(
    base_url: str, lease_id: str, timeout: float
) -> tuple[bool, dict | None]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            code, body = _get_json(base_url, "/status", timeout=10)
        except Exception:
            time.sleep(1)
            continue
        if code < 400 and body.get("ok"):
            failure = _status_close_failure(body, lease_id)
            if failure is not None:
                return False, failure
            if not _status_contains_lease(body, lease_id):
                return True, None
        time.sleep(1)
    return False, None


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test SWE-smith Docker worker")
    parser.add_argument("--worker-url", required=True)
    parser.add_argument(
        "--dataset",
        default=str(
            Path(__file__).resolve().parents[2]
            / "dataset"
            / "swesmith_smoke"
            / "swesmith_convert"
            / "smoke.jsonl"
        ),
    )
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Legacy override for all task timeouts",
    )
    parser.add_argument("--ensure-image-timeout", type=float, default=1800.0)
    parser.add_argument("--build-queue-timeout", type=float, default=1800.0)
    parser.add_argument("--reset-session-timeout", type=float, default=900.0)
    parser.add_argument("--eval-timeout", type=float, default=1200.0)
    parser.add_argument("--close-wait-timeout", type=float, default=240.0)
    parser.add_argument("--skip-evaluate", action="store_true")
    parser.add_argument("--skip-ref-check", action="store_true")
    parser.add_argument("--expect-score", type=float, default=None)
    parser.add_argument(
        "--expect-reason",
        default=None,
        help="Require evaluate.details.reason to match (for example test_exit_nonzero)",
    )
    args = parser.parse_args()
    if args.skip_evaluate and (
        args.expect_score is not None or args.expect_reason is not None
    ):
        parser.error(
            "--skip-evaluate cannot be combined with --expect-score/--expect-reason"
        )
    if args.expect_score is not None and not math.isfinite(args.expect_score):
        parser.error("--expect-score must be finite")

    if args.timeout is not None:
        ensure_image_timeout = args.timeout
        build_queue_timeout = args.timeout
        reset_session_timeout = args.timeout
        eval_timeout = args.timeout
    else:
        ensure_image_timeout = args.ensure_image_timeout
        build_queue_timeout = args.build_queue_timeout
        reset_session_timeout = args.reset_session_timeout
        eval_timeout = args.eval_timeout

    dataset = Path(args.dataset)
    row = _load_row(dataset, args.index)
    meta = row.get("metadata") or {}
    required = ["task_name", "task_path", "instruction", "image_name"]
    missing = [key for key in required if not str(meta.get(key) or "").strip()]
    if missing:
        raise SystemExit(f"[ERROR] row {args.index} missing metadata keys: {missing}")
    expected_runner = infer_test_runner(meta)
    expected_command = OFFICIAL_TEST_COMMANDS.get(
        str(meta.get("repo") or "").lower(), ""
    )
    try:
        expected_task_path = expected_swesmith_task_path(meta.get("task_name"))
    except ValueError as exc:
        raise SystemExit(f"[ERROR] selected row has invalid SWE-smith identity: {exc}") from exc
    if (
        str(meta.get("task_format_version") or "") != TASK_FORMAT_VERSION
        or str(meta.get("test_runner") or "") != expected_runner
        or str(meta.get("test_command") or "") != expected_command
        or str(meta.get("swesmith_instance_id") or "") != str(meta.get("task_name") or "")
        or str(meta.get("task_path") or "") != expected_task_path
    ):
        raise SystemExit(
            "[ERROR] selected row uses a stale/untrusted SWE-smith task format or "
            "test profile; re-run dataset conversion"
        )

    code, body = _get_json(args.worker_url, "/healthz", timeout=10)
    print(f"[smoke] healthz HTTP {code}: {json.dumps(body, ensure_ascii=False)[:500]}")
    if code >= 400 or not body.get("ok"):
        return 2

    task_key = f"{meta['task_name']}:{meta['task_path']}"
    request_id = f"swesmith-smoke-{uuid.uuid4().hex[:12]}"
    lease_id = ""
    result_code = 0
    try:
        code, body = _post_json(
            args.worker_url,
            "/allocate",
            {"task_key": task_key, "request_id": request_id},
            timeout=60,
        )
        print(f"[smoke] allocate HTTP {code}: {json.dumps(body, ensure_ascii=False)[:1000]}")
        if code >= 400 or not body.get("ok"):
            raise SmokeFailure(3)
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
                "ensure_image": ensure_image_timeout,
                "reset_session": reset_session_timeout,
                "eval": eval_timeout,
                "close_session": 90,
            },
        }
        code, body = _post_json(
            args.worker_url,
            "/reset",
            reset_payload,
            timeout=build_queue_timeout + ensure_image_timeout + reset_session_timeout + 600,
        )
        print(f"[smoke] reset HTTP {code}: {json.dumps(body, ensure_ascii=False)[:1500]}")
        if code >= 400 or not body.get("ok"):
            raise SmokeFailure(4)

        if not args.skip_ref_check:
            expected_ref = str(
                meta.get("swesmith_instance_id")
                or meta.get("swe_instance_id")
                or meta["task_name"]
            )
            ref_command = (
                "set -eu; repo_dir=''; "
                "for candidate in . /testbed /workspace; do "
                "if git -C \"$candidate\" rev-parse --is-inside-work-tree >/dev/null 2>&1; "
                "then repo_dir=\"$candidate\"; break; fi; done; "
                "test -n \"$repo_dir\"; "
                "printf '__SWESMITH_REF__=%s\\n' \"$(git -C \"$repo_dir\" branch --show-current)\""
            )
            code, ref_body = _post_json(
                args.worker_url,
                "/exec_tool",
                {
                    "lease_id": lease_id,
                    "tool_call": {
                        "name": "shell_exec",
                        "arguments": {"id": "swesmith-ref-check", "command": ref_command},
                    },
                },
                timeout=120,
            )
            print(f"[smoke] ref-check HTTP {code}: {json.dumps(ref_body, ensure_ascii=False)[:1500]}")
            expected_marker = f"__SWESMITH_REF__={expected_ref}"
            if code >= 400 or not ref_body.get("ok") or expected_marker not in str(ref_body.get("observation", "")):
                print(
                    f"[ERROR] worker checked out the wrong SWE-smith ref; expected {expected_ref!r}",
                    file=sys.stderr,
                )
                raise SmokeFailure(5)

        if not args.skip_evaluate:
            code, body = _post_json(
                args.worker_url,
                "/evaluate",
                {"lease_id": lease_id, "trajectory": {"source": "swesmith-smoke"}},
                timeout=eval_timeout + 30,
            )
            print(f"[smoke] evaluate HTTP {code}: {json.dumps(body, ensure_ascii=False)[:1500]}")
            if code >= 400 or not body.get("ok"):
                raise SmokeFailure(6)
            raw_score = body.get("score")
            if (
                isinstance(raw_score, bool)
                or not isinstance(raw_score, (int, float))
                or not math.isfinite(float(raw_score))
            ):
                print(
                    "[ERROR] evaluate response has no finite numeric score",
                    file=sys.stderr,
                )
                raise SmokeFailure(7)
            score = float(raw_score)
            if score not in {0.0, 1.0}:
                print(
                    f"[ERROR] SWE-smith evaluate score must be binary, got {score}",
                    file=sys.stderr,
                )
                raise SmokeFailure(7)
            if args.expect_score is not None:
                if abs(score - args.expect_score) > 1e-9:
                    print(
                        f"[ERROR] score mismatch: expected={args.expect_score} actual={score}",
                        file=sys.stderr,
                    )
                    raise SmokeFailure(7)
            if args.expect_reason is not None:
                details = body.get("details")
                actual_reason = (
                    str(details.get("reason"))
                    if isinstance(details, dict) and details.get("reason") is not None
                    else None
                )
                if actual_reason != args.expect_reason:
                    print(
                        "[ERROR] evaluate reason mismatch: "
                        f"expected={args.expect_reason!r} actual={actual_reason!r}",
                        file=sys.stderr,
                    )
                    raise SmokeFailure(7)
    except SmokeFailure as exc:
        result_code = exc.exit_code
    finally:
        if lease_id:
            try:
                code, body = _post_json(
                    args.worker_url, "/close", {"lease_id": lease_id}, timeout=180
                )
                print(
                    f"[smoke] close HTTP {code}: "
                    f"{json.dumps(body, ensure_ascii=False)[:1000]}"
                )
                close_accepted = (
                    code < 400 and body.get("ok") and body.get("found") is True
                )
                close_confirmed = False
                close_failure = None
                if not close_accepted:
                    print(
                        "[ERROR] close was not accepted for the allocated lease",
                        file=sys.stderr,
                    )
                else:
                    close_confirmed, close_failure = _wait_for_close(
                        args.worker_url, lease_id, args.close_wait_timeout
                    )
                if close_accepted and close_confirmed:
                    print(f"[smoke] close-confirmed lease={lease_id}")
                elif close_accepted:
                    close_accepted = False
                    if close_failure is not None:
                        print(
                            "[ERROR] close cleanup failed: "
                            f"{json.dumps(close_failure, ensure_ascii=False)}",
                            file=sys.stderr,
                        )
                    else:
                        print(
                            f"[ERROR] close cleanup did not finish within "
                            f"{args.close_wait_timeout:.0f}s for lease={lease_id}",
                            file=sys.stderr,
                        )
                if result_code == 0 and not close_accepted:
                    result_code = 8
            except Exception as exc:
                print(f"[ERROR] close request failed: {exc}", file=sys.stderr)
                if result_code == 0:
                    result_code = 8

    if result_code:
        return result_code
    print("[smoke] SWE-smith worker API path OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
