from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "dev"))

import prebuild_seta_worker  # noqa: E402
from prebuild_seta_worker import (  # noqa: E402
    _allocate_with_capacity_retry,
    _evaluation_failure_reason,
    _load_tasks,
    _schedule_tasks,
)


class SetaWorkerPreflightTest(unittest.TestCase):
    def test_seeded_warmup_matches_formal_rollout_order(self) -> None:
        tasks = _load_tasks(
            ROOT
            / "benchmarks"
            / "datasets"
            / "seta_env_convert"
            / "train_minus_eval12.filtered.jsonl",
            preserve_order=True,
        )
        scheduled = _schedule_tasks(
            tasks,
            shuffle_seed=42,
            skip_first=0,
            limit=8,
        )
        self.assertEqual(
            [str(task["task_name"]) for task in scheduled],
            ["654", "66", "323", "1280", "1023", "937", "307", "897"],
        )

    def test_capacity_backpressure_does_not_consume_task_attempt(self) -> None:
        responses = iter(
            [
                (429, {"ok": False, "code": "TOTAL_RUN_SLOTS_EXHAUSTED"}),
                (200, {"ok": True, "lease_id": "run-1"}),
            ]
        )
        with (
            mock.patch.object(
                prebuild_seta_worker,
                "_request_json",
                side_effect=lambda *_args, **_kwargs: next(responses),
            ),
            mock.patch.object(prebuild_seta_worker.time, "sleep"),
        ):
            code, body, retries = _allocate_with_capacity_retry(
                "http://worker", {"task_key": "task"}, timeout=60
            )
        self.assertEqual(code, 200)
        self.assertTrue(body["ok"])
        self.assertEqual(retries, 1)

    def test_accepts_real_zero_score(self) -> None:
        self.assertEqual(
            _evaluation_failure_reason(200, {"ok": True, "score": 0.0, "details": None}),
            "",
        )

    def test_rejects_infrastructure_failure_reason(self) -> None:
        self.assertEqual(
            _evaluation_failure_reason(
                200,
                {
                    "ok": True,
                    "score": 0.0,
                    "details": {"reason": "eval_parse_failed"},
                },
            ),
            "eval_parse_failed",
        )

    def test_rejects_invalid_score_and_http_failure(self) -> None:
        self.assertEqual(
            _evaluation_failure_reason(200, {"ok": True, "score": float("nan")}),
            "invalid_score",
        )
        self.assertEqual(
            _evaluation_failure_reason(503, {"ok": False}),
            "http_503",
        )


if __name__ == "__main__":
    unittest.main()
