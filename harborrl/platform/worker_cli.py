from __future__ import annotations

import argparse
import logging
import os

import uvicorn

import harborrl.platform.worker_app as worker_app
from harborrl.types import TaskTimeouts
from harborrl.platform.worker_pool import WorkerPool

logger = logging.getLogger("lightrl.env.worker")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C-layer: terminal env worker server")

    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument(
        "--port", type=int, default=int(os.getenv("ENV_SERVER_PORT", "18081"))
    )

    parser.add_argument(
        "--max-tasks", type=int, default=int(os.getenv("WORKER_MAX_TASKS", "16"))
    )
    parser.add_argument(
        "--max-runs-per-task",
        type=int,
        default=int(os.getenv("WORKER_MAX_RUNS_PER_TASK", "8")),
    )
    parser.add_argument(
        "--max-total-runs",
        type=int,
        default=int(os.getenv("WORKER_MAX_TOTAL_RUNS", "128")),
        help="Global active lease ceiling across all task keys",
    )
    parser.add_argument(
        "--run-idle-ttl",
        type=int,
        default=int(os.getenv("WORKER_RUN_IDLE_TTL", "600")),
        help="Seconds before an idle RunSlot is reaped",
    )

    parser.add_argument(
        "--output-root",
        type=str,
        default=os.getenv("TBENCH_OUTPUT_ROOT", "runs/env-worker/task_outputs"),
    )

    parser.add_argument(
        "--ensure-image-timeout",
        type=float,
        default=float(os.getenv("ENSURE_IMAGE_TIMEOUT", "1200.0")),
    )
    parser.add_argument(
        "--reset-session-timeout",
        type=float,
        default=float(os.getenv("RESET_SESSION_TIMEOUT", "600.0")),
    )
    parser.add_argument(
        "--close-session-timeout",
        type=float,
        default=float(os.getenv("CLOSE_SESSION_TIMEOUT", "60.0")),
    )
    parser.add_argument(
        "--eval-timeout", type=float, default=float(os.getenv("EVAL_TIMEOUT", "600.0"))
    )
    parser.add_argument(
        "--max-concurrent-closes",
        type=int,
        default=int(os.getenv("WORKER_MAX_CONCURRENT_CLOSES", "10")),
        help="Max concurrent Docker stop operations",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s %(levelname)s %(name)s] %(message)s"
    )

    worker_app.POOL = WorkerPool(
        max_tasks=args.max_tasks,
        max_runs_per_task=args.max_runs_per_task,
        max_total_runs=args.max_total_runs,
        run_idle_ttl=args.run_idle_ttl,
        output_root=args.output_root,
        default_timeouts=TaskTimeouts(
            ensure_image=float(args.ensure_image_timeout),
            reset_session=float(args.reset_session_timeout),
            close_session=float(args.close_session_timeout),
            eval=float(args.eval_timeout),
        ),
        max_concurrent_closes=args.max_concurrent_closes,
    )

    logger.info(
        "Starting worker server on %s:%s  max_tasks=%s  max_runs_per_task=%s  max_total_runs=%s",
        args.host,
        args.port,
        args.max_tasks,
        args.max_runs_per_task,
        args.max_total_runs,
    )

    uvicorn.run(worker_app.app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
