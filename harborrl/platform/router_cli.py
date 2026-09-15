from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import uvicorn

import harborrl.platform.router_app as router_app
from harborrl.platform.router import Router, _format_error, _parse_worker_urls_text

logger = logging.getLogger("lightrl.env.router")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="B-layer: terminal env router server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument(
        "--port", type=int, default=int(os.getenv("ROUTER_PORT", "18080"))
    )
    parser.add_argument(
        "--workers",
        type=str,
        default=os.getenv("WORKER_URLS", ""),
        help="Comma-separated worker URLs, e.g. http://w0:18081,http://w1:18081",
    )
    parser.add_argument(
        "--workers-file",
        type=str,
        default=os.getenv("WORKER_URLS_FILE", ""),
        help="Optional file containing worker URLs. The router hot-reloads it periodically.",
    )
    parser.add_argument(
        "--workers-reload-interval",
        type=float,
        default=float(os.getenv("WORKER_URLS_RELOAD_INTERVAL", "0")),
        help="Seconds between worker URL file reload checks. Set 0 to disable.",
    )
    parser.add_argument(
        "--forward-timeout",
        type=float,
        default=float(os.getenv("ROUTER_FORWARD_TIMEOUT", "1800.0")),  # 600→1800s for reset endpoint
        help="HTTP timeout (seconds) when forwarding to a worker",
    )
    parser.add_argument(
        "--forward-retries",
        type=int,
        default=int(os.getenv("ROUTER_FORWARD_RETRIES", "1")),
        help="Retries for transient worker connection errors",
    )
    parser.add_argument(
        "--forward-retry-backoff",
        type=float,
        default=float(os.getenv("ROUTER_FORWARD_RETRY_BACKOFF", "2.0")),  # 0.2→2.0s exponential backoff base
        help="Exponential backoff base (seconds) between worker retries",
    )
    parser.add_argument(
        "--pressure-cooldown",
        type=float,
        default=float(os.getenv("ROUTER_PRESSURE_COOLDOWN", "60.0")),
        help="Seconds to avoid a worker after pressure/unreachable allocate failures",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s %(levelname)s %(name)s] %(message)s"
    )

    worker_urls = [u.strip().rstrip("/") for u in args.workers.split(",") if u.strip()]
    if not worker_urls and args.workers_file:
        try:
            worker_urls = _parse_worker_urls_text(
                Path(args.workers_file).read_text(encoding="utf-8")
            )
        except OSError as exc:
            raise SystemExit(
                f"ERROR: failed to read --workers-file {args.workers_file}: {_format_error(exc)}"
            ) from exc
    if not worker_urls:
        raise SystemExit(
            "ERROR: --workers, WORKER_URLS env, or --workers-file must list at least one worker URL"
        )

    router_app.ROUTER = Router(
        worker_urls=worker_urls,
        forward_timeout=args.forward_timeout,
        forward_retries=args.forward_retries,
        forward_retry_backoff=args.forward_retry_backoff,
        pressure_cooldown=args.pressure_cooldown,
        workers_file=args.workers_file,
        workers_reload_interval=args.workers_reload_interval,
    )
    logger.info(
        "Starting router on %s:%s  workers=%s  workers_file=%s  workers_reload_interval=%s  forward_timeout=%s  forward_retries=%s  forward_retry_backoff=%s  pressure_cooldown=%s",
        args.host,
        args.port,
        worker_urls,
        args.workers_file,
        args.workers_reload_interval,
        args.forward_timeout,
        args.forward_retries,
        args.forward_retry_backoff,
        args.pressure_cooldown,
    )

    uvicorn.run(router_app.app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
