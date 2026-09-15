"""Managed SGLang serving lifecycle for evaluation runs.

``mode="external"`` means the endpoint is already running (e.g. behind a
site-specific relay); start/stop are no-ops and :func:`wait_ready` only polls.
``mode="managed"`` starts a local SGLang server with a command template,
tracks it with a pid file, and can switch models between batch runs.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

from harborrl.harnesses.eval.base import ServingSpec

DEFAULT_COMMAND_TEMPLATE = (
    "python3 -m sglang.launch_server --model-path {model_path} "
    "--served-model-name {served_name} --host 127.0.0.1 --port {port} "
    "--tp-size {tp_size} --trust-remote-code --mem-fraction-static {mem_fraction}"
)

_PID_FILE = "sglang.pid"
_LOG_FILE = "sglang.log"


def build_command(
    serving: ServingSpec,
    command_template: str = DEFAULT_COMMAND_TEMPLATE,
) -> list[str]:
    gpu_ids = ",".join(str(g) for g in serving.gpu_ids)
    rendered = command_template.format(
        model_path=serving.model_path,
        served_name=serving.model_name,
        port=serving.port,
        tp_size=serving.tp_size,
        gpu_ids=gpu_ids,
        mem_fraction=serving.mem_fraction,
    )
    cmd = rendered.split()
    cmd.extend(serving.extra_args)
    return cmd


def start_serving(
    serving: ServingSpec,
    work_dir: str | Path,
    *,
    command_template: str = DEFAULT_COMMAND_TEMPLATE,
    launcher: str = "nohup",
    tmux_session: str = "eval-sglang",
) -> None:
    """Start SGLang in the background; no-op for ``external`` mode."""
    if serving.mode != "managed":
        return
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    pid_file = work / _PID_FILE
    if pid_file.is_file():
        # Idempotent: a previous launch is already tracked.
        return
    cmd = build_command(serving, command_template)
    env = os.environ.copy()
    if serving.gpu_ids:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in serving.gpu_ids)

    if launcher == "tmux":
        shell_cmd = " ".join(cmd) + f" >> {work / _LOG_FILE} 2>&1"
        subprocess.run(
            ["tmux", "new-session", "-d", "-s", tmux_session, shell_cmd],
            check=True,
            env=env,
        )
        pid_file.write_text(f"tmux:{tmux_session}\n", encoding="utf-8")
    else:
        log = open(work / _LOG_FILE, "ab")  # noqa: SIM115 - kept open for the child
        proc = subprocess.Popen(
            cmd, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, env=env,
        )
        pid_file.write_text(f"pid:{proc.pid}\n", encoding="utf-8")


def wait_ready(serving: ServingSpec, timeout_s: float | None = None) -> bool:
    """Poll ``<api_base>/models`` until the server answers or the timeout hits."""
    if not serving.api_base:
        return False
    deadline = time.monotonic() + (timeout_s if timeout_s is not None else serving.health_timeout_s)
    url = serving.api_base.rstrip("/") + "/models"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:  # noqa: S310 - local endpoint
                json.loads(resp.read().decode("utf-8"))
                return True
        except Exception:  # connection refused, HTTP error, bad JSON: keep polling
            time.sleep(5)
    return False


def stop_serving(serving: ServingSpec, work_dir: str | Path) -> None:
    """Stop a managed server tracked by the pid file; no-op otherwise."""
    if serving.mode != "managed":
        return
    pid_file = Path(work_dir) / _PID_FILE
    if not pid_file.is_file():
        return
    token = pid_file.read_text(encoding="utf-8").strip()
    pid_file.unlink()
    kind, _, value = token.partition(":")
    if kind == "tmux":
        subprocess.run(["tmux", "kill-session", "-t", value], check=False)
        return
    try:
        pgid = os.getpgid(int(value))
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, ValueError):
        return


def switch_model(
    serving: ServingSpec,
    work_dir: str | Path,
    *,
    command_template: str = DEFAULT_COMMAND_TEMPLATE,
    launcher: str = "nohup",
    tmux_session: str = "eval-sglang",
) -> None:
    """Restart the managed server for the model in ``serving`` and wait ready."""
    if serving.mode != "managed":
        return
    stop_serving(serving, work_dir)
    start_serving(
        serving, work_dir,
        command_template=command_template, launcher=launcher, tmux_session=tmux_session,
    )
    if not wait_ready(serving):
        raise RuntimeError(
            f"SGLang did not become ready within {serving.health_timeout_s}s "
            f"for model {serving.model_name!r}"
        )
