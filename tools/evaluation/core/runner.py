"""Single-model evaluation orchestration.

Drives one harness run end to end:

1. ensure serving (managed: start + wait ready; external: poll only),
2. build the harness-native config and write it under ``output_dir``,
3. launch the runner as a subprocess with its log under ``output_dir``,
4. poll :meth:`BaseEvalHarness.progress` until finished,
5. collect and write the normalized ``eval_result.json``.
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from harborrl.harnesses.eval import create_eval_harness
from harborrl.harnesses.eval.base import EvalResult, EvalRunSpec, ServingSpec

from . import serving as serving_mod


def result_to_dict(result: EvalResult) -> dict:
    return dataclasses.asdict(result)


def run_eval(
    spec: EvalRunSpec,
    serving: ServingSpec,
    *,
    dry_run: bool = False,
    poll_interval_s: float = 30.0,
    serving_work_dir: str | Path | None = None,
    command_template: str = serving_mod.DEFAULT_COMMAND_TEMPLATE,
    launcher: str = "nohup",
    tmux_session: str = "eval-sglang",
    log=None,
) -> EvalResult | None:
    """Run one evaluation; returns the collected result (None on dry-run)."""
    log = log or (lambda msg: print(msg, flush=True))
    if not str(spec.output_dir).strip():
        # Keep direct ``run_eval(EvalRunSpec(...))`` callers on the same
        # categorized layout as the CLI/config path.
        from .config import default_run_dir

        spec = dataclasses.replace(
            spec, output_dir=str(default_run_dir("evaluation", spec.job_name))
        )
    harness = create_eval_harness(spec.harness)
    output_dir = Path(spec.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = harness.build_config(spec)
    config_path = output_dir / f"{spec.job_name}.config.json"
    cmd, process_env = harness.launch_command(spec, str(config_path))

    if dry_run:
        log(f"[dry-run] config ({config_path}):")
        log(json.dumps(config, indent=2, ensure_ascii=False))
        log(f"[dry-run] command: {' '.join(cmd)}")
        log(f"[dry-run] process env overlay: {json.dumps(process_env, indent=2)}")
        if serving.mode == "managed":
            log(f"[dry-run] serving command: {' '.join(serving_mod.build_command(serving, command_template))}")
        return None

    # Serving: managed servers are (re)started; external endpoints are polled.
    work = Path(serving_work_dir) if serving_work_dir else output_dir / "serving"
    if serving.mode == "managed":
        serving_mod.switch_model(
            serving, work,
            command_template=command_template, launcher=launcher, tmux_session=tmux_session,
        )
    elif serving.api_base and not serving_mod.wait_ready(serving, timeout_s=60.0):
        log(f"warning: external endpoint {serving.api_base} did not answer /models within 60s")

    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log(f"config written: {config_path}")

    env = os.environ.copy()
    env.update(process_env)
    log_path = output_dir / f"{spec.job_name}.log"
    with open(log_path, "ab") as log_file:
        proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env)
        log(f"launched pid={proc.pid}, log: {log_path}")
        while proc.poll() is None:
            progress = harness.progress(spec)
            log(
                f"progress: completed={progress.completed} running={progress.running} "
                f"pending={progress.pending} errored={progress.errored}"
            )
            if progress.finished:
                break
            time.sleep(poll_interval_s)
        if proc.poll() is None:
            # Runner marked finished but the process is still wrapping up.
            proc.wait(timeout=300)
    if proc.returncode != 0:
        log(f"warning: harness exited with code {proc.returncode}; attempting collect anyway")

    result = harness.collect(spec)
    result_path = output_dir / "eval_result.json"
    result_path.write_text(
        json.dumps(result_to_dict(result), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    log(f"eval result written: {result_path}")
    print(
        f"pass@1={result.pass_at_1} mean_reward={result.mean_reward} "
        f"completed={result.n_completed} errored={result.n_errored}",
        file=sys.stderr,
    )
    return result
