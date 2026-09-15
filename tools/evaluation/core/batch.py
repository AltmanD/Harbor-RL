"""Serial batch evaluation over multiple checkpoints.

Batch config schema::

    defaults:          # any single-run config keys (harness, run, serving, ...)
      harness: terminus-2
      ...
    serving:           # shared managed-serving knobs (port, tp_size, template)
      mode: managed
      port: 30000
    models:
      - name: ckpt-a
        model_path: /path/to/ckpt-a
        model_name: served-name-a
        overrides: {}  # deep-merged over defaults for this model only
    report:
      output: /path/to/compare   # writes compare.md / compare.csv

Each model gets its own ``output_dir`` (``<defaults.output_dir>/<name>``) so
per-run ``eval_result.json`` files never collide. If ``defaults.output_dir`` is
omitted, the batch defaults to ``runs/evaluation/<job_name>/``. Failures are
recorded and the batch continues with the next model.
"""

from __future__ import annotations

import json
from pathlib import Path

from harborrl.harnesses.eval.base import EvalResult

from .config import build_specs, default_run_dir, deep_merge
from .runner import run_eval
from .serving import DEFAULT_COMMAND_TEMPLATE


def run_batch(
    config: dict,
    *,
    dry_run: bool = False,
    poll_interval_s: float = 30.0,
    log=None,
) -> tuple[list[EvalResult], dict[str, str]]:
    """Run every model in ``config["models"]``; return (results, failures)."""
    log = log or (lambda msg: print(msg, flush=True))
    defaults = dict(config.get("defaults") or {})
    shared_serving = dict(config.get("serving") or {})
    models = config.get("models") or []
    if not models:
        raise ValueError("batch config has no models")

    serving_cfg = dict(defaults.get("serving") or {})
    serving_cfg = deep_merge(serving_cfg, shared_serving)

    results: list[EvalResult] = []
    failures: dict[str, str] = {}
    for model in models:
        name = str(model["name"])
        model_cfg = deep_merge(defaults, dict(model.get("overrides") or {}))
        model_serving = deep_merge(
            serving_cfg, dict(model_cfg.get("serving") or {})
        )
        model_serving["model_path"] = str(model.get("model_path", model_serving.get("model_path", "")))
        model_serving["model_name"] = str(model.get("model_name", name))
        model_cfg["serving"] = model_serving
        batch_job_name = str(model_cfg.get("job_name") or defaults.get("job_name") or "eval")
        model_cfg["job_name"] = f"{batch_job_name}-{name}"
        base_output = str(
            defaults.get("output_dir")
            or model_cfg.get("output_dir")
            or default_run_dir("evaluation", batch_job_name)
        )
        model_cfg["output_dir"] = str(Path(base_output) / name)

        log(f"=== model {name} ===")
        try:
            spec, serving = build_specs(model_cfg)
            result = run_eval(
                spec,
                serving,
                dry_run=dry_run,
                poll_interval_s=poll_interval_s,
                command_template=str(serving_cfg.get("command_template") or DEFAULT_COMMAND_TEMPLATE),
                launcher=str(serving_cfg.get("launcher", "nohup")),
                tmux_session=str(serving_cfg.get("tmux_session", "eval-sglang")),
                log=log,
            )
            if result is not None:
                results.append(result)
        except Exception as exc:  # keep the batch going
            failures[name] = f"{type(exc).__name__}: {exc}"
            log(f"model {name} FAILED: {failures[name]}")

    report_cfg = dict(config.get("report") or {})
    if report_cfg.get("output") and not dry_run:
        from .report import write_report

        write_report(results, str(report_cfg["output"]))
        log(f"report written: {report_cfg['output']}.md / .csv")

    if failures:
        log(f"batch finished with {len(failures)} failed model(s): {sorted(failures)}")
    return results, failures


def results_from_output_dirs(config: dict) -> list[dict]:
    """Load per-model ``eval_result.json`` files produced by a batch run."""
    defaults = dict(config.get("defaults") or {})
    batch_job_name = str(defaults.get("job_name") or "eval")
    base_output = Path(
        str(defaults.get("output_dir") or default_run_dir("evaluation", batch_job_name))
    )
    loaded = []
    for model in config.get("models") or []:
        path = base_output / str(model["name"]) / "eval_result.json"
        if path.is_file():
            loaded.append(json.loads(path.read_text(encoding="utf-8")))
    return loaded
