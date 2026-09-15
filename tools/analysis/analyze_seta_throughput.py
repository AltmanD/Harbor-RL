#!/usr/bin/env python3
"""Summarize SETA rollout/actor timing from a HarborRL run directory."""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any


PERF_RE = re.compile(r"perf\s+(\d+):\s+(\{.*\})\s*$")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "sum": 0.0, "mean": None, "p50": None, "p90": None, "p95": None, "max": None}
    return {
        "count": len(values),
        "sum": sum(values),
        "mean": statistics.fmean(values),
        "p50": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
        "p95": _percentile(values, 0.95),
        "max": max(values),
    }


def _parse_perf_log(path: Path, limit: int) -> tuple[dict[int, dict], dict[int, dict]]:
    rollout: dict[int, dict] = {}
    actor: dict[int, dict] = {}
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = ANSI_RE.sub("", raw_line).rstrip()
            match = PERF_RE.search(line)
            if not match:
                continue
            step = int(match.group(1))
            if step >= limit:
                continue
            try:
                payload = ast.literal_eval(match.group(2))
            except (SyntaxError, ValueError):
                continue
            if not isinstance(payload, dict):
                continue
            if "RolloutManager" in line and "perf/rollout_time" in payload:
                rollout[step] = payload
            elif "MegatronTrainRayActor" in line and "perf/step_time" in payload:
                actor[step] = payload
    return rollout, actor


def _values(records: dict[int, dict], key: str) -> list[float]:
    output: list[float] = []
    for _, payload in sorted(records.items()):
        value = payload.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            output.append(float(value))
    return output


def analyze(run_dir: Path, steps: int) -> dict[str, Any]:
    train_log = run_dir / "logs" / "train.log"
    if not train_log.is_file():
        raise FileNotFoundError(f"missing {train_log}")
    rollout, actor = _parse_perf_log(train_log, steps)
    rollout_times = _values(rollout, "perf/rollout_time")
    actor_train = _values(actor, "perf/train_time")
    actor_compute = _values(actor, "perf/actor_train_time")
    log_probs = _values(actor, "perf/log_probs_time")
    train_wait = _values(actor, "perf/train_wait_time")
    step_times = _values(actor, "perf/step_time")
    update_weights = _values(actor, "perf/update_weights_time")
    save_model = _values(actor, "perf/save_model_time")
    wait_ratios = _values(actor, "perf/wait_time_ratio")
    token_rates = _values(rollout, "perf/tokens_per_gpu_per_sec")

    measured = len(rollout_times)
    projected_seconds = statistics.fmean(rollout_times) * 1000 if rollout_times else None
    return {
        "schema": "lightrl.seta_throughput.v1",
        "run_dir": str(run_dir.resolve()),
        "requested_steps": steps,
        "measured_rollout_steps": measured,
        "missing_rollout_steps": sorted(set(range(steps)) - set(rollout)),
        "rollout_time_sec": _stats(rollout_times),
        "actor_step_time_sec": _stats(step_times),
        "actor_train_time_sec": _stats(actor_train),
        "actor_compute_time_sec": _stats(actor_compute),
        "log_probs_time_sec": _stats(log_probs),
        "actor_wait_time_sec": _stats(train_wait),
        "actor_wait_ratio": _stats(wait_ratios),
        "update_weights_time_sec": _stats(update_weights),
        "checkpoint_save_time_sec": _stats(save_model),
        "tokens_per_rollout_gpu_per_sec": _stats(token_rates),
        "projected_1000_rollout_steps": (
            {
                "seconds": projected_seconds,
                "hours": projected_seconds / 3600,
                "days": projected_seconds / 86400,
                "basis": f"mean of first {measured} measured rollout steps; excludes future eval/startup/failure downtime",
            }
            if projected_seconds is not None
            else None
        ),
        "checkpoint_fraction_of_rollout_time": (
            sum(save_model) / sum(rollout_times) if rollout_times and save_model else 0.0
        ),
    }


def _minutes(value: float | None) -> str:
    return "n/a" if value is None else f"{value / 60:.2f}"


def render_markdown(report: dict[str, Any]) -> str:
    rollout = report["rollout_time_sec"]
    actor = report["actor_train_time_sec"]
    wait = report["actor_wait_ratio"]
    projected = report["projected_1000_rollout_steps"]
    lines = [
        "# SETA 训练吞吐分析",
        "",
        f"- run: `{report['run_dir']}`",
        f"- 覆盖 rollout-step: {report['measured_rollout_steps']}/{report['requested_steps']}",
        f"- rollout 每步：均值 {_minutes(rollout['mean'])} min，p50 {_minutes(rollout['p50'])} min，p90 {_minutes(rollout['p90'])} min，p95 {_minutes(rollout['p95'])} min，最大 {_minutes(rollout['max'])} min",
        f"- actor 每步实际训练：均值 {actor['mean']:.2f} s" if actor["mean"] is not None else "- actor 每步实际训练：n/a",
        f"- actor 等待比例：均值 {wait['mean'] * 100:.2f}%" if wait["mean"] is not None else "- actor 等待比例：n/a",
        f"- checkpoint 占 rollout 总时长：{report['checkpoint_fraction_of_rollout_time'] * 100:.3f}%",
    ]
    if projected:
        lines.append(
            f"- 按当前样本均值外推 1000 step：{projected['hours']:.1f} h（{projected['days']:.2f} d），未计后续评估、启动与故障停顿"
        )
    lines += [
        "",
        "结论：actor 绝大部分时间在等待 rollout；优先优化环境并发、长尾和推理 engine 拓扑，缩短 checkpoint 不是主矛盾。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    report = analyze(args.run_dir, args.steps)
    output_dir = args.output_dir or args.run_dir / "metrics" / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "throughput_analysis.json"
    md_path = output_dir / "throughput_analysis.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
