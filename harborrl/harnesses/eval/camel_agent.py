"""slime ``eval_only`` evaluation harness adapter (camel-agent rollout chain).

Wraps the heavyweight slime evaluation entrypoint::

    python3 -u slime/eval_only.py --hf-checkpoint <HF目录> --load <torch-dist目录> \
        --prompt-data <jsonl> --num-rollout 0 \
        --custom-generate-function-path harborrl.rollout.entrypoint.generate \
        --custom-eval-rollout-log-function-path harborrl.misc.rollout_log.eval_rollout_log \
        --custom-config-path <rollout_config.yaml> ...

Unlike the Harbor adapters this does not talk to an external serving
endpoint: slime launches its own training/inference engines, so a run usually
needs the surrounding cluster launch scripts (GPU allocation, env vars).
``ServingSpec`` is ignored here.

Results land in ``<run_dir>/evaluations/<dataset>/step_XXXX/summary.json``
with keys like ``eval/pass_at_k``, ``eval/reward_best_at_k``,
``eval/task_count``, ``eval/response_unique_ratio`` and
``eval/response_pairwise_jaccard_distance``; per-task rows live in the
sibling ``tasks.jsonl``.

Harness-specific ``spec.extra`` keys:

- ``slime_root``: path of the slime repository (required).
- ``run_dir``: slime run directory that will contain ``evaluations/`` (required).
- ``hf_checkpoint`` / ``load`` / ``rollout_config``: wired to the
  corresponding slime flags.
- ``slime_args``: dict of extra slime CLI args, merged over the defaults.
- ``process_env``: extra env for the slime process.
"""

from __future__ import annotations

import json
from pathlib import Path

from harborrl.harnesses.eval.base import (
    BaseEvalHarness,
    EvalProgress,
    EvalResult,
    EvalRunSpec,
    TaskOutcome,
)


class CamelAgentEvalHarness(BaseEvalHarness):
    @property
    def name(self) -> str:
        return "camel_agent"

    def build_config(self, spec: EvalRunSpec) -> dict:
        extra = spec.extra
        args: dict = {
            "hf-checkpoint": extra.get("hf_checkpoint", ""),
            "load": extra.get("load", ""),
            "prompt-data": spec.dataset_path,
            "num-rollout": 0,
            "custom-generate-function-path": "harborrl.rollout.entrypoint.generate",
            "custom-eval-rollout-log-function-path": "harborrl.misc.rollout_log.eval_rollout_log",
        }
        if extra.get("rollout_config"):
            args["custom-config-path"] = extra["rollout_config"]
        args.update(extra.get("slime_args", {}))
        return args

    def launch_command(self, spec: EvalRunSpec, config_path: str) -> tuple[list[str], dict[str, str]]:
        slime_root = spec.extra.get("slime_root")
        if not slime_root:
            raise ValueError("camel_agent harness requires spec.extra['slime_root']")
        args = json.loads(Path(config_path).read_text(encoding="utf-8"))
        cmd = ["python3", "-u", str(Path(slime_root) / "eval_only.py")]
        for key, value in args.items():
            if value is None:
                continue
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{key}")
                continue
            cmd.extend([f"--{key}", str(value)])
        env = {str(k): str(v) for k, v in dict(spec.extra.get("process_env", {})).items()}
        return cmd, env

    def _summaries(self, spec: EvalRunSpec) -> list[Path]:
        run_dir = spec.extra.get("run_dir")
        if not run_dir:
            return []
        evals_dir = Path(run_dir) / "evaluations"
        if not evals_dir.is_dir():
            return []
        return sorted(evals_dir.glob("*/*/summary.json"))

    def progress(self, spec: EvalRunSpec) -> EvalProgress:
        summaries = self._summaries(spec)
        if not summaries:
            run_dir = spec.extra.get("run_dir")
            started = bool(run_dir) and Path(run_dir).is_dir()
            return EvalProgress(running=1 if started else 0)
        try:
            data = json.loads(summaries[-1].read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return EvalProgress(running=1)
        return EvalProgress(completed=int(data.get("eval/task_count") or 0), finished=True)

    def collect(self, spec: EvalRunSpec) -> EvalResult:
        summaries = self._summaries(spec)
        if not summaries:
            raise FileNotFoundError(
                f"no evaluation summary under {spec.extra.get('run_dir')}/evaluations"
            )
        summary_path = summaries[-1]
        data = json.loads(summary_path.read_text(encoding="utf-8"))

        outcomes: list[TaskOutcome] = []
        tasks_path = summary_path.parent / "tasks.jsonl"
        if tasks_path.is_file():
            for line in tasks_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                task_name = str(
                    row.get("task_id") or row.get("group") or row.get("prompt_index") or len(outcomes)
                )
                reward = row.get("best_reward_at_k", row.get("reward"))
                outcomes.append(
                    TaskOutcome(
                        task_name=task_name,
                        trial_name=task_name,
                        reward=float(reward) if reward is not None else None,
                    )
                )

        k = int(data.get("eval/k") or 1)
        return EvalResult(
            harness=self.name,
            model_name=spec.serving.model_name or str(spec.extra.get("hf_checkpoint", "")),
            job_name=spec.job_name,
            dataset=str(data.get("dataset") or spec.dataset_path),
            task_count=int(data.get("eval/task_count") or 0),
            pass_at_1=data.get("eval/pass_at_k"),
            mean_reward=data.get("eval/pass_at_k"),
            reward_best_at_k=data.get("eval/reward_best_at_k"),
            k=k,
            n_completed=int(data.get("eval/task_count") or 0),
            n_errored=0,
            exception_counts={},
            task_outcomes=outcomes,
            raw_result_path=str(summary_path),
            extras={
                "global_step": data.get("global_step"),
                "response_unique_ratio": data.get("eval/response_unique_ratio"),
                "response_pairwise_jaccard_distance": data.get(
                    "eval/response_pairwise_jaccard_distance"
                ),
            },
        )
