"""Harbor ``terminus-2`` evaluation harness adapter.

Generalizes the one-off TB-style Harbor runner: builds a Harbor job config,
launches ``harbor run --config <json>``, polls ``<jobs_dir>/<job>/result.json``
and normalizes the finished job into an :class:`EvalResult`.

Harbor ``result.json`` schema relied upon here:

- top level: ``finished_at`` (non-null when done), ``stats`` with
  ``n_completed_trials`` / ``n_running_trials`` / ``n_pending_trials`` /
  ``n_errored_trials`` and ``evals``.
- ``stats.evals["<agent>__<model>__<dataset>"]``: ``n_trials``, ``n_errors``,
  ``metrics: [{"mean": ...}]``, ``reward_stats: {"reward": {"0.0": [trial...]}}``,
  ``exception_stats: {"<ExceptionClass>": [trial...]}``.
- per-trial ``<job>/<task>__<id>/result.json``: ``verifier_result.rewards.reward``,
  ``exception_info``, ``agent_result``.

Harness-specific ``spec.extra`` keys:

- ``harbor_bin``: path/name of the harbor CLI (default ``"harbor"``).
- ``model_prefix``: prefix for ``model_name`` (default ``"openai/"``).
- ``openai_api_key``: API key placeholder sent to the agent (default ``"dummy"``).
- ``no_proxy``: NO_PROXY value for the harbor process (default localhost trio).
- ``environment_type`` / ``environment_delete``: docker environment knobs.
- ``extra_docker_compose``: list of compose override files.
- ``mounts``: list of docker mount dicts (becomes ``mounts_json``).
- ``process_env``: extra env for the harbor process itself.
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

_DEFAULT_NO_PROXY = "localhost,127.0.0.1,::1"


def _job_dir(spec: EvalRunSpec) -> Path:
    return Path(spec.output_dir) / spec.job_name


def _job_result_path(spec: EvalRunSpec) -> Path:
    return _job_dir(spec) / "result.json"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _exception_name(exception_info: object) -> str | None:
    if not exception_info:
        return None
    if isinstance(exception_info, dict):
        for key in ("exception_type", "type", "class_name", "name"):
            value = exception_info.get(key)
            if value:
                return str(value)
        return str(exception_info)
    return str(exception_info)


class Terminus2EvalHarness(BaseEvalHarness):
    """Harbor runner with the ``terminus-2`` agent over an OpenAI-compatible API."""

    agent_name = "terminus-2"
    default_model_prefix = "openai/"

    @property
    def name(self) -> str:
        return "terminus2"

    def build_agent(self, spec: EvalRunSpec) -> dict:
        extra = spec.extra
        api_base = spec.serving.api_base
        prefix = extra.get("model_prefix", self.default_model_prefix)
        env = dict(spec.environment)
        env.setdefault("OPENAI_API_KEY", str(extra.get("openai_api_key", "dummy")))
        if api_base:
            env.setdefault("OPENAI_BASE_URL", api_base)
        return {
            "name": self.agent_name,
            "model_name": f"{prefix}{spec.serving.model_name}",
            "kwargs": {
                "api_base": api_base,
                "record_terminal_session": False,
                "model_info": {
                    "max_input_tokens": spec.max_input_tokens,
                    "max_output_tokens": spec.max_output_tokens,
                    "input_cost_per_token": 0,
                    "output_cost_per_token": 0,
                },
            },
            "env": env,
        }

    def build_config(self, spec: EvalRunSpec) -> dict:
        extra = spec.extra
        environment: dict = {
            "type": extra.get("environment_type", "docker"),
            "delete": bool(extra.get("environment_delete", True)),
        }
        if extra.get("extra_docker_compose"):
            environment["extra_docker_compose"] = list(extra["extra_docker_compose"])
        if extra.get("mounts"):
            environment["mounts_json"] = list(extra["mounts"])
        if spec.environment:
            environment["env"] = dict(spec.environment)

        dataset: dict = {"path": spec.dataset_path}
        if spec.task_names:
            dataset["task_names"] = list(spec.task_names)

        return {
            "job_name": spec.job_name,
            "jobs_dir": spec.output_dir,
            "n_attempts": spec.n_attempts,
            "timeout_multiplier": spec.timeout_multiplier,
            "n_concurrent_trials": spec.concurrency,
            "retry": {"max_retries": spec.max_retries},
            "environment": environment,
            "agents": [self.build_agent(spec)],
            "datasets": [dataset],
        }

    def launch_command(self, spec: EvalRunSpec, config_path: str) -> tuple[list[str], dict[str, str]]:
        extra = spec.extra
        harbor_bin = str(extra.get("harbor_bin", "harbor"))
        no_proxy = str(extra.get("no_proxy", _DEFAULT_NO_PROXY))
        env = {
            # Harbor's LiteLLM client resolves these in the host-side agent
            # process; agents[].env alone only reaches the task container.
            "OPENAI_API_KEY": str(extra.get("openai_api_key", "dummy")),
            "OPENAI_BASE_URL": spec.serving.api_base,
            "NO_PROXY": no_proxy,
            "no_proxy": no_proxy,
        }
        env.update({str(k): str(v) for k, v in dict(extra.get("process_env", {})).items()})
        return [harbor_bin, "run", "--config", config_path], env

    def progress(self, spec: EvalRunSpec) -> EvalProgress:
        path = _job_result_path(spec)
        if not path.is_file():
            return EvalProgress()
        try:
            data = _load_json(path)
        except (OSError, json.JSONDecodeError):
            return EvalProgress()
        stats = data.get("stats") or {}
        return EvalProgress(
            completed=int(stats.get("n_completed_trials") or 0),
            running=int(stats.get("n_running_trials") or 0),
            pending=int(stats.get("n_pending_trials") or 0),
            errored=int(stats.get("n_errored_trials") or 0),
            finished=bool(data.get("finished_at")),
        )

    def collect(self, spec: EvalRunSpec) -> EvalResult:
        path = _job_result_path(spec)
        if not path.is_file():
            raise FileNotFoundError(f"missing harbor job result: {path}")
        data = _load_json(path)
        stats = data.get("stats") or {}
        evals = stats.get("evals") or {}
        eval_stats: dict = next(iter(evals.values()), {}) if evals else {}

        metrics = eval_stats.get("metrics") or []
        mean = metrics[0].get("mean") if metrics and isinstance(metrics[0], dict) else None
        exception_counts = {
            str(name): len(trials)
            for name, trials in (eval_stats.get("exception_stats") or {}).items()
        }
        reward_stats = (eval_stats.get("reward_stats") or {}).get("reward") or {}

        return EvalResult(
            harness=self.name,
            model_name=spec.serving.model_name,
            job_name=spec.job_name,
            dataset=spec.dataset_path,
            task_count=int(eval_stats.get("n_trials") or 0),
            pass_at_1=mean,
            mean_reward=mean,
            reward_best_at_k=None,
            k=spec.n_attempts,
            n_completed=int(stats.get("n_completed_trials") or 0),
            n_errored=int(stats.get("n_errored_trials") or 0),
            exception_counts=exception_counts,
            task_outcomes=self._collect_trials(spec, reward_stats),
            raw_result_path=str(path),
            extras={"eval_key": next(iter(evals), "") if evals else ""},
        )

    def _collect_trials(self, spec: EvalRunSpec, reward_stats: dict) -> list[TaskOutcome]:
        job_dir = _job_dir(spec)
        outcomes: list[TaskOutcome] = []
        # Trial names recorded in the job-level reward stats, even when a
        # per-trial result.json is missing.
        reward_by_trial: dict[str, float] = {}
        for reward, trials in reward_stats.items():
            try:
                value = float(reward)
            except (TypeError, ValueError):
                continue
            for trial in trials or []:
                reward_by_trial[str(trial)] = value

        trial_names = set(reward_by_trial)
        if job_dir.is_dir():
            for child in sorted(job_dir.iterdir()):
                if child.is_dir() and (child / "result.json").is_file():
                    trial_names.add(child.name)

        for trial_name in sorted(trial_names):
            reward = reward_by_trial.get(trial_name)
            exception = None
            trial_path = job_dir / trial_name / "result.json"
            if trial_path.is_file():
                try:
                    trial = _load_json(trial_path)
                except (OSError, json.JSONDecodeError):
                    trial = {}
                verifier = (trial.get("verifier_result") or {}).get("rewards") or {}
                if reward is None and verifier.get("reward") is not None:
                    reward = float(verifier["reward"])
                exception = _exception_name(trial.get("exception_info"))
            outcomes.append(
                TaskOutcome(
                    task_name=trial_name.split("__")[0],
                    trial_name=trial_name,
                    reward=reward,
                    exception=exception,
                )
            )
        return outcomes
