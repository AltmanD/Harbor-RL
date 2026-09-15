"""Shared dataclasses and abstract base for evaluation harness adapters.

An evaluation harness adapter wraps an *external* evaluation runner (for
example the Harbor CLI or the slime ``eval_only`` entrypoint) behind a small
synchronous interface:

1. :meth:`BaseEvalHarness.build_config` renders the harness-native config
   (e.g. a Harbor job config dict).
2. :meth:`BaseEvalHarness.launch_command` returns the command line plus the
   environment the *launching process* needs.
3. :meth:`BaseEvalHarness.progress` polls the runner's on-disk state.
4. :meth:`BaseEvalHarness.collect` normalizes the finished run into an
   :class:`EvalResult`.

Serving (e.g. an SGLang server) is deliberately *not* managed by adapters;
:class:`ServingSpec` only describes where the model is reachable and, for
``mode="managed"``, how a tool layer should start it.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field


@dataclass
class ServingSpec:
    """How the model under evaluation is served."""

    mode: str = "external"  # "external" (already running) | "managed" (tool layer starts it)
    api_base: str = ""  # OpenAI-compatible base URL, e.g. "http://127.0.0.1:30000/v1"
    model_path: str = ""  # local HF checkpoint path (managed mode)
    model_name: str = ""  # served model name
    port: int = 30000
    gpu_ids: list[int] = field(default_factory=list)
    tp_size: int = 1
    mem_fraction: float = 0.70
    extra_args: list[str] = field(default_factory=list)
    health_timeout_s: float = 900.0


@dataclass
class EvalRunSpec:
    """One evaluation run of one model on one dataset with one harness."""

    harness: str
    job_name: str
    dataset_path: str
    task_names: list[str] | None = None  # None -> whole dataset
    output_dir: str = ""  # resolved by the evaluator to runs/evaluation/<job> when omitted
    n_attempts: int = 1
    concurrency: int = 4
    max_retries: int = 1
    timeout_multiplier: float = 1.0
    max_input_tokens: int = 8192
    max_output_tokens: int = 8192
    environment: dict[str, str] = field(default_factory=dict)  # passed through to the task env
    serving: ServingSpec = field(default_factory=ServingSpec)
    extra: dict = field(default_factory=dict)  # escape hatch for harness-specific options


@dataclass
class EvalProgress:
    completed: int = 0
    running: int = 0
    pending: int = 0
    errored: int = 0
    finished: bool = False


@dataclass
class TaskOutcome:
    task_name: str
    trial_name: str
    reward: float | None = None
    exception: str | None = None


@dataclass
class EvalResult:
    """Normalized evaluation result across harnesses."""

    harness: str
    model_name: str
    job_name: str
    dataset: str
    task_count: int = 0
    pass_at_1: float | None = None
    mean_reward: float | None = None
    reward_best_at_k: float | None = None
    k: int = 1
    n_completed: int = 0
    n_errored: int = 0
    exception_counts: dict[str, int] = field(default_factory=dict)
    task_outcomes: list[TaskOutcome] = field(default_factory=list)
    raw_result_path: str = ""
    extras: dict = field(default_factory=dict)


class BaseEvalHarness(abc.ABC):
    """Synchronous adapter between the tool layer and an evaluation runner."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Canonical harness name."""

    @abc.abstractmethod
    def build_config(self, spec: EvalRunSpec) -> dict:
        """Render the harness-native configuration for ``spec``."""

    @abc.abstractmethod
    def launch_command(self, spec: EvalRunSpec, config_path: str) -> tuple[list[str], dict[str, str]]:
        """Return ``(argv, process_env)``; ``process_env`` overlays ``os.environ``."""

    @abc.abstractmethod
    def progress(self, spec: EvalRunSpec) -> EvalProgress:
        """Poll on-disk state; must tolerate a not-yet-started run."""

    @abc.abstractmethod
    def collect(self, spec: EvalRunSpec) -> EvalResult:
        """Normalize a finished run into an :class:`EvalResult`."""
