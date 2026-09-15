"""Task ingestion, conversion, and dataset preparation."""

from typing import Any

__all__ = ["TBenchTrainingTask", "load_terminal_bench_tasks"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from .load_tasks import TBenchTrainingTask, load_terminal_bench_tasks

        return {
            "TBenchTrainingTask": TBenchTrainingTask,
            "load_terminal_bench_tasks": load_terminal_bench_tasks,
        }[name]
    raise AttributeError(name)
