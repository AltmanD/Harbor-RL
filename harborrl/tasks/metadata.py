"""Task metadata helpers independent of the training backend."""
from typing import Any, Dict
from harborrl.types import TaskSpec

def _extract_task_meta(sample: Any) -> Dict[str, Any]:
    if isinstance(sample.prompt, dict):
        return sample.prompt

    metadata = sample.metadata or {}
    task_meta = metadata.get("task_meta") if isinstance(metadata, dict) else None
    if isinstance(task_meta, dict):
        return task_meta

    if isinstance(metadata, dict):
        return metadata

    return {}


def _make_task_spec(meta: Dict[str, Any]) -> TaskSpec:
    return TaskSpec(
        task_name=meta.get("task_name", "unknown"),
        task_path=meta.get("task_path", ""),
        instruction=meta.get("instruction", ""),
    )


