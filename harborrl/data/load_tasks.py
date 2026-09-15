import json
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel
import yaml


class TBenchTrainingTask(BaseModel):
    """Data model for a task which follows the same format as terminal-bench."""

    task_name: str
    task_path: Path
    instruction: str
    # test_weights: dict
    # dockerfile_contents: str
    # py_test_file_contents: str
    # max_test_timeout_sec: int = 300  # Default timeout
    # additional_files: Optional[dict] = None  # Maps file paths to contents


def load_terminal_bench_tasks(
    tasks_dir: Path,
    task_names: Optional[List[str]] = None,
) -> List[TBenchTrainingTask]:
    if task_names is None:
        print(f"tasks_dir {tasks_dir}")
        import os

        print(f"len {len(os.listdir(str(tasks_dir)))}")
        task_names = [p.name for p in tasks_dir.iterdir() if p.is_dir()]

    print(f"task names {task_names}")
    tasks = []
    from tqdm import tqdm

    for task_name in tqdm(task_names):
        try:
            task_path = tasks_dir / task_name
            task_yaml = task_path / "task.yaml"

            if not task_yaml.exists():
                raise FileNotFoundError(f"Task YAML file not found: {task_yaml}")

            with open(task_yaml, "r", encoding="utf-8") as f:
                task_data = yaml.safe_load(f)

            instruction = task_data.get("instruction")
            if not instruction:
                raise ValueError(f"Instruction not found in task YAML: {task_yaml}")

            # # Get max test timeout if specified
            # max_test_timeout_sec = task_data.get("max_test_timeout_sec", 300)

            # # Load test weights
            # test_weights_path = task_path / "test_weights.json"
            # if test_weights_path.exists():
            #     with open(test_weights_path, "r", encoding="utf-8") as f:
            #         test_weights = json.load(f)
            # else:
            #     test_weights = {"default": 1.0}

            # # Load Dockerfile
            # dockerfile_path = task_path / "Dockerfile"
            # if not dockerfile_path.exists():
            #     continue
            # with open(dockerfile_path, "r", encoding="utf-8") as f:
            #     dockerfile_contents = f.read()

            # # Load Python test file if it exists
            # py_test_file_path = task_path / "tests" / "test_outputs.py"
            # if not py_test_file_path.exists():
            #     # raise FileNotFoundError(f"Python test file not found: {py_test_file_path}")
            #     continue
            # with open(py_test_file_path, "r", encoding="utf-8") as f:
            #     py_test_file_contents = f.read()
            # if not py_test_file_contents:
            #     # raise ValueError(f"Python test file is empty for task: {task_name}")
            #     continue

            # # Load additional files if they exist
            # additional_files = {}
            # # List all files in the task directory (excluding standard files)
            # standard_files = {'Dockerfile', 'task.yaml', 'test_weights.json'}
            # standard_dirs = {'tests', '__pycache__'}

            # for item in task_path.iterdir():
            #     if item.is_file() and item.name not in standard_files:
            #         # Read the file and store with relative path
            #         rel_path = item.relative_to(task_path)
            #         with open(item, "r", encoding="utf-8") as f:
            #             additional_files[str(rel_path)] = f.read()
            #     elif item.is_dir() and item.name not in standard_dirs:
            #         # Recursively read files from subdirectories
            #         for subfile in item.rglob("*"):
            #             if subfile.is_file():
            #                 rel_path = subfile.relative_to(task_path)
            #                 try:
            #                     with open(subfile, "r", encoding="utf-8") as f:
            #                         additional_files[str(rel_path)] = f.read()
            #                 except UnicodeDecodeError:
            #                     # Skip binary files for now
            #                     pass

            tasks.append(
                TBenchTrainingTask(
                    task_name=task_name,
                    task_path=task_path,
                    instruction=instruction,
                    # test_weights=None,
                    # dockerfile_contents=None,
                    # py_test_file_contents=None,
                    # max_test_timeout_sec=None,
                    # additional_files=None,
                )
            )
        except Exception as e:
            print(f"Error loading task {task_name}: {e}")
            continue
    return tasks
