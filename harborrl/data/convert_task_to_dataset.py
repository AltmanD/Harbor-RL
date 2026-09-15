"""Convert Terminal Bench tasks to RLLM/VERL format."""

import os
import json
import pandas as pd
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm

from harborrl.data.load_tasks import TBenchTrainingTask, load_terminal_bench_tasks

ENVIRONMENT_DIR = Path(os.getenv("DATASET_DIR", "./benchmarks/environments"))
DATASET_DIR = Path(os.getenv("BENCHMARK_DATA_DIR", "./benchmarks/datasets"))


def convert_tasks(
    tasks_dir: List[Path],
    train_split: Optional[float] = None,
    system_prompt: Optional[str] = None,
    task_names: Optional[List[str]] = None,
    test_tasks_dir: Optional[Path] = None,
    format: Optional[str] = "jsonl",
    output_dir: Optional[Path] = None,
) -> None:
    """Convert terminal bench tasks to parquet format for VERL training.

    Args:
        tasks_dir: Directory containing terminal bench tasks (or train tasks if test_tasks_dir is provided)
        train_split: Fraction of data for training (ignored if test_tasks_dir is provided)
        system_prompt: System prompt to use
        task_names: Specific task names to convert
        test_tasks_dir: Directory containing test tasks for validation set
    """

    # Load tasks
    print(f"Loading tasks from {tasks_dir}")
    tasks = []
    for dir_path in tasks_dir:
        tasks.extend(load_terminal_bench_tasks(dir_path, task_names))
    print(f"Loaded {len(tasks)} tasks")

    if output_dir is None:
        output_dir = DATASET_DIR

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare data for parquet
    data_records = []

    print(f"len tasks {len(tasks)}")

    for task in tqdm(tasks, desc="Converting tasks"):
        # find path relative to outdir
        task_path = task.task_path.relative_to(ENVIRONMENT_DIR)
        print(f"Processing task: {task.task_name} at {task_path}")
        record = {
            "task_name": task.task_name,
            "task_path": str(task_path),
            "instruction": task.instruction,
            "data_source": "terminal_bench",  # For reward_fn_key
        }

        data_records.append(record)

    # Create DataFrame
    df = pd.DataFrame(data_records)

    # Split into train and validation
    # Use train_split parameter
    if train_split is None:
        train_split = 1.0
    n_train = int(len(df) * train_split)
    train_df = df[:n_train]
    val_df = df[n_train:]

    # Save to file(s)
    train_path = output_dir / f"train.{format}"
    val_path = output_dir / f"val.{format}"

    if format == "jsonl":
        train_wrapped = pd.DataFrame({"task": train_df.to_dict(orient="records")})
        val_wrapped = pd.DataFrame({"task": val_df.to_dict(orient="records")})
        train_wrapped.to_json(
            train_path, orient="records", lines=True, force_ascii=False
        )
        val_wrapped.to_json(val_path, orient="records", lines=True, force_ascii=False)
    elif format == "parquet":
        train_df.to_parquet(train_path, index=False)
        val_df.to_parquet(val_path, index=False)

    print(f"Saved {len(train_df)} training examples to {train_path}")
    print(f"Saved {len(val_df)} validation examples to {val_path}")


def main(tasks_dir, test_tasks_dir, train_split, depth, output_dir):

    # Check if directories exist
    if not tasks_dir.exists():
        print(f"Error: {tasks_dir} directory not found")
        return

    if depth == 0:
        tasks_dirs = [tasks_dir]
    elif depth == 1:
        tasks_dirs = [p for p in tasks_dir.iterdir() if p.is_dir()]
    print(f"Found {tasks_dirs} task directories at depth {depth}")

    if output_dir is None:
        output_dir = DATASET_DIR / f"{tasks_dir.name}_convert"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Writing converted dataset to {output_dir}")

    # Convert to parquet only (with extra_info)
    convert_tasks(
        tasks_dir=tasks_dirs,
        train_split=train_split,
        system_prompt=None,
        task_names=None,
        test_tasks_dir=test_tasks_dir,
        format="jsonl",
        output_dir=output_dir,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert Terminal Bench tasks to RLLM/VERL dataset format."
    )
    parser.add_argument(
        "--tasks_dir",
        type=str,
        help="Directory containing terminal bench tasks (or train tasks if --test_tasks_dir is provided)",
    )
    parser.add_argument(
        "--test_tasks_dir",
        type=str,
        default=None,
        help="Directory containing test tasks for validation set (if provided)",
    )
    parser.add_argument(
        "--train_split",
        type=float,
        default=1.0,
        help="Fraction of data for training (ignored if --test_tasks_dir is provided)",
    )
    parser.add_argument(
        "-d",
        "--depth",
        type=int,
        default=0,
        help="Depth of directory traversal ",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for converted dataset (default: <DATASET_DIR>/<tasks_dir_name>_convert)",
    )
    args = parser.parse_args()
    main(
        tasks_dir=Path(args.tasks_dir),
        test_tasks_dir=Path(args.test_tasks_dir) if args.test_tasks_dir else None,
        train_split=args.train_split,
        depth=args.depth,
        output_dir=args.output_dir,
    )
