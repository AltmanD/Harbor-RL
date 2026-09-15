# Script to download and prepare currently supported datasets for terminal agent training

import os
import shutil
import subprocess
from pathlib import Path

DATASET_DIR = Path(os.getenv("DATASET_DIR", "./benchmarks/environments"))


def _download_github_folder(
    repo_url, sparse_path, target_dir, branch="main", temp_suffix="temp"
):
    """
    General function to download a specific folder from a GitHub repository.

    Args:
        repo_url: GitHub repository URL (.git)
        sparse_path: Path within the repo to download
        target_dir: Local destination directory
        branch: Git branch to checkout (default: "main")
        temp_suffix: Suffix for temporary directory name
    """
    if target_dir.exists():
        print(f"Dataset already exists at {target_dir}. Skipping download.")
        return

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    temp_dir = DATASET_DIR / f"temp_{temp_suffix}"

    try:
        # Clone with sparse checkout
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--filter=blob:none",
                "--sparse",
                repo_url,
                str(temp_dir),
                "-b",
                branch,
            ],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(temp_dir), "sparse-checkout", "set", sparse_path],
            check=True,
        )

        # Move downloaded folder to target location
        shutil.move(str(temp_dir / sparse_path), str(target_dir))
        print(f"Successfully downloaded to {target_dir}")
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def download_seta_env():
    url = "https://github.com/camel-ai/seta-env.git"
    target_dir = DATASET_DIR / "seta_env"
    _download_github_folder(
        url, "Dataset", target_dir, branch="main", temp_suffix="seta_env"
    )


def download_tbench_core():
    url = "https://github.com/laude-institute/terminal-bench.git"
    target_dir = DATASET_DIR / "tbench_core"
    _download_github_folder(
        url, "tasks", target_dir, branch="main", temp_suffix="tbench_core"
    )


def download_tbench_test():
    url = "https://github.com/laude-institute/terminal-bench.git"
    target_dir = DATASET_DIR / "tbench_test"
    _download_github_folder(
        url,
        "tasks",
        target_dir,
        branch="dataset/terminal-bench-core/v0.1.x",
        temp_suffix="tbench_test",
    )


def download_tbench_adapted():
    url = "https://github.com/laude-institute/terminal-bench-datasets.git"
    raw_dir = DATASET_DIR / "tbench_adapted_raw"
    target_dir = DATASET_DIR / "tbench_adapted"

    if target_dir.exists():
        print(f"Dataset already exists at {target_dir}. Skipping download.")
        return

    # Download the raw datasets
    _download_github_folder(
        url, "datasets", raw_dir, branch="main", temp_suffix="tbench_adapted"
    )

    # Create target directory
    target_dir.mkdir(parents=True, exist_ok=True)

    # Create symbolic links with prefixed names
    for subfolder in raw_dir.iterdir():
        if subfolder.is_dir():
            subfolder_name = subfolder.name
            for task_folder in subfolder.iterdir():
                if task_folder.is_dir():
                    task_name = task_folder.name
                    prefixed_name = f"{subfolder_name}_{task_name}"
                    symlink_path = target_dir / prefixed_name
                    symlink_path.symlink_to(task_folder, target_is_directory=True)

    print(f"Successfully created symlinks in {target_dir}")


def download_data(ds_name):
    DATASET_DOWNLOADERS = {
        "seta_env": download_seta_env,
        "tbench_core": download_tbench_core,
        "tbench_test": download_tbench_test,
        "tbench_adapted": download_tbench_adapted,
    }
    if ds_name not in DATASET_DOWNLOADERS:
        raise ValueError(f"Dataset {ds_name} is not supported.")
    DATASET_DOWNLOADERS[ds_name]()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        download_data(sys.argv[1])
    else:
        print("Available datasets: seta_env, tbench_core, tbench_test, tbench_adapted")
        print("Usage: python download_data.py <dataset_name>")
