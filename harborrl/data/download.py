"""Download a sparse task directory from a public Git source."""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def download_sparse(repository, sparse_path, target, *, revision):
    """Clone one pinned directory without interpreting task contents."""
    target = Path(target).resolve()
    if target.exists():
        raise ValueError(f"target already exists: {target}")
    temporary = target.parent / f".{target.name}.download"
    if temporary.exists():
        raise ValueError(f"incomplete download already exists: {temporary}")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [
                "git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
                repository, str(temporary), "--branch", revision,
            ],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(temporary), "sparse-checkout", "set", sparse_path],
            check=True,
        )
        source = temporary / sparse_path
        if not source.is_dir():
            raise ValueError(f"sparse path not found: {sparse_path}")
        shutil.move(str(source), str(target))
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repository", help="public Git repository URL")
    parser.add_argument("sparse_path", help="directory inside the repository")
    parser.add_argument("target", type=Path, help="new local task-root path")
    parser.add_argument("--revision", required=True, help="branch or tag provided by the source")
    args = parser.parse_args(argv)
    if not args.revision.strip():
        parser.error("revision must be explicit; mutable defaults are unsafe")
    path = download_sparse(
        args.repository, args.sparse_path, args.target, revision=args.revision
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
