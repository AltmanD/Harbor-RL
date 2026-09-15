#!/usr/bin/env python3
"""Capture enough dirty-worktree state to reproduce an already launched run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CAPTURE_PATHS = [
    "benchmarks/environments/seta_env/327/data",
    "benchmarks/datasets/seta_env_convert/eval_fixed12.jsonl",
    "benchmarks/datasets/seta_env_convert/train_minus_eval12.filtered.jsonl",
    "benchmarks/datasets/seta_env_convert/eval_fixed48_v2.jsonl",
    "benchmarks/datasets/seta_env_convert/eval_fixed48_v2.manifest.json",
    "benchmarks/datasets/seta_env_convert/train_minus_eval48_v2.filtered.jsonl",
    "configs/evaluation/seta_fixed48_v2.yaml",
    "configs/evaluation/seta_fixed12_score_v1.yaml",
    "tests/harborrl/test_seta_fixed48_gate.py",
    "tests/harborrl/test_seta_fixed48_protocol.py",
    "tools/dev/prebuild_seta_worker.py",
    "tools/dev/smoke_seta_worker.py",
    "tools/evaluation/benchmarks/seta/build_seta_fixed_eval.py",
    "tools/evaluation/benchmarks/seta/compare_seta_fixed_eval.py",
    "tools/infra/capture_formal_run_source.py",
]


def _git(*args: str) -> bytes:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expand_paths(raw_paths: list[str]) -> list[Path]:
    files: set[Path] = set()
    for raw_path in raw_paths:
        path = (REPO_ROOT / raw_path).resolve()
        try:
            path.relative_to(REPO_ROOT)
        except ValueError as exc:
            raise ValueError(f"capture path escapes repository: {raw_path}") from exc
        if not path.exists():
            raise FileNotFoundError(path)
        if path.is_dir():
            files.update(candidate for candidate in path.rglob("*") if candidate.is_file())
        else:
            files.add(path)
    return sorted(files)


def _file_record(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": path.relative_to(REPO_ROOT).as_posix(),
        "sha256": _sha256_file(path),
        "size": stat.st_size,
        "mode": oct(stat.st_mode & 0o777),
    }


def _write_capture(run_dir: Path, include_paths: list[str]) -> dict[str, object]:
    run_dir = run_dir.resolve()
    try:
        run_dir.relative_to(REPO_ROOT / "runs")
    except ValueError as exc:
        raise ValueError(f"run directory must be below {REPO_ROOT / 'runs'}: {run_dir}") from exc
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)

    output_dir = run_dir / "reproducibility" / "source_state"
    output_dir.mkdir(parents=True, exist_ok=False)

    base_commit = _git("rev-parse", "HEAD").decode().strip()
    branch = _git("rev-parse", "--abbrev-ref", "HEAD").decode().strip()
    status = _git("status", "--porcelain=v1", "--untracked-files=all")
    diff = _git("diff", "--binary", "--no-ext-diff", "HEAD", "--")
    modified_names = [
        line.strip()
        for line in _git("diff", "--name-only", "HEAD", "--").decode().splitlines()
        if line.strip()
    ]
    modified_files = [REPO_ROOT / name for name in modified_names]
    included_files = _expand_paths(include_paths)

    (output_dir / "tracked.patch").write_bytes(diff)
    (output_dir / "git-status.txt").write_bytes(status)
    (output_dir / "base-commit.txt").write_text(base_commit + "\n", encoding="utf-8")

    archive_path = output_dir / "untracked-and-ignored-inputs.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for path in included_files:
            archive.add(path, arcname=path.relative_to(REPO_ROOT).as_posix(), recursive=False)

    file_records = [_file_record(path) for path in sorted(set(modified_files + included_files))]
    fingerprint_payload = json.dumps(
        {"base_commit": base_commit, "files": file_records},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    manifest: dict[str, object] = {
        "schema_version": 1,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "base_commit": base_commit,
        "branch": branch,
        "source_fingerprint_sha256": _sha256_bytes(fingerprint_payload),
        "tracked_patch_sha256": _sha256_bytes(diff),
        "git_status_sha256": _sha256_bytes(status),
        "inputs_archive_sha256": _sha256_file(archive_path),
        "dirty_tracked_files": modified_names,
        "captured_extra_files": [path.relative_to(REPO_ROOT).as_posix() for path in included_files],
        "files": file_records,
        "run_config_sha256": _sha256_file(run_dir / "config" / "run_config.json"),
        "meta_sha256": _sha256_file(run_dir / "meta.json"),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--include", action="append", default=[])
    args = parser.parse_args()

    manifests = []
    for run_dir in args.run_dir:
        manifest = _write_capture(run_dir, [*DEFAULT_CAPTURE_PATHS, *args.include])
        manifests.append(
            {
                "run_dir": str(run_dir),
                "source_fingerprint_sha256": manifest["source_fingerprint_sha256"],
                "tracked_patch_sha256": manifest["tracked_patch_sha256"],
                "file_count": len(manifest["files"]),
            }
        )
    print(json.dumps(manifests, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
