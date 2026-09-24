"""Probe a Harbor task with native Docker verifier semantics.

The legacy TerminalEnv comparison path is intentionally not part of the public
release.  This probe runs the untouched task verifier twice in isolated
containers and requires the same scalar reward, which keeps the check aligned
with the Native Harbor Runner boundary without importing the old environment
stack.
"""
from __future__ import annotations

import asyncio
import json
import subprocess
import tempfile
from pathlib import Path
from uuid import uuid4

from .inspector import inspect, tree_digest

PROFILE = "native-docker-v1"


def parse_reward(raw: str) -> dict:
    value = json.loads(raw)
    if not isinstance(value, dict) or "reward" not in value:
        raise ValueError("verifier must emit a JSON object with a reward key")
    return value


def docker(*args, timeout=900):
    return subprocess.run(
        ["docker", *map(str, args)],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=True,
    ).stdout


async def _run_verifier(source: Path, work: Path) -> float:
    uid = uuid4().hex
    image, container = "harborrl-probe:" + uid, "harborrl-probe-" + uid
    try:
        await asyncio.to_thread(docker, "build", "-t", image, source / "environment")
        run_args = ["run", "-d", "--name", container]
        environment = _task_environment(source)
        if "cpus" in environment:
            run_args += ["--cpus", str(environment["cpus"])]
        if "memory_mb" in environment:
            run_args += ["--memory", f"{environment['memory_mb']}m"]
        if environment.get("allow_internet") is False:
            run_args += ["--network", "none"]
        await asyncio.to_thread(
            docker, *run_args, "--entrypoint", "sh", image, "-c", "sleep infinity"
        )
        await asyncio.to_thread(
            docker, "exec", "-u", "root", container, "mkdir", "-p", "/tests"
        )
        await asyncio.to_thread(
            docker, "cp", str(source / "tests") + "/.", f"{container}:/tests"
        )
        try:
            await asyncio.to_thread(
                docker,
                "exec", "-u", "root", container, "sh", "-c",
                "mkdir -p /logs/verifier; rm -f /logs/verifier/reward.txt; bash /tests/test.sh",
            )
        except subprocess.CalledProcessError:
            pass
        raw = await asyncio.to_thread(
            docker, "exec", container, "cat", "/logs/verifier/reward.txt"
        )
        return parse_reward(raw)["reward"]
    finally:
        for args in (("rm", "-f", container), ("image", "rm", image)):
            try:
                await asyncio.to_thread(docker, *args, timeout=60)
            except (subprocess.SubprocessError, OSError):
                pass


def _task_environment(source: Path):
    import tomllib

    return tomllib.loads((source / "task.toml").read_text()).get("environment", {})


async def probe(source: Path, dataset: str, work: Path) -> dict:
    report = inspect(source, dataset)
    if report.status != "NEEDS_PROBE":
        raise ValueError(f"{report.status}: {report.reasons}")
    receipt = {
        "source_digest": report.ref.source_digest,
        "profile": PROFILE,
        "execution_backend": "native_docker",
        "success": False,
    }
    try:
        with tempfile.TemporaryDirectory(prefix="harbor-probe-", dir=work) as temp:
            first = await _run_verifier(source, Path(temp))
            second = await _run_verifier(source, Path(temp))
        receipt["native_reward"] = first
        receipt["repeat_reward"] = second
        if tree_digest(source) != receipt["source_digest"]:
            raise ValueError("source changed during probe")
        receipt["success"] = first == second
        if not receipt["success"]:
            receipt["error"] = "verifier reward is not deterministic"
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, subprocess.CalledProcessError):
            receipt["stderr"] = exc.stderr
    return receipt


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    result = asyncio.run(
        probe(args.task.resolve(), args.dataset, args.work_dir.resolve())
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump({args.task.name: result}, stream, indent=2)
    if not result["success"]:
        raise SystemExit(result.get("error", "probe failed"))


if __name__ == "__main__":
    main()
