"""Compare an untouched Harbor verifier with execution through TerminalEnv.

Run in the worker Python environment. All containers/images created here have
unique IDs; cleanup never prunes shared Docker resources.
"""

import argparse
import asyncio
import json
import os
from pathlib import Path
import subprocess
import tempfile
from uuid import uuid4
from .inspector import inspect, tree_digest, PROFILE, tomllib
from .materializer import prepare_probe
from .receipt import parse_reward


def docker(*args, timeout=900):
    return subprocess.run(
        ["docker", *map(str, args)],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=True,
    ).stdout


async def probe(source: Path, dataset: str, work: Path) -> dict:
    from harborrl.environments.terminal.runtime import TerminalEnv
    from harborrl.types import TaskSpec, RunContext, TaskTimeouts

    report = inspect(source, dataset)
    if report.status != "NEEDS_PROBE":
        raise ValueError(f"{report.status}: {report.reasons}")
    receipt = {
        "source_digest": report.ref.source_digest,
        "profile": PROFILE,
        "execution_backend": "terminal_env",
        "success": False,
    }
    uid = uuid4().hex
    image, container = "harborrl-probe:" + uid, "harborrl-probe-" + uid
    env = TerminalEnv(harbor_probe=True)
    old_dataset_dir = os.environ.get("DATASET_DIR")
    try:
        # Native layout keeps the original environment/ context and /tests path.
        await asyncio.to_thread(docker, "build", "-t", image, source / "environment")
        environment = tomllib.loads((source / "task.toml").read_text()).get(
            "environment", {}
        )
        run_args = ["run", "-d", "--name", container]
        if "cpus" in environment:
            run_args += ["--cpus", str(environment["cpus"])]
        if "memory_mb" in environment:
            run_args += ["--memory", f"{environment['memory_mb']}m"]
        if environment.get("allow_internet") is False:
            run_args += ["--network", "none"]
        await asyncio.to_thread(
            docker, *run_args, "--entrypoint", "sh", image, "-c", "sleep infinity"
        )
        await asyncio.to_thread(docker, "cp", source / "tests", f"{container}:/tests")
        try:
            await asyncio.to_thread(
                docker,
                "exec",
                "-u",
                "root",
                container,
                "sh",
                "-c",
                "mkdir -p /logs/verifier; rm -f /logs/verifier/reward.txt; bash /tests/test.sh",
            )
        except subprocess.CalledProcessError:
            # A task failure may still have a valid scalar receipt.
            pass
        raw = await asyncio.to_thread(
            docker, "exec", container, "cat", "/logs/verifier/reward.txt"
        )
        receipt["native_reward"] = parse_reward(raw)["raw_reward"]
        with tempfile.TemporaryDirectory(prefix="harbor-probe-", dir=work) as temp:
            row = prepare_probe(source, dataset, Path(temp) / "tasks")
            meta = row["metadata"]
            os.environ["DATASET_DIR"] = str(Path(temp) / "tasks")
            await env.reset(
                task_meta=meta,
                task_spec=TaskSpec(
                    meta["task_name"], meta["task_path"], meta["instruction"]
                ),
                run_ctx=RunContext(uid, 0, 0, Path(temp) / "logs"),
                timeouts=TaskTimeouts(),
            )
            receipt["materialized_reward"] = await env.evaluate()
            await env.close()
        if tree_digest(source) != receipt["source_digest"]:
            raise ValueError("source changed during probe")
        receipt["success"] = receipt["native_reward"] == receipt["materialized_reward"]
        if not receipt["success"]:
            receipt["error"] = "verifier reward mismatch"
    except Exception as exc:
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, subprocess.CalledProcessError):
            receipt["stderr"] = exc.stderr
    finally:
        try:
            await env.close()
        finally:
            for args in (("rm", "-f", container), ("image", "rm", image)):
                try:
                    await asyncio.to_thread(docker, *args, timeout=60)
                except (subprocess.SubprocessError, OSError):
                    pass
    if old_dataset_dir is None:
        os.environ.pop("DATASET_DIR", None)
    else:
        os.environ["DATASET_DIR"] = old_dataset_dir
    return receipt


def main():
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
    with args.output.open("x") as f:
        json.dump({args.task.name: result}, f, indent=2)
    if not result["success"]:
        raise SystemExit(result.get("error", "probe failed"))


if __name__ == "__main__":
    main()
