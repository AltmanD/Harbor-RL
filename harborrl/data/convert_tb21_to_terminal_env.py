"""Convert Terminal-Bench 2.x (tb2.1) tasks to the TB1-style env layout the
HarborRL online training runtime expects.

TB2.x task layout (per task)::

    <task>/task.toml              # [task]/[metadata]/[verifier]/[agent]/[environment]
    <task>/instruction.md
    <task>/environment/Dockerfile # build context = environment/
    <task>/environment/<assets>
    <task>/tests/test.sh          # uvx pytest --ctrf ... /tests/test_outputs.py
    <task>/tests/test_outputs.py
    <task>/solution/solve.sh

TB1-style output layout consumed by ``harborrl.environments.terminal``::

    <env_root>/<task>/task.yaml           # instruction + parser_name + timeouts
    <env_root>/<task>/docker-compose.yaml # stock template (T_BENCH_* env vars)
    <env_root>/</task>/Dockerfile         # copied; build context = task dir
    <env_root>/<task>/<assets>            # environment/ content hoisted to root
    <env_root>/<task>/tests/...           # verifier tests
    <env_root>/<task>/run-tests.sh        # adapted from tests/test.sh
    <env_root>/<task>/solution.sh         # optional, from solution/solve.sh

It also writes the rollout prompt JSONL (``task``/``metadata`` rows with
``data_source=terminal_bench``) used by ``ROLLOUT_PROMPT_DATA``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import sys
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

COMPOSE_TEMPLATE = """# You usually don't need to modify anything in this file, but you can use it to add
# more containers or configure the client container, if needed.

services:
  client:
    build:
      dockerfile: Dockerfile
    image: ${T_BENCH_TASK_DOCKER_CLIENT_IMAGE_NAME}
    container_name: ${T_BENCH_TASK_DOCKER_CLIENT_CONTAINER_NAME}
    command: [ "sh", "-c", "sleep infinity" ]
    environment:
      - TEST_DIR=${T_BENCH_TEST_DIR}
    volumes:
      - ${T_BENCH_TASK_LOGS_PATH}:${T_BENCH_CONTAINER_LOGS_PATH}
      - ${T_BENCH_TASK_AGENT_LOGS_PATH}:${T_BENCH_CONTAINER_AGENT_LOGS_PATH}
"""

TASK_YAML_TEMPLATE = """instruction: |-
{instruction}
author_name: {author_name}
author_email: {author_email}
difficulty: {difficulty}
category: {category}
tags: [{tags}]
parser_name: pytest
max_agent_timeout_sec: {agent_timeout}
max_test_timeout_sec: {test_timeout}
run_tests_in_same_shell: false
disable_asciinema: false
estimated_duration_sec:
expert_time_estimate_min: {expert_min}
junior_time_estimate_min: {junior_min}
"""


def _indent(text: str, prefix: str = "  ") -> str:
    return "\n".join(prefix + line if line.strip() else line for line in text.splitlines())


def _adapt_test_script(test_sh: str) -> str:
    """Turn a tb2.1 ``tests/test.sh`` into a TB1-style ``run-tests.sh``.

    The training runtime parses pytest stdout, so the CTRF report and the
    reward.txt side file are dropped; the pinned-dependency ``uvx`` invocation
    and the exit code are preserved.
    """

    script = test_sh
    script = re.sub(r"--ctrf\s+\S+\s*", "", script)
    script = script.replace("/tests/", "${TEST_DIR}/")
    # Drop the trailing reward.txt writer block while keeping pytest's exit
    # code as the script's exit code.
    script = re.sub(r"if\s+\[\s+\$\?\s*-eq\s*0\s+\];.*?^fi\s*$", "", script, flags=re.S | re.M)
    return script.rstrip() + "\n"


def convert_task(task_dir: Path, out_dir: Path, env_root_name: str) -> dict:
    """Convert one tb2.x task directory into the TB1-style env layout.

    Reads ``task.toml``/``instruction.md`` for metadata, writes ``task.yaml``
    plus the stock ``docker-compose.yaml``, hoists ``environment/`` (including
    its Dockerfile) to the task root so the task dir becomes the build
    context, merges ``tests/`` and adapts ``tests/test.sh`` into an executable
    ``run-tests.sh``, and copies ``solution/solve.sh`` when present.  An
    existing output task dir is replaced.

    Args:
        task_dir: tb2.x task directory containing ``task.toml``.
        out_dir: Environment root the converted task dir is created under.
        env_root_name: Name of that root, recorded in the prompt record's
            ``task_path`` so rollout configs can reference it.

    Returns:
        One rollout-prompt JSONL record (``task``/``metadata`` with
        ``data_source=terminal_bench``).

    Raises:
        FileNotFoundError: If ``task.toml``, ``environment/Dockerfile``, or
            ``tests/`` is missing.
        ValueError: If the task has no usable instruction text.
    """

    toml_path = task_dir / "task.toml"
    if not toml_path.is_file():
        raise FileNotFoundError(f"not a tb2.x task dir (missing task.toml): {task_dir}")
    spec = tomllib.loads(toml_path.read_text(encoding="utf-8"))
    task_section = spec.get("task") or {}
    metadata = spec.get("metadata") or {}
    full_name = str(task_section.get("name") or task_dir.name)
    short_name = full_name.split("/")[-1]
    instruction_path = task_dir / "instruction.md"
    if instruction_path.is_file():
        instruction = instruction_path.read_text(encoding="utf-8").strip()
    else:
        instruction = str(task_section.get("description") or "").strip()
    if not instruction:
        raise ValueError(f"empty instruction for {task_dir}")

    out_task = out_dir / short_name
    if out_task.exists():
        shutil.rmtree(out_task)
    out_task.mkdir(parents=True)

    environment = spec.get("environment") or {}
    agent = spec.get("agent") or {}
    verifier = spec.get("verifier") or {}
    keywords = task_section.get("keywords") or metadata.get("tags") or []
    task_yaml = TASK_YAML_TEMPLATE.format(
        instruction=_indent(instruction),
        author_name=metadata.get("author_name") or "anonymous",
        author_email=metadata.get("author_email") or "anonymous",
        difficulty=metadata.get("difficulty") or "medium",
        category=metadata.get("category") or "general",
        tags=", ".join(str(tag) for tag in keywords),
        agent_timeout=float(agent.get("timeout_sec") or 900.0),
        test_timeout=float(verifier.get("timeout_sec") or 900.0),
        expert_min=metadata.get("expert_time_estimate_min") or "",
        junior_min=metadata.get("junior_time_estimate_min") or "",
    )
    (out_task / "task.yaml").write_text(task_yaml, encoding="utf-8")
    (out_task / "docker-compose.yaml").write_text(COMPOSE_TEMPLATE, encoding="utf-8")

    env_dir = task_dir / "environment"
    if not (env_dir / "Dockerfile").is_file():
        raise FileNotFoundError(f"missing environment/Dockerfile in {task_dir}")
    for item in env_dir.iterdir():
        dest = out_task / item.name
        if item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest)

    tests_dir = task_dir / "tests"
    if not tests_dir.is_dir():
        raise FileNotFoundError(f"missing tests/ in {task_dir}")
    # Some tb2.1 tasks ship their own environment/tests tree; merge rather than
    # fail so the verifier tests always land in the TB1-expected location.
    shutil.copytree(tests_dir, out_task / "tests", dirs_exist_ok=True)
    test_sh = tests_dir / "test.sh"
    if test_sh.is_file():
        run_tests = _adapt_test_script(test_sh.read_text(encoding="utf-8"))
    else:
        run_tests = (
            "#!/bin/bash\nset -e\npython3 -m pytest ${TEST_DIR}/test_outputs.py -rA\n"
        )
    run_tests_path = out_task / "run-tests.sh"
    run_tests_path.write_text(run_tests, encoding="utf-8")
    run_tests_path.chmod(0o755)

    solve_sh = task_dir / "solution" / "solve.sh"
    if solve_sh.is_file():
        shutil.copy2(solve_sh, out_task / "solution.sh")

    return {
        "task": [{"content": instruction}],
        "metadata": {
            "task_name": short_name,
            "task_path": f"{env_root_name}/{short_name}",
            "data_source": "terminal_bench",
            "instruction": instruction,
            "tb21_full_name": full_name,
            "tb21_docker_image": environment.get("docker_image"),
            "tb21_allow_internet": bool(environment.get("allow_internet", False)),
        },
    }


def main() -> None:
    """CLI entry: convert a tb2.x task tree into a TB1-style env root.

    Iterates task dirs under ``--tasks-dir`` (optionally filtered by
    ``--tasks``/``--max-tasks``), converts each with :func:`convert_task`,
    collecting per-task failures into a skipped list instead of aborting, and
    writes the rollout prompt JSONL plus a ``*.manifest.json`` sidecar.
    Exits non-zero when no task was converted.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-dir", required=True, type=Path)
    parser.add_argument("--output-env-dir", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--tasks", default=None, help="Comma-separated task subset (short names).")
    parser.add_argument("--max-tasks", type=int, default=None)
    args = parser.parse_args()

    env_root_name = args.output_env_dir.name
    selected = None
    if args.tasks:
        selected = {item.strip() for item in args.tasks.split(",") if item.strip()}

    records: list[dict] = []
    skipped: list[str] = []
    task_dirs = sorted(
        path for path in args.tasks_dir.iterdir() if path.is_dir() and (path / "task.toml").is_file()
    )
    for task_dir in task_dirs:
        short = task_dir.name
        if selected is not None and short not in selected:
            continue
        try:
            records.append(convert_task(task_dir, args.output_env_dir, env_root_name))
        except (OSError, ValueError, FileNotFoundError) as exc:
            skipped.append(f"{short}: {exc}")
        if args.max_tasks is not None and len(records) >= args.max_tasks:
            break

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    manifest = {
        "tasks_dir": str(args.tasks_dir),
        "output_env_dir": str(args.output_env_dir),
        "output_jsonl": str(args.output_jsonl),
        "converted": len(records),
        "skipped": skipped,
    }
    args.output_jsonl.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False))
    if not records:
        sys.exit("no tasks converted")


if __name__ == "__main__":
    main()
