"""Publish a task without changing its verifier or Docker build context."""

import json
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib
import yaml

from .inspector import PROFILE, REWARD_PROFILE, inspect, tree_digest

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


def materialize(source: Path, dataset: str, root: Path, receipt: dict) -> dict:
    return _materialize(source, dataset, root, receipt, probe=False)


def prepare_probe(source: Path, dataset: str, root: Path) -> dict:
    """Build a candidate only in a caller-owned temporary probe directory."""
    return _materialize(source, dataset, root, None, probe=True)


def _materialize(source, dataset, root, receipt, *, probe):
    source, root = Path(source), Path(root).resolve()
    if root == source.resolve() or source.resolve() in root.parents:
        raise ValueError("output must not be inside the source task")
    report = inspect(source, dataset, receipt)
    if report.status != "SUPPORTED" and not (probe and report.status == "NEEDS_PROBE"):
        raise ValueError(f"{report.status}: {report.reasons}")
    import hashlib

    identity = hashlib.sha256(f"{dataset}/{report.ref.task}".encode()).hexdigest()[:20]
    name = f"{identity}-{report.ref.source_digest[:16]}"
    root.mkdir(parents=True, exist_ok=True)
    target = root / name
    if target.exists():
        raise FileExistsError(target)
    temp = Path(tempfile.mkdtemp(prefix=".materialize-", dir=root))
    try:
        shutil.copytree(source / "environment", temp / "environment")
        shutil.copytree(source / "tests", temp / "tests")
        # Terminal-Bench versions either flatten tests/ into TEST_DIR or retain
        # the directory. Preserve the verifier's original /tests layout in both.
        (temp / "run-tests.sh").write_text(
            "#!/bin/bash\nset -e\nmkdir -p /tests /logs/verifier\n"
            'if [ -f "${TEST_DIR}/tests/test.sh" ]; then\n'
            '  cp -a "${TEST_DIR}/tests/." /tests/\n'
            'elif [ "${TEST_DIR}" != /tests ]; then\n'
            '  cp -a "${TEST_DIR}/." /tests/\n'
            'fi\n'
            "rm -f /logs/verifier/reward.txt /logs/verifier/reward.json\n"
            "exec bash /tests/test.sh\n"
        )
        (temp / "run-tests.sh").chmod(0o755)
        spec = tomllib.loads((source / "task.toml").read_text())
        environment = spec.get("environment", {})
        compose = yaml.safe_load(COMPOSE_TEMPLATE)
        compose["services"]["client"]["build"] = {
            "context": "environment",
            "dockerfile": "Dockerfile",
        }
        client = compose["services"]["client"]
        if "cpus" in environment:
            client["cpus"] = environment["cpus"]
        if "memory_mb" in environment:
            client["mem_limit"] = f"{environment['memory_mb']}m"
        if environment.get("allow_internet") is False:
            client["network_mode"] = "none"
        (temp / "docker-compose.yaml").write_text(yaml.safe_dump(compose))
        instruction = (source / "instruction.md").read_text()
        (temp / "task.yaml").write_text(
            yaml.safe_dump(
                {
                    "instruction": instruction,
                    "difficulty": spec.get("metadata", {}).get("difficulty") or "medium",
                    "parser_name": "pytest",
                    "max_agent_timeout_sec": spec.get("agent", {}).get(
                        "timeout_sec", 900
                    ),
                    "max_test_timeout_sec": spec.get("verifier", {}).get(
                        "timeout_sec", 900
                    ),
                    "run_tests_in_same_shell": False,
                }
            )
        )
        if tree_digest(source) != report.ref.source_digest:
            raise ValueError("source changed during materialization")
        manifest = {
            "task_ref": asdict(report.ref),
            "profile": PROFILE,
            "reward_profile": REWARD_PROFILE,
            "probe": receipt,
            "materialized_digest": tree_digest(temp),
        }
        (temp / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        temp.rename(target)
        return {
            "task": [{"content": instruction}],
            "metadata": {
                "task_name": name,
                "task_path": str(target),
                "instruction": instruction,
                "data_source": "harbor_terminal",
                "harbor_task_ref": asdict(report.ref),
                "harbor_manifest": manifest,
            },
        }
    finally:
        if temp.exists():
            shutil.rmtree(temp)


def validate_materialized(path: Path, metadata: dict, *, allow_probe=False):
    if (path / "manifest.json").is_symlink():
        raise ValueError("manifest must not be a symlink")
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest != metadata.get("harbor_manifest"):
        raise ValueError("Harbor manifest does not match catalog")
    if manifest.get("task_ref") != metadata.get("harbor_task_ref"):
        raise ValueError("Harbor task identity does not match catalog")
    if (
        manifest.get("profile") != PROFILE
        or manifest.get("reward_profile") != REWARD_PROFILE
    ):
        raise ValueError("unsupported Harbor profile")
    if tree_digest(path, exclude=("manifest.json",)) != manifest.get(
        "materialized_digest"
    ):
        raise ValueError("materialized Harbor task was modified")
    proof = manifest.get("probe") or {}
    if not allow_probe and (
        proof.get("success") is not True
        or proof.get("source_digest") != manifest["task_ref"]["source_digest"]
    ):
        raise ValueError("Harbor task has no matching successful probe")
