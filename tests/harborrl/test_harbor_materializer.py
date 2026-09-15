from pathlib import Path
import pytest
import yaml
from harborrl.data.harbor.inspector import inspect, tree_digest
from harborrl.data.harbor.materializer import materialize
from harborrl.data.harbor.receipt import parse_reward


@pytest.fixture
def task(tmp_path):
    root = tmp_path / "source" / "same-name"
    (root / "environment").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "task.toml").write_text("[environment]\ncpus=1\n")
    (root / "instruction.md").write_text("Solve the task.")
    (root / "environment/Dockerfile").write_text("FROM busybox\nCOPY asset /asset\n")
    (root / "environment/asset").write_text("asset")
    (root / "tests/test.sh").write_text("echo 0.5 > /logs/verifier/reward.txt\n")
    return root


def receipt(task):
    return dict(
        source_digest=tree_digest(task),
        profile="terminal_text_v1",
        success=True,
        native_reward=0.5,
        materialized_reward=0.5,
        execution_backend="terminal_env",
    )


def test_probe_required_and_digest_bound(task):
    assert inspect(task, "a").status == "NEEDS_PROBE"
    proof = receipt(task)
    assert inspect(task, "a", proof).status == "SUPPORTED"
    (task / "instruction.md").write_text("Changed")
    assert inspect(task, "a", proof).status == "NEEDS_PROBE"


def test_preserves_verifier_context_and_dataset_identity(task, tmp_path):
    original = tree_digest(task)
    rows = [
        materialize(task, dataset, tmp_path / "output", receipt(task))
        for dataset in ("a", "b")
    ]
    assert rows[0]["metadata"]["task_path"] != rows[1]["metadata"]["task_path"]
    target = Path(rows[0]["metadata"]["task_path"])
    assert (target / "tests/test.sh").read_bytes() == (
        task / "tests/test.sh"
    ).read_bytes()
    assert (target / "environment/asset").read_text() == "asset"
    assert (
        yaml.safe_load((target / "docker-compose.yaml").read_text())["services"][
            "client"
        ]["build"]["context"]
        == "environment"
    )
    assert tree_digest(task) == original
    with pytest.raises(FileExistsError):
        materialize(task, "a", tmp_path / "output", receipt(task))


def test_rejects_unverified_compose(task):
    (task / "environment/compose.yaml").write_text("services: {}")
    assert inspect(task, "a").status == "UNSUPPORTED"


@pytest.mark.parametrize("text", ["", "nan", "inf", "-1", "1.5", "{}", "1\n0"])
def test_bad_reward_is_not_zero(text):
    with pytest.raises(ValueError):
        parse_reward(text)


def test_zero_is_valid_task_failure():
    assert parse_reward("0\n")["raw_reward"] == 0


@pytest.mark.asyncio
async def test_client_preserves_failure_receipt(monkeypatch):
    from harborrl.environments.client import TerminalEnvClient
    import harborrl.environments.client as client_module

    async def post(*args, **kwargs):
        return {
            "ok": False,
            "error": "missing reward",
            "details": {"exception_stage": "reward_parse", "raw_reward": None},
        }

    monkeypatch.setattr(client_module, "post", post)
    client = TerminalEnvClient("http://worker")
    with pytest.raises(RuntimeError):
        await client.evaluate("lease")
    assert client.last_evaluate_details["exception_stage"] == "reward_parse"


def test_runtime_rejects_modified_task(task, tmp_path):
    from harborrl.data.harbor.materializer import validate_materialized

    row = materialize(task, "a", tmp_path / "out", receipt(task))
    meta = row["metadata"]
    path = Path(meta["task_path"])
    validate_materialized(path, meta)
    (path / "tests/test.sh").write_text("echo 1 > /logs/verifier/reward.txt")
    with pytest.raises(ValueError, match="modified"):
        validate_materialized(path, meta)


def test_materializer_keeps_resource_and_network_settings(task, tmp_path):
    (task / "task.toml").write_text(
        "[environment]\ncpus=2\nmemory_mb=1024\nallow_internet=false\n"
    )
    row = materialize(task, "a", tmp_path / "out", receipt(task))
    compose = yaml.safe_load(
        (Path(row["metadata"]["task_path"]) / "docker-compose.yaml").read_text()
    )
    assert compose["services"]["client"]["cpus"] == 2
    assert compose["services"]["client"]["mem_limit"] == "1024m"
    assert compose["services"]["client"]["network_mode"] == "none"
