from __future__ import annotations

import ast
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from types import SimpleNamespace
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[2] / "harborrl" / "data" / "convert_swesmith.py"
SPEC = importlib.util.spec_from_file_location("convert_swesmith", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
converter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(converter)


def _item(instance_id: str = "owner__repo.abcdef12.mutation__token") -> dict:
    instruction = "Fix the failing test."
    return {
        "task": [{"role": "user", "content": instruction}],
        "metadata": {
            "task_name": instance_id,
            "swesmith_instance_id": instance_id,
            "task_path": f"swesmith_env/{instance_id}",
            "instruction": instruction,
            "data_source": "swesmith",
            "repo": "swesmith/owner__repo.abcdef12",
            "image_name": "ghcr.io/swe-smith/repo.python.x86_64:latest",
            "test_runner": "pytest",
            "test_command": "",
            "task_format_version": converter.TASK_FORMAT_VERSION,
            "FAIL_TO_PASS": ["tests/test_hidden.py::fail_case"],
            "PASS_TO_PASS": ["tests/test_existing.py::pass_case"],
        },
    }


def test_task_dir_checks_out_instance_ref_and_uses_format_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    item = _item()
    assert converter.create_terminal_bench_task_dir(item, tmp_path)

    task_dir = tmp_path / item["metadata"]["task_path"]
    dockerfile = (task_dir / "Dockerfile").read_text(encoding="utf-8")
    assert f"checkout --force {item['metadata']['task_name']}" in dockerfile
    assert "branch --show-current" in dockerfile
    assert "refs/agentic_rl/swesmith-task-stage" in dockerfile
    assert "refs/agentic_rl/swesmith-bug-stage" in dockerfile
    compose = (task_dir / "docker-compose.yaml").read_text(encoding="utf-8")
    assert compose.count("harborrl.pool-namespace") == 2
    marker = (task_dir / converter.TASK_FORMAT_MARKER).read_text(encoding="utf-8")
    marker_payload = json.loads(marker)
    assert marker_payload["format_version"] == converter.TASK_FORMAT_VERSION
    assert set(marker_payload["files"]) == set(converter.GENERATED_TASK_FILES)

    assert not converter.create_terminal_bench_task_dir(item, tmp_path)
    (task_dir / "run-tests.sh").write_text("tampered\n", encoding="utf-8")
    assert converter.create_terminal_bench_task_dir(item, tmp_path)
    assert "tampered" not in (task_dir / "run-tests.sh").read_text(encoding="utf-8")

    item["metadata"]["PASS_TO_PASS"].append("tests/test_new.py::test_new")
    assert converter.create_terminal_bench_task_dir(item, tmp_path)
    assert (task_dir / converter.TASK_FORMAT_MARKER).read_text(encoding="utf-8") != marker

    original_write = converter._write_text

    def interrupted_write(*_args, **_kwargs):
        raise OSError("simulated interrupted task rewrite")

    monkeypatch.setattr(converter, "_write_text", interrupted_write)
    with pytest.raises(OSError, match="simulated interrupted"):
        converter.create_terminal_bench_task_dir(item, tmp_path, overwrite=True)
    assert not (task_dir / converter.TASK_FORMAT_MARKER).exists()
    monkeypatch.setattr(converter, "_write_text", original_write)
    assert converter.create_terminal_bench_task_dir(item, tmp_path)


def test_run_tests_executes_pass_to_pass_after_failing_test(tmp_path: Path) -> None:
    item = _item()
    converter.create_terminal_bench_task_dir(item, tmp_path)
    task_dir = tmp_path / item["metadata"]["task_path"]

    call_log = tmp_path / "pytest-calls.log"
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$*\" >> \"${CALL_LOG}\"\n"
        "case \"$*\" in *fail_case*) exit 1 ;; *) exit 0 ;; esac\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    env = {
        **os.environ,
        "CALL_LOG": str(call_log),
        "PYTHON_BIN": str(fake_python),
        "TEST_DIR": str(task_dir / "tests"),
        "SWESMITH_RUN_PASS_TO_PASS": "1",
    }
    repo_dir = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", str(repo_dir)], check=True)
    subprocess.run(["git", "-C", str(repo_dir), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(repo_dir), "config", "user.name", "Test"], check=True)
    hidden_test = repo_dir / "tests" / "test_hidden.py"
    hidden_test.parent.mkdir()
    hidden_test.write_text("def test_hidden(): pass\n", encoding="utf-8")
    existing_test = repo_dir / "tests" / "test_existing.py"
    existing_test.write_text("def test_existing(): pass\n", encoding="utf-8")
    unselected_test = repo_dir / "tests" / "test_unselected.py"
    unselected_test.write_text("def test_unselected(): pass\n", encoding="utf-8")
    existing_test.chmod(0o755)
    source_file = repo_dir / "source.py"
    source_file.write_text("value = 'bug'\n", encoding="utf-8")
    root_hook = repo_dir / "conftest.py"
    root_hook.write_text("# trusted root hook\n", encoding="utf-8")
    nested_hook = repo_dir / "package" / "conftest.py"
    nested_hook.parent.mkdir()
    nested_hook.write_text("# trusted nested hook\n", encoding="utf-8")
    sitecustomize = repo_dir / "sitecustomize.py"
    sitecustomize.write_text("# trusted sitecustomize\n", encoding="utf-8")
    python_runner_controls = {
        "pytest.ini": "[pytest]\n",
        "pyproject.toml": "[tool.pytest.ini_options]\n",
        "setup.cfg": "[tool:pytest]\n",
        "tox.ini": "[tox]\n",
    }
    other_runner_controls = {
        "go.mod": "module trusted.example/repo\n",
        "go.sum": "trusted go sum\n",
        "pom.xml": "<project>trusted</project>\n",
        ".mvn/extensions.xml": "<extensions>trusted</extensions>\n",
        "package.json": '{"trusted":true}\n',
        "package-extra.json": '{"trusted":true}\n',
        "package-lock.json": '{"lockfileVersion":3}\n',
        "yarn.lock": "trusted yarn lock\n",
        "pnpm-lock.yaml": "lockfileVersion: trusted\n",
        "bun.lockb": "trusted bun lock\n",
        "composer.json": '{"trusted":true}\n',
        "composer.lock": '{"trusted":true}\n',
        "phpunit.xml.dist": "<phpunit>trusted</phpunit>\n",
        "Cargo.toml": "[package]\nname = \"trusted\"\n",
        "Cargo.lock": "trusted cargo lock\n",
        ".cargo/config.toml": "[build]\ntarget-dir = \"trusted\"\n",
    }
    runner_controls = {**python_runner_controls, **other_runner_controls}
    for relative, content in runner_controls.items():
        control_path = repo_dir / relative
        control_path.parent.mkdir(parents=True, exist_ok=True)
        control_path.write_text(content, encoding="utf-8")
    subprocess.run(["git", "-C", str(repo_dir), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo_dir), "commit", "-qm", "Bug Patch"], check=True)
    hidden_test.unlink()
    subprocess.run(["git", "-C", str(repo_dir), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo_dir), "commit", "-qm", "Remove F2P Tests"], check=True)
    subprocess.run(
        ["git", "-C", str(repo_dir), "update-ref", "refs/agentic_rl/swesmith-task-stage", "HEAD"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo_dir), "update-ref", "refs/agentic_rl/swesmith-bug-stage", "HEAD^"],
        check=True,
    )
    task_commit = subprocess.check_output(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD"], text=True
    ).strip()
    bug_commit = subprocess.check_output(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD^"], text=True
    ).strip()
    env["SWESMITH_TRUSTED_TASK_COMMIT"] = task_commit
    env["SWESMITH_TRUSTED_BUG_COMMIT"] = bug_commit
    # Simulate an agent moving the writable refs. The host-captured SHAs must
    # remain authoritative during evaluation.
    subprocess.run(
        ["git", "-C", str(repo_dir), "update-ref", "refs/agentic_rl/swesmith-bug-stage", "HEAD"],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(repo_dir),
            "config",
            "filter.agent-smudge.smudge",
            "sed 's/pass/SMUDGED/g'",
        ],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(repo_dir),
            "config",
            "filter.agent-smudge.clean",
            "cat",
        ],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(repo_dir),
            "config",
            "filter.agent-smudge.required",
            "true",
        ],
        check=True,
    )
    git_info_attributes = repo_dir / ".git" / "info" / "attributes"
    git_info_attributes.write_text("tests/*.py filter=agent-smudge\n", encoding="utf-8")
    filtered_blob = subprocess.check_output(
        [
            "git",
            "-C",
            str(repo_dir),
            "cat-file",
            "--filters",
            "--path=tests/test_existing.py",
            f"{bug_commit}:tests/test_existing.py",
        ],
        text=True,
    )
    assert "SMUDGED" in filtered_blob
    existing_test.write_text("def test_existing(): assert False\n", encoding="utf-8")
    existing_test.chmod(0o600)
    unselected_test.write_text("def test_unselected(): assert False\n", encoding="utf-8")
    source_file.write_text("value = 'agent fix'\n", encoding="utf-8")
    root_hook.write_text("raise RuntimeError('agent test hook')\n", encoding="utf-8")
    nested_hook.write_text("raise RuntimeError('nested agent test hook')\n", encoding="utf-8")
    sitecustomize_marker = tmp_path / "sitecustomize-executed"
    sitecustomize.write_text(
        f"from pathlib import Path\nPath({str(sitecustomize_marker)!r}).write_text('owned')\n",
        encoding="utf-8",
    )
    for relative in runner_controls:
        (repo_dir / relative).write_text("agent tamper\n", encoding="utf-8")
    untracked_hook = repo_dir / "attacker" / "conftest.py"
    untracked_hook.parent.mkdir()
    untracked_hook.write_text("raise RuntimeError('untracked agent hook')\n", encoding="utf-8")
    untracked_runner_control = repo_dir / "attacker" / "pytest.ini"
    untracked_runner_control.write_text(
        "[pytest]\naddopts = --ignore=tests\n", encoding="utf-8"
    )
    same_name_dir = repo_dir / "docs" / "conftest.py"
    same_name_dir.mkdir(parents=True)
    (same_name_dir / "README").write_text("not a Python hook\n", encoding="utf-8")
    symlink_hook = repo_dir / "linked" / "conftest.py"
    symlink_hook.parent.mkdir()
    symlink_hook.symlink_to(source_file)
    untracked_test = repo_dir / "tests" / "test_agent_added.py"
    untracked_test.write_text("def test_fake(): pass\n", encoding="utf-8")
    result = subprocess.run(
        ["bash", str(task_dir / "run-tests.sh")],
        cwd=repo_dir,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert hidden_test.is_file()
    assert hidden_test.read_bytes() == b"def test_hidden(): pass\n"
    assert existing_test.read_text(encoding="utf-8") == "def test_existing(): pass\n"
    assert existing_test.stat().st_mode & 0o777 == 0o755
    assert unselected_test.read_text(encoding="utf-8") == "def test_unselected(): pass\n"
    assert source_file.read_text(encoding="utf-8") == "value = 'agent fix'\n"
    assert root_hook.read_text(encoding="utf-8") == "# trusted root hook\n"
    assert nested_hook.read_text(encoding="utf-8") == "# trusted nested hook\n"
    assert sitecustomize.read_text(encoding="utf-8") == "# trusted sitecustomize\n"
    assert not sitecustomize_marker.exists()
    for relative, content in python_runner_controls.items():
        assert (repo_dir / relative).read_text(encoding="utf-8") == content
    for relative in other_runner_controls:
        assert (repo_dir / relative).read_text(encoding="utf-8") == "agent tamper\n"
    assert not untracked_hook.exists()
    assert not untracked_runner_control.exists()
    assert (same_name_dir / "README").read_text(encoding="utf-8") == "not a Python hook\n"
    assert not symlink_hook.is_symlink()
    assert not untracked_test.exists()
    calls = call_log.read_text(encoding="utf-8")
    assert "fail_case" in calls
    assert "pass_case" in calls
    assert "-I -m pytest -q -rA -- tests/test_hidden.py::fail_case" in calls


def test_secure_restore_refuses_symlink_parent(tmp_path: Path) -> None:
    item = _item()
    converter.create_terminal_bench_task_dir(item, tmp_path)
    task_dir = tmp_path / item["metadata"]["task_path"]
    repo_dir = tmp_path / "symlink-repo"
    subprocess.run(["git", "init", "-q", str(repo_dir)], check=True)
    subprocess.run(["git", "-C", str(repo_dir), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(repo_dir), "config", "user.name", "Test"], check=True)
    hidden_test = repo_dir / "tests" / "test_hidden.py"
    hidden_test.parent.mkdir()
    hidden_test.write_text("def test_hidden(): pass\n", encoding="utf-8")
    protected_config = repo_dir / "config" / "pyproject.toml"
    protected_config.parent.mkdir()
    protected_config.write_text("[tool.pytest.ini_options]\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo_dir), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo_dir), "commit", "-qm", "Bug Patch"], check=True)
    hidden_test.unlink()
    subprocess.run(["git", "-C", str(repo_dir), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo_dir), "commit", "-qm", "Remove F2P Tests"], check=True)
    task_commit = subprocess.check_output(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD"], text=True
    ).strip()
    bug_commit = subprocess.check_output(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD^"], text=True
    ).strip()

    protected_config.unlink()
    protected_config.parent.rmdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_config = outside / "pyproject.toml"
    outside_config.write_text("outside sentinel\n", encoding="utf-8")
    (repo_dir / "config").symlink_to(outside, target_is_directory=True)
    result = subprocess.run(
        ["bash", str(task_dir / "run-tests.sh")],
        cwd=repo_dir,
        env={
            **os.environ,
            "TEST_DIR": str(task_dir / "tests"),
            "SWESMITH_TRUSTED_TASK_COMMIT": task_commit,
            "SWESMITH_TRUSTED_BUG_COMMIT": bug_commit,
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "secure SWE-smith restore failed" in result.stderr
    assert outside_config.read_text(encoding="utf-8") == "outside sentinel\n"


def test_run_tests_requires_host_captured_commits(tmp_path: Path) -> None:
    item = _item()
    converter.create_terminal_bench_task_dir(item, tmp_path)
    task_dir = tmp_path / item["metadata"]["task_path"]
    repo_dir = tmp_path / "writable-ref-repo"
    subprocess.run(["git", "init", "-q", str(repo_dir)], check=True)
    subprocess.run(
        ["git", "-C", str(repo_dir), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo_dir), "config", "user.name", "Test"], check=True
    )
    hidden_test = repo_dir / "tests" / "test_hidden.py"
    hidden_test.parent.mkdir()
    hidden_test.write_text("def test_hidden(): pass\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo_dir), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo_dir), "commit", "-qm", "bug"], check=True)
    hidden_test.unlink()
    subprocess.run(["git", "-C", str(repo_dir), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo_dir), "commit", "-qm", "task"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo_dir),
            "update-ref",
            "refs/agentic_rl/swesmith-task-stage",
            "HEAD",
        ],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(repo_dir),
            "update-ref",
            "refs/agentic_rl/swesmith-bug-stage",
            "HEAD^",
        ],
        check=True,
    )
    result = subprocess.run(
        ["bash", str(task_dir / "run-tests.sh")],
        cwd=repo_dir,
        env={**os.environ, "TEST_DIR": str(task_dir / "tests")},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "host-captured SWE-smith task/bug commits" in result.stderr


def test_secure_restore_rejects_protected_symlink_blob(tmp_path: Path) -> None:
    item = _item()
    converter.create_terminal_bench_task_dir(item, tmp_path)
    task_dir = tmp_path / item["metadata"]["task_path"]
    repo_dir = tmp_path / "trusted-symlink-repo"
    subprocess.run(["git", "init", "-q", str(repo_dir)], check=True)
    subprocess.run(["git", "-C", str(repo_dir), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(repo_dir), "config", "user.name", "Test"], check=True)
    source = repo_dir / "source.py"
    source.write_text("value = 'trusted'\n", encoding="utf-8")
    hidden_link = repo_dir / "tests" / "test_hidden.py"
    hidden_link.parent.mkdir()
    hidden_link.symlink_to("../source.py")
    subprocess.run(["git", "-C", str(repo_dir), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo_dir), "commit", "-qm", "Bug Patch"], check=True)
    hidden_link.unlink()
    subprocess.run(["git", "-C", str(repo_dir), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo_dir), "commit", "-qm", "Remove F2P Tests"], check=True)
    result = subprocess.run(
        ["bash", str(task_dir / "run-tests.sh")],
        cwd=repo_dir,
        env={
            **os.environ,
            "TEST_DIR": str(task_dir / "tests"),
            "SWESMITH_TRUSTED_TASK_COMMIT": subprocess.check_output(
                ["git", "-C", str(repo_dir), "rev-parse", "HEAD"], text=True
            ).strip(),
            "SWESMITH_TRUSTED_BUG_COMMIT": subprocess.check_output(
                ["git", "-C", str(repo_dir), "rev-parse", "HEAD^"], text=True
            ).strip(),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "protected symlink is unsupported" in result.stderr
    assert source.read_text(encoding="utf-8") == "value = 'trusted'\n"


def test_pytest_runner_ignores_untracked_local_pytest_package(tmp_path: Path) -> None:
    item = _item()
    item["metadata"]["FAIL_TO_PASS"] = ["tests/test_hidden.py::test_hidden"]
    item["metadata"]["PASS_TO_PASS"] = ["tests/test_existing.py::test_existing"]
    converter.create_terminal_bench_task_dir(item, tmp_path)
    task_dir = tmp_path / item["metadata"]["task_path"]
    repo_dir = tmp_path / "isolated-pytest-repo"
    subprocess.run(["git", "init", "-q", str(repo_dir)], check=True)
    subprocess.run(["git", "-C", str(repo_dir), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(repo_dir), "config", "user.name", "Test"], check=True)
    tests_dir = repo_dir / "tests"
    tests_dir.mkdir()
    hidden_test = tests_dir / "test_hidden.py"
    hidden_test.write_text("def test_hidden(): assert True\n", encoding="utf-8")
    (tests_dir / "test_existing.py").write_text(
        "def test_existing(): assert True\n", encoding="utf-8"
    )
    subprocess.run(["git", "-C", str(repo_dir), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo_dir), "commit", "-qm", "Bug Patch"], check=True)
    hidden_test.unlink()
    subprocess.run(["git", "-C", str(repo_dir), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo_dir), "commit", "-qm", "Remove F2P Tests"], check=True)

    marker = tmp_path / "fake-pytest-executed"
    fake_pytest = repo_dir / "pytest"
    fake_pytest.mkdir()
    (fake_pytest / "__main__.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('owned')\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        ["bash", str(task_dir / "run-tests.sh")],
        cwd=repo_dir,
        env={
            **os.environ,
            "TEST_DIR": str(task_dir / "tests"),
            "SWESMITH_RUN_PASS_TO_PASS": "1",
            "PYTHON_BIN": sys.executable,
            "SWESMITH_TRUSTED_TASK_COMMIT": subprocess.check_output(
                ["git", "-C", str(repo_dir), "rev-parse", "HEAD"], text=True
            ).strip(),
            "SWESMITH_TRUSTED_BUG_COMMIT": subprocess.check_output(
                ["git", "-C", str(repo_dir), "rev-parse", "HEAD^"], text=True
            ).strip(),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert not marker.exists()


def test_invalid_image_name_is_rejected(tmp_path: Path) -> None:
    item = _item()
    item["metadata"]["image_name"] = "valid-image\nRUN echo injected"
    with pytest.raises(ValueError, match="invalid SWE-smith image_name"):
        converter.create_terminal_bench_task_dir(item, tmp_path)


def test_unsafe_or_mismatched_task_identity_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsafe SWE-smith instance_id"):
        converter.convert_sample(
            {
                "instance_id": "../outside",
                "repo": "swesmith/owner__repo.abcdef12",
                "image_name": "ghcr.io/swe-smith/repo:latest",
                "FAIL_TO_PASS": ["tests/test_bug.py::test_bug"],
            }
        )

    item = _item()
    item["metadata"]["task_path"] = "swesmith_env/other"
    with pytest.raises(ValueError, match="invalid SWE-smith task_path"):
        converter.create_terminal_bench_task_dir(item, tmp_path)

    item = _item()
    item["metadata"]["swesmith_instance_id"] = "other-ref"
    with pytest.raises(ValueError, match="instance ref must equal task_name"):
        converter.create_terminal_bench_task_dir(item, tmp_path)


def test_argument_parser_has_no_conflicting_options(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", [str(MODULE_PATH), "--dry-run", "--input-jsonl", "input.jsonl"])
    args = converter.parse_args()
    assert args.dry_run is True
    assert args.input_jsonl == "input.jsonl"


def test_go_subtests_are_not_misclassified_as_pytest(tmp_path: Path) -> None:
    sample = {
        "instance_id": "ariga__atlas.1afaaba2.mutation__token",
        "repo": "swesmith/ariga__atlas.1afaaba2",
        "image_name": "jyangballin/swesmith.x86_64.ariga_1776_atlas.1afaaba2",
        "FAIL_TO_PASS": [
            "TestFormatters",
            "TestFormatters/pressly/goose",
            "TestFormatters/flyway",
            'TestFormatters/value_"quoted"',
        ],
        "PASS_TO_PASS": ["TestExisting"],
    }
    assert converter.infer_test_runner(sample) == "go"
    item = converter.convert_sample(sample)
    converter.create_terminal_bench_task_dir(item, tmp_path)
    run_tests = (
        tmp_path / item["metadata"]["task_path"] / "run-tests.sh"
    ).read_text(encoding="utf-8")
    assert "PASSED swesmith_go::all_tests" in run_tests
    assert "env GOENV=off GOFLAGS= go test -json ./..." in run_tests
    assert run_tests.index("short test summary info") < run_tests.index(
        "PASSED swesmith_go::all_tests"
    )

    sample["FAIL_TO_PASS"] = ["TestPanicInHandler", "github.com/gin-gonic/gin"]
    assert converter.infer_test_runner(sample) == "go"


def test_go_runner_rejects_successful_command_with_zero_tests(tmp_path: Path) -> None:
    sample = {
        "instance_id": "cweill__gotests.16a93f6e.mutation__token",
        "repo": "swesmith/cweill__gotests.16a93f6e",
        "image_name": "ghcr.io/swe-smith/gotests:latest",
        "FAIL_TO_PASS": ["TestHidden"],
        "PASS_TO_PASS": ["TestExisting"],
    }
    item = converter.convert_sample(sample)
    converter.create_terminal_bench_task_dir(item, tmp_path)
    task_dir = tmp_path / item["metadata"]["task_path"]

    repo_dir = tmp_path / "go-repo"
    subprocess.run(["git", "init", "-q", str(repo_dir)], check=True)
    subprocess.run(
        ["git", "-C", str(repo_dir), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo_dir), "config", "user.name", "Test"], check=True
    )
    hidden_test = repo_dir / "hidden_test.go"
    hidden_test.write_text("package example\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo_dir), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo_dir), "commit", "-qm", "Bug Patch"], check=True)
    hidden_test.unlink()
    subprocess.run(["git", "-C", str(repo_dir), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(repo_dir), "commit", "-qm", "Remove F2P Tests"],
        check=True,
    )
    task_commit = subprocess.check_output(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD"], text=True
    ).strip()
    bug_commit = subprocess.check_output(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD^"], text=True
    ).strip()

    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    fake_go = fake_bin / "go"
    fake_go.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' '{\"Action\":\"pass\",\"Package\":\"example\"}'\n",
        encoding="utf-8",
    )
    fake_go.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ.get('PATH', '')}",
        "TEST_DIR": str(task_dir / "tests"),
        "SWESMITH_TRUSTED_TASK_COMMIT": task_commit,
        "SWESMITH_TRUSTED_BUG_COMMIT": bug_commit,
    }
    result = subprocess.run(
        ["bash", str(task_dir / "run-tests.sh")],
        cwd=repo_dir,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "executed zero tests" in result.stdout


def test_unknown_double_colon_profile_fails_closed() -> None:
    sample = {
        "repo": "swesmith/future__rust.abcdef12",
        "image_name": "ghcr.io/swe-smith/future-rust:latest",
        "FAIL_TO_PASS": ["crate::module::test_case"],
        "PASS_TO_PASS": ["crate::module::existing_case"],
    }
    assert converter.infer_test_runner(sample) == "unsupported"


def test_custom_env_dir_must_keep_swesmith_env_name(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.jsonl"
    raw_path.write_text(
        '{"instance_id":"owner__repo.abcdef12.mutation__token",'
        '"repo":"swesmith/owner__repo.abcdef12",'
        '"image_name":"ghcr.io/swe-smith/repo:latest",'
        '"FAIL_TO_PASS":["tests/test_bug.py::test_bug"]}\n',
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            str(MODULE_PATH),
            "--input-jsonl",
            str(raw_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--create-env-dirs",
            "--env-dir",
            str(tmp_path / "wrong-name"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "must end with /swesmith_env" in (result.stdout + result.stderr)


@pytest.mark.parametrize("repo", sorted(converter.OFFICIAL_TEST_COMMANDS))
def test_fixed_profiles_use_official_commands(repo: str, tmp_path: Path) -> None:
    sample = {
        "instance_id": f"{repo.removeprefix('swesmith/')}" + ".mutation__token",
        "repo": repo,
        "image_name": "jyangballin/swesmith.x86_64.owner_1776_repo.abcdef12",
        "FAIL_TO_PASS": ["example test"],
        "PASS_TO_PASS": ["existing test"],
    }
    item = converter.convert_sample(sample)
    expected_runner = converter.SPECIAL_TEST_RUNNERS.get(repo, "command")
    assert item["metadata"]["test_runner"] == expected_runner
    assert item["metadata"]["test_command"] == converter.OFFICIAL_TEST_COMMANDS[repo]
    converter.create_terminal_bench_task_dir(item, tmp_path)
    run_tests = (
        tmp_path / item["metadata"]["task_path"] / "run-tests.sh"
    ).read_text(encoding="utf-8")
    assert converter.OFFICIAL_TEST_COMMANDS[repo] in run_tests
    assert "SWESMITH_TEST_COMMAND" not in run_tests
    if expected_runner == "command":
        assert "bash --noprofile --norc -c" in run_tests
    if repo == "swesmith/un33k__python-slugify.872b3750":
        assert 'return path == "test.py"' in run_tests
    if repo == "swesmith/markedjs__marked.dbf29d91":
        assert '".test.js"' in run_tests


def test_special_python_profiles_match_official_execution_semantics(tmp_path: Path) -> None:
    cases = {
        "swesmith/un33k__python-slugify.872b3750": "command",
        "swesmith/tornadoweb__tornado.d5ac65c1": "command",
        "swesmith/python__mypy.e93f06ce": "mypy",
        "swesmith/pydantic__pydantic.acb0f10f": "pytest_uv",
    }
    for index, (repo, runner) in enumerate(cases.items()):
        sample = {
            "instance_id": f"special__repo.abcdef1{index}.mutation__token",
            "repo": repo,
            "image_name": "ghcr.io/swe-smith/special:latest",
            "FAIL_TO_PASS": ["tests/test_bug.py::test_bug"],
            "PASS_TO_PASS": ["tests/test_existing.py::test_existing"],
        }
        item = converter.convert_sample(sample)
        assert item["metadata"]["test_runner"] == runner
        converter.create_terminal_bench_task_dir(item, tmp_path)

    mypy_script = (
        tmp_path
        / "swesmith_env/special__repo.abcdef12.mutation__token/run-tests.sh"
    ).read_text(encoding="utf-8")
    assert '"${PYTHON_BIN}" -I -m pytest --color=no -rA -k "${expression}"' in mypy_script
    pydantic_script = (
        tmp_path
        / "swesmith_env/special__repo.abcdef13.mutation__token/run-tests.sh"
    ).read_text(encoding="utf-8")
    assert "/root/.local/bin/uv run python -I -m pytest" in pydantic_script
    assert "--disable-warnings --color=no --tb=no --verbose" in pydantic_script


def test_mido_test_paths_match_official_profile() -> None:
    item = converter.convert_sample(
        {
            "instance_id": "mido__mido.a0158ff9.mutation__token",
            "repo": "swesmith/mido__mido.a0158ff9",
            "image_name": "ghcr.io/swe-smith/mido:latest",
            "FAIL_TO_PASS": ["../dev/tests/test_midifiles.py::test_bug"],
            "PASS_TO_PASS": ["../dev/tests/test_messages.py::test_existing"],
        }
    )
    assert item["metadata"]["FAIL_TO_PASS"] == [
        "tests/test_midifiles.py::test_bug"
    ]
    assert item["metadata"]["PASS_TO_PASS"] == [
        "tests/test_messages.py::test_existing"
    ]


def test_converter_rejects_empty_fail_to_pass_cap(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.jsonl"
    raw_path.write_text("{}\n", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(MODULE_PATH),
            "--dry-run",
            "--input-jsonl",
            str(raw_path),
            "--max-fail-to-pass-count",
            "0",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "must be -1 (unlimited) or a positive integer" in (
        result.stdout + result.stderr
    )


def test_full_conversion_rejects_rows_without_fail_to_pass(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.jsonl"
    raw_path.write_text(
        json.dumps(
            {
                "instance_id": "owner__repo.abcdef12.mutation__token",
                "repo": "swesmith/owner__repo.abcdef12",
                "image_name": "ghcr.io/swe-smith/repo:latest",
                "FAIL_TO_PASS": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            str(MODULE_PATH),
            "--input-jsonl",
            str(raw_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--fail-on-too-few-tests",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "too_few_fail_to_pass=1" in (result.stdout + result.stderr)


def test_command_runner_is_fixed_and_emits_parseable_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = "swesmith/example__command-repo.abcdef12"
    monkeypatch.setitem(
        converter.OFFICIAL_TEST_COMMANDS,
        repo,
        ": node; grep -Fqx '{\"trusted\":true}' package.json && "
        "printf 'official-command-ran\\nTests: 1\\n'",
    )
    item = converter.convert_sample(
        {
            "instance_id": "example__command-repo.abcdef12.mutation__token",
            "repo": repo,
            "image_name": "ghcr.io/swe-smith/command-repo:latest",
            "problem_statement": "Fix it.",
            "FAIL_TO_PASS": ["hidden test"],
            "PASS_TO_PASS": ["existing test"],
        }
    )
    converter.create_terminal_bench_task_dir(item, tmp_path)
    task_dir = tmp_path / item["metadata"]["task_path"]

    repo_dir = tmp_path / "command-repo"
    subprocess.run(["git", "init", "-q", str(repo_dir)], check=True)
    subprocess.run(["git", "-C", str(repo_dir), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(repo_dir), "config", "user.name", "Test"], check=True)
    hidden_test = repo_dir / "tests" / "hidden.txt"
    hidden_test.parent.mkdir()
    hidden_test.write_text("hidden\n", encoding="utf-8")
    command_profile = repo_dir / "package.json"
    command_profile.write_text('{"trusted":true}\n', encoding="utf-8")
    subprocess.run(["git", "-C", str(repo_dir), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo_dir), "commit", "-qm", "Bug Patch"], check=True)
    hidden_test.unlink()
    subprocess.run(["git", "-C", str(repo_dir), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo_dir), "commit", "-qm", "Remove F2P Tests"], check=True)
    task_commit = subprocess.check_output(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD"], text=True
    ).strip()
    bug_commit = subprocess.check_output(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD^"], text=True
    ).strip()
    env = {
        **os.environ,
        "TEST_DIR": str(task_dir / "tests"),
        "SWESMITH_TRUSTED_TASK_COMMIT": task_commit,
        "SWESMITH_TRUSTED_BUG_COMMIT": bug_commit,
        "SWESMITH_TEST_COMMAND": "false",
    }
    command_profile.write_text('{"scripts":{"test":"agent-controlled"}}\n', encoding="utf-8")
    result = subprocess.run(
        ["bash", str(task_dir / "run-tests.sh")],
        cwd=repo_dir,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "official-command-ran" in result.stdout
    assert "PASSED swesmith_command::all_tests" in result.stdout
    assert command_profile.read_text(encoding="utf-8") == '{"trusted":true}\n'
    assert result.stdout.index("short test summary info") < result.stdout.index(
        "PASSED swesmith_command::all_tests"
    )
    assert hidden_test.is_file()

    monkeypatch.setitem(
        converter.OFFICIAL_TEST_COMMANDS,
        repo,
        "printf 'official-command-ran-but-zero-tests\\n'",
    )
    zero_test_item = converter.convert_sample(
        {
            "instance_id": "example__command-repo.abcdef12.mutation__token",
            "repo": repo,
            "image_name": "ghcr.io/swe-smith/command-repo:latest",
            "problem_statement": "Fix it.",
            "FAIL_TO_PASS": ["hidden test"],
            "PASS_TO_PASS": ["existing test"],
        }
    )
    converter.create_terminal_bench_task_dir(
        zero_test_item, tmp_path, overwrite=True
    )
    zero_test_result = subprocess.run(
        ["bash", str(task_dir / "run-tests.sh")],
        cwd=repo_dir,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert zero_test_result.returncode == 1
    assert "executed zero tests" in zero_test_result.stdout


def test_pool_namespace_keeps_seta_and_swesmith_objects_separate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    terminal_env_path = MODULE_PATH.parents[1] / "environments" / "terminal" / "runtime.py"
    tree = ast.parse(terminal_env_path.read_text(encoding="utf-8"))
    helper_names = {
        "_clean_docker_label",
        "_current_pool_namespace",
        "_pool_scoped_trial_name",
        "_matches_pool_namespace",
    }
    helper_nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in helper_names
    ]
    namespace = {
        "os": os,
        "re": re,
        "_POOL_NAMESPACE_RE": re.compile(r"^[a-z0-9][a-z0-9_-]{0,62}$"),
    }
    helper_module = ast.Module(body=helper_nodes, type_ignores=[])
    exec(compile(helper_module, str(terminal_env_path), "exec"), namespace)
    matches = namespace["_matches_pool_namespace"]
    scoped_name = namespace["_pool_scoped_trial_name"]

    monkeypatch.delenv("TERMINAL_RL_POOL_NAMESPACE", raising=False)
    assert scoped_name("task", "uid") == "task.uid.slime-run"
    assert matches("")
    assert matches("default")
    assert not matches("swesmith")

    monkeypatch.setenv("TERMINAL_RL_POOL_NAMESPACE", "swesmith")
    assert scoped_name("task", "uid") == "swesmith.task.uid.slime-run"
    assert matches("swesmith")
    assert not matches("")
    assert not matches("default")

    monkeypatch.setenv("TERMINAL_RL_POOL_NAMESPACE", "a/b")
    with pytest.raises(ValueError, match="TERMINAL_RL_POOL_NAMESPACE"):
        scoped_name("task", "uid")


def test_non_default_destructive_cleanup_requires_matching_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    terminal_env_path = MODULE_PATH.parents[1] / "environments" / "terminal" / "runtime.py"
    source = terminal_env_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    helper_names = {
        "_DockerCleanupDeadlineExceeded",
        "_clean_docker_label",
        "_current_pool_namespace",
        "_docker_cleanup_command_timeout",
        "_pool_scoped_trial_name",
        "_matches_pool_namespace",
        "_docker_object_pool_namespace_state",
        "_docker_object_matches_pool_namespace",
        "_compose_project_pool_namespace_state",
        "_compose_project_matches_pool_namespace",
        "_container_compose_project_state",
        "_remove_owned_container_for_reset",
        "_terminal_stop_ownership_verified",
    }
    helper_nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        and node.name in helper_names
    ]

    class FakeSubprocess:
        outputs: list[SimpleNamespace] = []
        calls: list[list[str]] = []

        @classmethod
        def run(cls, command, **_kwargs):
            cls.calls.append(command)
            return cls.outputs.pop(0)

    namespace = {
        "os": os,
        "re": re,
        "subprocess": FakeSubprocess,
        "time": time,
        "_POOL_NAMESPACE_RE": re.compile(r"^[a-z0-9][a-z0-9_-]{0,62}$"),
    }
    exec(
        compile(ast.Module(body=helper_nodes, type_ignores=[]), str(terminal_env_path), "exec"),
        namespace,
    )
    owns_object = namespace["_docker_object_matches_pool_namespace"]
    owns_project = namespace["_compose_project_matches_pool_namespace"]
    object_state = namespace["_docker_object_pool_namespace_state"]
    remove_owned = namespace["_remove_owned_container_for_reset"]
    stop_verified = namespace["_terminal_stop_ownership_verified"]

    monkeypatch.delenv("TERMINAL_RL_POOL_NAMESPACE", raising=False)
    assert owns_object("container", "legacy-seta", timeout=1)
    assert owns_project("legacy-seta", timeout=1)
    assert FakeSubprocess.calls == []

    FakeSubprocess.outputs = [
        SimpleNamespace(returncode=0, stdout="swesmith\n", stderr=""),
    ]
    assert object_state("container", "foreign", timeout=1) == "foreign"

    monkeypatch.setenv("TERMINAL_RL_POOL_NAMESPACE", "swesmith")
    FakeSubprocess.outputs = [SimpleNamespace(returncode=0, stdout="swesmith\n", stderr="")]
    assert owns_object("container", "owned", timeout=1)
    FakeSubprocess.outputs = [SimpleNamespace(returncode=0, stdout="default\n", stderr="")]
    assert not owns_object("network", "foreign", timeout=1)
    FakeSubprocess.outputs = [
        SimpleNamespace(returncode=0, stdout="swesmith\n", stderr=""),
        SimpleNamespace(returncode=0, stdout="swesmith\n", stderr=""),
        SimpleNamespace(returncode=0, stdout="", stderr=""),
    ]
    assert owns_project("owned-project", timeout=1)
    FakeSubprocess.outputs = [
        SimpleNamespace(returncode=0, stdout="swesmith\n", stderr=""),
        SimpleNamespace(returncode=0, stdout="default\n", stderr=""),
        SimpleNamespace(returncode=0, stdout="", stderr=""),
    ]
    assert not owns_project("mixed-project", timeout=1)
    FakeSubprocess.outputs = [
        SimpleNamespace(returncode=0, stdout="", stderr=""),
        SimpleNamespace(returncode=0, stdout="", stderr=""),
        SimpleNamespace(returncode=0, stdout="", stderr=""),
    ]
    assert not owns_project("unprovable-project", timeout=1)

    FakeSubprocess.outputs = [
        SimpleNamespace(returncode=1, stdout="", stderr="No such container"),
    ]
    assert object_state("container", "missing", timeout=1) == "absent"
    FakeSubprocess.outputs = [
        SimpleNamespace(returncode=1, stdout="", stderr="daemon unavailable"),
    ]
    assert object_state("container", "unknown", timeout=1) == "unknown"

    FakeSubprocess.outputs = [
        SimpleNamespace(returncode=0, stdout="swesmith\tcontainer-id\n", stderr=""),
        SimpleNamespace(returncode=0, stdout="", stderr=""),
        SimpleNamespace(returncode=1, stdout="", stderr="No such container"),
    ]
    assert remove_owned("owned", timeout=1)
    assert FakeSubprocess.calls[-2][-1] == "container-id"
    FakeSubprocess.outputs = [
        SimpleNamespace(returncode=0, stdout="default\tforeign-id\n", stderr=""),
    ]
    with pytest.raises(RuntimeError, match="refusing reset pre-cleanup"):
        remove_owned("foreign", timeout=1)

    FakeSubprocess.outputs = [
        SimpleNamespace(
            returncode=0,
            stdout="swesmith\towned-project\n",
            stderr="",
        ),
        SimpleNamespace(returncode=0, stdout="swesmith\n", stderr=""),
        SimpleNamespace(returncode=0, stdout="swesmith\n", stderr=""),
        SimpleNamespace(returncode=0, stdout="", stderr=""),
    ]
    assert stop_verified("owned", timeout=1)
    FakeSubprocess.outputs = [
        SimpleNamespace(returncode=0, stdout="swesmith\towned-project\n", stderr=""),
        SimpleNamespace(returncode=0, stdout="swesmith\n", stderr=""),
        SimpleNamespace(returncode=0, stdout="default\n", stderr=""),
        SimpleNamespace(returncode=0, stdout="", stderr=""),
    ]
    with pytest.raises(RuntimeError, match="compose project"):
        stop_verified("mixed", timeout=1)

    compact = " ".join(source.split())
    assert "_remove_owned_container_for_reset( container_name, timeout=5" in compact
    assert "container_ids.append(container_id)" in source
    assert "trial_name = _pool_scoped_trial_name(" in source
    assert 'compose_env["TERMINAL_RL_POOL_NAMESPACE"] = namespace' in source
    assert "_terminal_stop_ownership_verified" in source
    assert "non-default Docker pool requires a single static Compose" in source
    assert '["docker", "volume", "rm", volume_name]' in source
    assert (
        "_compose_project_matches_pool_namespace( project, timeout=command_timeout, "
        "deadline=deadline )"
        in compact
    )
    assert source.count('Label "harborrl.pool-namespace"') >= 5


def test_non_default_pool_requires_labels_on_all_compose_objects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    item = _item()
    converter.create_terminal_bench_task_dir(item, tmp_path)
    compose_path = tmp_path / item["metadata"]["task_path"] / "docker-compose.yaml"

    terminal_env_path = MODULE_PATH.parents[1] / "environments" / "terminal" / "runtime.py"
    tree = ast.parse(terminal_env_path.read_text(encoding="utf-8"))
    helper_names = {"_current_pool_namespace", "_compose_declares_pool_namespace"}
    helper_nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in helper_names
    ]
    namespace = {
        "os": os,
        "Path": Path,
        "_POOL_NAMESPACE_RE": re.compile(r"^[a-z0-9][a-z0-9_-]{0,62}$"),
    }
    exec(
        compile(
            ast.Module(body=helper_nodes, type_ignores=[]),
            str(terminal_env_path),
            "exec",
        ),
        namespace,
    )
    declares_namespace = namespace["_compose_declares_pool_namespace"]

    monkeypatch.setenv("TERMINAL_RL_POOL_NAMESPACE", "swesmith")
    assert declares_namespace(compose_path)

    generated_compose = compose_path.read_text(encoding="utf-8")
    monkeypatch.setenv("COMPOSE_OVERRIDE_PATH", "/tmp/unsafe-override.yaml")
    assert not declares_namespace(compose_path)
    monkeypatch.delenv("COMPOSE_OVERRIDE_PATH", raising=False)

    compose_path.write_text(
        generated_compose.replace(
            "networks:\n  default:\n    labels:\n",
            "networks:\n  default:\n    name: seta-shared-network\n    labels:\n",
        ),
        encoding="utf-8",
    )
    assert not declares_namespace(compose_path)

    compose_path.write_text(
        generated_compose.replace(
            "container_name: ${T_BENCH_TASK_DOCKER_CLIENT_CONTAINER_NAME}",
            "container_name: seta-shared-container",
        ),
        encoding="utf-8",
    )
    assert not declares_namespace(compose_path)

    compose_path.write_text(
        generated_compose.replace(
            "    command: [ \"sh\", \"-c\", \"sleep infinity\" ]\n",
            "    command: [ \"sh\", \"-c\", \"sleep infinity\" ]\n"
            "    network_mode: host\n",
        ),
        encoding="utf-8",
    )
    assert not declares_namespace(compose_path)

    compose_path.write_text(
        generated_compose.replace(
            "      - ${T_BENCH_TASK_LOGS_PATH}:${T_BENCH_CONTAINER_LOGS_PATH}\n",
            "      - /var/lib/seta:/foreign\n",
        ),
        encoding="utf-8",
    )
    assert not declares_namespace(compose_path)
    compose_path.write_text(
        "services:\n"
        "  client:\n"
        "    labels:\n"
        "      - harborrl.pool-namespace=swesmith\n"
        "    volumes:\n"
        "      - /anonymous\n"
        "networks:\n"
        "  default:\n"
        "    labels:\n"
        "      - harborrl.pool-namespace=swesmith\n",
        encoding="utf-8",
    )
    assert not declares_namespace(compose_path)

    compose_path.write_text(
        "services:\n"
        "  client:\n"
        "    labels:\n"
        "      - harborrl.pool-namespace=swesmith\n"
        "    volumes:\n"
        "      - cache:/cache\n"
        "networks:\n"
        "  default:\n"
        "    labels:\n"
        "      - harborrl.pool-namespace=swesmith\n"
        "volumes:\n"
        "  cache: {}\n",
        encoding="utf-8",
    )
    assert not declares_namespace(compose_path)

    compose_path.write_text(generated_compose, encoding="utf-8")
    compose_path.write_text(
        compose_path.read_text(encoding="utf-8").replace(
            "networks:\n  default:\n    labels:\n"
            "      - harborrl.pool-namespace=${TERMINAL_RL_POOL_NAMESPACE:-default}\n",
            "networks:\n  default: {}\n",
        ),
        encoding="utf-8",
    )
    assert not declares_namespace(compose_path)


def test_non_default_cleanup_requires_absent_container_and_projects() -> None:
    terminal_env_path = MODULE_PATH.parents[1] / "environments" / "terminal" / "runtime.py"
    source = terminal_env_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_docker_cleanup_postcondition"
    )
    namespace: dict[str, object] = {}
    exec(
        compile(ast.Module(body=[helper], type_ignores=[]), str(terminal_env_path), "exec"),
        namespace,
    )
    postcondition = namespace["_docker_cleanup_postcondition"]

    namespace["_docker_object_pool_namespace_state"] = (
        lambda *_args, **_kwargs: "absent"
    )
    namespace["_compose_project_pool_namespace_state"] = (
        lambda *_args, **_kwargs: "absent"
    )
    assert postcondition(
        client_container_name="swesmith.task.uid.slime-run",
        project_names={"swesmith-task-uid-slime-run"},
        timeout=1,
    ) == (True, [])

    namespace["_compose_project_pool_namespace_state"] = (
        lambda project, **_kwargs: "owned" if project == "leaked" else "absent"
    )
    ok, remaining = postcondition(
        client_container_name="swesmith.task.uid.slime-run",
        project_names={"clean", "leaked"},
        timeout=1,
    )
    assert not ok
    assert remaining == ["project:leaked=owned"]

    compact = " ".join(source.split())
    assert "force_cleanup_completed = await _force_remove_docker_objects_async(" in compact
    assert "not force_cleanup_started or not force_cleanup_completed" in compact
    assert "Docker cleanup could not be verified" in source


def test_non_default_cleanup_disabled_is_not_reported_complete() -> None:
    terminal_env_path = MODULE_PATH.parents[1] / "environments" / "terminal" / "runtime.py"
    tree = ast.parse(terminal_env_path.read_text(encoding="utf-8"))
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_force_remove_docker_objects"
    )
    namespace = {
        "_env_bool": lambda *_args, **_kwargs: False,
        "_current_pool_namespace": lambda: "swesmith",
    }
    exec(
        compile(ast.Module(body=[helper], type_ignores=[]), str(terminal_env_path), "exec"),
        namespace,
    )
    force_cleanup = namespace["_force_remove_docker_objects"]
    kwargs = {
        "trial_name": "swesmith.task.uid.slime-run",
        "client_container_name": "swesmith.task.uid.slime-run",
        "reason": "test",
    }
    assert force_cleanup(**kwargs) is False
    namespace["_current_pool_namespace"] = lambda: "default"
    assert force_cleanup(**kwargs) is True


def test_worker_captures_trusted_commits_before_agent_access() -> None:
    terminal_env_path = MODULE_PATH.parents[1] / "environments" / "terminal" / "runtime.py"
    tree = ast.parse(terminal_env_path.read_text(encoding="utf-8"))
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_capture_swesmith_stage_commits"
    )
    calls = []

    class FakeSubprocess:
        @staticmethod
        def run(args, **kwargs):
            calls.append((args, kwargs))
            return SimpleNamespace(
                returncode=0,
                stdout=(
                    "__SWESMITH_COMMITS__="
                    + "a" * 40
                    + " "
                    + "b" * 40
                    + "\n"
                ),
                stderr="",
            )

    namespace = {
        "subprocess": FakeSubprocess,
        "_GIT_COMMIT_RE": re.compile(r"^[0-9a-fA-F]{40,64}$"),
    }
    exec(
        compile(ast.Module(body=[helper], type_ignores=[]), str(terminal_env_path), "exec"),
        namespace,
    )
    capture = namespace["_capture_swesmith_stage_commits"]
    assert capture("safe-container") == ("a" * 40, "b" * 40)
    assert calls[0][0][:5] == ["docker", "exec", "-u", "root", "safe-container"]
    assert calls[0][1]["check"] is False


def test_swesmith_reward_is_binary_while_existing_reward_stays_dense() -> None:
    terminal_env_path = MODULE_PATH.parents[1] / "environments" / "terminal" / "runtime.py"
    tree = ast.parse(terminal_env_path.read_text(encoding="utf-8"))
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_terminal_test_reward"
    )

    class Status:
        PASSED = "passed"
        FAILED = "failed"

    namespace = {"UnitTestStatus": Status}
    exec(
        compile(ast.Module(body=[helper], type_ignores=[]), str(terminal_env_path), "exec"),
        namespace,
    )
    reward = namespace["_terminal_test_reward"]
    results = {"a": Status.PASSED, "b": Status.FAILED}
    assert reward(results, "swesmith") == (0.0, 1)
    assert reward({"a": Status.PASSED}, "swesmith") == (1.0, 1)
    assert reward(results, "seta") == (0.5, 1)


def test_worker_launcher_disables_host_wide_shim_cleanup() -> None:
    remote_dir = MODULE_PATH.parents[2] / "deploy" / "workers"
    launcher = (remote_dir / "run_pool_server_swesmith.sh").read_text(
        encoding="utf-8"
    )
    pool_server = (
        MODULE_PATH.parents[1] / "platform" / "worker_pool.py"
    ).read_text(encoding="utf-8")
    assert 'WORKER_SHIM_CLEANUP_ENABLED="${WORKER_SHIM_CLEANUP_ENABLED:-0}"' in launcher
    assert 'WORKER_ORPHAN_DOCKER_SWEEP="${WORKER_ORPHAN_DOCKER_SWEEP:-1}"' in launcher
    assert 'if not _env_bool("WORKER_SHIM_CLEANUP_ENABLED", True):' in pool_server
    assert 'SWESMITH_WORKER_DATA_PREFLIGHT:-1' in launcher
    assert "formal SWE-smith worker requires SWESMITH_RUN_PASS_TO_PASS=1" in launcher
    assert "SWESMITH_ENV_DIR must be DATASET_DIR/swesmith_env" in launcher
    assert "COMPOSE_OVERRIDE_PATH is unsupported" in launcher
    assert "requires a non-default Docker pool namespace" in launcher
    assert 'TERMINAL_RL_POOL_NAMESPACE}" == "default"' in launcher
    assert "import yaml" in launcher
    assert 'data_preflight=ok format=v' in launcher
    assert 'export POOL_SERVER_VENV' in launcher
    platform_dir = MODULE_PATH.parents[1] / "platform"
    launcher_parts = [platform_dir / "slime_train.sh"] + sorted(
        (platform_dir / "slime_train").glob("lib_*.sh")
    )
    training = "\n".join(part.read_text(encoding="utf-8") for part in launcher_parts)
    smoke = (
        MODULE_PATH.parents[2] / "tools" / "dev" / "smoke_swesmith_worker.py"
    ).read_text(encoding="utf-8")
    assert 'ROUTER_FORWARD_TIMEOUT:-5100' in training
    assert 'ENV_RESET_HTTP_TIMEOUT:-5400' in training
    assert "foreign_rows += 1" in training
    assert "reset_session_timeout + 600" in smoke
    assert 'if _current_pool_namespace() == "default":' in (
        MODULE_PATH.parents[1] / "environments" / "terminal" / "runtime.py"
    ).read_text(encoding="utf-8")


def test_worker_launcher_rejects_default_pool_namespace() -> None:
    launcher = MODULE_PATH.parents[2] / "deploy" / "workers" / "run_pool_server_swesmith.sh"
    result = subprocess.run(
        ["bash", str(launcher)],
        env={**os.environ, "TERMINAL_RL_POOL_NAMESPACE": "default"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "requires a non-default Docker pool namespace" in result.stderr


def test_downloader_full_mode_requires_canonical_bounded_source(
    tmp_path: Path,
) -> None:
    script = MODULE_PATH.parent / "download_swesmith.sh"
    base_env = {
        **os.environ,
        "MODE": "full",
        "INSTALL_DATASETS": "0",
        "CREATE_ENV_DIRS": "0",
        "OUTPUT_DIR": str(tmp_path / "output"),
        "DATASET_REVISION": "main",
    }
    disabled = subprocess.run(
        ["bash", str(script)],
        env={**base_env, "ALLOW_FULL": "0"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert disabled.returncode == 1
    assert "disabled by default" in disabled.stdout

    moving_revision = subprocess.run(
        ["bash", str(script)],
        env={
            **base_env,
            "ALLOW_FULL": "1",
            "ALLOW_MOVING_REVISION": "0",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert moving_revision.returncode == 1
    assert "requires audited revision" in moving_revision.stderr

    canonical_env = {
        **base_env,
        "ALLOW_FULL": "1",
        "DATASET_REVISION": converter.CANONICAL_SWESMITH_REVISION,
    }
    wrong_test_caps = subprocess.run(
        ["bash", str(script)],
        env={**canonical_env, "MAX_FAIL_TO_PASS_COUNT": "49"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert wrong_test_caps.returncode == 1
    assert "requires audited training caps F2P=50/P2P=200" in wrong_test_caps.stderr

    local_source = subprocess.run(
        ["bash", str(script)],
        env={**canonical_env, "INPUT_JSONL": str(tmp_path / "raw.jsonl")},
        capture_output=True,
        text=True,
        check=False,
    )
    assert local_source.returncode == 1
    assert "rejects local input" in local_source.stderr

    smoke_named_train = subprocess.run(
        ["bash", str(script)],
        env={
            **base_env,
            "MODE": "smoke",
            "OUTPUT_NAME": "train.jsonl",
            "ALLOW_SMOKE_TRAIN_NAME": "0",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert smoke_named_train.returncode == 1
    assert "smoke mode cannot write train.jsonl" in smoke_named_train.stderr


def test_formal_artifact_manifest_binds_canonical_full_jsonl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prompt = tmp_path / "train.jsonl"
    prompt.write_text(json.dumps(_item()) + "\n", encoding="utf-8")
    digest = converter._file_sha256(prompt)
    revision = converter.CANONICAL_SWESMITH_REVISION
    source_files = [
        {"path": path, "sha256": sha256, "bytes": size}
        for path, sha256, size in converter.CANONICAL_SWESMITH_SOURCE_FILES
    ]
    stats = {
        "manifest_schema_version": 2,
        "source_dataset": "SWE-bench/SWE-smith",
        "source": "SWE-bench/SWE-smith:train",
        "split": "train",
        "source_revision": revision,
        "processed": 1,
        "converted": 1,
        "max_samples": None,
        "max_source_samples": None,
        "skipped": {},
        "conversion_mode": "full",
        "dataset_revision": revision,
        "artifact_rows": 1,
        "artifact_sha256": digest,
        "artifact_bytes": prompt.stat().st_size,
        "converter_sha256": converter._converter_fingerprint(),
        "source_backend": "parquet",
        "hf_endpoint": "https://huggingface.co",
        "source_files": source_files,
        "runner_counts": {"pytest": 1},
        "max_fail_to_pass_count": converter.CANONICAL_MAX_FAIL_TO_PASS_COUNT,
        "max_pass_to_pass_count": converter.CANONICAL_MAX_PASS_TO_PASS_COUNT,
        "truncated_fail_to_pass": 0,
        "truncated_pass_to_pass": 0,
        "env_errors": 0,
    }
    stats_path = tmp_path / "convert_stats.json"
    stats_path.write_text(json.dumps(stats), encoding="utf-8")

    with pytest.raises(ValueError, match="requires 59136 rows"):
        converter.validate_swesmith_artifact_manifest(
            prompt,
            stats_path=stats_path,
            require_full=True,
            artifact_rows=1,
            artifact_sha256=digest,
        )

    monkeypatch.setattr(converter, "CANONICAL_SWESMITH_ROWS", 1)
    monkeypatch.setattr(converter, "CANONICAL_SWESMITH_RUNNER_COUNTS", {"pytest": 1})
    monkeypatch.setattr(
        converter, "CANONICAL_SWESMITH_ARTIFACT_BYTES", prompt.stat().st_size
    )
    monkeypatch.setattr(converter, "CANONICAL_SWESMITH_ARTIFACT_SHA256", digest)
    monkeypatch.setattr(converter, "CANONICAL_TRUNCATED_FAIL_TO_PASS", 0)
    monkeypatch.setattr(converter, "CANONICAL_TRUNCATED_PASS_TO_PASS", 0)
    validated = converter.validate_swesmith_artifact_manifest(
        prompt,
        stats_path=stats_path,
        require_full=True,
        artifact_rows=1,
        artifact_sha256=digest,
    )
    assert validated["dataset_revision"] == revision

    for field, value, message in (
        ("conversion_mode", "smoke", "conversion_mode=full"),
        ("source", "local:/tmp/raw.jsonl", "pinned HF source"),
        ("dataset_revision", "b" * 40, "audited dataset revision"),
        ("converter_sha256", "0" * 64, "different converter"),
        ("artifact_sha256", "0" * 64, "artifact SHA256"),
        ("max_samples", 1, "rejects capped source conversion"),
        ("processed", 2, "gap-free conversion"),
        ("skipped", {"unsupported_runner": 1}, "gap-free conversion"),
        ("max_fail_to_pass_count", 49, "audited F2P/P2P caps"),
        ("truncated_pass_to_pass", -1, "invalid truncated_pass_to_pass"),
        ("env_errors", 1, "generation was incomplete"),
    ):
        broken = dict(stats)
        broken[field] = value
        stats_path.write_text(json.dumps(broken), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            converter.validate_swesmith_artifact_manifest(
                prompt,
                stats_path=stats_path,
                require_full=True,
                artifact_rows=1,
                artifact_sha256=digest,
            )

    stats_path.write_text(json.dumps(stats), encoding="utf-8")
    with pytest.raises(ValueError, match="observed JSONL rows and SHA256"):
        converter.validate_swesmith_artifact_manifest(
            prompt, stats_path=stats_path, require_full=True
        )


def test_smoke_close_failure_is_not_reported_as_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    smoke_path = MODULE_PATH.parents[2] / "tools" / "dev" / "smoke_swesmith_worker.py"
    smoke_spec = importlib.util.spec_from_file_location("smoke_swesmith_worker", smoke_path)
    assert smoke_spec is not None and smoke_spec.loader is not None
    smoke = importlib.util.module_from_spec(smoke_spec)
    smoke_spec.loader.exec_module(smoke)

    item = _item()
    dataset = tmp_path / "smoke.jsonl"
    dataset.write_text(json.dumps(item) + "\n", encoding="utf-8")
    responses = iter(
        [
            (200, {"ok": True, "lease_id": "lease-1"}),
            (200, {"ok": True, "user_msg": "ready"}),
            (500, {"ok": False, "error": "close failed"}),
        ]
    )
    monkeypatch.setattr(smoke, "_get_json", lambda *_args, **_kwargs: (200, {"ok": True}))
    monkeypatch.setattr(smoke, "_post_json", lambda *_args, **_kwargs: next(responses))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(smoke_path),
            "--worker-url",
            "http://worker:18082",
            "--dataset",
            str(dataset),
            "--skip-ref-check",
            "--skip-evaluate",
        ],
    )
    assert smoke.main() == 8


def test_smoke_zero_score_requires_expected_evaluation_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    smoke_path = MODULE_PATH.parents[2] / "tools" / "dev" / "smoke_swesmith_worker.py"
    smoke_spec = importlib.util.spec_from_file_location("smoke_swesmith_worker", smoke_path)
    assert smoke_spec is not None and smoke_spec.loader is not None
    smoke = importlib.util.module_from_spec(smoke_spec)
    smoke_spec.loader.exec_module(smoke)

    item = _item()
    dataset = tmp_path / "smoke.jsonl"
    dataset.write_text(json.dumps(item) + "\n", encoding="utf-8")
    responses = iter(
        [
            (200, {"ok": True, "lease_id": "lease-1"}),
            (200, {"ok": True, "user_msg": "ready"}),
            (
                200,
                {
                    "ok": True,
                    "score": 0.0,
                    "details": {"reason": "eval_timeout"},
                },
            ),
            (200, {"ok": True, "found": True}),
        ]
    )
    monkeypatch.setattr(smoke, "_get_json", lambda *_args, **_kwargs: (200, {"ok": True}))
    monkeypatch.setattr(smoke, "_post_json", lambda *_args, **_kwargs: next(responses))
    monkeypatch.setattr(
        smoke, "_wait_for_close", lambda *_args, **_kwargs: (True, None)
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(smoke_path),
            "--worker-url",
            "http://worker:18082",
            "--dataset",
            str(dataset),
            "--skip-ref-check",
            "--expect-score",
            "0",
            "--expect-reason",
            "test_exit_nonzero",
        ],
    )
    assert smoke.main() == 7


def test_smoke_rejects_expectations_when_evaluate_is_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    smoke_path = MODULE_PATH.parents[2] / "tools" / "dev" / "smoke_swesmith_worker.py"
    smoke_spec = importlib.util.spec_from_file_location("smoke_swesmith_worker", smoke_path)
    assert smoke_spec is not None and smoke_spec.loader is not None
    smoke = importlib.util.module_from_spec(smoke_spec)
    smoke_spec.loader.exec_module(smoke)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(smoke_path),
            "--worker-url",
            "http://worker:18082",
            "--skip-evaluate",
            "--expect-score",
            "0",
        ],
    )
    with pytest.raises(SystemExit) as exc_info:
        smoke.main()
    assert exc_info.value.code == 2


@pytest.mark.parametrize("raw_score", [float("nan"), float("inf"), True, "0"])
def test_smoke_rejects_non_finite_or_non_numeric_response_scores(
    raw_score: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    smoke_path = MODULE_PATH.parents[2] / "tools" / "dev" / "smoke_swesmith_worker.py"
    smoke_spec = importlib.util.spec_from_file_location(
        "smoke_swesmith_worker", smoke_path
    )
    assert smoke_spec is not None and smoke_spec.loader is not None
    smoke = importlib.util.module_from_spec(smoke_spec)
    smoke_spec.loader.exec_module(smoke)
    dataset = tmp_path / "smoke.jsonl"
    dataset.write_text(json.dumps(_item()) + "\n", encoding="utf-8")
    responses = iter(
        [
            (200, {"ok": True, "lease_id": "lease-1"}),
            (200, {"ok": True, "user_msg": "ready"}),
            (
                200,
                {
                    "ok": True,
                    "score": raw_score,
                    "details": {"reason": "test_exit_nonzero"},
                },
            ),
            (200, {"ok": True, "found": True}),
        ]
    )
    monkeypatch.setattr(smoke, "_get_json", lambda *_args, **_kwargs: (200, {"ok": True}))
    monkeypatch.setattr(smoke, "_post_json", lambda *_args, **_kwargs: next(responses))
    monkeypatch.setattr(
        smoke, "_wait_for_close", lambda *_args, **_kwargs: (True, None)
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(smoke_path),
            "--worker-url",
            "http://worker:18082",
            "--dataset",
            str(dataset),
            "--skip-ref-check",
            "--expect-score",
            "0",
            "--expect-reason",
            "test_exit_nonzero",
        ],
    )
    assert smoke.main() == 7


def test_smoke_rejects_non_finite_expected_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    smoke_path = MODULE_PATH.parents[2] / "tools" / "dev" / "smoke_swesmith_worker.py"
    smoke_spec = importlib.util.spec_from_file_location(
        "smoke_swesmith_worker", smoke_path
    )
    assert smoke_spec is not None and smoke_spec.loader is not None
    smoke = importlib.util.module_from_spec(smoke_spec)
    smoke_spec.loader.exec_module(smoke)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(smoke_path),
            "--worker-url",
            "http://worker:18082",
            "--expect-score",
            "nan",
        ],
    )
    with pytest.raises(SystemExit) as exc_info:
        smoke.main()
    assert exc_info.value.code == 2


@pytest.mark.parametrize("evaluate_body", [{"ok": True}, {"ok": True, "score": 0.5}])
def test_smoke_requires_binary_score_even_without_expectation(
    evaluate_body: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    smoke_path = MODULE_PATH.parents[2] / "tools" / "dev" / "smoke_swesmith_worker.py"
    smoke_spec = importlib.util.spec_from_file_location(
        "smoke_swesmith_worker", smoke_path
    )
    assert smoke_spec is not None and smoke_spec.loader is not None
    smoke = importlib.util.module_from_spec(smoke_spec)
    smoke_spec.loader.exec_module(smoke)
    dataset = tmp_path / "smoke.jsonl"
    dataset.write_text(json.dumps(_item()) + "\n", encoding="utf-8")
    responses = iter(
        [
            (200, {"ok": True, "lease_id": "lease-1"}),
            (200, {"ok": True, "user_msg": "ready"}),
            (200, evaluate_body),
            (200, {"ok": True, "found": True}),
        ]
    )
    monkeypatch.setattr(smoke, "_get_json", lambda *_args, **_kwargs: (200, {"ok": True}))
    monkeypatch.setattr(smoke, "_post_json", lambda *_args, **_kwargs: next(responses))
    monkeypatch.setattr(
        smoke, "_wait_for_close", lambda *_args, **_kwargs: (True, None)
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(smoke_path),
            "--worker-url",
            "http://worker:18082",
            "--dataset",
            str(dataset),
            "--skip-ref-check",
        ],
    )
    assert smoke.main() == 7


def test_smoke_observes_worker_close_cleanup_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    smoke_path = MODULE_PATH.parents[2] / "tools" / "dev" / "smoke_swesmith_worker.py"
    smoke_spec = importlib.util.spec_from_file_location(
        "smoke_swesmith_worker", smoke_path
    )
    assert smoke_spec is not None and smoke_spec.loader is not None
    smoke = importlib.util.module_from_spec(smoke_spec)
    smoke_spec.loader.exec_module(smoke)
    failure = {
        "lease_id": "lease-1",
        "reason": "close_exception",
        "error": "cleanup postcondition failed",
    }
    monkeypatch.setattr(
        smoke,
        "_get_json",
        lambda *_args, **_kwargs: (
            200,
            {"ok": True, "pool": {"recent_close_failures": [failure]}},
        ),
    )
    assert smoke._wait_for_close("http://worker:18082", "lease-1", 1) == (
        False,
        failure,
    )

    pool_source = (
        MODULE_PATH.parents[1] / "platform" / "worker_pool.py"
    ).read_text(encoding="utf-8")
    assert '"recent_close_failures"' in pool_source
    assert "_record_close_failure(" in pool_source
