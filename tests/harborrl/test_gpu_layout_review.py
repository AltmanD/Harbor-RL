"""Failure-path regressions found while integrating the GPU layout branch."""

import json
from types import SimpleNamespace

from harborrl import cli
from harborrl.config import launch_plan, load_config
from harborrl.platform.run_leases import close_run_leases
from test_cleanup_entrypoints import profile


def test_cleanup_continues_past_incomplete_lease_record(tmp_path, monkeypatch):
    leases = tmp_path / "leases"
    leases.mkdir()
    (leases / "0-interrupted.json").write_text('{"endpoint":')
    (leases / "1-valid.json").write_text(
        json.dumps(
            {
                "endpoint": "http://worker:18081",
                "lease_id": "owned-lease",
            }
        )
    )
    closed = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return b'{"ok": true}'

    class Opener:
        def open(self, req, timeout):
            closed.append(json.loads(req.data)["lease_id"])
            return Response()

    monkeypatch.setattr(
        "harborrl.platform.run_leases.request.build_opener", lambda *a: Opener()
    )
    close_run_leases(tmp_path)
    assert closed == ["owned-lease"]
    outcomes = json.loads((tmp_path / "lease-cleanup.json").read_text())
    assert any(
        row.get("record") == "0-interrupted.json" and row.get("error")
        for row in outcomes
    )
    assert any(
        row.get("lease_id") == "owned-lease" and row.get("result", {}).get("ok")
        for row in outcomes
    )


def test_colocate_doctor_rejects_missing_preload_library(tmp_path, monkeypatch):
    plan = launch_plan(
        load_config(
            profile(tmp_path),
            [
                "deployment.layout=colocate",
                "deployment.actor_gpus=4",
                "deployment.rollout_gpus=4",
            ],
        )
    )
    spec = SimpleNamespace(origin=str(tmp_path / "torch_memory_saver" / "__init__.py"))
    monkeypatch.setattr(
        cli.importlib.util,
        "find_spec",
        lambda name: spec if name == "torch_memory_saver" else None,
    )
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)
    # The ELF loader warns but may exit successfully when LD_PRELOAD is missing.
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(
            returncode=0,
            stdout="2.6.0",
            stderr="LD_PRELOAD cannot be preloaded: ignored",
        ),
    )

    class Opener:
        def open(self, *a, **k):
            raise OSError("no worker needed for this preflight test")

    monkeypatch.setattr(cli.request, "build_opener", lambda *a: Opener())
    checks = cli.doctor(plan)
    preload_checks = [
        row
        for row in checks
        if row.get("dependency") == "memory saver CUDA preload ABI"
    ]
    assert len(preload_checks) == 1
    assert preload_checks[0]["ok"] is False


def test_failed_lease_write_does_not_publish_partial_record(tmp_path, monkeypatch):
    from pathlib import Path
    import pytest
    from harborrl.platform.run_leases import record_lease

    monkeypatch.setenv("RUN_DIR", str(tmp_path))
    monkeypatch.setenv("HARBORRL_VERIFY_POLICY_POOL", "1")
    original_write = Path.write_text

    def interrupted_write(path, data, *args, **kwargs):
        original_write(path, data[:5], *args, **kwargs)
        raise OSError("interrupted lease write")

    monkeypatch.setattr(Path, "write_text", interrupted_write)
    with pytest.raises(OSError, match="interrupted"):
        record_lease("http://worker:18081", "owned-lease")
    assert not list((tmp_path / "leases").iterdir())
