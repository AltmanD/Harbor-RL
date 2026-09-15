from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backends" / "slime"))

from slime.utils.arguments import _resolve_eval_datasets


CONVERTED = ROOT / "benchmarks" / "datasets" / "seta_env_convert"


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _task_id(row: dict) -> str:
    return str(row["metadata"]["task_name"])


def test_fixed48_is_disjoint_stratified_and_manifested():
    eval_path = CONVERTED / "eval_fixed48_v2.jsonl"
    train_path = CONVERTED / "train_minus_eval48_v2.filtered.jsonl"
    manifest_path = CONVERTED / "eval_fixed48_v2.manifest.json"
    eval_rows = _rows(eval_path)
    train_rows = _rows(train_path)
    manifest = json.loads(manifest_path.read_text())

    eval_ids = {_task_id(row) for row in eval_rows}
    train_ids = {_task_id(row) for row in train_rows}
    assert len(eval_rows) == len(eval_ids) == 48
    assert not eval_ids.intersection(train_ids)
    assert manifest["eval_task_count"] == 48
    assert manifest["training_task_count"] == len(train_rows)
    assert manifest["seed"] == 20260808

    difficulties = Counter()
    for task_id in eval_ids:
        task = yaml.safe_load(
            (
                ROOT
                / "benchmarks"
                / "environments"
                / "seta_env"
                / task_id
                / "task.yaml"
            ).read_text()
        )
        difficulties[str(task["difficulty"]).lower()] += 1
    assert difficulties == {"hard": 25, "medium": 22, "easy": 1}

    digest = hashlib.sha256(eval_path.read_bytes()).hexdigest()
    assert digest == "e34436edf263e33c22622174f8c67f509a29ff0e382da47d125daa9290bed03e"


def test_fixed48_eval_config_has_separate_exploit_and_explore_streams():
    config = yaml.safe_load(
        (ROOT / "configs" / "evaluation" / "seta_fixed48_v2.yaml").read_text()
    )
    datasets = {item["name"]: item for item in config["eval"]["datasets"]}
    assert config["eval"]["defaults"]["max_response_len"] == 8192
    assert datasets["seta_fixed48_exploit"]["temperature"] == 0.0
    assert datasets["seta_fixed48_exploit"]["n_samples_per_eval_prompt"] == 1
    assert datasets["seta_fixed48_explore"]["temperature"] == 1.0
    assert datasets["seta_fixed48_explore"]["n_samples_per_eval_prompt"] == 8


def test_fixed48_eval_config_resolves_in_runtime_environment(monkeypatch):
    monkeypatch.setenv("REPO_ROOT", str(ROOT))
    args = SimpleNamespace(
        eval_config=str(ROOT / "configs" / "evaluation" / "seta_fixed48_v2.yaml"),
        eval_prompt_data=None,
    )
    datasets = _resolve_eval_datasets(args)
    assert [dataset.name for dataset in datasets] == [
        "seta_fixed48_exploit",
        "seta_fixed48_explore",
    ]
    assert all(Path(dataset.path).is_file() for dataset in datasets)
    assert datasets[0].n_samples_per_eval_prompt == 1
    assert datasets[1].n_samples_per_eval_prompt == 8
    assert all(dataset.max_response_len == 8192 for dataset in datasets)


def test_formal_launcher_bounds_shared_worker_concurrency():
    launcher = (ROOT / "runs" / "refactor" / "launch_seta_fixed48_v2.sh").read_text()
    assert 'ENV_REMOTE_MAX_ACTIVE_TASKS="${ENV_REMOTE_MAX_ACTIVE_TASKS:-12}"' in launcher
    assert 'ENV_REMOTE_MAX_ACTIVE_RUNS="${ENV_REMOTE_MAX_ACTIVE_RUNS:-12}"' in launcher
    assert 'ENV_REMOTE_ADMISSION_TIMEOUT="${ENV_REMOTE_ADMISSION_TIMEOUT:-0}"' in launcher
    assert 'EVAL_ROLLOUT_MAX_CONCURRENCY="${EVAL_ROLLOUT_MAX_CONCURRENCY:-4}"' in launcher
    assert 'MAX_CKPT_KEEP="${MAX_CKPT_KEEP:-1}"' in launcher
