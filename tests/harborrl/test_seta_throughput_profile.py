from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _load_analyzer():
    path = ROOT / "tools" / "analysis" / "analyze_seta_throughput.py"
    spec = importlib.util.spec_from_file_location("analyze_seta_throughput", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_throughput_analyzer_separates_rollout_and_actor(tmp_path: Path):
    module = _load_analyzer()
    run_dir = tmp_path / "run"
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True)
    (log_dir / "train.log").write_text(
        "(RolloutManager pid=1) perf 0: {'perf/rollout_time': 120.0, 'perf/tokens_per_gpu_per_sec': 50.0}\n"
        "(MegatronTrainRayActor pid=2) perf 0: {'perf/step_time': 125.0, 'perf/train_time': 5.0, "
        "'perf/actor_train_time': 4.0, 'perf/log_probs_time': 1.0, 'perf/train_wait_time': 120.0, "
        "'perf/wait_time_ratio': 0.96, 'perf/update_weights_time': 0.2, 'perf/save_model_time': 2.0}\n",
        encoding="utf-8",
    )

    report = module.analyze(run_dir, 1)

    assert report["measured_rollout_steps"] == 1
    assert report["rollout_time_sec"]["mean"] == 120.0
    assert report["actor_wait_ratio"]["mean"] == 0.96
    assert report["projected_1000_rollout_steps"]["hours"] == 120000 / 3600


def test_fixed12_default_profile_is_sequential_and_four_engine():
    launcher = (ROOT / "runs" / "refactor" / "launch_seta_fixed12_score_8g.sh").read_text()

    assert 'SETA_EXECUTION_PROFILE:-sequential-throughput-v1' in launcher
    assert "sequential-env12-v1)" in launcher
    assert 'ENV_REMOTE_MAX_ACTIVE_RUNS="${ENV_REMOTE_MAX_ACTIVE_RUNS:-32}"' in launcher
    assert 'ENV_REMOTE_MAX_RUNS_PER_TASK="${ENV_REMOTE_MAX_RUNS_PER_TASK:-8}"' in launcher
    assert 'ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}"' in launcher
    assert 'SLIME_USE_FAULT_TOLERANCE="${SLIME_USE_FAULT_TOLERANCE:-1}"' in launcher
    assert "ACTOR_GPUS + ROLLOUT_GPUS != NUM_GPUS" in launcher
    assert "ACTOR_GPUS % TP_SIZE" in launcher
    assert "ROLLOUT_GPUS % ROLLOUT_NUM_GPUS_PER_ENGINE" in launcher


def test_slime_args_exposes_opt_in_observability_and_fault_tolerance():
    launcher_lib = (ROOT / "harborrl" / "platform" / "slime_train" / "lib_args.sh").read_text()
    launch_runtime = (ROOT / "harborrl" / "platform" / "slime_train" / "lib_launch.sh").read_text()

    assert "--sglang-server-concurrency" in launcher_lib
    assert "--use-fault-tolerance" in launcher_lib
    assert "--save-debug-rollout-data" in launcher_lib
    assert '"seta_execution_profile"' in launch_runtime
    assert "SLIME_USE_FAULT_TOLERANCE" in launch_runtime
    assert '\\"NCCL_SOCKET_IFNAME\\": \\"${NCCL_SOCKET_IFNAME}\\"' in launch_runtime
    assert '\\"GLOO_SOCKET_IFNAME\\": \\"${GLOO_SOCKET_IFNAME}\\"' in launch_runtime
