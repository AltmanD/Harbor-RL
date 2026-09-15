from __future__ import annotations

import importlib.util
from pathlib import Path


TESTS_DIR = Path(__file__).resolve().parent
SCRIPT_PATH = TESTS_DIR.parents[1] / "tools" / "analysis" / "analyze_trajectories.py"
SPEC = importlib.util.spec_from_file_location("analyze_trajectories", SCRIPT_PATH)
assert SPEC is not None
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
analyze = MODULE.analyze


def test_analyze_reports_tau2_non_solo_follow_up(tmp_path: Path):
    run_dir = tmp_path / "run"
    traj_dir = run_dir / "trajectories" / "sample_0"
    traj_dir.mkdir(parents=True)
    (run_dir / "metrics").mkdir(parents=True)

    (traj_dir / "traj.json").write_text(
        """
{
  "info": {
    "task_name": "tau2_mock_create_task_1",
    "status": "Status.TRUNCATED",
    "uid": "abc123",
    "num_turns": 2
  },
  "reward": {
    "accuracy": 0.0,
    "raw_score": 0.0
  },
  "turns": [
    {
      "turn_idx": 0,
      "context_messages": [
        {
          "role": "user",
          "content": "<instructions>\\nYou are solving a tau2-bench task in non-solo mode.\\n</instructions>"
        }
      ],
      "assistant_output": "Let me confirm that for you."
    },
    {
      "turn_idx": 1,
      "context_messages": [],
      "assistant_output": "Done.",
      "env_user_message": "Thanks, that helps."
    }
  ]
}
""".strip()
    )

    report = analyze(run_dir=run_dir)

    tau2_summary = report["tau2_non_solo_summary"]
    assert tau2_summary["n_tau2_trajectories"] == 1
    assert tau2_summary["n_non_solo_trajectories"] == 1
    assert tau2_summary["n_non_solo_with_env_user_message"] == 1
    assert tau2_summary["n_non_solo_without_env_user_message"] == 0
    assert tau2_summary["sample_non_solo_with_env_user_message"][0]["dir"] == "sample_0"
    assert tau2_summary["sample_non_solo_with_env_user_message"][0]["env_user_message_turns"] == [1]
