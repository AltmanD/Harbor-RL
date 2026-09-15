import json

from slime.world_model.seta_dataset import (
    build_data_manifest,
    load_terminal_transitions,
    transitions_from_seta_trajectory,
    transitions_from_tb21_trajectory,
)


def _trajectory():
    return {
        "info": {
            "uid": "trajectory-1",
            "task_name": "42",
            "data_source": "terminal_bench",
            "status": "Status.COMPLETED",
            "rollout_id": 3,
            "train_step": 6,
        },
        "reward": {"score": 1.0},
        "turns": [
            {
                "turn_idx": 0,
                "context_messages": [{"role": "user", "content": "inspect"}],
                "assistant_output": "run pwd",
                "tool_calls": [{"tool_name": "bash", "args": {"command": "pwd"}, "result": "/tmp"}],
            },
            {
                "turn_idx": 1,
                "context_messages": [
                    {"role": "user", "content": "inspect"},
                    {"role": "tool", "content": "/tmp"},
                ],
                "assistant_output": "done",
                "tool_calls": [],
            },
        ],
    }


def test_seta_transition_boundaries_include_next_context():
    transitions = transitions_from_seta_trajectory(_trajectory(), source_path="/tmp/traj.json")

    assert len(transitions) == 2
    assert transitions[0].action_text.startswith("run pwd")
    assert "/tmp" in transitions[0].feedback_text
    assert transitions[0].has_next is True
    assert transitions[0].next_context_messages[-1]["role"] == "tool"
    assert transitions[1].done is True
    assert transitions[1].has_next is False


def test_load_terminal_transitions_reads_trajectory_directory(tmp_path):
    run = tmp_path / "sample"
    run.mkdir()
    (run / "traj.json").write_text(json.dumps(_trajectory()), encoding="utf-8")

    transitions = load_terminal_transitions(tmp_path, max_transitions=1)

    assert len(transitions) == 1
    assert transitions[0].trajectory_id == "trajectory-1"


def test_tb21_atif_adapter_emits_agent_transitions_and_terminal_reward(tmp_path):
    trial = tmp_path / "task__trial"
    (trial / "verifier").mkdir(parents=True)
    (trial / "verifier" / "reward.txt").write_text("1\n", encoding="utf-8")
    payload = {
        "schema_version": "ATIF-v1.7",
        "session_id": "tb21-session",
        "steps": [
            {"step_id": 1, "source": "user", "message": "inspect"},
            {
                "step_id": 2,
                "source": "agent",
                "message": '{"commands": [{"keystrokes": "pwd\\n"}]}',
                "observation": {"results": [{"content": "/tmp"}]},
            },
            {
                "step_id": 3,
                "source": "agent",
                "message": '{"commands": [{"keystrokes": "exit\\n"}]}',
                "observation": {"results": [{"content": "done"}]},
            },
        ],
    }
    path = trial / "trajectory.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    transitions = transitions_from_tb21_trajectory(payload, source_path=str(path))

    assert len(transitions) == 2
    assert transitions[0].data_source == "tb21"
    assert transitions[0].has_next is True
    assert transitions[1].done is True
    assert transitions[0].context_messages[0]["role"] == "user"
    assert transitions[0].reward is None
    assert transitions[1].reward == 1.0
    assert "tmp" in transitions[0].feedback_text
    manifest = build_data_manifest(transitions, requested_source="tb21")
    assert manifest["trajectory_count"] == 1
    assert manifest["terminal_transition_count"] == 1


def test_tb21_raw_tool_call_result_is_marked_for_clean_feedback_filter(tmp_path):
    trial = tmp_path / "task__raw"
    (trial / "verifier").mkdir(parents=True)
    (trial / "verifier" / "reward.txt").write_text("1\n", encoding="utf-8")
    payload = {
        "session_id": "tb-raw",
        "steps": [
            {"source": "user", "message": "inspect"},
            {
                "source": "agent",
                "message": "run pwd",
                "tool_calls": [
                    {
                        "tool_name": "bash",
                        "result": {"stdout": "/tmp", "stderr": "", "exit_code": 0},
                    }
                ],
                # A pane capture is also present, but clean feedback must use
                # the per-call result above rather than this screen snapshot.
                "observation": {"results": [{"content": "root@host:/app#"}]},
            },
        ],
    }
    path = trial / "trajectory.json"
    transitions = transitions_from_tb21_trajectory(payload, source_path=str(path))

    assert len(transitions) == 1
    assert transitions[0].feedback_text.startswith("<tool_result name=bash>")
    assert "/tmp" in transitions[0].feedback_text
    assert "root@host" not in transitions[0].feedback_text

    path.write_text(json.dumps(payload), encoding="utf-8")
    clean = load_terminal_transitions(path, require_tool_feedback=True, data_source="tb21")
    assert len(clean) == 1


def test_tb21_pane_capture_does_not_satisfy_clean_feedback_filter(tmp_path):
    trial = tmp_path / "task__pane"
    trial.mkdir(parents=True)
    payload = {
        "session_id": "tb-pane",
        "steps": [
            {"source": "user", "message": "inspect"},
            {
                "source": "agent",
                "message": "run pwd",
                "observation": {"results": [{"content": "root@host:/app#"}]},
            },
        ],
    }
    path = trial / "trajectory.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert load_terminal_transitions(path, require_tool_feedback=True, data_source="tb21") == []


def test_loader_prioritizes_tb21_and_reads_multiple_records_from_jsonl(tmp_path):
    tb_trial = tmp_path / "tb" / "task"
    (tb_trial / "verifier").mkdir(parents=True)
    (tb_trial / "verifier" / "reward.txt").write_text("0.5\n", encoding="utf-8")
    payload = {
        "session_id": "tb-session",
        "steps": [
            {"source": "user", "message": "u"},
            {"source": "agent", "message": "a", "observation": "o"},
        ],
    }
    (tb_trial / "trajectory.json").write_text(json.dumps(payload), encoding="utf-8")
    records = tmp_path / "records.jsonl"
    records.write_text(
        "\n".join(
            json.dumps(
                {
                    "trajectory_id": f"r-{idx}",
                    "turn_idx": 0,
                    "context_text": "ctx",
                    "action_text": "act",
                    "feedback_text": "fb",
                }
            )
            for idx in range(2)
        )
        + "\n",
        encoding="utf-8",
    )
    loaded = load_terminal_transitions(
        tmp_path,
        max_trajectories=3,
        data_source="auto",
    )
    assert loaded[0].data_source == "tb21"
    assert {row.trajectory_id for row in loaded} == {"tb-session", "r-0", "r-1"}
