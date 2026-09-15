import argparse
import json

from slime.world_model.online_learner import run_online, _load_snapshot_transitions
from slime.world_model.replay_buffer import TrajectoryReplayBuffer
from slime.world_model.seta_dataset import TerminalTransition


def _transition(trajectory_id: str, turn_idx: int, reward: float | None = None) -> TerminalTransition:
    return TerminalTransition(
        trajectory_id=trajectory_id,
        task_name=f"task-{trajectory_id}",
        data_source="terminal_bench",
        turn_idx=turn_idx,
        context_messages=[{"role": "user", "content": f"goal {trajectory_id}"}],
        action_text=f"bash({trajectory_id}:{turn_idx})",
        feedback_text=f"<tool_result>out {turn_idx}</tool_result>",
        next_context_messages=None if turn_idx == 2 else [{"role": "user", "content": "next"}],
        done=turn_idx == 2,
        reward=reward if turn_idx == 2 else None,
        status="completed",
        source_path=f"/tmp/{trajectory_id}/trajectory.json",
    )


def _transitions(num_trajectories: int = 8, turns: int = 3) -> list[TerminalTransition]:
    rows: list[TerminalTransition] = []
    for traj in range(num_trajectories):
        for turn in range(turns):
            rows.append(_transition(f"traj-{traj}", turn, reward=float(traj % 2)))
    return rows


def _write_records_jsonl(path, transitions: list[TerminalTransition]) -> None:
    records = []
    for row in transitions:
        records.append(
            {
                "uid": row.trajectory_id,
                "task_name": row.task_name,
                "turn_idx": row.turn_idx,
                "context_messages": row.context_messages,
                "action_text": row.action_text,
                "feedback_text": row.feedback_text,
                "next_context_messages": row.next_context_messages,
                "done": row.done,
                "reward_score": row.reward,
                "status": row.status,
            }
        )
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


def _online_args(output_dir, snapshot_dir, val_path, **overrides):
    defaults = dict(
        snapshot_dir=str(snapshot_dir),
        val_input=str(val_path),
        val_data_source="records",
        val_max_trajectories=None,
        val_max_transitions=None,
        output_dir=str(output_dir),
        encoder="hash",
        hash_hidden_dim=16,
        hf_model=None,
        hf_local_files_only=False,
        hf_dtype="auto",
        device="cpu",
        hidden_layer=-1,
        action_pool="mean",
        max_context_tokens=64,
        max_action_tokens=32,
        max_feedback_tokens=32,
        encode_batch_size=4,
        backprop_to_llm=False,
        latent_dim=8,
        adapter_dim=None,
        predictor_type="adaln",
        predictor_depth=1,
        predictor_num_heads=2,
        predictor_mlp_ratio=2.0,
        stop_grad_target=True,
        batch_size=4,
        lr=1e-3,
        llm_lr=1e-6,
        weight_decay=0.0,
        sigreg_coef=0.09,
        action_contrast_coef=0.1,
        alignment_coef=0.1,
        value_coef=0.0,
        gamma=0.99,
        gradient_clip=1.0,
        replay_buffer_size=64,
        replay_ratio=0.5,
        seed=42,
        poll_interval_sec=0.01,
        idle_timeout_sec=0.0,
        timeout_sec=0.0,
        max_snapshots=0,
        once=True,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _write_snapshot(path, transitions, step: int) -> None:
    buffer = TrajectoryReplayBuffer(buffer_size=64, seed=1)
    buffer.push(transitions, current_step=step)
    buffer.save(path)


def test_load_snapshot_transitions_roundtrip(tmp_path):
    _write_snapshot(tmp_path / "world_model_replay_0.pt", _transitions(2, 3), 0)
    loaded = _load_snapshot_transitions(tmp_path / "world_model_replay_0.pt")
    assert len(loaded) == 6
    assert {row.trajectory_id for row in loaded} == {"traj-0", "traj-1"}


def test_online_learner_once_processes_snapshots(tmp_path):
    snapshot_dir = tmp_path / "rollout"
    snapshot_dir.mkdir()
    _write_snapshot(snapshot_dir / "world_model_replay_0.pt", _transitions(3, 3)[:9], 0)
    _write_snapshot(snapshot_dir / "world_model_replay_1.pt", _transitions(6, 3), 1)

    val_path = tmp_path / "val.jsonl"
    _write_records_jsonl(val_path, _transitions(2, 3))

    output_dir = tmp_path / "out"
    args = _online_args(output_dir, snapshot_dir, val_path)

    summary = run_online(args)

    assert summary["snapshots_seen"] == 2
    assert summary["transition_count"] == 18  # second snapshot re-pushes traj-0..2, deduped
    assert summary["cumulative_steps"] > 0
    rows = [
        json.loads(line)
        for line in (output_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 2  # one metrics row per snapshot
    assert rows[-1]["replay_stats"]["wm_replay_total_sampled"] > 0
    assert rows[-1]["val_loss"] is not None
    assert (output_dir / "latent_world_model_online.pt").is_file()
    assert (output_dir / "replay_buffer_online.pt").is_file()
    assert (output_dir / "online_summary.json").is_file()
