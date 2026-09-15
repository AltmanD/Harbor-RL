import argparse
import json

import pytest
import torch

from slime.world_model.replay_buffer import TrajectoryReplayBuffer
from slime.world_model.seta_dataset import TerminalTransition
from slime.world_model.stream_latent import (
    compose_arm_batches,
    plan_stream,
    run_stream,
)


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
    """Write the world-model records JSONL schema the loader expects."""

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


def test_plan_stream_groups_trajectories_and_balances():
    transitions = _transitions(num_trajectories=8, turns=3)
    train = list(range(len(transitions)))

    chunks = plan_stream(transitions, train, 4, seed=42)

    assert sum(len(chunk) for chunk in chunks) == len(train)
    assert sorted(index for chunk in chunks for index in chunk) == train
    # trajectory-contiguous: each chunk holds whole trajectories
    for chunk in chunks:
        traj_ids = {transitions[index].trajectory_id for index in chunk}
        for traj_id in traj_ids:
            members = [
                index for index in train if transitions[index].trajectory_id == traj_id
            ]
            assert all(index in chunk for index in members)
    # balanced within one trajectory group (3 transitions)
    sizes = [len(chunk) for chunk in chunks]
    assert max(sizes) - min(sizes) <= 3


def test_plan_stream_is_deterministic_for_same_seed():
    transitions = _transitions()
    train = list(range(len(transitions)))
    assert plan_stream(transitions, train, 4, seed=7) == plan_stream(transitions, train, 4, seed=7)


def test_compose_arm_batches_equal_compute_and_replay_mix():
    transitions = _transitions(num_trajectories=4, turns=3)
    id_to_index = {row.transition_id: index for index, row in enumerate(transitions)}
    chunk = list(range(6))
    buffer = TrajectoryReplayBuffer(buffer_size=16, seed=42)
    buffer.push(transitions[:6], current_step=0)

    fresh_batches = compose_arm_batches(
        arm="noreplay",
        chunk_indices=chunk,
        buffer=None,
        id_to_index=id_to_index,
        batch_size=4,
        replay_ratio=0.0,
        seed=0,
        current_step=0,
    )
    replay_batches = compose_arm_batches(
        arm="replay",
        chunk_indices=chunk,
        buffer=buffer,
        id_to_index=id_to_index,
        batch_size=4,
        replay_ratio=0.5,
        seed=0,
        current_step=0,
    )

    assert len(fresh_batches) == len(replay_batches) == 2
    assert all(len(batch) == 4 for batch in fresh_batches + replay_batches)
    # noreplay arm touches only chunk indices
    assert {index for batch in fresh_batches for index in batch} <= set(chunk)
    # replay arm samples from the buffer (total_sampled>0) and keeps at least
    # one fresh transition per batch
    assert buffer.total_sampled > 0
    for batch in replay_batches:
        assert any(index in chunk for index in batch)


def test_compose_arm_batches_without_buffer_falls_back_to_fresh():
    transitions = _transitions(num_trajectories=2, turns=3)
    id_to_index = {row.transition_id: index for index, row in enumerate(transitions)}
    batches = compose_arm_batches(
        arm="replay",
        chunk_indices=[0, 1, 2],
        buffer=None,
        id_to_index=id_to_index,
        batch_size=4,
        replay_ratio=0.5,
        seed=0,
        current_step=0,
    )
    assert len(batches) == 1
    assert set(batches[0]) <= {0, 1, 2}


def _stream_args(tmp_path, output_dir, **overrides):
    defaults = dict(
        input=str(tmp_path),
        supplement_input=[],
        data_source="records",
        min_turns=1,
        exclude_terminal=False,
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
        backprop_to_llm=False,
        save_updated_llm=False,
        max_trajectories=None,
        max_transitions=None,
        require_tool_feedback=False,
        latent_dim=8,
        adapter_dim=None,
        predictor_type="adaln",
        predictor_depth=1,
        predictor_num_heads=2,
        predictor_mlp_ratio=2.0,
        stop_grad_target=True,
        batch_size=4,
        encode_batch_size=2,
        lr=1e-3,
        llm_lr=1e-6,
        weight_decay=0.0,
        sigreg_coef=0.09,
        action_contrast_coef=0.1,
        alignment_coef=0.1,
        value_coef=0.0,
        gamma=0.99,
        gradient_clip=1.0,
        val_ratio=0.25,
        seed=42,
        stream_chunks=3,
        stream_arms=["noreplay", "replay"],
        stream_epochs_per_chunk=1,
        stream_keep_source_order=False,
        replay_buffer_size=64,
        replay_ratio=0.5,
        replay_warmup_chunks=0,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_run_stream_end_to_end_hash(tmp_path):
    transitions_path = tmp_path / "records.jsonl"
    _write_records_jsonl(transitions_path, _transitions(8, 3))
    output_dir = tmp_path / "out"
    args = _stream_args(tmp_path, output_dir)

    summary = run_stream(args)

    assert set(summary["arms"]) == {"noreplay", "replay"}
    rows = [
        json.loads(line)
        for line in (output_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 2 * 3  # arms x chunks
    per_arm = {}
    for row in rows:
        per_arm.setdefault(row["arm"], []).append(row)
        assert row["train_loss"] == pytest.approx(row["train_loss"])  # not NaN
        assert row["val_loss"] is not None
    # equal compute: identical step counts and fresh-sample budgets
    for arm in ("noreplay", "replay"):
        assert [row["cumulative_steps"] for row in per_arm[arm]] == [
            row["cumulative_steps"] for row in per_arm["noreplay"]
        ]
    replay_final = per_arm["replay"][-1]["replay_stats"]
    assert replay_final["wm_replay_total_sampled"] > 0
    assert replay_final["wm_replay_total_admitted"] == summary["train_count"]
    for arm in ("noreplay", "replay"):
        assert (output_dir / f"latent_world_model_{arm}.pt").is_file()
        assert (output_dir / f"predictions_{arm}.jsonl").is_file()
    assert (output_dir / "replay_buffer_replay.pt").is_file()


def test_run_stream_rejects_multi_arm_backprop(tmp_path):
    args = _stream_args(tmp_path, tmp_path / "out", encoder="hf-policy", hf_model="x", backprop_to_llm=True)
    with pytest.raises(ValueError, match="single --stream-arms"):
        run_stream(args)


def test_run_stream_replay_warmup_ramps_ratio(tmp_path):
    transitions_path = tmp_path / "records.jsonl"
    _write_records_jsonl(transitions_path, _transitions(8, 3))
    output_dir = tmp_path / "out"
    args = _stream_args(
        tmp_path,
        output_dir,
        stream_arms=["replay"],
        replay_warmup_chunks=3,
        stream_chunks=3,
    )
    run_stream(args)
    rows = [
        json.loads(line)
        for line in (output_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    ratios = [row["replay_ratio_effective"] for row in rows]
    assert ratios == sorted(ratios)
    assert ratios[0] < ratios[-1] == args.replay_ratio
