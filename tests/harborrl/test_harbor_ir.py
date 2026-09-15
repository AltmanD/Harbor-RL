from pathlib import Path
import sys

import pytest
from harborrl.types import Interaction
from harborrl.trajectories.producers.interactive import InteractiveProducer
from harborrl.trajectories.store import save, load
from harborrl.trajectories.validation import require_rl_ready


def fixture():
    turns = [
        Interaction(
            turn_idx=i,
            input_ids=[1, 2],
            output_token_ids=list(range(12)),
            output_token_logprobs=[-0.3] * 12,
            messages=[{"role": "user", "content": "task"}],
            generation_meta={"weight_version": "7"},
        )
        for i in range(2)
    ]
    ctx = dict(
        task={"dataset": "a", "task": "b", "source_digest": "abc"},
        trajectory_id="trajectory",
        attempt_id="attempt",
        group_id="3",
        policy={
            "weight_version": "7",
            "tokenizer": "tok",
            "chat_template": "template",
            "logprob_source": "sglang-pinned",
            "logprob_semantics": "raw_model",
        },
        execution_status="completed",
    )
    return turns, InteractiveProducer.build(
        turns, [{"tool": "terminal", "result": "ok"}], {"raw_reward": 0.5}, ctx
    )


def test_store_roundtrip_and_immutable(tmp_path):
    _, ir = fixture()
    path = save(ir, tmp_path)
    assert load(path) == ir
    with pytest.raises(FileExistsError):
        save(ir, tmp_path)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda ir: ir.turns[0]["output_token_logprobs"].pop(),
        lambda ir: ir.turns[0]["generation_meta"].update(weight_version="8"),
        lambda ir: ir.evaluation.update(raw_reward=None),
        lambda ir: ir.evaluation.update(raw_reward=float("nan")),
        lambda ir: ir.evaluation.update(raw_reward=1.5),
        lambda ir: ir.evaluation.update(exception_stage="timeout"),
        lambda ir: ir.policy.update(logprob_semantics="unknown"),
        lambda ir: ir.task.clear(),
        lambda ir: ir.turns[0].update(output_token_ids=None),
        lambda ir: ir.turns[0].update(generation_meta=None),
    ],
)
def test_invalid_ir_rejected(mutation):
    _, ir = fixture()
    mutation(ir)
    with pytest.raises(ValueError):
        require_rl_ready(ir, weight_version="7", group_id="3")


def test_stale_policy_and_wrong_group_rejected():
    _, ir = fixture()
    for weight, group in (("8", "3"), ("7", "4")):
        with pytest.raises(ValueError):
            require_rl_ready(ir, weight_version=weight, group_id=group)


def test_producer_copies_raw_facts():
    turns, ir = fixture()
    turns[0].input_ids.clear()
    assert ir.turns[0]["input_ids"] == [1, 2]
    assert ir.events[0]["result"] == "ok"


def test_export_matches_old_builder():
    pytest.importorskip("torch")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backends/slime"))
    from slime.utils.types import Sample
    from harborrl.rollout.sample_builder import _build_samples
    from harborrl.rollout.exporters.slime import SlimeExporter, ExportConfig

    turns, ir = fixture()
    sample = Sample(group_index=3, index=9, metadata={})
    old = _build_samples(turns, sample, 0.5, Sample.Status.COMPLETED)
    new = SlimeExporter.export(
        ir, sample, ExportConfig(), weight_version="7", group_id="3"
    )
    for left, right in zip(old, new, strict=True):
        for field in (
            "tokens",
            "rollout_log_probs",
            "loss_mask",
            "reward",
            "group_index",
            "index",
            "status",
        ):
            assert getattr(left, field) == getattr(right, field)
    assert all(sample.weight_versions == ["7"] for sample in new)
    with pytest.raises(ValueError, match="configuration"):
        SlimeExporter.export(
            ir,
            sample,
            ExportConfig(discount=float("nan")),
            weight_version="7",
            group_id="3",
        )
    assert ir.derived == {}


def test_invalid_nonfinite_raw_values_can_be_archived(tmp_path):
    _, ir = fixture()
    ir.evaluation["raw_reward"] = float("nan")
    restored = load(save(ir, tmp_path))
    assert restored.evaluation["raw_reward"] == {"nonfinite_float": "nan"}
    with pytest.raises(ValueError):
        require_rl_ready(restored, weight_version="7", group_id="3")
