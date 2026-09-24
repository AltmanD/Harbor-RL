"""Slime v0.3.2 rollout function hook for complete Native Harbor groups."""
from __future__ import annotations

import asyncio
import math

from harborrl.backends.slime_v032.versions import lock_policy_version
from harborrl.export.native import export_training_batch


def sampling_params(args):
    max_new_tokens = getattr(args, "rollout_max_response_len", None)
    if type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError("native rollout requires a positive rollout_max_response_len")
    return {
        "temperature": getattr(args, "rollout_temperature", 1.0),
        "top_p": getattr(args, "rollout_top_p", 1.0),
        "top_k": getattr(args, "rollout_top_k", -1),
        "max_new_tokens": max_new_tokens,
        "stop": list(getattr(args, "rollout_stop", None) or []),
    }


def assert_locked_policy_versions(ir_groups, policy_version):
    """Fail closed if any generation escaped the locked weight version."""
    for group in ir_groups:
        for ir in group:
            if not isinstance(ir, dict):
                raise ValueError("native rollout did not return trajectory evidence")
            version = ir.get("identity", {}).get("policy_version")
            if policy_version is not None and version != policy_version:
                raise ValueError("native rollout observed a stale policy version")


async def run_native_groups(args, groups, policy_version, rollout_id):
    """Run every group concurrently and return only complete IR evidence."""
    from harborrl.rollout.native_generate import close_runtime, generate_group

    params = sampling_params(args)
    try:
        results = await asyncio.gather(*(
            generate_group(args, group, params, rollout_id=rollout_id) for group in groups
        ))
    except BaseException:
        close_runtime()
        raise
    ir_groups = []
    for group in results:
        trajectories = []
        for sample in group:
            metadata = getattr(sample, "metadata", None)
            ir = metadata.get("native_ir") if isinstance(metadata, dict) else None
            if not isinstance(ir, dict):
                raise ValueError("native rollout did not return trajectory evidence")
            trajectories.append(ir)
        ir_groups.append(trajectories)
    assert_locked_policy_versions(ir_groups, policy_version)
    return ir_groups


def to_slime_samples(batch, ir_groups=None, *, sample_factory=None):
    """Expand one Slime training sample per Native policy turn."""
    if sample_factory is None:
        from slime.utils.types import Sample

        def sample_factory(**fields):
            fields["status"] = Sample.Status.COMPLETED
            return Sample(**fields)

    output = []
    index = 0
    for group_index, (group, trajectories) in enumerate(zip(batch.groups, ir_groups or [None] * len(batch.groups), strict=True)):
        by_id = {
            ir["identity"]["trajectory_id"]: ir
            for ir in (trajectories or [])
        }
        group_samples = []
        for unit in group.trajectories:
            for span in unit.turns:
                group_samples.append(sample_factory(
                    group_index=group_index,
                    index=index,
                    rollout_id=unit.rollout_id,
                    tokens=list(span.tokens),
                    response_length=span.response_length,
                    response="native Harbor trial",
                    reward={"score": unit.reward},
                    loss_mask=list(span.loss_mask[span.prompt_length:]),
                    rollout_log_probs=list(span.old_logprobs),
                    weight_versions=[span.policy_version],
                    metadata={
                        "native_ir": by_id.get(unit.trajectory_id),
                        "native_turn": span.turn_index,
                    },
                ))
                index += 1
        output.append(group_samples)
    if any(sample.metadata.get("native_ir") is None for group in output for sample in group):
        raise ValueError("Slime turn samples require native IR metadata")
    return output


def collect_metrics(batch, *, policy_version=None):
    rewards = [
        unit.reward
        for group in batch.groups
        for unit in group.trajectories
    ]
    metrics = {
        "native_group_count": float(len(batch.groups)),
        "native_trajectory_count": float(len(rewards)),
        "native_turn_count": float(sum(
            len(unit.turns) for group in batch.groups for unit in group.trajectories
        )),
        "native_reward_mean": math.fsum(rewards) / len(rewards),
        "native_reward_std": math.sqrt(math.fsum(
            (reward - math.fsum(rewards) / len(rewards)) ** 2 for reward in rewards
        ) / len(rewards)),
    }
    if policy_version is not None:
        metrics["native_policy_version"] = policy_version
    return metrics


def generate_rollout(
    args,
    rollout_id,
    data_source,
    evaluation=False,
    *,
    runner=None,
    output_factory=None,
    sample_factory=None,
):
    """Slime ``--rollout-function-path`` hook; groups stay atomic."""
    if evaluation:
        raise NotImplementedError("Native evaluation rollout is not part of v1")
    if type(rollout_id) is not int or rollout_id < 0:
        raise ValueError("Slime must provide a nonnegative rollout id")
    batch_size = getattr(args, "rollout_batch_size", None)
    expected_size = getattr(args, "n_samples_per_prompt", None)
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("native rollout requires a positive rollout batch size")
    if type(expected_size) is not int or expected_size < 2:
        raise ValueError("native rollout requires a complete group size")
    groups = [list(group) for group in data_source.get_samples(batch_size)]
    if len(groups) != batch_size:
        raise ValueError("native rollout batch is incomplete")
    if any(len(group) != expected_size for group in groups):
        raise ValueError("native rollout group is incomplete before launch")
    if runner is None:
        async def runner(args_, groups_, policy_version):
            return await run_native_groups(args_, groups_, policy_version, rollout_id)
    ir_groups = asyncio.run(runner(args, groups, None))
    policy_version = lock_policy_version(args)
    assert_locked_policy_versions(ir_groups, policy_version)
    batch = export_training_batch(
        ir_groups,
        group_size=expected_size,
        rollout_id_base=rollout_id * batch_size,
    )
    if output_factory is None:
        from slime.rollout.base_types import RolloutFnTrainOutput
        output_factory = RolloutFnTrainOutput
    return output_factory(
        samples=to_slime_samples(batch, ir_groups, sample_factory=sample_factory),
        metrics=collect_metrics(batch, policy_version=policy_version),
    )
