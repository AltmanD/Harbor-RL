"""Convert Native IR samples into Slime v0.3.2 standard training fields."""
from __future__ import annotations

import math
from collections import defaultdict

from harborrl.export.native import export_training_batch
from harborrl.trajectories.native import Identity, finite, read_json, require_ready

STANDARD_FIELDS = (
    "tokens",
    "response_lengths",
    "rewards",
    "raw_reward",
    "truncated",
    "sample_indices",
    "rollout_ids",
    "loss_masks",
    "rollout_log_probs",
    "rollout_mask_sums",
)


def _flatten(samples):
    result = []
    for sample in samples:
        result.extend(sample) if isinstance(sample, (list, tuple)) else result.append(sample)
    return result


def _load_ir(value):
    if isinstance(value, str):
        return read_json(value)
    if isinstance(value, dict):
        return value
    raise ValueError("Slime sample metadata must carry the native IR or its path")


def _slot_rank(slot_id):
    try:
        return (0, int(slot_id), "")
    except (TypeError, ValueError):
        return (1, 0, str(slot_id))


def training_batch_from_samples(args, samples):
    """Revalidate complete sample evidence and rebuild the neutral batch."""
    samples = _flatten(samples)
    if not samples:
        raise ValueError("native training requires at least one trajectory turn")
    groups = defaultdict(dict)
    turn_samples = {}
    group_rollout_ids = {}
    for sample in samples:
        metadata = getattr(sample, "metadata", None)
        if (not isinstance(metadata, dict) or "native_ir" not in metadata
                or type(metadata.get("native_turn")) is not int):
            raise ValueError("Slime sample is missing native IR turn metadata")
        ir = _load_ir(metadata["native_ir"])
        metadata["native_ir"] = ir
        require_ready(ir)
        identity = ir["identity"]
        group_key = Identity(**identity).group_key
        rollout_id = getattr(sample, "rollout_id", None)
        if type(rollout_id) is not int or rollout_id < 0:
            raise ValueError("native turn samples require a Slime rollout id")
        previous = group_rollout_ids.setdefault(identity["group_id"], rollout_id)
        if previous != rollout_id:
            raise ValueError("one native group spans multiple Slime rollout ids")
        key = (group_key, identity["trajectory_id"], metadata["native_turn"])
        if key in turn_samples:
            raise ValueError("duplicate native turn")
        turn_samples[key] = sample
        groups[identity["group_id"]][identity["trajectory_id"]] = ir
    ordered_groups = [list(group.values()) for group in groups.values()]
    rollout_ids = [group_rollout_ids[group_id] for group_id in groups]
    if len(set(rollout_ids)) != len(rollout_ids):
        raise ValueError("distinct native groups must not share one Slime rollout id")
    expected_size = getattr(args, "n_samples_per_prompt", None)
    batch = export_training_batch(
        ordered_groups,
        group_size=expected_size,
        rollout_ids=rollout_ids,
    )
    return batch, turn_samples


def convert_samples_to_train_data(args, samples):
    """Slime ``--custom-convert-samples-to-train-data-path`` hook."""
    batch, turn_samples = training_batch_from_samples(args, samples)
    units = {
        (unit.group_key, unit.trajectory_id): (unit, group)
        for group in batch.groups
        for unit in group.trajectories
    }
    spans = {
        (unit.group_key, unit.trajectory_id, span.turn_index): (unit, span)
        for unit, _ in units.values()
        for span in unit.turns
    }
    if set(spans) != set(turn_samples):
        raise ValueError("incomplete native turns or mixed policy batch")
    data = {field: [] for field in STANDARD_FIELDS}
    for running_index, ((group_key, trajectory_id, turn_index), sample) in enumerate(sorted(
        turn_samples.items(),
        key=lambda item: (
            units[item[0][:2]][0].rollout_id,
            _slot_rank(_trajectory_slot(turn_samples[item[0]])),
            item[0][2],
        ),
    )):
        unit, _group = units[(group_key, trajectory_id)]
        span = spans[(group_key, trajectory_id, turn_index)][1]
        metadata = sample.metadata
        ir = metadata["native_ir"]
        response_mask = list(span.loss_mask[span.prompt_length:])
        if (list(getattr(sample, "tokens", [])) != list(span.tokens)
                or list(getattr(sample, "rollout_log_probs", [])) != list(span.old_logprobs)
                or list(getattr(sample, "loss_mask", [])) != response_mask):
            raise ValueError("Slime sample differs from native evidence")
        sample_index = getattr(sample, "index", None)
        if sample_index is None:
            sample_index = running_index
        data["tokens"].append(list(span.tokens))
        data["response_lengths"].append(span.response_length)
        data["rewards"].append(unit.advantage)
        data["raw_reward"].append(unit.reward)
        data["truncated"].append(int(ir.get("termination") == "budget_truncated"))
        data["sample_indices"].append(sample_index)
        data["rollout_ids"].append(unit.rollout_id)
        data["loss_masks"].append(response_mask)
        data["rollout_log_probs"].append(list(span.old_logprobs))
        data["rollout_mask_sums"].append(unit.token_count)
    for group in batch.groups:
        weight_sum = math.fsum(
            1.0 / (group.group_size * unit.token_count)
            for unit in group.trajectories
            for span in unit.turns
            for _ in range(span.response_length)
        )
        if not math.isclose(weight_sum, 1.0, rel_tol=1e-7, abs_tol=1e-9):
            raise ValueError(f"native trajectory weights must sum to one per group, got {weight_sum}")
    if any(not finite(data["rewards"][index]) for index in range(len(data["rewards"]))):
        raise ValueError("nonfinite native advantage")
    return data


def _trajectory_slot(sample):
    ir = sample.metadata["native_ir"]
    return ir["identity"]["slot_id"]

