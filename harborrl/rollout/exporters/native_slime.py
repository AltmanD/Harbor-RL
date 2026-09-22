"""Native IR to Slime data, with no reward postprocessing or turn regrouping."""
from __future__ import annotations

import math
from types import SimpleNamespace

from harborrl.rollout.exporters.native import export_group
from harborrl.trajectories.native import read_json


def _flatten(samples):
    result = []
    for sample in samples:
        result.extend(sample) if isinstance(sample, (list, tuple)) else result.append(sample)
    return result


def samples_from_groups(groups):
    from slime.utils.types import Sample
    samples = []
    for group_index, group in enumerate(groups):
        export = export_group(group, len(group))
        by_id = {ir['identity']['trajectory_id']: ir for ir in group}
        for record in export['records']:
            ir = by_id[record['identity']['trajectory_id']]
            samples.append(Sample(group_index=group_index, index=len(samples),
                tokens=record['tokens'], response_length=record['response_length'],
                response='native policy response', reward={'score': record['training_reward']},
                loss_mask=record['loss_mask'][record['prompt_length']:],
                rollout_log_probs=record['old_logprobs'], weight_versions=[record['identity']['policy_version']],
                status=Sample.Status.COMPLETED,
                metadata={'native_ir': ir, 'native_turn': record['turn_index']}))
    return samples


def _expanded_turn(record, ir):
    return SimpleNamespace(tokens=record['tokens'], response_length=record['response_length'],
                           loss_mask=record['loss_mask'][record['prompt_length']:],
                           rollout_log_probs=record['old_logprobs'],
                           metadata={'native_ir': ir, 'native_turn': record['turn_index']})


def convert_samples(args, samples):
    # Revalidate the complete IR and exact turn coverage at the training boundary.
    from collections import defaultdict
    from harborrl.trajectories.native import finite, require_ready
    groups, turn_keys, trajectory_owners = defaultdict(dict), set(), {}
    samples = _flatten(samples)
    if not samples:
        raise ValueError("native training requires at least one trajectory")
    expanded_mode = None
    for sample in samples:
        ir = sample.metadata['native_ir']
        if isinstance(ir, str):
            ir = read_json(ir)
        require_ready(ir)
        ident = ir['identity']
        trajectory_id = ident['trajectory_id']
        is_turn = 'native_turn' in sample.metadata
        if not is_turn and trajectory_id in trajectory_owners:
            raise ValueError('duplicate native trajectory')
        trajectory_owners[trajectory_id] = sample
        groups[ident['group_id']][ident['trajectory_id']] = ir
        sample.metadata['native_ir'] = ir
        if expanded_mode is None:
            expanded_mode = is_turn
        elif expanded_mode != is_turn:
            raise ValueError('mixed trajectory and turn samples')
        if is_turn:
            key = (trajectory_id, sample.metadata['native_turn'])
            if key in turn_keys:
                raise ValueError('duplicate native turn')
            turn_keys.add(key)
    records = {}
    versions = set()
    for group in groups.values():
        exported = export_group(list(group.values()), args.n_samples_per_prompt)
        for record in exported['records']:
            key = (record['identity']['trajectory_id'], record['turn_index'])
            records[key] = record
            versions.add(record['identity']['policy_version'])
    if len(versions) != 1:
        raise ValueError('incomplete native turns or mixed policy batch')
    if expanded_mode and set(records) != turn_keys:
        raise ValueError('incomplete native turns or mixed policy batch')
    turn_samples = []
    if expanded_mode:
        turn_samples = samples
    else:
        for (trajectory_id, turn_index), record in records.items():
            turn_samples.append(_expanded_turn(record, trajectory_owners[trajectory_id].metadata['native_ir']))
        turn_samples.sort(key=lambda sample: (sample.metadata['native_ir']['identity']['group_id'],
                                               sample.metadata['native_ir']['identity']['slot_id'],
                                               sample.metadata['native_turn']))
    keys = ('tokens', 'response_lengths', 'rewards', 'raw_reward', 'loss_masks',
            'rollout_log_probs', 'native_advantages', 'native_token_weights', 'truncated',
            'native_padding')
    data = {key: [] for key in keys}
    for sample in turn_samples:
        ir = sample.metadata['native_ir']
        if isinstance(ir, str):
            ir = read_json(ir)
        r = records[(ir['identity']['trajectory_id'], sample.metadata['native_turn'])]
        if (sample.tokens != r['tokens'] or sample.rollout_log_probs != r['old_logprobs']
                or sample.loss_mask != r['loss_mask'][r['prompt_length']:]):
            raise ValueError('Slime sample differs from native evidence')
        values = (r['tokens'], r['response_length'], r['training_reward'], r['training_reward'],
                  r['loss_mask'][r['prompt_length']:], r['old_logprobs'], r['advantages'],
                  [w / len(groups) for w in r['token_weights']], int(ir['termination'] == 'budget_truncated'),
                  False)
        for key, value in zip(data, values):
            data[key].append(value)
    weight_sum = math.fsum(math.fsum(weights) for weights in data['native_token_weights'])
    if not finite(weight_sum) or not math.isclose(weight_sum, 1.0, rel_tol=1e-7, abs_tol=1e-9):
        raise ValueError(f'native trajectory weights must sum to one, got {weight_sum}')
    for advantages, weights, mask in zip(data['native_advantages'], data['native_token_weights'],
                                          data['loss_masks']):
        if (not len(advantages) == len(weights) == len(mask)
                or any(weight < 0 for weight in weights)
                or any(bool(flag) != (weight > 0) for flag, weight in zip(mask, weights))):
            raise ValueError('invalid native token advantage, weight or loss mask')
    return data


def clipped_loss(new_logprobs, old_logprobs, advantages, weights, epsilon=.2, loss_masks=None):
    """Global sum of weighted PPO losses; usable on CPU for gradient contracts."""
    import torch
    if not (new_logprobs.shape == old_logprobs.shape == advantages.shape == weights.shape):
        raise ValueError('native token tensors must have identical shapes')
    if loss_masks is not None and loss_masks.shape != weights.shape:
        raise ValueError('native loss mask must match token weights')
    if not 0 < epsilon < float('inf'):
        raise ValueError('native clipping epsilon must be finite and positive')
    if any(not torch.isfinite(t).all() for t in (new_logprobs, old_logprobs, advantages, weights)) or (weights < 0).any():
        raise ValueError('nonfinite native loss input or negative weight')
    ratio = (new_logprobs - old_logprobs).exp()
    loss = torch.maximum(-advantages * ratio, -advantages * ratio.clamp(1-epsilon, 1+epsilon))
    if loss_masks is not None:
        weights = weights * loss_masks.to(device=weights.device, dtype=weights.dtype)
    return (loss * weights).sum()


def loss_function(args, batch, logits, unused_reducer):
    import torch
    from slime.backends.megatron_utils.loss import get_log_probs_and_entropy
    _, output = get_log_probs_and_entropy(logits, args=args, unconcat_tokens=batch['unconcat_tokens'],
        total_lengths=batch['total_lengths'], response_lengths=batch['response_lengths'],
        with_entropy=False, max_seq_lens=batch.get('max_seq_lens'))
    old_logprobs, advantages, weights = (
        torch.cat(batch[key]) for key in ('rollout_log_probs', 'native_advantages', 'native_token_weights'))
    loss = clipped_loss(torch.cat(output['log_probs']), old_logprobs, advantages, weights,
                        args.eps_clip, torch.cat(batch['loss_masks']))
    # loss_function's standard scaling divides by GBS; undo that division.
    # The native weights already sum to one across the *entire* rollout batch.
    return loss * batch.get('dynamic_global_batch_size', args.global_batch_size), {
        'native_pg_loss': loss.detach(), 'native_weight_sum': weights.sum().detach(),
        'native_padding': float(sum(bool(flag) for flag in batch.get('native_padding', [])))}
