"""Align filtered training batches without adding loss-bearing samples."""
from copy import deepcopy


def pad_filtered_batch(data, global_batch_size):
    count = len(data['tokens'])
    if count == 0 or global_batch_size <= 0:
        raise ValueError('training batch and global batch size must be positive')
    padding = (-count) % global_batch_size
    if not padding:
        return 0
    for key, values in data.items():
        if isinstance(values, list) and len(values) == count:
            values.extend(deepcopy(values[0]) for _ in range(padding))
    for index in range(count, count + padding):
        data['loss_masks'][index] = [0] * data['response_lengths'][index]
        data['rewards'][index] = 0.0
        for key in ('native_advantages', 'native_token_weights'):
            if key in data:
                data[key][index] = [0.0] * data['response_lengths'][index]
        if 'native_padding' in data:
            data['native_padding'][index] = True
        if 'per_is_weights' in data:
            data['per_is_weights'][index] = 0.0
    return padding
