import math

import torch

from src.kvcache import cache_clone, cache_expand, cache_from_layers, cache_concat


def recompute_epic(malicious_cache, picm, ratio, *, include_context=False):
    if not math.isfinite(ratio) or not 0 <= ratio <= 1:
        raise ValueError('Recomputation ratio must be finite and in [0, 1].')
    length = malicious_cache.layers[0].keys.shape[2]
    count = int(length * ratio)
    batch_size = malicious_cache.layers[0].keys.shape[0]
    if count == length:
        if include_context:
            return cache_expand(picm.cache_group['full'], batch_size)
        return cache_expand(picm.cache_group['benign'], batch_size)
    if count == 0:
        if include_context:
            return cache_concat([cache_expand(picm.cache_group['context'], batch_size), malicious_cache])
        return cache_clone(malicious_cache)

    def merged():
        for idx, (benign, malicious) in enumerate(zip(
            picm.cache_group['benign'].layers, malicious_cache.layers, strict=True
        )):
            key_parts = [benign.keys[:, :, :count].expand(batch_size, -1, -1, -1), malicious.keys[:, :, count:]]
            value_parts = [benign.values[:, :, :count].expand(batch_size, -1, -1, -1), malicious.values[:, :, count:]]
            if include_context:
                context = picm.cache_group['context'].layers[idx]
                key_parts.insert(0, context.keys.expand(batch_size, -1, -1, -1))
                value_parts.insert(0, context.values.expand(batch_size, -1, -1, -1))
            yield torch.cat(key_parts, 2), torch.cat(value_parts, 2)
    return cache_from_layers(merged())
