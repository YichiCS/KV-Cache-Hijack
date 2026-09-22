import torch

from src.kvcache import cache_clone, cache_expand, cache_concat
from src.kvcache.recomps.cacheblend import (
    _build_full_cache, _finalize_cache, _finish_layer, _get_backend,
    _get_input_embeds, _mask, _project, _update, _validate, _window,
)


def recompute_random(model, malicious_cache, picm, ratio=0.1, positions=None, *, include_context=False):
    _validate(model, malicious_cache, picm, ratio)
    apply_rope, attention_fn = _get_backend(model)
    first = malicious_cache.layers[0].keys
    batch_size, _, length, _ = first.shape
    count = int(length * ratio)
    if count in (0, length):
        result = cache_clone(malicious_cache) if count == 0 else cache_expand(picm.cache_group['benign'], batch_size)
        if include_context:
            result = cache_concat([cache_expand(picm.cache_group['context'], batch_size), result])
        return result
    if positions is None:
        positions = torch.randperm(length, device=first.device)[:count].sort().values
    elif positions.ndim != 1 or positions.numel() != count:
        raise ValueError('Random positions must be a 1-D tensor matching the recomputation budget.')
    context_length = picm.cache_group['context'].get_seq_length()
    full_cache = _build_full_cache(model, malicious_cache, picm, batch_size)
    hidden = _get_input_embeds(model, picm, batch_size, first.device).index_select(1, positions)
    positions = (positions + context_length).expand(batch_size, -1)
    cos, sin = model.model.rotary_emb(hidden, position_ids=positions)
    key_positions = torch.arange(context_length + length, device=first.device)
    masks = {}
    for idx, layer in enumerate(model.model.layers):
        residual = hidden
        hidden = layer.input_layernorm(hidden)
        last = idx == len(model.model.layers) - 1
        queries, keys, values = _project(layer, hidden, cos, sin, apply_rope, not last)
        keys, values = _update(full_cache.layers[idx], keys, values, positions)
        if not last:
            window = _window(model, layer)
            if window not in masks:
                masks[window] = _mask(positions, key_positions, hidden.dtype, window)
            hidden = _finish_layer(model, layer, residual, queries, keys, values, masks[window], attention_fn)
    return full_cache if include_context else _finalize_cache(full_cache, context_length)
