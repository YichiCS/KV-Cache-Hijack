"""Sparse recomputation with absolute-position masks and owned cache storage."""
import math
from importlib import import_module

import torch

from src.kvcache import cache_clone, cache_expand, cache_concat, cache_from_layers


def _next_layer_budget(remaining_budget, remaining_layers, available_tokens):
    if remaining_budget <= 0 or remaining_layers <= 0 or available_tokens <= 0:
        return 0
    return min(available_tokens, math.ceil(remaining_budget / remaining_layers))


def _validate(model, cache, picm, ratio):
    if not math.isfinite(ratio) or not 0 <= ratio <= 1:
        raise ValueError(f"Recomputation ratio must be finite and in [0, 1], got {ratio}.")
    if len(cache.layers) != len(model.model.layers) or not cache.layers:
        raise ValueError("Cache must contain one initialized layer per model layer.")
    if cache.layers[0].keys.shape[2] == 0:
        raise ValueError("Recomputation requires a non-empty cache.")
    if len(picm.cache_group['context'].layers) != len(cache.layers):
        raise ValueError("Context and cache layer counts must match.")


def _get_backend(model):
    kind = model.config.model_type
    if kind not in {"llama", "qwen3", "mistral", "ministral"}:
        raise ValueError(f"Unsupported recomputation model_type={kind}.")
    module = import_module(f"transformers.models.{kind}.modeling_{kind}")
    impl = model.config._attn_implementation
    # Arbitrarily selected query positions require an explicit 4-D mask.
    if impl not in {"sdpa", "eager"}:
        raise ValueError(f"Sparse recomputation requires sdpa or eager attention, got {impl}.")
    return module.apply_rotary_pos_emb, module.ALL_ATTENTION_FUNCTIONS.get_interface(
        impl, module.eager_attention_forward
    )


def _build_full_cache(model, malicious_cache, picm, batch_size):
    return cache_from_layers((
        torch.cat((context.keys.expand(batch_size, -1, -1, -1), layer.keys), dim=2),
        torch.cat((context.values.expand(batch_size, -1, -1, -1), layer.values), dim=2),
    ) for context, layer in zip(picm.cache_group['context'].layers, malicious_cache.layers, strict=True))


def _get_input_embeds(model, picm, batch_size, device):
    # Cache only immutable embeddings for this sample; the model is frozen.
    if not hasattr(picm, '_recomp_embeds'):
        with torch.no_grad():
            picm._recomp_embeds = model.get_input_embeddings()(picm.ids_group['cache']).detach()
    return picm._recomp_embeds.to(device).expand(batch_size, -1, -1)


def _finalize_cache(full_cache, context_length):
    return cache_from_layers((layer.keys[:, :, context_length:, :], layer.values[:, :, context_length:, :])
                        for layer in full_cache.layers)


def _gather(x, indices, dim):
    shape = [indices.shape[0]] + [1] * (x.ndim - 1)
    shape[dim] = indices.shape[1]
    expanded = list(x.shape)
    expanded[dim] = indices.shape[1]
    return x.gather(dim, indices.reshape(shape).expand(expanded))


def _update(layer, keys, values, positions):
    indices = positions[:, None, :, None].expand_as(keys)
    # Input caches are never mutated. During backward, functional scatter avoids
    # invalidating tensors saved by attention in earlier layers.
    if torch.is_grad_enabled():
        layer.keys = layer.keys.scatter(2, indices, keys)
        layer.values = layer.values.scatter(2, indices, values)
    else:
        layer.keys.scatter_(2, indices, keys)
        layer.values.scatter_(2, indices, values)
    return layer.keys, layer.values


def _window(model, layer):
    attn = layer.self_attn
    if model.config.model_type == 'mistral':
        return model.config.sliding_window
    return getattr(attn, 'sliding_window', None)


def _mask(positions, key_positions, dtype, window=None):
    # Query positions may be sparse and differ between candidate batches.
    allowed = key_positions[None, None, :] <= positions[:, :, None]
    if window is not None:
        allowed &= key_positions[None, None, :] > positions[:, :, None] - window
    return torch.zeros(allowed.shape, dtype=dtype, device=positions.device).masked_fill_(
        ~allowed, torch.finfo(dtype).min
    ).unsqueeze(1)


def _project(layer, hidden_states, cos, sin, apply_rope, need_query=True):
    attn = layer.self_attn
    shape = (*hidden_states.shape[:-1], -1, attn.head_dim)
    keys = attn.k_proj(hidden_states).view(shape)
    values = attn.v_proj(hidden_states).view(shape).transpose(1, 2)
    if hasattr(attn, 'k_norm'):
        keys = attn.k_norm(keys)
    keys = keys.transpose(1, 2)
    if need_query:
        queries = attn.q_proj(hidden_states).view(shape)
        if hasattr(attn, 'q_norm'):
            queries = attn.q_norm(queries)
        queries = queries.transpose(1, 2)
    else:
        queries = keys  # RoPE is identical for Q/K; Q result is discarded.
    queries, keys = apply_rope(queries, keys, cos, sin)
    return queries, keys, values


def _finish_layer(model, layer, residual, queries, keys, values, mask, attention_fn):
    attn = layer.self_attn
    output, _ = attention_fn(
        attn, queries, keys, values, mask,
        dropout=0.0 if not attn.training else attn.attention_dropout,
        scaling=attn.scaling, sliding_window=_window(model, layer),
    )
    hidden = residual + attn.o_proj(output.reshape(*residual.shape[:-1], -1).contiguous())
    return hidden + layer.mlp(layer.post_attention_layernorm(hidden))


def recompute_cacheblend(model, malicious_cache, picm, ratio=0.15, return_stats=False, *, include_context=False):
    _validate(model, malicious_cache, picm, ratio)
    apply_rope, attention_fn = _get_backend(model)
    first = malicious_cache.layers[0].keys
    batch_size, _, cache_length, _ = first.shape
    num_layers = len(malicious_cache.layers)
    max_budget = cache_length * num_layers
    total_budget = math.ceil(max_budget * ratio)
    context_length = picm.cache_group['context'].get_seq_length()
    if ratio in (0, 1):
        result = (cache_clone(malicious_cache) if ratio == 0 else
                  cache_expand(picm.cache_group['benign'], batch_size))
        if include_context:
            result = cache_concat([cache_expand(picm.cache_group['context'], batch_size), result])
        counts = [cache_length if ratio == 1 else 0] * num_layers
        positions = torch.arange(counts[0], device=first.device).expand(batch_size, -1)
        selected = [positions] * num_layers
    else:
        full_cache = _build_full_cache(model, malicious_cache, picm, batch_size)
        hidden = _get_input_embeds(model, picm, batch_size, first.device)
        positions = torch.arange(context_length, context_length + cache_length, device=first.device).expand(batch_size, -1)
        key_positions = torch.arange(context_length + cache_length, device=first.device)
        # RoPE depends on positions, not hidden values. Gather as the set shrinks.
        cos, sin = model.model.rotary_emb(hidden, position_ids=positions)
        remaining = total_budget
        counts, selected = [], []
        masks = {}
        for idx, layer in enumerate(model.model.layers):
            keep = _next_layer_budget(remaining, num_layers - idx, positions.shape[1])
            counts.append(keep)
            if keep == 0:
                if return_stats:
                    selected.append(positions[:, :0])
                continue
            residual = hidden
            hidden = layer.input_layernorm(hidden)
            queries, keys, values = _project(layer, hidden, cos, sin, apply_rope, idx < num_layers - 1)
            if keep < positions.shape[1]:
                with torch.no_grad():
                    old = _gather(full_cache.layers[idx].keys, positions, 2)
                    scores = (keys.float() - old.float()).square().mean(dim=(1, 3))
                    chosen = scores.topk(keep, dim=1, sorted=False).indices.sort(dim=1).values
                positions = positions.gather(1, chosen)
                residual = _gather(residual, chosen, 1)
                queries, keys, values = (_gather(x, chosen, 2) for x in (queries, keys, values))
                cos, sin = (_gather(x, chosen, 1) for x in (cos, sin))
                masks.clear()
            remaining -= keep
            if return_stats:
                selected.append(positions - context_length)
            keys, values = _update(full_cache.layers[idx], keys, values, positions)
            # The final hidden state is unused: only K/V are returned.
            if idx < num_layers - 1:
                window = _window(model, layer)
                if window not in masks:
                    masks[window] = _mask(positions, key_positions, hidden.dtype, window)
                hidden = _finish_layer(model, layer, residual, queries, keys, values, masks[window], attention_fn)
        result = full_cache if include_context else _finalize_cache(full_cache, context_length)
    if not return_stats:
        return result
    return result, dict(total_budget=total_budget, realized_budget=sum(counts), max_budget=max_budget,
                        layer_counts=counts,
                        selected_by_layer=[p[0] if batch_size == 1 else p for p in selected])
