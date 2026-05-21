import math

import torch

from transformers import Cache, CacheLayerMixin, DynamicCache
from transformers.masking_utils import create_causal_mask


def _next_layer_budget(remaining_budget, remaining_layers, available_tokens):
    if remaining_budget <= 0 or remaining_layers <= 0 or available_tokens <= 0:
        return 0
    # Keep the remaining budget feasible for later layers while carrying any unused
    # budget forward automatically.
    return min(available_tokens, math.ceil(remaining_budget / remaining_layers))


class _MutableFullLayer(CacheLayerMixin):
    is_compileable = False
    is_sliding = False

    def __init__(self, keys, values):
        super().__init__()
        self.keys = keys
        self.values = values
        self.device = keys.device
        self.dtype = keys.dtype
        self.max_cache_len = keys.shape[2]
        self.is_initialized = True

    def lazy_initialization(self, key_states, value_states):
        raise RuntimeError("Mutable cache layers are initialized eagerly.")

    def update(self, key_states, value_states, cache_kwargs=None):
        if cache_kwargs is None or "cache_position" not in cache_kwargs:
            raise ValueError("CacheBlend cache update requires cache_position.")
        cache_position = cache_kwargs["cache_position"]
        self.keys.index_copy_(2, cache_position, key_states)
        self.values.index_copy_(2, cache_position, value_states)
        return self.keys, self.values

    def get_mask_sizes(self, cache_position):
        return self.max_cache_len, 0

    def get_seq_length(self):
        return self.max_cache_len

    def get_max_cache_shape(self):
        return self.max_cache_len


class _MutableFullCache(Cache):
    def __init__(self, layers):
        super().__init__(layers=layers)

    def __iter__(self):
        for layer in self.layers:
            yield layer.keys, layer.values, None


def _get_backend(model):
    if model.config.model_type == "qwen3":
        from transformers.models.qwen3.modeling_qwen3 import (
            ALL_ATTENTION_FUNCTIONS,
            apply_rotary_pos_emb,
            create_sliding_window_causal_mask,
        )

        return ALL_ATTENTION_FUNCTIONS, apply_rotary_pos_emb, create_sliding_window_causal_mask
    if model.config.model_type == "llama":
        from transformers.models.llama.modeling_llama import ALL_ATTENTION_FUNCTIONS, apply_rotary_pos_emb

        return ALL_ATTENTION_FUNCTIONS, apply_rotary_pos_emb, None
    raise ValueError(f"CacheBlend only supports llama/qwen3, got model_type={model.config.model_type}.")


def _get_attention_fn(attn_registry, attn):
    impl = attn.config._attn_implementation
    if impl not in attn_registry.valid_keys():
        raise ValueError(f"Unsupported attention implementation: {impl}")
    return attn_registry[impl]


def _build_full_cache(model, malicious_cache, picm, batch_size):
    return _MutableFullCache(
        [
            _MutableFullLayer(
                keys=torch.cat(
                    [
                        context_layer.keys if batch_size == 1 else context_layer.keys.expand(batch_size, -1, -1, -1),
                        malicious_layer.keys,
                    ],
                    dim=2,
                ),
                values=torch.cat(
                    [
                        context_layer.values if batch_size == 1 else context_layer.values.expand(batch_size, -1, -1, -1),
                        malicious_layer.values,
                    ],
                    dim=2,
                ),
            )
            for context_layer, malicious_layer in zip(picm.cache_group["context"].layers, malicious_cache.layers)
        ]
    )


def _get_input_embeds(model, picm, batch_size, device):
    with torch.no_grad():
        input_embeds = model.get_input_embeddings()(picm.ids_group["cache"])
    if input_embeds.device != device:
        input_embeds = input_embeds.to(device)
    return input_embeds.expand(batch_size, -1, -1)


def _finalize_cache(full_cache, context_length):
    result = DynamicCache()
    for layer_idx, layer in enumerate(full_cache.layers):
        result.update(layer.keys[:, :, context_length:, :], layer.values[:, :, context_length:, :], layer_idx)
    return result


def recompute_cacheblend(model, malicious_cache, picm, ratio=0.15, return_stats=False):
    layers = malicious_cache.layers
    if not layers:
        raise ValueError("CacheBlend requires a non-empty malicious cache.")
    if ratio <= 0:
        raise ValueError(f"CacheBlend requires ratio > 0, got {ratio}.")
    cache_length = layers[0].keys.shape[2]
    if cache_length <= 0:
        raise ValueError("CacheBlend requires cache_length > 0.")
    num_layers = len(layers)
    max_budget = cache_length * num_layers
    total_budget = min(max_budget, math.ceil(max_budget * ratio))
    if total_budget <= 0:
        raise ValueError(f"CacheBlend selected zero tokens: cache_length={cache_length}, ratio={ratio}.")
    attn_registry, apply_rope, sliding_mask_fn = _get_backend(model)
    batch_size = layers[0].keys.shape[0]
    device = layers[0].keys.device
    context_length = picm.cache_group["context"].layers[0].keys.shape[2]
    full_cache = _build_full_cache(model, malicious_cache, picm, batch_size)
    hidden_states = _get_input_embeds(model, picm, batch_size, device)
    positions = torch.arange(context_length, context_length + cache_length, device=device)
    base_model = model.model
    model_layers = base_model.layers
    attn_fns = [_get_attention_fn(attn_registry, layer.self_attn) for layer in model_layers]
    selected_by_layer = [] if return_stats else None
    actual_layer_counts = [] if return_stats else None
    remaining_budget = total_budget

    for layer_idx, layer in enumerate(model_layers):
        if not positions.numel():
            if selected_by_layer is not None:
                selected_by_layer.append(positions)
                actual_layer_counts.append(0)
            continue
        keep_count = _next_layer_budget(
            remaining_budget=remaining_budget,
            remaining_layers=len(model_layers) - layer_idx,
            available_tokens=positions.numel(),
        )
        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)
        position_ids = positions.unsqueeze(0).expand(batch_size, -1)
        cos, sin = base_model.rotary_emb(hidden_states, position_ids=position_ids)
        attn = layer.self_attn
        query_states = attn.q_proj(hidden_states).view(*hidden_states.shape[:-1], -1, attn.head_dim)
        key_states = attn.k_proj(hidden_states).view(*hidden_states.shape[:-1], -1, attn.head_dim)
        value_states = attn.v_proj(hidden_states).view(*hidden_states.shape[:-1], -1, attn.head_dim)
        if hasattr(attn, "q_norm"):
            query_states = attn.q_norm(query_states)
        if hasattr(attn, "k_norm"):
            key_states = attn.k_norm(key_states)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        query_states, key_states = apply_rope(query_states, key_states, cos, sin)
        scores = (key_states.float() - full_cache.layers[layer_idx].keys.index_select(2, positions).float()).pow(2).mean(dim=(0, 1, 3))
        keep_count = min(int(keep_count), scores.numel())
        if keep_count == 0:
            positions = positions[:0]
            if selected_by_layer is not None:
                selected_by_layer.append(positions)
                actual_layer_counts.append(0)
            hidden_states = hidden_states[:, :0, :]
            continue
        chosen = (
            torch.arange(scores.numel(), device=device)
            if keep_count == scores.numel()
            else torch.sort(torch.topk(scores, k=keep_count, largest=True, sorted=False).indices).values
        )
        remaining_budget -= keep_count
        positions = positions.index_select(0, chosen)
        if selected_by_layer is not None:
            selected_by_layer.append(positions)
            actual_layer_counts.append(keep_count)
        if not positions.numel():
            hidden_states = hidden_states[:, :0, :]
            continue
        residual = residual.index_select(1, chosen)
        hidden_states = hidden_states.index_select(1, chosen)
        position_ids = position_ids.index_select(1, chosen)
        query_states = query_states.index_select(2, chosen)
        key_states = key_states.index_select(2, chosen)
        value_states = value_states.index_select(2, chosen)
        key_states, value_states = full_cache.update(key_states, value_states, layer_idx, {"cache_position": positions})
        attn_mask = create_causal_mask(
            config=model.config,
            input_embeds=hidden_states,
            attention_mask=None,
            cache_position=positions,
            past_key_values=full_cache,
            position_ids=position_ids,
        )
        if getattr(layer, "attention_type", None) == "sliding_attention":
            if sliding_mask_fn is None:
                raise ValueError("Sliding attention is only supported for qwen3.")
            attn_mask = sliding_mask_fn(
                config=model.config,
                input_embeds=hidden_states,
                attention_mask=None,
                cache_position=positions,
                past_key_values=full_cache,
                position_ids=position_ids,
            )
        attn_output, _ = attn_fns[layer_idx](
            attn,
            query_states,
            key_states,
            value_states,
            attn_mask,
            dropout=0.0 if not attn.training else attn.attention_dropout,
            scaling=attn.scaling,
            sliding_window=getattr(attn, "sliding_window", None),
        )
        hidden_states = residual + attn.o_proj(attn_output.reshape(*hidden_states.shape[:-1], -1).contiguous())
        hidden_states = hidden_states + layer.mlp(layer.post_attention_layernorm(hidden_states))
    result = _finalize_cache(full_cache, context_length)
    if not return_stats:
        return result
    return result, {
        "total_budget": total_budget,
        "realized_budget": sum(actual_layer_counts),
        "max_budget": max_budget,
        "layer_counts": actual_layer_counts,
        "selected_by_layer": [positions.sub(context_length) for positions in selected_by_layer],
    }
