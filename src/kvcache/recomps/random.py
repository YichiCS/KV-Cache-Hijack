import torch

from transformers.masking_utils import create_causal_mask

from src.kvcache.recomps.cacheblend import (
    _build_full_cache,
    _finalize_cache,
    _get_attention_fn,
    _get_backend,
    _get_input_embeds,
)


def recompute_random(model, malicious_cache, picm, ratio=0.1):
    layers = malicious_cache.layers
    if not layers:
        raise ValueError("Random recomputation requires a non-empty malicious cache.")
    if ratio <= 0:
        raise ValueError(f"Random recomputation requires ratio > 0, got {ratio}.")
    cache_length = layers[0].keys.shape[2]
    if cache_length <= 0:
        raise ValueError("Random recomputation requires cache_length > 0.")
    num_replace = min(cache_length, int(cache_length * ratio))
    if num_replace <= 0:
        raise ValueError(
            f"Random recomputation selected zero tokens: cache_length={cache_length}, ratio={ratio}."
        )
    attn_registry, apply_rope, sliding_mask_fn = _get_backend(model)
    batch_size = layers[0].keys.shape[0]
    device = layers[0].keys.device
    context_length = picm.cache_group["context"].layers[0].keys.shape[2]
    full_cache = _build_full_cache(model, malicious_cache, picm, batch_size)
    chosen = torch.sort(torch.randperm(cache_length, device=device)[:num_replace]).values
    positions = torch.arange(context_length, context_length + cache_length, device=device).index_select(0, chosen)
    hidden_states = _get_input_embeds(model, picm, batch_size, device).index_select(1, chosen)
    base_model = model.model
    model_layers = base_model.layers
    attn_fns = [_get_attention_fn(attn_registry, layer.self_attn) for layer in model_layers]
    position_ids = positions.unsqueeze(0).expand(batch_size, -1)
    if positions.numel():
        cos, sin = base_model.rotary_emb(hidden_states, position_ids=position_ids)
        causal_mask = create_causal_mask(
            config=model.config,
            input_embeds=hidden_states,
            attention_mask=None,
            cache_position=positions,
            past_key_values=full_cache,
            position_ids=position_ids,
        )
        sliding_mask = (
            sliding_mask_fn(
                config=model.config,
                input_embeds=hidden_states,
                attention_mask=None,
                cache_position=positions,
                past_key_values=full_cache,
                position_ids=position_ids,
            )
            if sliding_mask_fn is not None
            and any(getattr(layer, "attention_type", None) == "sliding_attention" for layer in model_layers)
            else None
        )
    else:
        cos = sin = causal_mask = sliding_mask = None

    for layer_idx, layer in enumerate(model_layers):
        if not positions.numel():
            break
        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)
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
        key_states, value_states = full_cache.update(key_states, value_states, layer_idx, {"cache_position": positions})
        attn_mask = sliding_mask if getattr(layer, "attention_type", None) == "sliding_attention" else causal_mask
        if attn_mask is None:
            raise ValueError("Sliding attention is only supported for qwen3.")
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
    return _finalize_cache(full_cache, context_length)
