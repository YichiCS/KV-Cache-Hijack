import math

import torch
from transformers import DynamicCache

from src.utils.llm import (
    build_kv_cache,
    tokenizer_template, 
) 



def cache_clone(cache):
    cache_copy = DynamicCache()
    for layer_idx, (key, value, _) in enumerate(cache):
        cache_copy.update(
            key_states=key,
            value_states=value,
            layer_idx=layer_idx,
        )
    return cache_copy


def cache_split(cache, k):
    cache_a = DynamicCache()
    cache_b = DynamicCache()

    for layer_idx, (key, value, _) in enumerate(cache):
        cache_a.update(
            key_states=key[:, :, :k, :],
            value_states=value[:, :, :k, :],
            layer_idx=layer_idx,
        )
        cache_b.update(
            key_states=key[:, :, k:, :],
            value_states=value[:, :, k:, :],
            layer_idx=layer_idx,
        )

    return cache_a, cache_b


def cache_expand(cache, batch_size):
    expanded_cache = DynamicCache()
    for layer_idx, (key, value, _) in enumerate(cache):
        expanded_cache.update(
            key_states=key.expand(batch_size, -1, -1, -1),
            value_states=value.expand(batch_size, -1, -1, -1),
            layer_idx=layer_idx,
        )
    return expanded_cache


def cache_concat(cache_list):
    if not cache_list:
        raise ValueError("cache_concat requires at least one cache.")

    cache = DynamicCache()
    num_layers = len(list(cache_list[0]))

    resolved = [list(item) for item in cache_list]
    for layer_idx in range(num_layers):
        keys = [cache_item[layer_idx][0] for cache_item in resolved]
        values = [cache_item[layer_idx][1] for cache_item in resolved]
        cache.update(
            key_states=torch.cat(keys, dim=2),
            value_states=torch.cat(values, dim=2),
            layer_idx=layer_idx,
        )
    return cache


def cache_merge(cache_a, cache_b, mask):
    merged_cache = DynamicCache()
    for layer_idx, (layer_a, layer_b, layer_mask) in enumerate(zip(cache_a, cache_b, mask)):
        key_a, value_a = layer_a[:2]
        key_b, value_b = layer_b[:2]
        layer_mask = layer_mask.to(key_a.device)
        merged_cache.update(
            key_states=torch.where(layer_mask, key_a, key_b),
            value_states=torch.where(layer_mask, value_a, value_b),
            layer_idx=layer_idx,
        )
    return merged_cache

class PICacheManager:
    def __init__(self, sample, model, tokenizer, device, args):
        self.args = args
        self.sample = sample
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
        self.ids_group = self.cache_locate(
            context=self.sample['context'],
            context_wp=tokenizer_template(
                tokenizer=self.tokenizer, 
                sample=self.sample, 
                args=args
            ),
            args=args
        )
        self.ids_group["target"] = tokenizer.encode(
            self.sample["target"], 
            return_tensors="pt", 
            add_special_tokens=False
        )[:, :1].to(self.device)
        
        self.cache_group = self.cache_precompute()
    
    def cache_locate(self, context, context_wp, args):
        context_end = context_wp.find(context) + len(context)
        ids_wp = self.tokenizer(context_wp, return_tensors="pt", add_special_tokens=False)
        len_contex_ids = ids_wp.char_to_token(0, context_end - 1) + 1
        
        num_chunks = len_contex_ids // args.chunk_size
        num_cache_chunks = min(num_chunks, math.ceil((len_contex_ids * args.cache_ratio) / args.chunk_size))
        
        assert num_cache_chunks > 0
        
        cache_end = num_chunks * args.chunk_size
        cache_begin = (num_chunks - num_cache_chunks) * args.chunk_size

        return {
            "context": ids_wp["input_ids"][:, :cache_begin].to(self.device),
            "cache": ids_wp["input_ids"][:, cache_begin:cache_end].to(self.device),
            "query": ids_wp["input_ids"][:, cache_end:].to(self.device),
        }
        
    def cache_precompute(self, ):
        full_cache = build_kv_cache(
            model=self.model, 
            input_ids=torch.cat([
                self.ids_group["context"], 
                self.ids_group["cache"]
            ], dim=1)
        )
        context_cache, benign_cache = cache_split(
            cache=full_cache, 
            k=self.ids_group["context"].shape[1]
        )
        return {
            "full": full_cache, 
            "context": context_cache, 
            "benign": benign_cache, 
        }
