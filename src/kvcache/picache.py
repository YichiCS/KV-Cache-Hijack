import math

import torch
from transformers import DynamicCache
from transformers.cache_utils import DynamicLayer

from src.utils.llm import (
    build_kv_cache,
    tokenizer_template, 
) 



def cache_from_layers(pairs):
    """Wrap K/V tensors without DynamicLayer.update's initial empty cat copy.

    Each wrapper owns its layer metadata; tensor storage and autograd edges are
    preserved. This adapter is tested against the pinned Transformers version.
    """
    cache = DynamicCache()
    for keys, values in pairs:
        layer = DynamicLayer()
        layer.keys, layer.values = keys, values
        layer.dtype, layer.device = keys.dtype, keys.device
        layer.is_initialized = True
        cache.layers.append(layer)
    return cache


def cache_clone(cache):
    """Copy the append-only container, sharing tensors (not a deep tensor clone).

    DynamicCache updates replace tensors via cat; views are safe for append/crop.
    Callers performing in-place tensor writes must allocate their own storage.
    All utilities use full, untruncated caches with positions starting at zero.
    """
    return cache_from_layers((layer.keys, layer.values) for layer in cache.layers)


def cache_slice(cache, start=0, end=None):
    return cache_from_layers((layer.keys[:, :, start:end, :], layer.values[:, :, start:end, :])
                        for layer in cache.layers)


def cache_split(cache, k):
    return cache_slice(cache, end=k), cache_slice(cache, start=k)


def cache_expand(cache, batch_size):
    return cache_from_layers((layer.keys.expand(batch_size, -1, -1, -1),
                         layer.values.expand(batch_size, -1, -1, -1)) for layer in cache.layers)


def cache_concat(cache_list):
    if not cache_list:
        raise ValueError("cache_concat requires at least one cache.")
    num_layers = len(cache_list[0].layers)
    if any(len(cache.layers) != num_layers for cache in cache_list):
        raise ValueError("Cannot concatenate caches with different layer counts.")
    return cache_from_layers((
        torch.cat([cache.layers[i].keys for cache in cache_list], dim=2),
        torch.cat([cache.layers[i].values for cache in cache_list], dim=2),
    ) for i in range(num_layers))


def cache_merge(cache_a, cache_b, mask):
    def merged():
        for a, b, m in zip(cache_a.layers, cache_b.layers, mask, strict=True):
            m = m.to(a.keys.device)
            yield torch.where(m, a.keys, b.keys), torch.where(m, a.values, b.values)
    return cache_from_layers(merged())

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
        context_start = context_wp.find(context)
        if not context or context_start < 0:
            raise ValueError("Non-empty context must appear in the rendered chat template.")
        if args.chunk_size <= 0 or not 0 < args.cache_ratio <= 1:
            raise ValueError("chunk_size must be positive and cache_ratio must be in (0, 1].")
        context_end = context_start + len(context)
        ids_wp = self.tokenizer(context_wp, return_tensors="pt", add_special_tokens=False)
        # Offset mappings handle trailing whitespace not assigned to a token.
        while context_end > context_start and ids_wp.char_to_token(0, context_end - 1) is None:
            context_end -= 1
        last_token = ids_wp.char_to_token(0, context_end - 1)
        if last_token is None:
            raise ValueError("Could not locate context tokens; a fast tokenizer is required.")
        len_contex_ids = last_token + 1
        
        num_chunks = len_contex_ids // args.chunk_size
        num_cache_chunks = min(num_chunks, math.ceil((len_contex_ids * args.cache_ratio) / args.chunk_size))
        
        if num_cache_chunks <= 0:
            raise ValueError("Context is shorter than chunk_size; use a smaller chunk_size.")
        
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
