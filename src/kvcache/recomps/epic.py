import torch

from transformers import DynamicCache

from src.kvcache import cache_clone, cache_expand


def recompute_epic(malicious_cache, picm, ratio):
    length = malicious_cache.layers[0].keys.shape[2]
    num_replace = int(length * ratio)
    if num_replace <= 0:
        return cache_clone(malicious_cache)
    batch_size = malicious_cache.layers[0].keys.shape[0]
    benign_cache = picm.cache_group["benign"] if batch_size == 1 else cache_expand(picm.cache_group["benign"], batch_size)
    if num_replace >= length:
        merged = DynamicCache()
        for layer_idx, layer in enumerate(benign_cache):
            merged.update(layer[0], layer[1], layer_idx)
        return merged
    merged = DynamicCache()
    for layer_idx, (benign_layer, malicious_layer) in enumerate(zip(benign_cache, malicious_cache)):
        merged.update(
            torch.cat([benign_layer[0][:, :, :num_replace, :], malicious_layer[0][:, :, num_replace:, :]], dim=2),
            torch.cat([benign_layer[1][:, :, :num_replace, :], malicious_layer[1][:, :, num_replace:, :]], dim=2),
            layer_idx,
        )
    return merged
