import math

from src.kvcache import cache_concat, cache_expand
from src.kvcache.recomps.cacheblend import recompute_cacheblend
from src.kvcache.recomps.epic import recompute_epic
from src.kvcache.recomps.random import recompute_random
from src.kvcache.recomps.vanilla import recompute_vanilla


def cache_recomputation(model, malicious_cache, picm, ratio, method, *, include_context=False, random_positions=None):
    if not math.isfinite(ratio) or not 0 <= ratio <= 1:
        raise ValueError(f'Recomputation ratio must be finite and in [0, 1], got {ratio}.')
    if method == 'vanilla':
        result = recompute_vanilla(malicious_cache)
    elif method == 'epic':
        return recompute_epic(malicious_cache, picm, ratio, include_context=include_context)
    elif method == 'random':
        return recompute_random(model, malicious_cache, picm, ratio, positions=random_positions,
                                include_context=include_context)
    elif method == 'cacheblend':
        return recompute_cacheblend(model, malicious_cache, picm, ratio, include_context=include_context)
    else:
        raise ValueError(f'Unknown method: {method}')
    if include_context:
        return cache_concat([cache_expand(picm.cache_group['context'], result.layers[0].keys.shape[0]), result])
    return result


__all__ = ['cache_recomputation']
