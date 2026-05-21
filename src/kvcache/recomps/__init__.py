from src.kvcache.recomps.cacheblend import recompute_cacheblend
from src.kvcache.recomps.epic import recompute_epic
from src.kvcache.recomps.random import recompute_random
from src.kvcache.recomps.vanilla import recompute_vanilla

# Input ***** Output Cache
def cache_recomputation(model, malicious_cache, picm, ratio, method):
    if method == "vanilla":
        return recompute_vanilla(malicious_cache)
    if method == "epic":
        return recompute_epic(malicious_cache, picm, ratio)
    if method == "random":
        return recompute_random(model, malicious_cache, picm, ratio)
    if method == "cacheblend":
        return recompute_cacheblend(model, malicious_cache, picm, ratio)
    raise ValueError(f"Unknown method: {method}")

__all__ = ["cache_recomputation"]
