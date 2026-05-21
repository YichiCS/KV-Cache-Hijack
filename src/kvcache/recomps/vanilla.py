from src.kvcache import cache_clone


def recompute_vanilla(malicious_cache):
    """No recomputation — return a clone of the malicious cache directly."""
    return cache_clone(malicious_cache)
