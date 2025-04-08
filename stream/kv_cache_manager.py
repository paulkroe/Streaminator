# kv_cache_manager.py

class KVCacheManager:
    @staticmethod
    def clone(kv_cache):
        # kv_cache is a list of (key, value) pairs (one per transformer layer).
        # Clone each tensor so that subsequent updates do not interfere with others.
        return [ (k.clone().detach(), v.clone().detach()) for (k, v) in kv_cache ]