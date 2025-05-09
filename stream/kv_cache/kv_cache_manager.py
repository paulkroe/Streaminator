# kv_cache_manager.py

class KVCacheManager:
    @staticmethod
    def clone(kv_cache):
        """
        Clone the KV cache. It is a list of (key, value) pairs (one per transformer layer). 
        Clone each tensor so that subsequent updates do not interfere with others.

        Args:
            kv_cache: The KV cache to clone.

        Returns:
            A cloned KV cache.
        """
        return [ (k.clone().detach(), v.clone().detach()) for (k, v) in kv_cache ]