# kv_cache_wrapper.py

class KVCacheWrapper:
    """
    Wrap the past_key_values object (a tuple of (keys, values) for each layer)
    in a KVCacheWrapper that exposes additional methods.

    Args:
        past: The past_key_values object.
    """
    def __init__(self, past):
        self.past = past

    def get_seq_length(self):
        """
        Get the sequence length from the first layer's keys.
        """
        return self.past[0][0].shape[2]

    def update(self, new_keys, new_values, layer_idx, cache_kwargs=None):
        """
        Optionally implement an update method if the model calls it.
        A simple implementation could be to update the keys and values by concatenation.
        For example:
            self.past[layer_idx] = (
                self._cat_along_seq(self.past[layer_idx][0], new_keys),
                self._cat_along_seq(self.past[layer_idx][1], new_values),
            )
        
        Args:
            new_keys: The new keys to update the cache with.
            new_values: The new values to update the cache with.
            layer_idx: The layer index to update the cache with.
            cache_kwargs: Additional cache kwargs.

        Returns:
            The updated keys and values.
        """
        # This is a minimal implementation; you may need to adjust it to your needs.
        k_old, v_old = self.past[layer_idx]
        import torch
        new_k = torch.cat([k_old, new_keys], dim=2)
        new_v = torch.cat([v_old, new_values], dim=2)
        self.past = list(self.past)
        self.past[layer_idx] = (new_k, new_v)
        self.past = tuple(self.past)
        return new_k, new_v

    def __getitem__(self, idx):
        return self.past[idx]

    def __iter__(self):
        return iter(self.past)

    def __len__(self):
        return len(self.past)

    @staticmethod
    def wrap(past, model):
        """
        If the model is a Llama-style model (i.e. model.config.model_type starts with 'llama'),
        return an instance of KVCacheWrapper wrapping past.
        Otherwise, return past unchanged.

        Args:
            past: The past_key_values object.
            model: The model to use for generation.

        Returns:
            An instance of KVCacheWrapper wrapping past if the model is a Llama-style model,
            otherwise return past unchanged.
        """
        if hasattr(model.config, "model_type") and model.config.model_type.lower().startswith("llama"):
            return KVCacheWrapper(past)
        else:
            return past

class _LlamaKVCacheLayerWrapper:
    """
    A wrapper for a single layer's KV cache that provides the update method
    as expected by Llama-style models.
    The underlying past is a tuple (keys, values) where keys has shape
    [batch, n_heads, seq_len, head_dim] and values similarly.

    Args:
        past_layer: The past_key_values object.
    """
    def __init__(self, past_layer):
        self.key, self.value = past_layer  # Unpack the tuple

    def get_seq_length(self):
        """
        Get the sequence length from the key tensor.
        """
        return self.key.shape[2]

    def update(self, new_key, new_value, layer_idx, cache_kwargs):
        """
        Update the cache by concatenating new_key and new_value along the sequence dimension.
        This method is expected to be called by the model's self-attention.

        Args:
            new_key: The new key to update the cache with.
            new_value: The new value to update the cache with.
            layer_idx: The layer index to update the cache with.
            cache_kwargs: Additional cache kwargs.

        Returns:
            The updated keys and values.
        """
        # new_key and new_value are expected to have shape [batch, n_heads, 1, head_dim]
        self.key = self._cat_along_seq(self.key, new_key)
        self.value = self._cat_along_seq(self.value, new_value)
        return self.key, self.value

    def _cat_along_seq(self, tensor, new_tensor):
        """
        Concatenate along dimension 2 (the sequence dimension).

        Args:
            tensor: The tensor to concatenate along the sequence dimension.
            new_tensor: The tensor to concatenate along the sequence dimension.

        Returns:
            The concatenated tensor.
        """
        return tensor if new_tensor is None else  self._safe_cat(tensor, new_tensor)

    def _safe_cat(self, tensor, new_tensor):
        """
        Concatenate two tensors along dimension 2 (the sequence dimension).

        Args:
            tensor: The tensor to concatenate along the sequence dimension.
            new_tensor: The tensor to concatenate along the sequence dimension.

        Returns:
            The concatenated tensor.
        """
        import torch
        return torch.cat([tensor, new_tensor], dim=2)

    # To support iteration if needed
    def __iter__(self):
        return iter((self.key, self.value))

    def __getitem__(self, idx):
        # Allow indexing (for example, if the model treats the layer cache as a tuple)
        if idx == 0:
            return self.key
        elif idx == 1:
            return self.value
        else:
            raise IndexError("Index out of range for Llama KV cache layer wrapper.")

    def __len__(self):
        return 2