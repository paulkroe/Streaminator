import torch
import torch.nn.functional as F
import json
import gc
from collections import deque
from torch.nn.utils.rnn import pad_sequence
from ..utils.prompt_sampler import PromptSampler
from ..kv_cache.kv_cache_manager import KVCacheManager
from ..kv_cache.kv_cache_wrapper import KVCacheWrapper
from ..utils.sequence import Sequence
import pynvml
from tqdm import tqdm
import gc

def _generate_step_with_kv(self, proposals=None):
    """
    Unified KV generation: single-step (proposals=None) or γ-step speculative.
    Returns:
        logits: Tensor of shape (B, vocab) or (B, γ, vocab)
        new_per_seq_kv: List of length B, each element is a list of (k,v) pairs
    """
    self.profiler.start("generate_step_with_kv")
    if not self.active_seqs:
        self.profiler.stop("generate_step_with_kv")
        return None, None

    B = len(self.active_seqs)
    input_ids, position_ids, all_idle = self._build_input_ids(proposals)
    if all_idle:
        self.profiler.stop("generate_step_with_kv")
        return None, None

    # batch existing KV
    batched = self._get_batched_kv()

    # make the attention mask
    base_mask = self._build_leftpad_attention_mask()   # (B, L_past)
    # +1 for the “real” next token
    token_mask = torch.ones(B, 1, device=self.device)
    mask = torch.cat([base_mask, token_mask], dim=1)

    if proposals is not None:
        # +gamma for the speculative steps
        gamma = proposals[0].shape[0]
        pad = torch.ones(B, gamma, device=self.device)
        mask = torch.cat([mask, pad], dim=1)

    # model forward
    with torch.no_grad():
        out = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=KVCacheWrapper.wrap(batched, self.model),
            use_cache=True,
            attention_mask=mask
        )

    # slice off the logits we need
    if proposals is None:
        # (B, 1, V) -> (B, V)
        logits = out.logits[:, -1, :]
        new_len = 1
    else:
        # (B, 1+gamma, V) -> keep only the speculative portion -> (B, gamma, V)
        logits = out.logits[:, -(gamma + 1):, :]
        new_len = gamma + 1
    
    # unpack only the KV for those new tokens
    new_per_seq_kv = []
    for i in range(B):
        seq_kv = []
        for layer_k, layer_v in out.past_key_values:
            # layer_k, layer_v shape: (B, heads, seq_len, head_dim)
            # take the very last new_len positions
            k_slice = layer_k[i : i+1, :, -new_len :, :]
            v_slice = layer_v[i : i+1, :, -new_len :, :]
            seq_kv.append((k_slice, v_slice))
        new_per_seq_kv.append(seq_kv)

    self.profiler.stop("generate_step_with_kv")
    return logits, new_per_seq_kv