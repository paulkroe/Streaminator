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

def _generate_step_without_kv(self, proposals=None):
    """
    Unified non-KV generation: single-step (proposals=None or [])
    or gamma-step speculative.

    Args:
        self: The StreamManager instance.
        proposals: List of length B, each element is a tensor of shape (gamma)
    Returns:
        logits: Tensor of shape (B, vocab) if no proposals,
                or (B, gamma+1, vocab) if proposals are provided.
    """
    self.profiler.start("generate_step_without_kv")
    if not self.active_seqs:
        self.profiler.stop("generate_step_without_kv")
        return None, None

    # Build raw sequences: [prompt + generated_tokens (+ proposals)]
    sequences = []
    for i, seq in enumerate(self.active_seqs):
        prompt = seq.prompt_tokens.to(self.device)
        gen = (
            torch.tensor(seq.generated_tokens, device=self.device)
            if getattr(seq, "generated_tokens", None)
            else torch.empty(0, dtype=torch.long, device=sestream/generation/generate_step_with_kv.pylf.device)
        )
        tokens = torch.cat([prompt, gen], dim=0)

        if proposals:  # only if the list is non-empty
            prop = proposals[i].to(self.device)  # (gamma,)
            tokens = torch.cat([tokens, prop], dim=0)

        sequences.append(tokens)

    # Left-pad every sequence so they all share the same max_len
    pad_id = self.tokenizer.eos_token_id
    max_len = max(seq.shape[0] for seq in sequences)

    padded_seqs = []
    masks       = []
    for seq in sequences:
        pad_len = max_len - seq.shape[0]
        # left-pad with pad_id
        left_pad   = torch.full((pad_len,), pad_id, device=self.device, dtype=torch.long)
        padded     = torch.cat([left_pad, seq], dim=0)
        # mask: 0 for pad, 1 for real
        m = torch.cat([
            torch.zeros(pad_len, device=self.device, dtype=torch.long),
            torch.ones(seq.shape[0], device=self.device, dtype=torch.long)
        ], dim=0)

        padded_seqs.append(padded)
        masks.append(m)

    input_ids      = torch.stack(padded_seqs, dim=0)  # (B, max_len)
    attention_mask = torch.stack(masks, dim=0)        # (B, max_len)

    # 3) Forward
    with torch.no_grad():
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False
        )

    # Slice off the logits
    if not proposals:
        # single-step → last position
        logits = out.logits[:, -1, :]              # (B, V)
    else:
        gamma  = proposals[0].shape[0]
        # keep the last (gamma+1) positions → (B, gamma+1, V)
        logits = out.logits[:, -(gamma + 1):, :]

    self.profiler.stop("generate_step_without_kv")
    return logits, None