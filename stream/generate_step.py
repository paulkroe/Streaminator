import torch
import torch.nn.functional as F
import json
import gc
from collections import deque
from torch.nn.utils.rnn import pad_sequence
from .prompt_sampler import PromptSampler
from .kv_cache_manager import KVCacheManager
from .kv_cache_wrapper import KVCacheWrapper
from .sequence import Sequence
from .ngram import NGram
import pynvml
from tqdm import tqdm
import gc

def _generate_step_without_kv(self, proposals=None):
    """
    Unified non-KV generation: single-step (proposals=None or [])
    or gamma-step speculative.
    Returns:
        logits: Tensor of shape (B, vocab) if no proposals,
                or (B, gamma+1, vocab) if proposals are provided.
    """
    if not self.active_seqs:
        return None, None

    # 1) Build raw sequences: [prompt + generated_tokens (+ proposals)]
    sequences = []
    for i, seq in enumerate(self.active_seqs):
        prompt = seq.prompt_tokens.to(self.device)
        gen = (
            torch.tensor(seq.generated_tokens, device=self.device)
            if getattr(seq, "generated_tokens", None)
            else torch.empty(0, dtype=torch.long, device=self.device)
        )
        tokens = torch.cat([prompt, gen], dim=0)

        if proposals:  # only if the list is non-empty
            prop = proposals[i].to(self.device)  # (gamma,)
            tokens = torch.cat([tokens, prop], dim=0)

        sequences.append(tokens)

    # 2) Left-pad every sequence so they all share the same max_len
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

    # 4) Slice off the logits
    if not proposals:
        # single-step → last position
        logits = out.logits[:, -1, :]              # (B, V)
    else:
        gamma  = proposals[0].shape[0]
        # keep the last (gamma+1) positions → (B, gamma+1, V)
        logits = out.logits[:, -(gamma + 1):, :]

    return logits, None

def _build_input_ids(self, proposals=None):
    B = len(self.active_seqs)
    # 1) collect the next “real” token and its position
    toks, pos, idle = [], [], 0
    eff_len = lambda s: s.get_valid_length() + (1 if s.is_finished() else 0)
    for s in self.active_seqs:
        t = s.next_input_token() or self.tokenizer.eos_token_id
        if s.is_finished():
            idle += 1
        toks.append(t)
        pos.append(eff_len(s))
    if idle == B:
        return None, None, True

    # shape (B,1)
    input_ids = torch.tensor(toks, device=self.device).unsqueeze(1)
    position_ids = torch.tensor(pos, device=self.device).unsqueeze(1)

    if proposals is not None:
        # 2) append gamma speculative tokens along dim=1
        gamma = proposals[0].shape[0]
        proposal_tensor = torch.stack(proposals, dim=0).to(self.device)  # (B,γ)
        input_ids = torch.cat([input_ids, proposal_tensor], dim=1)

        # build the positions for those γ tokens: eff_len+1…eff_len+γ
        # position_ids currently is shape (B,1)
        offsets = torch.arange(1, gamma+1, device=self.device).unsqueeze(0)  # (1,γ)
        proposal_pos = position_ids + offsets  # (B,γ)
        position_ids = torch.cat([position_ids, proposal_pos], dim=1)

    return input_ids, position_ids, False

def _generate_step_with_kv(self, proposals=None):
    """
    Unified KV generation: single-step (proposals=None) or γ-step speculative.
    Returns:
        logits: Tensor of shape (B, vocab) or (B, γ, vocab)
        new_per_seq_kv: List of length B, each element is a list of (k,v) pairs
    """
    if not self.active_seqs:
        return None, None

    B = len(self.active_seqs)
    input_ids, position_ids, all_idle = self._build_input_ids(proposals)
    if all_idle:
        return None, None

    # 1) batch existing KV
    batched = self._get_batched_kv()

    # 2) make the attention mask
    base_mask = self._build_leftpad_attention_mask()   # (B, L_past)
    # +1 for the “real” next token
    token_mask = torch.ones(B, 1, device=self.device)
    mask = torch.cat([base_mask, token_mask], dim=1)

    if proposals is not None:
        # +γ for the speculative steps
        gamma = proposals[0].shape[0]
        pad = torch.ones(B, gamma, device=self.device)
        mask = torch.cat([mask, pad], dim=1)

    # 3) model forward
    # print(batched[0][0].shape)
    with torch.no_grad():
        out = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=KVCacheWrapper.wrap(batched, self.model),
            use_cache=True,
            attention_mask=mask
        )
    # print(out.past_key_values[0][0].shape)

    # 4) slice off the logits we need
    if proposals is None:
        # (B, 1, V) → (B, V)
        logits = out.logits[:, -1, :]
        new_len = 1
    else:
        # (B, 1+γ, V) → keep only the speculative portion → (B, γ, V)
        logits = out.logits[:, -(gamma + 1):, :]
        new_len = gamma + 1
    
    # 5) unpack only the KV for those new tokens
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

    return logits, new_per_seq_kv
