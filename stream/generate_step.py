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
    Unified non-KV generation: single-step (proposals=None) or γ-step speculative.
    Returns:
        logits: Tensor of shape (B, vocab) if proposals is None,
                or (B, γ, vocab) if proposals is provided.
    """
    if not self.active_seqs:
        return None, None

    # 1) Build input sequences: prompt + generated_tokens [+ proposals]
    sequences = []
    for i, seq in enumerate(self.active_seqs):
        # bring prompt_tokens onto the right device
        prompt = seq.prompt_tokens.to(self.device)
        # collected past outputs
        gen = torch.tensor(seq.generated_tokens, device=self.device) \
              if getattr(seq, 'generated_tokens', None) else torch.empty(0, dtype=torch.long, device=self.device)

        seq_tokens = torch.cat([prompt, gen], dim=0)

        if proposals is not None:
            # proposals[i] should be a 1D tensor of length γ
            prop = proposals[i].to(self.device)
            seq_tokens = torch.cat([seq_tokens, prop], dim=0)

        sequences.append(seq_tokens)

    # 2) Pad to batch and build attention mask
    padded = pad_sequence(
        sequences,
        batch_first=True,
        padding_value=self.tokenizer.eos_token_id
    ).to(self.device)
    mask = (padded != self.tokenizer.eos_token_id).long().to(self.device)

    # 3) Forward pass (no KV)
    with torch.no_grad():
        out = self.model(
            input_ids=padded,
            attention_mask=mask,
            use_cache=False
        )

    # 4) Slice out the logits we need
    if proposals is None:
        # single-token case → gather the logit at each sequence’s last real position
        logits = torch.stack([
            out.logits[i, seq.size(0) - 1]
            for i, seq in enumerate(sequences)
        ], dim=0)
    else:
        # speculative γ-step case → return the last γ logits
        gamma = proposals[0].shape[0]
        logits = out.logits[:, -gamma:, :]

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
    else:
        # (B, 1+γ, V) → keep only the speculative portion → (B, γ, V)
        logits = out.logits[:, -gamma:, :]

    # determine how many new tokens were just appended
    new_len = 1 if proposals is None else gamma

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
