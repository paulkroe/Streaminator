import torch

def _build_leftpad_attention_mask(self):
    """
    Build the leftpad attention mask.

    Args:
        self: The StreamManager instance.
    
    Returns:
        The leftpad attention mask.
    """
    self.profiler.start("_build_leftpad_attention_mask")
    if not self.active_seqs:
        self.profiler.stop("_build_leftpad_attention_mask")
        return None
    T = self.active_seqs[0].kv_cache[0][0].shape[2]
    B = len(self.active_seqs)
    mask = torch.zeros(B, T+1, device=self.device)
    for i,seq in enumerate(self.active_seqs):
        v = seq.get_valid_length()
        mask[i, T-v:]=1
    self.profiler.stop("_build_leftpad_attention_mask")
    return mask.long()

def _build_input_ids(self, proposals=None):
    """
    Build the input IDs.

    Args:
        self: The StreamManager instance.
        proposals: The proposals to build the input IDs with.

    Returns:
        The input IDs.
    """
    self.profiler.start("_build_input_ids")
    B = len(self.active_seqs)
    # collect the next “real” token and its position
    toks, pos, idle = [], [], 0
    eff_len = lambda s: s.get_valid_length() + (1 if s.is_finished() else 0)
    for s in self.active_seqs:
        t = s.next_input_token() or self.tokenizer.eos_token_id
        if s.is_finished():
            idle += 1
        toks.append(t)
        pos.append(eff_len(s))
    if idle == B:
        self.profiler.stop("_build_input_ids")
        return None, None, True

    # shape (B,1)
    input_ids = torch.tensor(toks, device=self.device).unsqueeze(1)
    position_ids = torch.tensor(pos, device=self.device).unsqueeze(1)

    if proposals is not None:
        # append gamma speculative tokens along dim=1
        gamma = proposals[0].shape[0]
        proposal_tensor = torch.stack(proposals, dim=0).to(self.device)  # (B,γ)
        input_ids = torch.cat([input_ids, proposal_tensor], dim=1)

        # build the positions for those γ tokens: eff_len+1…eff_len+γ
        # position_ids currently is shape (B,1)
        offsets = torch.arange(1, gamma+1, device=self.device).unsqueeze(0)  # (1,γ)
        proposal_pos = position_ids + offsets  # (B,γ)
        position_ids = torch.cat([position_ids, proposal_pos], dim=1)
    self.profiler.stop("_build_input_ids")
    return input_ids, position_ids, False

def _get_batched_kv(self):
    """
    Get the batched KV.

    Args:
        self: The StreamManager instance.
    
    Returns:
        The batched KV.
    """
    self.profiler.start("_get_batched_kv")
    if not self.active_seqs:
        self.profiler.stop("_get_batched_kv")
        return None
    layers = len(self.active_seqs[0].kv_cache)
    batched=[]
    for i in range(layers):
        ks = [s.kv_cache[i][0] for s in self.active_seqs]
        vs = [s.kv_cache[i][1] for s in self.active_seqs]
        batched.append((torch.cat(ks,0), torch.cat(vs,0)))
    self.profiler.stop("_get_batched_kv")
    return tuple(batched)