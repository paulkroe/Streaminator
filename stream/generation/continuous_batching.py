import torch
from ..utils.prompt_sampler import PromptSampler

def _run_generation_continuous(self):
    """
    Continuous generation loop with full n-gram distributions for speculative decoding.
    """
    self.profiler.start("_run_generation_continuous")
    self._refill_active_seqs()
    step = 0
    while self.active_seqs or self.prompt_deque:
        self._cleanup_and_refill()
        if self.spec_decoding:
            self.log_accepted_token = [None] * len(self.active_seqs)
            self.log_q_level = [None] * len(self.active_seqs)

            # Build proposals and collect q-distributions
            proposals, q_dists = [], []
            for i, seq in enumerate(self.active_seqs):
                ngram = self.ngram_registry[seq.qid]['model']
                self.log_q_level[i] = self.ngram_registry[seq.qid]['count']
                ctx = seq.full_input()[-self.ngram_order:]    # 1D tensor
                xs, dists = [], []
                cur = ctx.clone()
                for _ in range(self.gamma):
                    dist = ngram(cur)                # [vocab_size]
                    tok_id = dist.argmax().view(1)
                    xs.append(tok_id)
                    dists.append(dist)
                    cur = torch.cat([cur, tok_id], dim=0)
                proposals.append(torch.cat(xs, dim=0))     # [gamma]
                q_dists.append(torch.stack(dists, dim=0))  # [gamma, vocab_size]

            # Run LM and split KV caches
            B = len(self.active_seqs)
            if self.use_kv_cache:
                p_logits_all, per_seq_past = self._generate_step_with_kv(proposals)
            else:
                p_logits_all, _ = self._generate_step_without_kv(proposals)
                per_seq_past = [None] * B
            # Process each sequence
            for i, seq in enumerate(self.active_seqs):

                # get the full (1+gamma)-step block of LM probabilities
                p_block = torch.softmax(p_logits_all[i], dim=-1)  # shape: (1+gamma, vocab_size)
                ids     = proposals[i]                            # shape: (gamma,)
                gamma   = ids.shape[0]
                idx     = torch.arange(gamma, device=ids.device)

                # acceptance test on the gamma draft tokens
                p_token_probs = p_block[:gamma, ids]              # p1...p_gamma vs. q1...q_gamma
                q_token_probs = q_dists[i][idx, ids]
                mask = self._accept_speculative(q_token_probs, p_token_probs.squeeze(1))

                # count accepted proposals and append them
                n = int(mask.sum().item())                        # number of accepted proposals
                accepted_ids = ids[mask].tolist()
                for tok in accepted_ids:
                    seq.append_token(int(tok))
                    self.log_accepted_token[i] = tok


                # sample the correction from p_{n+1}, or from the difference if n < gamma
                base = p_block[n]                                 # this is p_{n+1}
                if n < gamma:
                    draft = q_dists[i][n]                         # q_{n+1}
                    diff  = (base - draft).clamp(min=0.0)
                    if (diff.sum().item() > 0):
                        final_dist = diff / diff.sum()
                    else:
                        final_dist = base
                else:
                    # all gamma were accepted → sample from p_{gamma+1} directly
                    final_dist = base
                max_token = final_dist.argmax()
                t = PromptSampler.sample_token(final_dist, is_dist=True) # TODO: not using temperature
                seq.append_token(int(t))

                # update KV-cache: append exactly n draft frames + the 1 correction frame
                if self.use_kv_cache:
                    indices = list(range(n)) + [n]
                    old_kv        = seq.kv_cache
                    new_kv_slices = per_seq_past[i]
                    updated_kv    = []
                    for (k_old, v_old), (k_new_all, v_new_all) in zip(old_kv, new_kv_slices):
                        k_append = k_new_all[..., indices, :]
                        v_append = v_new_all[..., indices, :]
                        updated_kv.append((
                            torch.cat([k_old, k_append], dim=2),
                            torch.cat([v_old, v_append], dim=2)
                        ))
                    seq.kv_cache = updated_kv
        else:
            # regular single‐token continuous generation
            logits, new_past = (
                self._generate_step_with_kv() if self.use_kv_cache
                else self._generate_step_without_kv()
            )
            if logits is None:
                step += 1
                continue

            for i, seq in enumerate(self.active_seqs):
                # sample one token
                tok = PromptSampler.sample_token(logits[i])
                seq.append_token(int(tok))

                # update KV‐cache by appending exactly the new slice
                if self.use_kv_cache:
                    old_kv        = seq.kv_cache             # List[(k_old, v_old)]
                    new_kv_slices = new_past[i]              # List[(k_new, v_new)]
                    updated_kv    = []

                    for (k_old, v_old), (k_new, v_new) in zip(old_kv, new_kv_slices):
                        # here k_new/v_new have shape (1, heads, 1, head_dim)
                        # just take that “1” step and concat
                        k_cat = torch.cat([k_old, k_new], dim=2)
                        v_cat = torch.cat([v_old, v_new], dim=2)
                        updated_kv.append((k_cat, v_cat))

                    seq.kv_cache = updated_kv

        step += 1
        self._log_gpu_stats(step)
    self.profiler.stop("_run_generation_continuous")