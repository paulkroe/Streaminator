import torch
from ..utils.prompt_sampler import PromptSampler

def _run_generation_static(self):
    """
    Static loop supports spec_decoding flag: if True, fall back to continuous speculative,
    else do standard static.
    """

    self.profiler.start("_run_generation_static")
    
    if self.spec_decoding:
        raise NotImplementedError(
            "We need to realign the KV cache anyways when doing speculative decoding. "
            "Please use continuous generation directly."
        )

    
    self._refill_active_seqs()
    step = 0

    while self.active_seqs or self.prompt_deque:
        if step % self.refill_period == 0:
            self._cleanup_and_refill()

        logits, new_past = (
            self._generate_step_with_kv() if self.use_kv_cache
            else self._generate_step_without_kv()
        )

        if logits is None:
            step += 1
            continue

        for i, seq in enumerate(self.active_seqs):
            # 1) sample and append the token
            tok = PromptSampler.sample_token(logits[i])
            seq.append_token(int(tok))

            # 2) if using KV, append the new slice rather than replace
            if self.use_kv_cache and new_past is not None:
                old_kv        = seq.kv_cache         # List[(k_old, v_old)]
                new_kv_slices = new_past[i]          # List[(k_new, v_new)]
                updated_kv    = []

                for (k_old, v_old), (k_new, v_new) in zip(old_kv, new_kv_slices):
                    # k_new/v_new have shape (1, heads, 1, head_dim)
                    k_cat = torch.cat([k_old, k_new], dim=2)
                    v_cat = torch.cat([v_old, v_new], dim=2)
                    updated_kv.append((k_cat, v_cat))

                seq.kv_cache = updated_kv

        step += 1
        self._log_gpu_stats(step)
    self.profiler.stop("_run_generation_static")
