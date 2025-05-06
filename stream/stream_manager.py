# stream_manager.py
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

pynvml.nvmlInit()

class StreamManager:
    def __init__(
        self,
        model,
        tokenizer,
        stream_width=8,
        max_length=50,
        refill_period=5,
        use_kv_cache=True,
        continuous_batching=True,
        spec_decoding=True,
        logger=None,
        debug=False
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.stream_width = stream_width
        self.max_length = max_length
        self.refill_period = refill_period
        self.use_kv_cache = use_kv_cache
        self.continuous_batching = continuous_batching
        self.spec_decoding = spec_decoding
        self.debug = debug
        self.logger = logger

        self.ngram_registry = {}
        self.next_qid = 0
        self.n_ngram = 0

        self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        self.prompt_deque = deque()
        self.active_seqs = []
        self.results = {}
        self.prompt_order = []
        self.device = next(model.parameters()).device

        if self.use_kv_cache:
            self.prompt_kv_cache = {}

        if self.logger:
            self.events = self.logger.Table(columns=["step", "prompt"])

        self.len_queue = 0

        # TODO this should be a parameter
        self.gamma = 1

    from .generate_step import _generate_step_with_kv, _generate_step_without_kv, _build_input_ids

    def _log_gpu_stats(self, step):
        if not self.logger:
            return
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
        except pynvml.NVMLError as e:
            self._debug_print(f"Failed to log GPU stats: {e}")
        stream_utilization = sum(not seq.is_finished() for seq in self.active_seqs) / self.stream_width * 100
        self.logger.log({
            "gpu SM utilization (%)": util.gpu,
            "gpu memory (MB)": mem_info.used / 1024**2,
            "gpu memory usage (%)": mem_info.used / mem_info.total * 100,
            "generation step": step,
            "stream utilization (%)": stream_utilization,
        })

    def _debug_print(self, msg):
        if self.debug:
            print(f"[DEBUG] {msg}")

    def enqueue_prompt(self, prompt_text, num_completions=1):
        # assign a unique qid for this prompt batch
        qid = self.next_qid
        self.next_qid += 1

        # track how many sequences will use this n-gram
        if self.spec_decoding:
            self.ngram_registry[qid] = {'model': None, 'ref_count': num_completions}

        self.len_queue += num_completions
        self._debug_print(f"Enqueue prompt: {prompt_text[:60]!r}, num_completions={num_completions}")
        self.prompt_order.append((prompt_text, qid))
        self.prompt_deque.append((prompt_text, num_completions, qid))

    def _prefill_prompt(self, prompt_text, num_completions, qid):
        """
        Prefill a single completion for `prompt_text`, sampling one token.
        On-use of KV caching, initialize and update the cache; otherwise, skip cache calls.
        """
        self._debug_print(f"Prefilling prompt: {prompt_text[:60]!r}")
        inputs = self.tokenizer(prompt_text, return_tensors='pt').to(self.device)
        prompt_tokens = inputs.input_ids[0]

        # First forward pass to get prefix logits (with or without caching)
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                use_cache=self.use_kv_cache,
            )

        # Build sequence object
        seq = Sequence(
            prompt_text=prompt_text,
            max_length=self.max_length,
            eos_token_id=self.tokenizer.eos_token_id,
            qid=qid
        )

        seq.set_prompt_tokens(prompt_tokens.clone())

        # Lazy-create the n-gram model if needed
        if self.spec_decoding and self.ngram_registry[qid]['model'] is None:
            self.ngram_registry[qid]['model'] = NGram(self.tokenizer, self.n_ngram)
            self.ngram_registry[qid]['model'].train([prompt_text])

        # Sample the first non-prompt token
        prefix_logits = outputs.logits[0, -1]
        first_token = PromptSampler.sample_token(prefix_logits)
        if self.use_kv_cache:
            # Initialize KV cache from the first pass
            cache = KVCacheManager.clone(outputs.past_key_values)
            self.prompt_kv_cache.setdefault(prompt_text, cache)
            seq.kv_cache = cache

            # Run second pass to update cache with the newly sampled token
            single_tok = torch.tensor([[first_token]], device=self.device)
            with torch.no_grad():
                out2 = self.model(
                    input_ids=single_tok,
                    past_key_values=KVCacheWrapper.wrap(seq.kv_cache, self.model),
                    use_cache=True,
                )
            seq.kv_cache = KVCacheManager.clone(out2.past_key_values)

        # Append the sampled token and return
        seq.append_token(first_token)
        return [seq]

    def _refill_active_seqs(self):
        eff_len = lambda s: s.get_valid_length() + (1 if s.is_finished() else 0)
        # trim old
        for seq in self.active_seqs:
            L = eff_len(seq)
            seq.kv_cache = [(k[..., -L:, :], v[..., -L:, :]) for k, v in seq.kv_cache]
        # add new
        needed = self.stream_width - len(self.active_seqs)
        for _ in range(needed):
            if not self.prompt_deque:
                break
            p, c, qid = self.prompt_deque.popleft()
            new = self._prefill_prompt(p, 1, qid)
            self.active_seqs += new
            if c > 1:
                self.prompt_deque.append((p, c-1, qid))
        # pad to equal length
        if self.active_seqs:
            maxL = max(eff_len(s) for s in self.active_seqs)
            for seq in self.active_seqs:
                L = eff_len(seq)
                pad = maxL - L
                if pad>0:
                    seq.kv_cache = [(
                        F.pad(k, (0,0,pad,0)),
                        F.pad(v, (0,0,pad,0))
                    ) for k,v in seq.kv_cache]

    def _get_batched_kv(self):
        if not self.active_seqs:
            return None
        layers = len(self.active_seqs[0].kv_cache)
        batched=[]
        for i in range(layers):
            ks = [s.kv_cache[i][0] for s in self.active_seqs]
            vs = [s.kv_cache[i][1] for s in self.active_seqs]
            batched.append((torch.cat(ks,0), torch.cat(vs,0)))
        return tuple(batched)

    def _build_leftpad_attention_mask(self):
        if not self.active_seqs:
            return None
        T = self.active_seqs[0].kv_cache[0][0].shape[2]
        B = len(self.active_seqs)
        mask = torch.zeros(B, T+1, device=self.device)
        for i,seq in enumerate(self.active_seqs):
            v = seq.get_valid_length()
            mask[i, T-v:]=1
        return mask.long()

    def _accept_speculative(self, q_logits: torch.Tensor, p_logits: torch.Tensor) -> torch.BoolTensor:
        '''
        Given q_i(x) and p_i(x) for i in [1..gamma],
        returns a boolean mask of length gamma indicating which speculative tokens to accept contiguously.
        '''
        assert q_logits.shape == p_logits.shape
        r = torch.rand_like(q_logits)
        accept = r < (p_logits / q_logits)
        # enforce contiguous acceptance: stop at first False
        if not accept.all():
            first_false = (~accept).nonzero()[0].item()
            accept[first_false:] = False
        return accept

    def _cleanup_and_refill(self):
        still_active = []
        for seq in self.active_seqs:
            if seq.is_finished() or self.tokenizer.eos_token_id in seq.generated_tokens:
                # Collect final text
                text = self.tokenizer.decode(seq.get_final_generation(), skip_special_tokens=True)
                self.results.setdefault(seq.prompt_text, []).append(text)
                # Free KV cache and sequence
                # seq.kv_cache = None
                if self.spec_decoding:
                    reg = self.ngram_registry[seq.qid]
                    reg['ref_count'] -= 1
                    if reg['ref_count'] == 0:
                        del self.ngram_registry[seq.qid]
                if self.use_kv_cache:
                    self.prompt_kv_cache.pop(seq.prompt_text, None)
                del seq
                self.pbar.update(1)
            else:
                still_active.append(seq)
        # Force GC and clear PyTorch cache to release GPU memory
        gc.collect()
        torch.cuda.empty_cache()

        self.active_seqs = still_active
        self._refill_active_seqs()
 
    def _run_generation_static(self):
        """
        Static loop supports spec_decoding flag: if True, fall back to continuous speculative,
        else do standard static.
        """
        if self.spec_decoding:
            raise NotImplementedError(
                "We need to realign the KV cache anyways when doing speculative decoding. "
                "Please use continuous generation directly."
            )

        # original static
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


    def _run_generation_continuous(self):
        """
        Continuous generation loop with full n-gram distributions for speculative decoding.
        """
        self._refill_active_seqs()
        step = 0
        while self.active_seqs or self.prompt_deque:
            self._cleanup_and_refill()
            if self.spec_decoding:
                # 1) Build proposals and collect q-distributions
                proposals, q_dists = [], []
                for seq in self.active_seqs:
                    ngram = self.ngram_registry[seq.qid]['model']
                    # TODO: full_input include padding i think
                    ctx = seq.full_input()                # 1D tensor
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

                # 2) Run LM and split KV caches
                B = len(self.active_seqs)
                if self.use_kv_cache:
                    p_logits_all, per_seq_past = self._generate_step_with_kv(proposals)
                else:
                    p_logits_all, _ = self._generate_step_without_kv(proposals)
                    per_seq_past = [None] * B
                # 3) Process each sequence
                for i, seq in enumerate(self.active_seqs):
                    p_probs = torch.softmax(p_logits_all[i], dim=-1)  # [gamma, vocab_size]
                    ids = proposals[i]                                 # [gamma]
                    idx = torch.arange(self.gamma, device=ids.device)
                    p_token_probs = p_probs[idx, ids]
                    q_token_probs = q_dists[i][idx, ids]

                    # acceptance mask
                    mask = self._accept_speculative(q_token_probs, p_token_probs)
                    
                    # append accepted proposal tokens
                    accepted_ids = ids[mask].tolist()
                    for tok in accepted_ids:
                        seq.append_token(int(tok))

                    # determine final index: last accepted, or 0
                    accepted_idxs = mask.nonzero(as_tuple=False).view(-1)
                    if accepted_idxs.numel() > 0:
                        final_idx = accepted_idxs[-1].item()
                    else:
                        final_idx = 0

                    # sample final token from LM at final_idx
                    final_dist = (p_probs[final_idx] - q_dists[i][final_idx]).clamp(min=0.0)
                    total = final_dist.sum()
                    
                    if total.item() > 0:
                        # renormalize to sum to 1
                        final_dist = final_dist / total
                    else:
                        # fall back in the degenerate case
                        final_dist = p_probs[final_idx]

                    t = PromptSampler.sample_token(final_dist)
                    seq.append_token(int(t))

                    if self.use_kv_cache:
                        old_kv        = seq.kv_cache               # List of (k_old, v_old)
                        new_kv_slices = per_seq_past[i]             # List of (k_new_all, v_new_all)
                        updated_kv    = []

                        # mask: Boolean tensor of shape [γ], true where proposal was accepted
                        accepted_idxs = mask.nonzero(as_tuple=False).view(-1)  # e.g. tensor([0, 2, ...])
                        # final_idx was computed above: the index in [0..γ-1] from which we sampled
                        # Build a Python list of all positions to append: accepted proposals + the final token
                        indices = accepted_idxs.tolist() + [final_idx]

                        for (k_old, v_old), (k_new_all, v_new_all) in zip(old_kv, new_kv_slices):
                            # k_new_all/v_new_all have shape (1, heads, γ, head_dim)
                            # gather exactly those slices
                            k_append = k_new_all[..., indices, :]  # (1, heads, len(indices), head_dim)
                            v_append = v_new_all[..., indices, :]

                            # concatenate onto the old cache along the sequence dim (dim=2)
                            k_cat = torch.cat([k_old, k_append], dim=2)
                            v_cat = torch.cat([v_old, v_append], dim=2)
                            updated_kv.append((k_cat, v_cat))

                        # sanity-check: cache should grow by len(indices)
                        prev_len = seq.kv_cache[0][0].shape[2]
                        new_len  = updated_kv[0][0].shape[2]
                        assert new_len - prev_len == len(indices), (
                            f"KV grew by {new_len - prev_len} but expected {len(indices)}"
                        )

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
                    # 1) sample one token
                    tok = PromptSampler.sample_token(logits[i])
                    seq.append_token(int(tok))

                    # 2) update KV‐cache by appending exactly the new slice
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

    def run_generation_loop(self):
        self.pbar = tqdm(total=self.len_queue, desc="Generating...")
        if self.continuous_batching: self._run_generation_continuous()
        else: self._run_generation_static()
        self.save_results("generation_results.json")
        self.pbar.close()
        self.len_queue = 0

    def save_results(self, filename):
        ordered={}
        for p, _ in self.prompt_order:
            ordered[p]=self.results.get(p,[])
        with open(filename,'w') as f:
            json.dump(ordered,f,indent=2)
        print(f"Results saved to {filename}")
