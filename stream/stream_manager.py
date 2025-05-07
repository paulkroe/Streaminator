# stream_manager.py
import torch
import torch.nn.functional as F
import json
import gc
from collections import deque, defaultdict
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
        prompt_training_only=False,
        spec_decoding=True,
        ngram_order=3,
        gamma=1,
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
        self.prompt_training_only = prompt_training_only
        self.debug = debug
        self.logger = logger

        self.ngram_registry = {}
        self.next_qid = 0
        self.ngram_order = ngram_order
        self.gamma = gamma
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
            self.acceptance_dict = defaultdict(int)
            self.completion_level_acceptance = defaultdict(int)
            self.completion_level_count = defaultdict(int)

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
        n_active_seqs = len(self.active_seqs)
        
        acceptance_rate = 0
        if self.spec_decoding:

            if n_active_seqs > 0:
                accepted_proposals = sum(1 for i in self.log_accepted_token if i is not None)
                acceptance_rate = (accepted_proposals / n_active_seqs) * 100
            
            for i, (level, token) in enumerate(zip(self.log_q_level, self.log_accepted_token)):
                self.completion_level_count[level] += 1
                if token is not None:
                    self.completion_level_acceptance[level] += 1
                    self.acceptance_dict[token] += 1


        self.logger.log({
            "gpu SM utilization (%)": util.gpu,
            "gpu memory (MB)": mem_info.used / 1024**2,
            "gpu memory usage (%)": mem_info.used / mem_info.total * 100,
            "generation step": step,
            "stream utilization (%)": stream_utilization,
            "acceptance rate (%)": acceptance_rate,
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
            self.ngram_registry[qid] = {'model': None, 'count': 0, 'num_completions': num_completions}

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
            self.ngram_registry[qid]['model'] = NGram(self.tokenizer, self.ngram_order)
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

    def _accept_speculative(self, q_probs: torch.Tensor, p_probs: torch.Tensor) -> torch.BoolTensor:
        """
        Given draft-model probabilities q_i and full-model probabilities p_i for
        i=1..γ, return a boolean mask of length γ indicating which proposed tokens
        to accept contiguously.

        q_probs.shape = (γ,)
        p_probs.shape = (γ,)  or  (γ+1,)
        """
        # both must be 1D
        assert q_probs.ndim == 1 and p_probs.ndim == 1

        gamma = q_probs.size(0)

        # if p_probs has one extra entry (the correction step), drop it
        if p_probs.size(0) == gamma + 1:
            p_slice = p_probs[:gamma]
        else:
            # must match exactly γ
            assert p_probs.size(0) == gamma, f"p_probs must be length γ or γ+1, got {p_probs.size(0)}"
            p_slice = p_probs

        # draw uniforms
        r = torch.rand_like(q_probs)

        # accept_i ~ Bernoulli(p_i / q_i)
        accept = r < (p_slice / q_probs)

        # enforce contiguous acceptance
        first_false = (~accept).nonzero(as_tuple=False)
        if first_false.numel() > 0:
            idx = first_false[0].item()
            accept[idx:] = False

        return accept


    def _cleanup_and_refill(self):
        still_active = []
        for seq in self.active_seqs:
            if seq.is_finished() or self.tokenizer.eos_token_id in seq.generated_tokens:
                # Collect final text
                tokens = seq.get_final_generation()
                text = self.tokenizer.decode(tokens, skip_special_tokens=True)
                self.results.setdefault(seq.prompt_text, []).append(text)
                if self.spec_decoding and not self.prompt_training_only:
                    self.ngram_registry[seq.qid]['model'].train([tokens], tokenized=True)
                # Free KV cache and sequence
                # seq.kv_cache = None
                if self.spec_decoding:
                    reg = self.ngram_registry[seq.qid]
                    reg['count'] += 1
                    if reg['count'] == reg['num_completions']:
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
                self.log_accepted_token = [None] * len(self.active_seqs)
                self.log_q_level = [None] * len(self.active_seqs)

                # 1) Build proposals and collect q-distributions
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

                # 2) Run LM and split KV caches
                B = len(self.active_seqs)
                if self.use_kv_cache:
                    p_logits_all, per_seq_past = self._generate_step_with_kv(proposals)
                else:
                    p_logits_all, _ = self._generate_step_without_kv(proposals)
                    per_seq_past = [None] * B
                # 3) Process each sequence
                for i, seq in enumerate(self.active_seqs):
                    # print("*******************************************")

                    # get the full (1+γ)-step block of LM probabilities
                    p_block = torch.softmax(p_logits_all[i], dim=-1)  # shape: (1+γ, vocab_size)
                    ids     = proposals[i]                            # shape: (γ,)
                    gamma   = ids.shape[0]
                    idx     = torch.arange(gamma, device=ids.device)

                    # acceptance test on the γ draft tokens
                    p_token_probs = p_block[:gamma, ids]              # p1...p_gamma vs. q1...q_gamma
                    q_token_probs = q_dists[i][idx, ids]
                    mask = self._accept_speculative(q_token_probs, p_token_probs.squeeze(1))

                    # count accepted proposals and append them
                    n = int(mask.sum().item())                        # number of accepted proposals
                    accepted_ids = ids[mask].tolist()
                    for tok in accepted_ids:
                        seq.append_token(int(tok))
                        self.log_accepted_token[i] = tok
                        # print("ACCEPTED")
                        # print(f"seq: {i}, Accepted token: {tok}, {self.tokenizer.decode([tok])}")
                        # print(f"alternative: {self.tokenizer.decode(PromptSampler.sample_token(p_token_probs[0]))}")
                        # print(f"seq: {i}, p_token_probs: {p_token_probs[mask].tolist()}")
                        # print(f"seq: {i}, q_token_probs: {q_token_probs[mask].tolist()}")

                    # print(f"seq: {i}, n={n}, mask={mask.tolist()}")

                    # sample the correction from p_{n+1}, or from the difference if n < γ
                    base = p_block[n]                                 # this is p_{n+1}
                    if n < gamma:
                        draft = q_dists[i][n]                         # q_{n+1}
                        diff  = (base - draft).clamp(min=0.0)
                        if (diff.sum().item() > 0):
                            final_dist = diff / diff.sum()
                        else:
                            final_dist = base
                    else:
                        # all γ were accepted → sample from p_{γ+1} directly
                        final_dist = base
                    # print(final_dist[:100])
                    max_token = final_dist.argmax()
                    # print(f"seq: {i}, max token: {max_token}, {self.tokenizer.decode([max_token])}")
                    t = PromptSampler.sample_token(final_dist, is_dist=True) # TODO: not using temperature
                    seq.append_token(int(t))
                    # print(f"seq: {i}, Final token: {t}, {self.tokenizer.decode([t])}")
                    # assert 0

                    # 4) update KV-cache: append exactly n draft frames + the 1 correction frame
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

            step += 1
            self._log_gpu_stats(step)

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
