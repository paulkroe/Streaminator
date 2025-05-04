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
        self.n_ngram = 3   

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

        # Number of tokens to generate speculatively per step
        self.gamma = 4

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
        if self.spec_decoding and qid in self.ngram_registry and self.ngram_registry[qid]['model'] is None:
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

    def _align_kv_cache_lengths(self):
        """
        Ensures all KV caches in active_seqs have the same sequence length
        by padding shorter ones to match the longest.
        """
        if not self.active_seqs or not self.use_kv_cache:
            return

        # Check if all sequences have KV caches
        if any(not hasattr(seq, 'kv_cache') or not seq.kv_cache for seq in self.active_seqs):
            if self.debug:
                print("Some sequences don't have KV caches, skipping alignment")
            return
            
        # Get the third dimension (sequence length) of each KV cache
        seq_lengths = []
        for seq in self.active_seqs:
            if seq.kv_cache and len(seq.kv_cache) > 0:
                # Get sequence length from the first layer's key
                seq_lengths.append(seq.kv_cache[0][0].shape[2])
        
        if not seq_lengths:
            return
            
        # Find the maximum length
        maxL = max(seq_lengths)
        
        if self.debug:
            print(f"\n--- Aligning KV caches to max length {maxL} ---")
            kv_shapes = []
            for i, seq in enumerate(self.active_seqs):
                if seq.kv_cache and len(seq.kv_cache) > 0:
                    kv_shapes.append((i, seq.kv_cache[0][0].shape[2]))
            print(f"Before alignment: {kv_shapes}")
        
        # Pad each sequence's KV cache to maxL
        for seq in self.active_seqs:
            if not seq.kv_cache or not len(seq.kv_cache):
                continue
                
            # Get current sequence length from KV cache
            currL = seq.kv_cache[0][0].shape[2]
            pad = maxL - currL
            
            if pad > 0:
                seq.kv_cache = [(
                    F.pad(k, (0, 0, pad, 0)),
                    F.pad(v, (0, 0, pad, 0))
                ) for k, v in seq.kv_cache]
        
        if self.debug:
            kv_shapes = []
            for i, seq in enumerate(self.active_seqs):
                if seq.kv_cache and len(seq.kv_cache) > 0:
                    kv_shapes.append((i, seq.kv_cache[0][0].shape[2]))
            print(f"After alignment: {kv_shapes}")

    def _refill_active_seqs(self):
        eff_len = lambda s: s.get_valid_length() + (1 if s.is_finished() else 0)
        # trim old
        for seq in self.active_seqs:
            L = eff_len(seq)
            if self.debug:
                print(f"Trimming seq {seq.qid} to length {L}, valid_length={seq.get_valid_length()}, finished={seq.is_finished()}")
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
        
        # Ensure all KV caches are of the same length
        self._align_kv_cache_lengths()

    def _get_batched_kv(self):
        if not self.active_seqs:
            return None
        
        # Always ensure alignment before batching
        self._align_kv_cache_lengths()
        
        layers = len(self.active_seqs[0].kv_cache)
        batched=[]
        
        # Debug: Print lengths of all KV caches
        if self.debug:
            print("\n--- KV Cache Dimensions After Final Alignment ---")
            for i, seq in enumerate(self.active_seqs):
                for layer_idx, (k, v) in enumerate(seq.kv_cache):
                    if layer_idx == 0:  # Only print first layer to avoid too much output
                        print(f"Seq {i}, Layer 0: K shape {k.shape}")
        
        for i in range(layers):
            ks = [s.kv_cache[i][0] for s in self.active_seqs]
            vs = [s.kv_cache[i][1] for s in self.active_seqs]
            
            # Debug: Check if all tensors in this layer have same dimensions
            if self.debug and len(ks) > 1:
                shapes_k = [k.shape for k in ks]
                shapes_v = [v.shape for v in vs]
                print(f"Layer {i} Key shapes: {shapes_k}")
                print(f"Layer {i} Value shapes: {shapes_v}")
            
            try:
                batched.append((torch.cat(ks,0), torch.cat(vs,0)))
            except RuntimeError as e:
                if self.debug:
                    print(f"Error concatenating at layer {i}: {e}")
                    print(f"Key tensor shapes: {[k.shape for k in ks]}")
                    print(f"Value tensor shapes: {[v.shape for v in vs]}")
                raise  # Re-raise the exception
                
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
                if self.debug:
                    print(f"Finished sequence: {seq.qid}")
                    print(f"Final text: {self.tokenizer.decode(seq.generated_tokens, skip_special_tokens=False)}")
                # Collect final text
                text = self.tokenizer.decode(seq.get_final_generation(), skip_special_tokens=True)
                if not text.strip():  # If text is empty or just whitespace
                    print(f"WARNING: Empty completion text for sequence {seq.qid}")
                    if seq.generated_tokens:
                        # Fallback: use all generated tokens
                        print(f"Using fallback: all generated tokens for seq {seq.qid}")
                        text = self.tokenizer.decode(seq.generated_tokens, skip_special_tokens=True)
                
                if self.debug:
                    print(f"Adding completion for prompt '{seq.prompt_text[:30]}...': '{text[:30]}...'")
                self.results.setdefault(seq.prompt_text, []).append(text)
                # Free KV cache and sequence
                # seq.kv_cache = None
                if self.spec_decoding and seq.qid in self.ngram_registry:
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
        
        # First trim KV caches for space efficiency (before adding new sequences)
        if self.use_kv_cache and self.active_seqs:
            eff_len = lambda s: s.get_valid_length() + (1 if s.is_finished() else 0)
            for seq in self.active_seqs:
                if not seq.kv_cache:
                    continue
                L = eff_len(seq)
                if self.debug:
                    print(f"Trimming seq {seq.qid} to length {L}, valid_length={seq.get_valid_length()}, finished={seq.is_finished()}")
                seq.kv_cache = [(k[..., -L:, :], v[..., -L:, :]) for k, v in seq.kv_cache]
        
        # Add new sequences
        needed = self.stream_width - len(self.active_seqs)
        for _ in range(needed):
            if not self.prompt_deque:
                break
            p, c, qid = self.prompt_deque.popleft()
            new = self._prefill_prompt(p, 1, qid)
            self.active_seqs += new
            if c > 1:
                self.prompt_deque.append((p, c-1, qid))
        
        # Ensure all KV caches are of the same length after adding new sequences
        self._align_kv_cache_lengths()

    def _generate_step_with_kv(self, proposals=None):
        """
        Unified KV generation: one-token or gamma-step proposals.
        """
        if not self.active_seqs:
            return None, None
        B = len(self.active_seqs)
        # non-speculative
        if proposals is None:
            toks, pos, idle = [], [], 0
            eff_len = lambda s: s.get_valid_length() + (1 if s.is_finished() else 0)
            for s in self.active_seqs:
                t = s.next_input_token() or self.tokenizer.eos_token_id
                if s.is_finished(): idle += 1
                toks.append(t); pos.append(eff_len(s))
            if idle == B:
                return None, None
            input_ids = torch.tensor(toks, device=self.device).unsqueeze(1)
            position_ids = torch.tensor(pos, device=self.device).unsqueeze(1)
            batched = self._get_batched_kv()
            mask = self._build_leftpad_attention_mask()
            token_mask = torch.ones(B, 1, device=self.device)
            mask = torch.cat([mask, token_mask], dim=1)
            with torch.no_grad():
                out = self.model(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    past_key_values=KVCacheWrapper.wrap(batched, self.model),
                    use_cache=True,
                    attention_mask=mask
                )
            logits = out.logits[:, -1]
            new_past = out.past_key_values
            per = []
            for i in range(B):
                seq_kv = []
                for layer in new_past:
                    seq_kv.append((layer[0][i:i+1], layer[1][i:i+1]))
                per.append(seq_kv)
            return logits, per
        # speculative proposals
        gamma = proposals[0].shape[0]
        proposal_tensor = torch.stack(proposals, dim=0).to(self.device)
        batched = self._get_batched_kv()
        base_mask = self._build_leftpad_attention_mask()
        token_mask = torch.ones(B, gamma, device=self.device)
        mask = torch.cat([base_mask, token_mask], dim=1)
        with torch.no_grad():
            out = self.model(
                input_ids=proposal_tensor,
                past_key_values=KVCacheWrapper.wrap(batched, self.model),
                use_cache=True,
                attention_mask=mask
            )
        return out.logits, out.past_key_values

    def _generate_step_without_kv(self, proposals=None):
        """
        Unified non-KV generation: one-token or proposals.
        """
        if not self.active_seqs:
            return None, None
        if proposals is None:
            sequences = []
            for seq in self.active_seqs:
                gen = torch.tensor(seq.generated_tokens, device=self.device)
                sequences.append(torch.cat([seq.prompt_tokens, gen]))
            padded = pad_sequence(sequences, batch_first=True,
                                  padding_value=self.tokenizer.eos_token_id).to(self.device)
            mask = (padded != self.tokenizer.eos_token_id).long().to(self.device)
            with torch.no_grad():
                out = self.model(
                    input_ids=padded,
                    attention_mask=mask,
                    use_cache=False
                )
            logits = torch.stack([out.logits[i, seq.size(0)-1] for i, seq in enumerate(sequences)])
            return logits, None
        # speculative proposals
        batch_inputs = []
        for i, seq in enumerate(self.active_seqs):
            prefix = torch.cat([seq.prompt_tokens, torch.tensor(seq.generated_tokens, device=self.device)])
            batch_inputs.append(torch.cat([prefix, proposals[i].to(self.device)]))
        padded = pad_sequence(batch_inputs, batch_first=True,
                              padding_value=self.tokenizer.eos_token_id).to(self.device)
        mask = (padded != self.tokenizer.eos_token_id).long().to(self.device)
        with torch.no_grad():
            out = self.model(
                input_ids=padded,
                attention_mask=mask,
                use_cache=False
            )
        return out.logits, None

    def _run_generation_static(self):
        """
        Static loop supports spec_decoding flag: if True, fall back to continuous speculative,
        else do standard static.
        """
        if self.spec_decoding:
            # simplest: use continuous speculative loop in static mode
            return self._run_generation_continuous()
        # original static
        self._refill_active_seqs()
        step = 0
        while self.active_seqs or self.prompt_deque:
            if step % self.refill_period == 0:
                self._cleanup_and_refill()
            logits, new_past = (self._generate_step_with_kv() if self.use_kv_cache
                                 else self._generate_step_without_kv())
            if logits is None:
                step += 1
                continue
            for i, seq in enumerate(self.active_seqs):
                tok = PromptSampler.sample_token(logits[i])
                seq.append_token(tok)
                if self.use_kv_cache and new_past is not None:
                    seq.kv_cache = new_past[i]
            step += 1
            self._log_gpu_stats(step)
        for seq in self.active_seqs:
            txt = self.tokenizer.decode(seq.get_final_generation(), skip_special_tokens=True)
            self.results.setdefault(seq.prompt_text, []).append(txt)

    def _run_generation_continuous(self):
        """
        Continuous generation loop with full n-gram distributions for speculative decoding.
        """
        self._refill_active_seqs()
        step = 0
        while self.active_seqs or self.prompt_deque:
            if step % self.refill_period == 0:
                self._cleanup_and_refill()
                print("================")
            if self.spec_decoding:
                # 1) Build proposals and collect q-distributions
                proposals, q_dists = [], []
                for seq in self.active_seqs:
                    ngram = self.ngram_registry[seq.qid]['model']
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
                    p_logits_all, new_past_all = self._generate_step_with_kv(proposals)
                    per_seq_past = []
                    for seq_idx in range(B):
                        seq_kv = []
                        for layer_k, layer_v in new_past_all:
                            seq_kv.append((layer_k[seq_idx:seq_idx+1], layer_v[seq_idx:seq_idx+1]))
                        per_seq_past.append(seq_kv)
                else:
                    p_logits_all, _ = self._generate_step_without_kv(proposals)
                    per_seq_past = [None] * B

                # 3) Process each sequence
                next_active = []
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
                    if self.debug:
                        print(f"Sequence {i} (qid={seq.qid}): Accepted {len(accepted_ids)} tokens")
                    for tok in accepted_ids:
                        seq.append_token(int(tok))

                    # determine final index: last accepted, or 0
                    accepted_idxs = mask.nonzero(as_tuple=False).view(-1)
                    if accepted_idxs.numel() > 0:
                        final_idx = accepted_idxs[-1].item()
                    else:
                        final_idx = 0

                    # sample final token from LM at final_idx
                    final_dist = p_probs[final_idx]
                    t = PromptSampler.sample_token(final_dist)
                    seq.append_token(int(t))

                    # update KV cache
                    if self.use_kv_cache:
                        if self.debug:
                            old_shapes = [k.shape for k, _ in seq.kv_cache]
                            new_shapes = [k.shape for k, _ in per_seq_past[i]]
                            print(f"Seq {i} (qid={seq.qid}): Old KV key shapes: {old_shapes}")
                            print(f"Seq {i} (qid={seq.qid}): New KV key shapes: {new_shapes}")
                        seq.kv_cache = per_seq_past[i]
                    
                    # Check if sequence is finished
                    if seq.is_finished():
                        # Don't add to next_active
                        if self.debug:
                            print(f"Sequence {i} (qid={seq.qid}) is finished and removed")
                        continue
                    next_active.append(seq)
                
                # Update active sequences
                self.active_seqs = next_active
                
                # Now do a single cleanup and refill after all sequences are processed
                self._cleanup_and_refill()

            else:
                # regular single-token continuous generation
                logits, new_past = (
                    self._generate_step_with_kv() if self.use_kv_cache
                    else self._generate_step_without_kv()
                )
                if logits is None:
                    step += 1
                    continue

                next_active = []
                for i, seq in enumerate(self.active_seqs):
                    tok = PromptSampler.sample_token(logits[i])
                    seq.append_token(tok)
                    if self.use_kv_cache:
                        seq.kv_cache = new_past[i]
                    if seq.is_finished():
                        txt = self.tokenizer.decode(seq.get_final_generation(), skip_special_tokens=True)
                        if self.debug:
                            print(f"Non-spec mode: Adding completion for prompt '{seq.prompt_text[:30]}...': '{txt[:30]}...'")
                            print(f"Generated tokens: {seq.generated_tokens}")
                        self.results.setdefault(seq.prompt_text, []).append(txt)
                        self.pbar.update(1)
                        if self.spec_decoding and seq.qid in self.ngram_registry:
                            reg = self.ngram_registry[seq.qid]
                            reg['ref_count'] -= 1
                            if reg['ref_count'] == 0:
                                del self.ngram_registry[seq.qid]
                    else:
                        next_active.append(seq)
                self.active_seqs = next_active

            step += 1
            self._log_gpu_stats(step)

        # finalize leftovers
        for seq in self.active_seqs:
            txt = self.tokenizer.decode(seq.get_final_generation(), skip_special_tokens=True)
            if self.debug:
                print(f"Finalizing leftover sequence: '{txt[:30]}...'")
                print(f"Generated tokens: {seq.generated_tokens}")
            self.results.setdefault(seq.prompt_text, []).append(txt)

    def run_generation_loop(self):
        self.pbar = tqdm(total=self.len_queue, desc="Generating...")
        if self.continuous_batching: self._run_generation_continuous()
        else: self._run_generation_static()
        
        # Validate results dictionary
        print("\n--- RESULTS DICTIONARY VALIDATION ---")
        if not self.results:
            print("ERROR: Results dictionary is empty!")
        else:
            print(f"Results contains {len(self.results)} prompts with completions:")
            for prompt, completions in self.results.items():
                print(f"  Prompt: '{prompt[:30]}...' has {len(completions)} completions")
                for i, completion in enumerate(completions):
                    print(f"    Completion {i+1}: '{completion[:60]}...'")
        
        self.save_results("generation_results.json")
        self.pbar.close()
        self.len_queue = 0

    def save_results(self, filename):
        ordered = {}
        for p, _ in self.prompt_order:
            completions = self.results.get(p, [])
            # Fail-safe: If no completions, add an empty string as completion
            if not completions:
                print(f"WARNING: No completions for prompt '{p[:30]}...', adding empty string")
                completions = [""]
            ordered[p] = completions
            
        with open(filename, 'w') as f:
            json.dump(ordered, f, indent=2)
        print(f"Results saved to {filename}")
        
        # Final verification
        print(f"Final results contains {len(ordered)} prompts")
        for prompt, completions in ordered.items():
            if not completions:
                print(f"ERROR: Still no completions for '{prompt[:30]}...'")
            elif all(not c.strip() for c in completions):
                print(f"WARNING: Only empty completions for '{prompt[:30]}...'")
            else:
                print(f"OK: Prompt '{prompt[:30]}...' has {len(completions)} completions")
