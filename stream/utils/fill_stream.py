import torch
from torch import nn
import torch.nn.functional as F
from ..utils.sequence import Sequence
from ..speculator_models.ngram import NGram
from ..utils.prompt_sampler import PromptSampler
from ..kv_cache.kv_cache_manager import KVCacheManager
from ..kv_cache.kv_cache_wrapper import KVCacheWrapper
import gc

def enqueue_prompt(self, prompt_text, num_completions=1):
    """
    Enqueue a prompt for generation.

    Args:
        prompt_text: The prompt to enqueue.
        num_completions: The number of completions to enqueue.

    Returns:
        The number of sequences enqueued.
    """
    self.profiler.start("enqueue_prompt")
    # assign a unique qid for this prompt batch
    qid = self.next_qid
    self.next_qid += 1

    # track how many sequences will use this n-gram
    if self.spec_decoding:
        self.ngram_registry[qid] = {'model': None, 'count': 0, 'num_completions': num_completions}

    self.len_queue += num_completions
    self.prompt_order.append((prompt_text, qid))
    self.prompt_deque.append((prompt_text, num_completions, qid))
    self.profiler.stop("enqueue_prompt")

def _prefill_prompt(self, prompt_text, num_completions, qid):
    """
    Prefill a single completion for `prompt_text`, sampling one token.
    On-use of KV caching, initialize and update the cache; otherwise, skip cache calls.

    Args:
        prompt_text: The prompt to prefill.
        num_completions: The number of completions to prefill.
        qid: The unique ID for this prompt batch.

    Returns:
        The number of sequences enqueued.
    """
    self.profiler.start("_prefill_prompt")
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
        self.ngram_registry[qid]['model'] = NGram(self.tokenizer, self.ngram_order, self.profiler)
        if self.no_prompt_training:
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
    self.profiler.stop("_prefill_prompt")
    return [seq]

def _refill_active_seqs(self):
    """
    Refill the active sequences with new tokens.

    Args:
        self: The StreamManager instance.

    Returns:
        The number of sequences enqueued.
    """
    self.profiler.start("_refill_active_seqs")
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

    self.profiler.stop("_refill_active_seqs")

def _cleanup_and_refill(self):
    """
    Cleanup and refill the active sequences.

    Args:
        self: The StreamManager instance.

    Returns:
        The number of sequences enqueued.
    """
    self.profiler.start("_cleanup_and_refill")
    still_active = []
    for seq in self.active_seqs:
        if seq.is_finished() or self.tokenizer.eos_token_id in seq.generated_tokens:
            # Collect final text
            tokens = seq.get_final_generation()
            text = self.tokenizer.decode(tokens, skip_special_tokens=True)
            self.results.setdefault(seq.prompt_text, []).append(text)
            if self.spec_decoding and self.no_generation_training:
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
    self.profiler.stop("_cleanup_and_refill")