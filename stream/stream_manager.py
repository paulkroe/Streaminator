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
import pynvml
from tqdm import tqdm
from utils.logging_utils import Logger
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
        continuous_batching=False,
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

        # Initialize Logger
        self.logger = logger or Logger(enable_wandb=False, debug=debug)

        self.prompt_deque = deque()
        self.active_seqs = []
        self.results = {}
        self.device = next(model.parameters()).device

        # Initialize next_qid to track unique query IDs
        self.next_qid = 0

        # Initialize ngram_registry to track n-grams and their metadata
        self.ngram_registry = {}

        # Initialize len_queue to track the total number of completions in the queue
        self.len_queue = 0

        # Initialize prompt_order to track the order of prompts
        self.prompt_order = []

        # Initialize prompt_kv_cache to store KV caches for prompts
        self.prompt_kv_cache = {}

    def _log_gpu_stats(self, step):
        """
        Log GPU stats using the Logger.
        """
        self.logger.log_gpu_stats(step, self.active_seqs, self.stream_width)

    def _debug_print(self, msg):
        """
        Print debug messages using the Logger.
        """
        self.logger.debug_print(msg)

    def enqueue_prompt(self, prompt_text, num_completions=1):
        """
        Enqueue a new prompt for generation.

        Args:
            prompt_text (str): The prompt text to enqueue.
            num_completions (int): The number of completions to generate for this prompt.
        """
        # Assign a unique qid for this prompt batch
        qid = self.next_qid
        self.next_qid += 1

        # Track how many sequences will use this n-gram
        self.ngram_registry[qid] = {'model': None, 'ref_count': num_completions}

        # Update the queue length
        self.len_queue += num_completions

        # Debug print
        self._debug_print(f"Enqueue prompt: {prompt_text[:60]!r}, num_completions={num_completions}")

        # Track the order of prompts
        self.prompt_order.append((prompt_text, qid))

        # Add the prompt to the deque
        self.prompt_deque.append((prompt_text, num_completions, qid))

    def _prefill_prompt(self, prompt_text, num_completions, qid):
        """
        Prefill a single completion for `prompt_text`, sampling one token.
        On-use of KV caching, initialize and update the cache; otherwise, skip cache calls.
        """
        self._debug_print(f"Prefilling prompt: {prompt_text[:60]!r}")

        # Tokenize the input prompt and move it to the appropriate device (e.g., GPU or CPU)
        inputs = self.tokenizer(prompt_text, return_tensors='pt').to(self.device)
        prompt_tokens = inputs.input_ids[0]  # Extract the tokenized input IDs

        # Perform the first forward pass through the model to obtain logits
        # This pass may or may not use KV caching, depending on the configuration
        with torch.no_grad():  # Disable gradient computation for inference
            outputs = self.model(
                **inputs,  # Pass the tokenized inputs to the model
                use_cache=self.use_kv_cache,  # Enable or disable KV caching
            )

        # Create a new Sequence object to manage the generation process for this prompt
        seq = Sequence(
            prompt_text=prompt_text,  # The original prompt text
            max_length=self.max_length,  # Maximum length for the generated sequence
            eos_token_id=self.tokenizer.eos_token_id,  # End-of-sequence token ID
            qid=qid  # Unique query ID for tracking this sequence
        )

        # Store the tokenized prompt in the Sequence object
        seq.set_prompt_tokens(prompt_tokens.clone())  # Clone to avoid unintended modifications

        # Lazy-create the n-gram model if needed
        # if self.ngram_registry[qid]['model'] is None:
        #     self.ngram_registry[qid]['model'] = NGram(-1, -1, prompt_text)

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
        """
        Combine the KV caches of all active sequences into a single batched format.
        This allows efficient processing of multiple sequences in parallel.
        """
        if not self.active_seqs:
            return None
        layers = len(self.active_seqs[0].kv_cache)  # Number of layers in the KV cache
        batched = []
        for i in range(layers):
            # Collect keys and values for the current layer from all active sequences
            ks = [s.kv_cache[i][0] for s in self.active_seqs]
            vs = [s.kv_cache[i][1] for s in self.active_seqs]
            # Concatenate keys and values along the batch dimension
            batched.append((torch.cat(ks, 0), torch.cat(vs, 0)))
        return tuple(batched)

    def _build_leftpad_attention_mask(self):
        """
        Build a left-padded attention mask for all active sequences.
        This ensures that only valid tokens are attended to during generation.
        """
        if not self.active_seqs:
            return None
        T = self.active_seqs[0].kv_cache[0][0].shape[2]  # Sequence length in the KV cache
        B = len(self.active_seqs)  # Number of active sequences
        mask = torch.zeros(B, T + 1, device=self.device)  # Initialize the mask with zeros
        for i, seq in enumerate(self.active_seqs):
            v = seq.get_valid_length()  # Get the valid length of the sequence
            mask[i, T - v:] = 1  # Set the valid positions to 1
        return mask.long()

    def _generate_step_with_kv(self):
        """
        Perform a single generation step using KV caching for active sequences.
        This method updates the KV cache and generates the next token for each sequence.
        """
        if not self.active_seqs:
            return None, None

        toks, pos, idle = [], [], 0
        eff_len = lambda s: s.get_valid_length() + (1 if s.is_finished() else 0)

        # Prepare input tokens and position IDs for all active sequences
        for s in self.active_seqs:
            t = s.next_input_token() or self.tokenizer.eos_token_id  # Get the next token or EOS token
            if s.is_finished():
                idle += 1  # Count finished sequences
            toks.append(t)
            pos.append(eff_len(s))

        # If all sequences are finished, return early
        if idle == len(self.active_seqs):
            return None, None

        # Convert tokens and positions to tensors
        input_ids = torch.tensor(toks, device=self.device).unsqueeze(1)
        position_ids = torch.tensor(pos, device=self.device).unsqueeze(1)

        # Get batched KV cache and attention mask
        batched = self._get_batched_kv()
        mask = self._build_leftpad_attention_mask()
        token_mask = torch.ones(len(self.active_seqs), 1, device=self.device)
        mask = torch.cat([mask, token_mask], 1)

        # Perform the forward pass through the model
        with torch.no_grad():
            out = self.model(
                input_ids=input_ids,
                position_ids=position_ids,
                past_key_values=KVCacheWrapper.wrap(batched, self.model),
                use_cache=True,
                attention_mask=mask,
            )

        logits = out.logits[:, -1]  # Extract logits for the last token
        new_past = out.past_key_values  # Extract updated KV cache

        # Split the updated KV cache back into per-sequence format
        per = []
        for i in range(len(self.active_seqs)):
            seq_kv = []
            for layer in new_past:
                seq_kv.append((layer[0][i:i + 1], layer[1][i:i + 1]))
            per.append(seq_kv)

        return logits, per

    def _generate_step_without_kv(self):
        """
        Perform a single generation step without using KV caching.
        This method processes all active sequences in a single forward pass.
        """
        if not self.active_seqs:
            return None, None

        sequences_inputs = []

        # Prepare input tensors for all active sequences
        for seq in self.active_seqs:
            gen_tensor = torch.tensor(seq.generated_tokens, device=self.device)  # Generated tokens
            full_input = torch.cat([seq.prompt_tokens, gen_tensor])  # Combine prompt and generated tokens
            sequences_inputs.append(full_input)

        # Pad sequences to the same length and create an attention mask
        padded = pad_sequence(sequences_inputs, batch_first=True, padding_value=self.tokenizer.eos_token_id).to(self.device)
        attention_mask = (padded != self.tokenizer.eos_token_id).long().to(self.device)

        # Perform the forward pass through the model
        with torch.no_grad():
            outputs = self.model(
                input_ids=padded,
                attention_mask=attention_mask,
                use_cache=False,
            )

        logits_list = []

        # Extract logits for the last token of each sequence
        for i, seq in enumerate(self.active_seqs):
            seq_len = sequences_inputs[i].shape[0]  # Length of the current sequence
            last_logits = outputs.logits[i, seq_len - 1, :]  # Logits for the last token
            logits_list.append(last_logits)

        logits_tensor = torch.stack(logits_list, dim=0)  # Stack logits into a single tensor
        return logits_tensor, None

    def _accept_speculative(self, q_logits: torch.Tensor, p_logits: torch.Tensor) -> torch.BoolTensor:
        '''
        Given q_i(x) and p_i(x) for i in [1..gamma],
        returns a boolean mask of length gamma indicating which speculative tokens to accept contiguously.
        '''
        print(q_logits.shape, p_logits.shape)
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
        Static generation loop refactored to mirror the continuous version's flow:
         - Refill at regular intervals
         - Generate one step per iteration
         - Clean up finished sequences and free their KV caches
         - Log GPU stats each generation step
        """
        # Initial refill
        self._refill_active_seqs()
        step_counter = 0

        # Continue until all prompts are processed
        while self.active_seqs or self.prompt_deque:
            # Refill at period boundaries
            if step_counter % self.refill_period == 0:
                self._cleanup_and_refill()

            # Single generation step
            if self.use_kv_cache:
                logits, new_past_list = self._generate_step_with_kv()
            else:
                logits, new_past_list = self._generate_step_without_kv()

            # If no logits returned, skip and advance step
            if logits is None:
                step_counter += 1
                continue

            # Append next token and update KV caches
            for i, seq in enumerate(self.active_seqs):
                token = PromptSampler.sample_token(logits[i])
                seq.append_token(token)
                if self.use_kv_cache and new_past_list is not None:
                    seq.kv_cache = new_past_list[i]

            step_counter += 1
            self._log_gpu_stats(step_counter)

        # Finalize any remaining sequences
        for seq in self.active_seqs:
            text = self.tokenizer.decode(seq.get_final_generation(), skip_special_tokens=True)
            self.results.setdefault(seq.prompt_text, []).append(text)

    def _run_generation_continuous(self):
        self._refill_active_seqs()
        step=0
        while self.active_seqs or self.prompt_deque:
            if step % self.refill_period==0:
                self._cleanup_and_refill()
            # generation step
            if self.use_kv_cache:
                logits, new_past = self._generate_step_with_kv()
            else:
                logits, _ = self._generate_step_without_kv()
            if logits is None:
                step+=1; continue
            for i,seq in enumerate(self.active_seqs):
                tok = PromptSampler.sample_token(logits[i])
                seq.append_token(tok)
                if self.use_kv_cache:
                    seq.kv_cache = new_past[i]
            step+=1
            self._log_gpu_stats(step)
        # finalize remaining
        for seq in self.active_seqs:
            txt=self.tokenizer.decode(seq.get_final_generation(), skip_special_tokens=True)
            self.results.setdefault(seq.prompt_text,[]).append(txt)

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
