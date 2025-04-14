import torch
import json
from collections import deque
from torch.nn.utils.rnn import pad_sequence
from .prompt_sampler import PromptSampler
from .kv_cache_manager import KVCacheManager
from .kv_cache_wrapper import KVCacheWrapper
from .sequence import Sequence

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
        wandb_logging=False,
        debug=False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.stream_width = stream_width
        self.max_length = max_length
        self.refill_period = refill_period
        self.use_kv_cache = use_kv_cache
        self.continuous_batching = continuous_batching
        self.wandb_logging = wandb_logging
        self.debug = debug

        if self.wandb_logging:
            try:
                import wandb
                self.wandb = wandb
            except ImportError:
                raise ImportError("wandb_logging is enabled, but wandb is not installed.")

        self.prompt_deque = deque()
        self.active_seqs = []
        self.results = {}
        self.device = next(model.parameters()).device

    def _debug_print(self, msg):
        if self.debug:
            print(f"[DEBUG] {msg}")

    def enqueue_prompt(self, prompt_text, num_completions=1):
        self._debug_print(f"Enqueue prompt: {prompt_text[:60]!r}, num_completions={num_completions}")
        self.prompt_deque.append((prompt_text, num_completions))

    def _prefill_prompt(self, prompt_text, num_completions):
        self._debug_print(f"Prefilling prompt: {prompt_text[:60]!r}")
        inputs = self.tokenizer(prompt_text, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # e.g. shape [prompt_len]
        prompt_tokens = inputs["input_ids"][0]

        # forward pass on entire prompt
        with torch.no_grad():
            outputs = self.model(**inputs, use_cache=self.use_kv_cache)

        # create a Sequence
        seq = Sequence(
            prompt_text=prompt_text,
            max_length=self.max_length,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        # store the prompt tokens, fill length_mask for them
        seq.set_prompt_tokens(prompt_tokens.clone())

        if self.use_kv_cache:
            seq.kv_cache = KVCacheManager.clone(outputs.past_key_values)

        # sample 1 token from the final logits
        prefix_logits = outputs.logits[0, -1, :]
        token_id = PromptSampler.sample_token(prefix_logits)

        # do a second forward pass *with that single token*
        # so the KV includes it
        single_tok = torch.tensor([[token_id]], device=self.device)
        if self.use_kv_cache:
            with torch.no_grad():
                out2 = self.model(
                    input_ids=single_tok,
                    past_key_values=KVCacheWrapper.wrap(seq.kv_cache, self.model),
                    use_cache=True
                )
            # the new kv_cache now has shape [prompt_len+1]
            seq.kv_cache = KVCacheManager.clone(out2.past_key_values)

        # now append that token to the Sequence
        seq.append_token(token_id)

        # return a list of (possibly) multiple sequences if you want num_completions>1
        return [seq]
    
    def _refill_active_seqs(self):
        # Step 1: Determine how many new sequences are needed.
        num_active = len(self.active_seqs)
        num_new_needed = self.stream_width - num_active

        # Step 2: For all currently active sequences, trim their KV caches to contain only the valid tokens.
        # This step removes previously added left-padding (and the corresponding dummy values) from the KV caches.
        for seq in self.active_seqs:
            valid_length = seq.get_valid_length()
            trimmed_kv = []
            for (k, v) in seq.kv_cache:
                # k and v are assumed to have shape [1, num_heads, seq_len, head_dim].
                # Here we extract only the last 'valid_length' tokens along the time dimension.
                trimmed_k = k[..., -valid_length:, :]
                trimmed_v = v[..., -valid_length:, :]
                trimmed_kv.append((trimmed_k, trimmed_v))
            seq.kv_cache = trimmed_kv

        # Step 3: Add new sequences from the prompt deque until we have enough to fill the stream.
        for _ in range(num_new_needed):
            if not self.prompt_deque:
                break
            # Pop one prompt (handling count > 1)
            prompt_text, count = self.prompt_deque.popleft()
            new_seqs = self._prefill_prompt(prompt_text, 1)
            for new_seq in new_seqs:
                self.active_seqs.append(new_seq)
            if count > 1:
                # Reinsert with a decremented count.
                self.prompt_deque.append((prompt_text, count - 1))
        
        # Step 4: Determine the maximum valid length among all active sequences.
        if self.active_seqs:
            max_valid_length = max(seq.get_valid_length() for seq in self.active_seqs)
        else:
            max_valid_length = 0

        # Step 5: Left-pad all active sequences' KV caches to match the maximum valid length.
        # For sequences that are shorter, add dummy (zeroed) tensors on the left.
        for seq in self.active_seqs:
            current_valid_length = seq.get_valid_length()
            pad_size = max_valid_length - current_valid_length
            if pad_size > 0:
                padded_kv = []
                for (k, v) in seq.kv_cache:
                    # Pad on the left along the time dimension (assumed dim=2)
                    padded_k = torch.nn.functional.pad(k, (0, 0, pad_size, 0), mode="constant", value=0)
                    padded_v = torch.nn.functional.pad(v, (0, 0, pad_size, 0), mode="constant", value=0)
                    padded_kv.append((padded_k, padded_v))
                seq.kv_cache = padded_kv

    def _get_batched_kv(self):
        """
        Batch KV caches from active sequences.
        Assumes that all active sequences have aligned KV caches.
        The resulting batched KV cache for each layer will have shape:
          [B, num_heads, seq_len, head_dim]
        """
        if not self.active_seqs:
            return None

        num_layers = len(self.active_seqs[0].kv_cache)
        for seq in self.active_seqs:
            if len(seq.kv_cache) != num_layers:
                raise ValueError("Mismatched # of layers in KV caches among sequences.")

        batched = []
        for layer_idx in range(num_layers):
            layer_keys = []
            layer_values = []
            for seq in self.active_seqs:
                # Each kv cache here is assumed to be aligned.
                (k, v) = seq.kv_cache[layer_idx]  # shape: [1, heads, seq_len, head_dim]
                layer_keys.append(k)
                layer_values.append(v)
            batched_keys = torch.cat(layer_keys, dim=0)
            batched_values = torch.cat(layer_values, dim=0)
            batched.append((batched_keys, batched_values))
        return tuple(batched)

    def _build_leftpad_attention_mask(self):
        """
        Build an attention mask [B, max_seq_len] for the current active batch,
        consistent with left-padded KV caches.
        """
        if not self.active_seqs:
            return None

        max_seq_len = max(seq.get_valid_length() for seq in self.active_seqs)
        print(f"max_seq_len: {max_seq_len}")
        if max_seq_len == 0:
            return None

        attention_mask = torch.zeros(len(self.active_seqs), max_seq_len, dtype=torch.long, device=self.device)
        for i, seq in enumerate(self.active_seqs):
            seq_len = seq.get_valid_length()
            if seq_len > 0:
                start = max_seq_len - seq_len
                attention_mask[i, start:] = 1
        return attention_mask

    def _generate_step_with_kv(self):
        if not self.active_seqs:
            return None, None

        # 1. Collect "last token" for each sequence.
        input_tokens = []
        positions = []
        num_idle_seqs = 0
        for seq in self.active_seqs:
            last_tok = seq.next_input_token()
            if last_tok is None:
                last_tok = self.tokenizer.eos_token_id
                print("almost idle")
            if seq.is_finished():
                last_tok = self.tokenizer.eos_token_id
                num_idle_seqs += 1
                print("idle")
            input_tokens.append(last_tok)
            pos = len(seq.prompt_tokens) + len(seq.generated_tokens)
            positions.append(max(pos, 0))
        print(len(self.active_seqs), self.stream_width, num_idle_seqs)
        if num_idle_seqs == len(self.active_seqs):
            # All sequences are finished (past their eos); nothing to generate.
            return None, None
        print("----")

        input_ids = torch.tensor(input_tokens, dtype=torch.long, device=self.device).unsqueeze(1)
        position_ids = torch.tensor(positions, dtype=torch.long, device=self.device).unsqueeze(1)

        # 2. Get the batched KV cache (which is assumed to be aligned)
        batched_kv = self._get_batched_kv()
        if batched_kv is None:
            return None, None

        past_for_model = KVCacheWrapper.wrap(batched_kv, self.model)

        # 3. Build the attention mask and add one extra column for the new token.
        attention_mask = self._build_leftpad_attention_mask()
        if attention_mask is None:
            return None, None
        token_mask = torch.ones(attention_mask.size(0), 1, dtype=attention_mask.dtype, device=attention_mask.device)
        attention_mask = torch.cat([attention_mask, token_mask], dim=1)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,         # shape [B, 1]
                position_ids=position_ids,   # shape [B, 1]
                past_key_values=past_for_model,
                use_cache=True,
                attention_mask=attention_mask,
            )

        # 4. Re-split the new past KV into a list per sequence.
        logits_full = outputs.logits  # shape [B, 1, vocab_size]
        new_past_full = outputs.past_key_values
        batch_size = len(self.active_seqs)

        new_past_list = []
        for i in range(batch_size):
            seq_past = []
            for layer_i in range(len(new_past_full)):
                k_layer = new_past_full[layer_i][0][i].unsqueeze(0)
                v_layer = new_past_full[layer_i][1][i].unsqueeze(0)
                seq_past.append((k_layer, v_layer))
            new_past_list.append(seq_past)

        return logits_full[:, -1, :], new_past_list

    def _generate_step_without_kv(self):
        if not self.active_seqs:
            return None, None

        sequences_inputs = []
        for seq in self.active_seqs:
            gen_tensor = torch.tensor(seq.generated_tokens, device=self.device)
            full_input = torch.cat([seq.prompt_tokens, gen_tensor])
            sequences_inputs.append(full_input)

        padded = pad_sequence(sequences_inputs, batch_first=True, padding_value=self.tokenizer.eos_token_id).to(self.device)
        attention_mask = (padded != self.tokenizer.eos_token_id).long().to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=padded,
                attention_mask=attention_mask,
                use_cache=False,
            )

        logits_list = []
        for i, seq in enumerate(self.active_seqs):
            seq_len = sequences_inputs[i].shape[0]
            last_logits = outputs.logits[i, seq_len - 1, :]
            logits_list.append(last_logits)

        logits_tensor = torch.stack(logits_list, dim=0)
        return logits_tensor, None

    def run_generation_loop(self):
        if self.continuous_batching:
            self._run_generation_continuous()
        else:
            self._run_generation_static()
        self.save_results("generation_results.json")

    def _run_generation_continuous(self):
        # Initially, try to refill the active batch.
        self._refill_active_seqs()
        step_counter = 0

        # Continue until there are no active sequences and no prompts waiting.
        while self.active_seqs or self.prompt_deque:

            # Every 'refill_period' steps, process finished sequences and refill.
            if step_counter % self.refill_period == 0:               
                still_active = []
                for seq in self.active_seqs:
                    # Here we check if the sequence has ever generated an EOS token.
                    # This assumes that seq.generated_tokens is a list/tensor of token IDs.
                    if self.tokenizer.eos_token_id in seq.generated_tokens or seq.is_finished():
                        # Trim the sequence to only include tokens up to (and including) the first EOS.
                        trimmed_tokens = seq.get_final_generation()
                        gen_text = self.tokenizer.decode(trimmed_tokens, skip_special_tokens=True)
                        if seq.prompt_text not in self.results:
                            self.results[seq.prompt_text] = []
                        self.results[seq.prompt_text].append(gen_text)
                        # Do not add this sequence back into still_active.
                    else:
                        # If no EOS has been generated, continue working on this sequence.
                        still_active.append(seq)

                # Update the active sequences: finished ones have been removed.
                self.active_seqs = still_active
                # Refill active sequences up to stream_width (i.e. add new prompts as needed).
                self._refill_active_seqs()
                for seq in still_active:
                    print(f"{len(seq.generated_tokens)} / {seq.max_length} tokens generated")
                print("---")
            # If there are no sequences left, simply advance the step counter.
            if not self.active_seqs:
                step_counter += 1
                continue

            # Generate one token for each sequence.
            # Notice we do not check here if a sequence is finished;
            # we always sample and append a new token.
            if self.use_kv_cache:
                logits, new_past_list = self._generate_step_with_kv()
            else:
                logits, new_past_list = self._generate_step_without_kv()

            # If generation returned nothing, skip this step.
            if logits is None:
                step_counter += 1
                continue

            # For every active sequence, sample and append the new token.
            # Even if a sequence already contains an EOS, we continue appending (and later trim).
            for i, seq in enumerate(self.active_seqs):
                token_logits = logits[i]
                next_token = PromptSampler.sample_token(token_logits)
                seq.append_token(next_token)
                # Update the KV cache if needed.
                if self.use_kv_cache and new_past_list is not None:
                    seq.kv_cache = new_past_list[i]

            step_counter += 1

        # After exiting the loop, if any active sequences remain, trim them and store their result.
        for seq in self.active_seqs:
            trimmed_tokens = seq.get_final_generation()  # This should trim tokens after EOS if any.
            gen_text = self.tokenizer.decode(trimmed_tokens, skip_special_tokens=True)
            if seq.prompt_text not in self.results:
                self.results[seq.prompt_text] = []
            self.results[seq.prompt_text].append(gen_text)
    
    def _run_generation_static(self):
        while self.active_seqs or self.prompt_deque:
            self._refill_active_seqs()
            if not self.active_seqs:
                continue

            step_counter = 0
            while self.active_seqs:
                if self.wandb_logging:
                    frac = len(self.active_seqs) / self.stream_width
                    self.wandb.log({"non_padding_fraction": frac})

                if self.use_kv_cache:
                    logits, new_past_list = self._generate_step_with_kv()
                else:
                    logits, new_past_list = self._generate_step_without_kv()

                if logits is None:
                    step_counter += 1
                    break

                finished_indices = []
                for i, seq in enumerate(self.active_seqs):
                    token_logits = logits[i]
                    next_token = PromptSampler.sample_token(token_logits)
                    seq.append_token(next_token)
                    self._debug_print(
                        f"[Static step {step_counter}] Seq {i} appended {next_token} -> "
                        f"{self.tokenizer.decode([next_token], skip_special_tokens=True)!r}"
                    )
                    if self.use_kv_cache and new_past_list is not None:
                        seq.kv_cache = new_past_list[i]

                    if seq.is_finished():
                        finished_indices.append(i)

                for idx in reversed(finished_indices):
                    seq = self.active_seqs[idx]
                    text = self.tokenizer.decode(seq.generated_tokens, skip_special_tokens=True)
                    if seq.prompt_text not in self.results:
                        self.results[seq.prompt_text] = []
                    self.results[seq.prompt_text].append(text)
                    self._debug_print(f"Sequence {idx} finished. Final text: {text!r}")
                    del self.active_seqs[idx]

                step_counter += 1

    def save_results(self, filename):
        with open(filename, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to {filename}")