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

    def _align_new_sequence(self, seq):
        """
        Align a new sequence's KV cache with the current active sequences.
        When a new prompt is added, if there are already active sequences,
        pad this new sequence’s KV cache (along the time dimension) so that it
        matches the maximum valid length among the active sequences.
        """
        if not self.active_seqs:
            # No active sequences exist; nothing to align.
            return seq

        current_max_len = max(s.get_valid_length() for s in self.active_seqs)
        seq_valid_len = seq.get_valid_length()
        if seq_valid_len < current_max_len:
            pad_size = current_max_len - seq_valid_len
            new_cache = []
            for (k, v) in seq.kv_cache:
                # Left-pad the time dimension (dimension 2)
                k = torch.nn.functional.pad(k, (0, 0, pad_size, 0), "constant", 0)
                v = torch.nn.functional.pad(v, (0, 0, pad_size, 0), "constant", 0)
                new_cache.append((k, v))
            seq.kv_cache = new_cache
        return seq

    def _refill_active_seqs(self):
        while len(self.active_seqs) < self.stream_width and self.prompt_deque:
            if len(self.prompt_deque) > 1:
                prompt_text, count = self.prompt_deque.popleft()
                new_seqs = self._prefill_prompt(prompt_text, 1)
                for seq in new_seqs:
                    aligned_seq = self._align_new_sequence(seq)
                    self.active_seqs.append(aligned_seq)
                if count > 1:
                    self.prompt_deque.append((prompt_text, count - 1))
            else:
                prompt_text, count = self.prompt_deque[0]
                new_seqs = self._prefill_prompt(prompt_text, 1)
                for seq in new_seqs:
                    aligned_seq = self._align_new_sequence(seq)
                    self.active_seqs.append(aligned_seq)
                if count > 1:
                    self.prompt_deque[0] = (prompt_text, count - 1)
                else:
                    self.prompt_deque.popleft()

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
        for i, seq in enumerate(self.active_seqs):
            last_tok = seq.next_input_token()
            if last_tok is None:
                last_tok = self.tokenizer.eos_token_id
            if seq.is_finished():
                last_tok = self.tokenizer.eos_token_id
                print(f"====================Sequence finished, using EOS token: {last_tok} for {i}==============================")
            input_tokens.append(last_tok)
            pos = len(seq.prompt_tokens) + len(seq.generated_tokens) - 1
            positions.append(max(pos, 0))

        input_ids = torch.tensor(input_tokens, dtype=torch.long, device=self.device).unsqueeze(1)
        position_ids = torch.tensor(positions, dtype=torch.long, device=self.device).unsqueeze(1)

        # 2. Get the batched KV cache (which is assumed to be aligned)
        batched_kv = self._get_batched_kv()
        if batched_kv is None:
            return None, None

        past_for_model = KVCacheWrapper.wrap(batched_kv, self.model)
        print(f"past_for_model shape: {past_for_model[0][0].shape}")

        # 3. Build the attention mask and add one extra column for the new token.
        attention_mask = self._build_leftpad_attention_mask()
        print(f"attention_mask shape: {attention_mask.shape}")
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
        print(f"new_past_full shape: {new_past_full[1][0].shape}")
        batch_size = len(self.active_seqs)

        new_past_list = []
        for i in range(batch_size):
            seq_past = []
            for layer_i in range(len(new_past_full)):
                k_layer = new_past_full[layer_i][0][i].unsqueeze(0)
                v_layer = new_past_full[layer_i][1][i].unsqueeze(0)
                seq_past.append((k_layer, v_layer))
            new_past_list.append(seq_past)
        print(f"new_past_list[0][1][0] shape: {new_past_list[0][1][0].shape}")

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
        self._refill_active_seqs()
        step_counter = 0

        while self.active_seqs or self.prompt_deque:
            if step_counter % self.refill_period == 0:
                # Remove finished sequences and store final results.
                still_active = []
                for seq in self.active_seqs:
                    if not seq.is_finished():
                        still_active.append(seq)
                    else:
                        gen_text = self.tokenizer.decode(seq.get_final_generation(), skip_special_tokens=True)
                        if seq.prompt_text not in self.results:
                            self.results[seq.prompt_text] = []
                        self.results[seq.prompt_text].append(gen_text)
                self.active_seqs = still_active

                # Refill active sequences up to stream_width.
                self._refill_active_seqs()

            if not self.active_seqs:
                step_counter += 1
                continue

            if self.use_kv_cache:
                logits, new_past_list = self._generate_step_with_kv()
            else:
                logits, new_past_list = self._generate_step_without_kv()

            if logits is None:
                step_counter += 1
                continue

            for i, seq in enumerate(self.active_seqs):
                if not seq.is_finished():
                    token_logits = logits[i]
                    next_token = PromptSampler.sample_token(token_logits)
                else:
                    next_token = self.tokenizer.eos_token_id

                seq.append_token(next_token)

                if self.use_kv_cache and new_past_list is not None:
                    seq.kv_cache = new_past_list[i]

            step_counter += 1

        # Dump any final results.
        for seq in self.active_seqs:
            gen_text = self.tokenizer.decode(seq.get_final_generation(), skip_special_tokens=True)
            if seq.prompt_text not in self.results:
                self.results[seq.prompt_text] = []
            self.results[seq.prompt_text].append(gen_text)

        self.save_results("generation_results.json")

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