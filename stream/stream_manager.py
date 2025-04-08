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
        """
        :param model: The Hugging Face model.
        :param tokenizer: The corresponding tokenizer.
        :param stream_width: Maximum number of active sequences (batch size).
        :param max_length: Maximum tokens to generate per sequence.
        :param refill_period: Number of generation steps between checking for new prompts.
        :param use_kv_cache: If True, reuse past key–value caches; if False, recompute full context.
        :param continuous_batching: If True, refill continuously; if False, process a full batch before swapping.
        :param wandb_logging: If True, log metrics to wandb.
        :param debug: If True, print debugging information (shapes, tokens, partial completions, etc.).
        """
        self.model = model
        self.tokenizer = tokenizer
        self.stream_width = stream_width
        self.max_length = max_length
        self.refill_period = refill_period
        self.use_kv_cache = use_kv_cache
        self.continuous_batching = continuous_batching
        self.wandb_logging = wandb_logging
        self.debug = debug  # <- Store debug flag

        if self.wandb_logging:
            try:
                import wandb
                self.wandb = wandb
            except ImportError:
                raise ImportError("wandb_logging is enabled, but wandb is not installed.")

        # Use a deque to hold prompts as tuples (prompt_text, remaining_completions)
        self.prompt_deque = deque()
        self.active_seqs = []  # List of active Sequence objects.
        self.results = {}      # Dictionary to accumulate results: prompt -> list of completions.
        self.device = next(model.parameters()).device

    def _debug_print(self, msg):
        """Helper to print debug information if self.debug is True."""
        if self.debug:
            print(f"[DEBUG] {msg}")

    def enqueue_prompt(self, prompt_text, num_completions=1):
        """Enqueue a prompt with its requested number of completions."""
        self._debug_print(f"Enqueue prompt: {prompt_text[:60]!r}..., num_completions={num_completions}")
        self.prompt_deque.append((prompt_text, num_completions))

    def _prefill_prompt(self, prompt_text, num_completions):
        """
        Tokenize the prompt, run it through the model to compute the prefix KV cache (if using KV caching),
        then initialize a Sequence with a cloned KV cache and first sampled token.
        Also store the tokenized prompt in seq.prompt_tokens for recomputation when KV caching is off.

        We'll sample 1 token from the final logits of the prompt's forward pass so that the new Sequence
        is already 1 token into the generation by the time it enters the main loop.
        """
        self._debug_print(f"Prefilling prompt: {prompt_text[:60]!r}...")
        inputs = self.tokenizer(prompt_text, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        prompt_tokens = inputs['input_ids'][0]  # [prompt_length]

        with torch.no_grad():
            outputs = self.model(**inputs, use_cache=self.use_kv_cache)

        prefix_kv = outputs.past_key_values if self.use_kv_cache else None

        # Sample 1 token from the last logits
        prefix_logits = outputs.logits[0, -1, :]
        token_id = PromptSampler.sample_token(prefix_logits)

        # Create a new Sequence object
        seq = Sequence(prompt_text, max_length=self.max_length, eos_token_id=self.tokenizer.eos_token_id)
        seq.prompt_tokens = prompt_tokens.clone()

        if prefix_kv:
            seq.kv_cache = KVCacheManager.clone(prefix_kv)
            # Debug shapes
            num_layers = len(seq.kv_cache)
            for layer_idx in range(num_layers):
                k, v = seq.kv_cache[layer_idx]
                self._debug_print(f"   Layer {layer_idx} k.shape={k.shape}, v.shape={v.shape}")

        seq.append_token(token_id)
        self._debug_print(f"   First sampled token_id={token_id} -> {self.tokenizer.decode([token_id])!r}")
        return [seq]

    def _refill_active_seqs(self):
        """
        Refill active sequences from the prompt deque until the stream is full.
        To promote diversity, if there are multiple distinct prompts waiting,
        add only one completion per prompt at a time.
        """
        while len(self.active_seqs) < self.stream_width and self.prompt_deque:
            if len(self.prompt_deque) > 1:
                prompt_text, count = self.prompt_deque.popleft()
                new_seqs = self._prefill_prompt(prompt_text, 1)
                self.active_seqs.extend(new_seqs)
                if count > 1:
                    self.prompt_deque.append((prompt_text, count - 1))
            else:
                prompt_text, count = self.prompt_deque[0]
                new_seqs = self._prefill_prompt(prompt_text, 1)
                self.active_seqs.extend(new_seqs)
                if count > 1:
                    self.prompt_deque[0] = (prompt_text, count - 1)
                else:
                    self.prompt_deque.popleft()

    def _stack_past_kv(self):
        """
        Stack the KV caches from active sequences.
        Pad the key and value tensors along the sequence dimension so they match.

        Each seq.kv_cache is a list (one per layer) of tuples (key, value),
        each shaped [1, n_heads, seq_len, head_dim].

        Returns a tuple of length num_layers, each item is (stacked_k, stacked_v):
          stacked_k, stacked_v shapes -> [batch_size, n_heads, max_seq_len, head_dim].
        """
        if not self.active_seqs:
            return None

        # All active sequences must have the same number of layers in the KV
        num_layers = len(self.active_seqs[0].kv_cache)

        # Basic shape check
        for seq in self.active_seqs:
            if len(seq.kv_cache) != num_layers:
                raise ValueError("Mismatched number of layers in KV caches among sequences.")

        batched_past = []
        for layer in range(num_layers):
            layer_keys = []
            layer_values = []

            # Determine max sequence length among all sequences for this layer
            max_seq_len = max(seq.kv_cache[layer][0].shape[2] for seq in self.active_seqs)

            for seq in self.active_seqs:
                k, v = seq.kv_cache[layer]
                seq_len = k.shape[2]

                if seq_len < max_seq_len:
                    pad_size = max_seq_len - seq_len
                    k = torch.nn.functional.pad(k, (0, 0, 0, pad_size), "constant", 0)
                    v = torch.nn.functional.pad(v, (0, 0, 0, pad_size), "constant", 0)
                layer_keys.append(k)
                layer_values.append(v)

            stacked_keys = torch.cat(layer_keys, dim=0)
            stacked_values = torch.cat(layer_values, dim=0)
            batched_past.append((stacked_keys, stacked_values))

        # Debug shapes
        if self.debug:
            for layer_idx, (k_, v_) in enumerate(batched_past):
                self._debug_print(
                    f"KV layer {layer_idx} after stacking: k.shape={k_.shape}, v.shape={v_.shape}"
                )

        return tuple(batched_past)

    def _generate_step_with_kv(self):
        """
        Perform one generation step using KV caching.

        We'll feed exactly 1 new token per sequence as input_ids,
        pass the stacked past_key_values, and rely on the model's internal
        causal mask to handle partial prefix lengths. 
        We'll not build a big custom attention_mask to avoid dimension mismatches.
        """
        if not self.active_seqs:
            return None, None

        # Gather the last token appended to each sequence
        input_tokens = [seq.next_input_token() for seq in self.active_seqs]
        input_ids = torch.tensor(input_tokens, dtype=torch.long, device=self.device).unsqueeze(1)

        # Past KVs
        stacked_past = self._stack_past_kv()
        past_for_model = KVCacheWrapper.wrap(stacked_past, self.model)

        self._debug_print(f"_generate_step_with_kv: input_ids={input_tokens}")

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=None,  # Rely on standard causal mask.
                use_cache=True,
                past_key_values=past_for_model,
            )

        logits_full = outputs.logits  # shape [batch_size, 1, vocab_size]
        new_past_full = outputs.past_key_values
        batch_size = len(self.active_seqs)
        # Resplit new past for each seq
        new_past_list = []
        for i in range(batch_size):
            seq_past = []
            for layer_i in range(len(new_past_full)):
                k_layer = new_past_full[layer_i][0][i].unsqueeze(0)
                v_layer = new_past_full[layer_i][1][i].unsqueeze(0)
                seq_past.append((k_layer, v_layer))
            new_past_list.append(seq_past)

        logits_tensor = logits_full[:, -1, :]  # shape [batch_size, vocab_size]
        return logits_tensor, new_past_list

    def _generate_step_without_kv(self):
        """
        Perform one generation step WITHOUT KV caching: 
        Recompute from scratch with each sequence's full context (prompt + generated tokens).
        """
        if not self.active_seqs:
            return None, None

        sequences_inputs = []
        for seq in self.active_seqs:
            gen_tensor = torch.tensor(seq.generated_tokens, device=self.device)
            full_input = torch.cat([seq.prompt_tokens, gen_tensor])
            sequences_inputs.append(full_input)

        padded_inputs = pad_sequence(
            sequences_inputs, batch_first=True, padding_value=self.tokenizer.eos_token_id
        ).to(self.device)
        attention_mask = (padded_inputs != self.tokenizer.eos_token_id).long()

        with torch.no_grad():
            outputs = self.model(
                input_ids=padded_inputs,
                attention_mask=attention_mask.to(self.device),
                use_cache=False,
            )

        # Extract final logits for each sequence
        logits_list = []
        for i, seq in enumerate(self.active_seqs):
            seq_len = sequences_inputs[i].shape[0]
            last_logits = outputs.logits[i, seq_len - 1, :]
            logits_list.append(last_logits)

        logits_tensor = torch.stack(logits_list, dim=0)
        return logits_tensor, None

    def run_generation_loop(self):
        """
        Main generation loop.
        Continues until both the active sequences and the prompt deque are empty.
        """
        if self.continuous_batching:
            self._run_generation_continuous()
        else:
            self._run_generation_static()
        self.save_results("generation_results.json")

    def _run_generation_continuous(self):
        """Run the generation loop with continuous (dynamic) batching."""
        self._refill_active_seqs()
        step_counter = 0

        while self.active_seqs or self.prompt_deque:
            non_padding_fraction = len(self.active_seqs) / self.stream_width
            if self.wandb_logging:
                self.wandb.log({"non_padding_fraction": non_padding_fraction})

            # Refill at intervals
            if step_counter % self.refill_period == 0:
                self._refill_active_seqs()

            if not self.active_seqs:
                step_counter += 1
                continue

            if self.use_kv_cache:
                logits, new_past_list = self._generate_step_with_kv() # new_past_list [stream_width][num_layers][2][1, n_heads, seq_len, head_dim]

            else:
                logits, new_past_list = self._generate_step_without_kv()

            if logits is None:
                # no active seqs
                step_counter += 1
                continue

            finished_indices = []
            for i, seq in enumerate(self.active_seqs):
                token_logits = logits[i]
                next_token = PromptSampler.sample_token(token_logits)
                seq.append_token(next_token)
                self._debug_print(
                    f"[Step {step_counter}] Seq {i} appended token {next_token} ({self.tokenizer.decode([next_token])!r})"
                )

                if self.use_kv_cache and new_past_list is not None:
                    seq.kv_cache = new_past_list[i]

                if seq.is_finished():
                    finished_indices.append(i)

            # Remove finished sequences
            for idx in reversed(finished_indices):
                seq = self.active_seqs[idx]
                generated_text = self.tokenizer.decode(seq.generated_tokens, skip_special_tokens=True)
                if seq.prompt_text not in self.results:
                    self.results[seq.prompt_text] = []
                self.results[seq.prompt_text].append(generated_text)
                self._debug_print(f"Sequence {idx} finished. Final text: {generated_text!r}")
                del self.active_seqs[idx]

            step_counter += 1

    def _run_generation_static(self):
        """Static batching mode: process active batch fully, then move on."""
        while self.active_seqs or self.prompt_deque:
            self._refill_active_seqs()
            if not self.active_seqs:
                continue

            step_counter = 0
            while self.active_seqs:
                non_padding_fraction = len(self.active_seqs) / self.stream_width
                if self.wandb_logging:
                    self.wandb.log({"non_padding_fraction": non_padding_fraction})

                if self.use_kv_cache:
                    logits, new_past_list = self._generate_step_with_kv()
                else:
                    logits, new_past_list = self._generate_step_without_kv()

                if logits is None:
                    break

                finished_indices = []
                for i, seq in enumerate(self.active_seqs):
                    token_logits = logits[i]
                    next_token = PromptSampler.sample_token(token_logits)
                    seq.append_token(next_token)
                    self._debug_print(
                        f"[Static step {step_counter}] Seq {i} appended token {next_token} ({self.tokenizer.decode([next_token])!r})"
                    )

                    if self.use_kv_cache and new_past_list is not None:
                        seq.kv_cache = new_past_list[i]

                    if seq.is_finished():
                        finished_indices.append(i)

                for idx in reversed(finished_indices):
                    seq = self.active_seqs[idx]
                    generated_text = self.tokenizer.decode(seq.generated_tokens, skip_special_tokens=True)
                    if seq.prompt_text not in self.results:
                        self.results[seq.prompt_text] = []
                    self.results[seq.prompt_text].append(generated_text)
                    self._debug_print(f"Sequence {idx} finished. Final text: {generated_text!r}")
                    del self.active_seqs[idx]

                step_counter += 1

    def save_results(self, filename):
        """Save the accumulated generation results to a JSON file."""
        with open(filename, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to {filename}")