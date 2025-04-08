# stream_manager.py

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
        wandb_logging=False
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
        """
        self.model = model
        self.tokenizer = tokenizer
        self.stream_width = stream_width
        self.max_length = max_length
        self.refill_period = refill_period
        self.use_kv_cache = use_kv_cache
        self.continuous_batching = continuous_batching
        self.wandb_logging = wandb_logging
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

    def enqueue_prompt(self, prompt_text, num_completions=1):
        """Enqueue a prompt with its requested number of completions."""
        self.prompt_deque.append((prompt_text, num_completions))

    def _prefill_prompt(self, prompt_text, num_completions):
        """
        Tokenize the prompt, run it through the model to compute the prefix KV cache (if using KV caching),
        then initialize a Sequence with a cloned KV cache and first sampled token.
        Also store the tokenized prompt in seq.prompt_tokens for recomputation when KV caching is off.
        """
        inputs = self.tokenizer(prompt_text, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        prompt_tokens = inputs['input_ids'][0]  # [prompt_length]
        
        if self.use_kv_cache:
            with torch.no_grad():
                outputs = self.model(**inputs, use_cache=True)
            prefix_kv = outputs.past_key_values  # List of (key, value) pairs per layer.
        else:
            with torch.no_grad():
                outputs = self.model(**inputs, use_cache=False)

        prefix_logits = outputs.logits[0, -1, :]
        token_id = PromptSampler.sample_token(outputs.logits[0, -1, :])
        seq = Sequence(prompt_text, max_length=self.max_length, eos_token_id=self.tokenizer.eos_token_id)
        seq.prompt_tokens = prompt_tokens.clone()
        if self.use_kv_cache:
            seq.kv_cache = KVCacheManager.clone(prefix_kv)
        seq.append_token(token_id)
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
        Pad the key and value tensors along the sequence dimension so that they all have the same length.
        Each seq.kv_cache is a list (one per layer) of tuples (key, value) with shape [1, n_heads, seq_len, head_dim].
        """
        if not self.active_seqs:
            return None
        num_layers = len(self.active_seqs[0].kv_cache)
        batched_past = []
        for layer in range(num_layers):
            layer_keys = []
            layer_values = []
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
            batched_keys = torch.cat(layer_keys, dim=0)
            batched_values = torch.cat(layer_values, dim=0)
            batched_past.append((batched_keys, batched_values))
        return tuple(batched_past)

    def _generate_step_with_kv(self):
        """
        Perform one generation step using KV caching.
        Returns:
            logits_tensor: Tensor of shape [batch, vocab_size]
            new_past: A list (length=batch) of updated KV caches.
        """
        if not self.active_seqs:
            return None, None

        input_tokens = [seq.next_input_token() for seq in self.active_seqs]
        input_ids = torch.tensor(input_tokens, dtype=torch.long, device=self.device).unsqueeze(1)
        attention_mask = torch.ones_like(input_ids, device=self.device)
        past = self._stack_past_kv()
        # Use the KVCacheWrapper to wrap past appropriately.
        past = KVCacheWrapper.wrap(past, self.model)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                past_key_values=past,
            )
        logits_full = outputs.logits  # [batch, 1, vocab_size]
        new_past_full = outputs.past_key_values
        logits = [logits_full[i, -1, :] for i in range(len(self.active_seqs))]
        new_past = []
        for i in range(len(self.active_seqs)):
            seq_new_past = []
            for layer in range(len(new_past_full)):
                k_layer = new_past_full[layer][0][i].unsqueeze(0)
                v_layer = new_past_full[layer][1][i].unsqueeze(0)
                seq_new_past.append((k_layer, v_layer))
            new_past.append(seq_new_past)
        logits_tensor = torch.stack(logits, dim=0)
        return logits_tensor, new_past

    def _generate_step_without_kv(self):
        """
        Perform one generation step without KV caching.
        Recompute the full context for each active sequence.
        """
        sequences_inputs = []
        for seq in self.active_seqs:
            gen_tensor = torch.tensor(seq.generated_tokens, device=self.device)
            full_input = torch.cat([seq.prompt_tokens, gen_tensor])
            sequences_inputs.append(full_input)
        padded_inputs = pad_sequence(sequences_inputs, batch_first=True, padding_value=self.tokenizer.eos_token_id)
        attention_mask = (padded_inputs != self.tokenizer.eos_token_id).long()
        with torch.no_grad():
            outputs = self.model(
                input_ids=padded_inputs,
                attention_mask=attention_mask,
                use_cache=False,
            )
        logits_list = []
        for i, seq in enumerate(self.active_seqs):
            seq_len = sequences_inputs[i].shape[0]
            logits_list.append(outputs.logits[i, seq_len - 1, :])
        logits_tensor = torch.stack(logits_list, dim=0)
        return logits_tensor, None

    def run_generation_loop(self):
        """
        Main generation loop.
        Continues until both the active sequences and the prompt deque are empty.
        If continuous_batching is True, the engine refills the active stream periodically;
        otherwise, it processes a full batch until completion before swapping in new prompts.
        After generation, results are saved to a file.
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

            if step_counter % self.refill_period == 0:
                self._refill_active_seqs()

            if not self.active_seqs:
                step_counter += 1
                continue

            if self.use_kv_cache:
                logits, new_past_list = self._generate_step_with_kv()
            else:
                logits, _ = self._generate_step_without_kv()

            finished_indices = []
            for i, seq in enumerate(self.active_seqs):
                token_logits = logits[i]
                next_token = PromptSampler.sample_token(token_logits)
                seq.append_token(next_token)
                if self.use_kv_cache:
                    seq.kv_cache = new_past_list[i]
                if seq.is_finished():
                    finished_indices.append(i)

            for idx in sorted(finished_indices, reverse=True):
                seq = self.active_seqs[idx]
                generated_text = self.tokenizer.decode(seq.generated_tokens, skip_special_tokens=True)
                if seq.prompt_text not in self.results:
                    self.results[seq.prompt_text] = []
                self.results[seq.prompt_text].append(generated_text)
                del self.active_seqs[idx]

            step_counter += 1

    def _run_generation_static(self):
        """
        Run the generation loop in static batching mode.
        Process the entire active batch until all sequences are complete,
        then refill a new batch from the prompt deque.
        """
        while self.active_seqs or self.prompt_deque:
            self._refill_active_seqs()
            while self.active_seqs:
                non_padding_fraction = len(self.active_seqs) / self.stream_width
                if self.wandb_logging:
                    self.wandb.log({"non_padding_fraction": non_padding_fraction})

                if self.use_kv_cache:
                    logits, new_past_list = self._generate_step_with_kv()
                else:
                    logits, _ = self._generate_step_without_kv()

                finished_indices = []
                for i, seq in enumerate(self.active_seqs):
                    token_logits = logits[i]
                    next_token = PromptSampler.sample_token(token_logits)
                    seq.append_token(next_token)
                    if self.use_kv_cache:
                        seq.kv_cache = new_past_list[i]
                    if seq.is_finished():
                        finished_indices.append(i)

                for idx in sorted(finished_indices, reverse=True):
                    seq = self.active_seqs[idx]
                    generated_text = self.tokenizer.decode(seq.generated_tokens, skip_special_tokens=True)
                    if seq.prompt_text not in self.results:
                        self.results[seq.prompt_text] = []
                    self.results[seq.prompt_text].append(generated_text)
                    del self.active_seqs[idx]

    def save_results(self, filename):
        """Save the accumulated generation results to a JSON file."""
        with open(filename, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to {filename}")