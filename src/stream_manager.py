# stream_manager.py

import torch
from queue import Queue
from prompt_sampler import PromptSampler
from kv_cache_manager import KVCacheManager
from sequence import Sequence

class StreamManager:
    def __init__(self, model, tokenizer, stream_width=8, max_length=50, refill_period=5):
        """
        :param model: The Hugging Face model.
        :param tokenizer: The corresponding tokenizer.
        :param stream_width: Maximum number of active sequences at once.
        :param max_length: Maximum tokens to generate per sequence.
        :param refill_period: Number of generation steps between checking for new prompts.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.stream_width = stream_width  
        self.max_length = max_length
        self.refill_period = refill_period  # Number of steps between checking the prompt queue.
        self.active_seqs = []  # list of active Sequence objects
        self.prompt_queue = Queue()  # holds (prompt_text, num_completions) pairs
        self.device = next(model.parameters()).device

    def enqueue_prompt(self, prompt_text, num_completions=1):
        """Enqueue a prompt with its requested number of completions."""
        self.prompt_queue.put((prompt_text, num_completions))

    def _prefill_prompt(self, prompt_text, num_completions):
        """Tokenize the prompt, run it through the model to compute the prefix KV cache,
        then initialize Sequence objects with a cloned KV cache and first sampled token."""
        inputs = self.tokenizer(prompt_text, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs, use_cache=True)
        prefix_kv = outputs.past_key_values  # list of (key, value) pairs per layer
        # Get logits for the last token in the prompt (from the first sample).
        prefix_logits = outputs.logits[0, -1, :]
        
        sequences = []
        # For each requested completion, sample the first token and clone the KV cache.
        for _ in range(num_completions):
            token_id = PromptSampler.sample_token(prefix_logits)
            seq = Sequence(prompt_text, max_length=self.max_length, eos_token_id=self.tokenizer.eos_token_id)
            seq.kv_cache = KVCacheManager.clone(prefix_kv)
            seq.append_token(token_id)
            sequences.append(seq)
        return sequences

    def _refill_active_seqs(self):
        """Attempt to refill active sequences from the prompt queue until stream_width is reached."""
        while len(self.active_seqs) < self.stream_width and not self.prompt_queue.empty():
            prompt_text, num_completions = self.prompt_queue.get()
            new_seqs = self._prefill_prompt(prompt_text, num_completions)
            self.active_seqs.extend(new_seqs)

    def _stack_past_kv(self):
        """
        Stack the KV caches from active sequences.
        Because sequences may have different lengths, we pad the key and value tensors along the sequence length dimension.
        Each seq.kv_cache is a list (one per layer) of tuples (key, value) with shape [1, n_heads, seq_len, head_dim].
        """
        num_layers = len(self.active_seqs[0].kv_cache)
        batched_past = []
        for layer in range(num_layers):
            layer_keys = []
            layer_values = []
            # Find the maximum sequence length for this layer.
            max_seq_len = max(seq.kv_cache[layer][0].shape[2] for seq in self.active_seqs)
            for seq in self.active_seqs:
                k, v = seq.kv_cache[layer]  # Each with shape [1, n_heads, seq_len, head_dim]
                seq_len = k.shape[2]
                if seq_len < max_seq_len:
                    # Pad along the sequence length (dim=2) with zeros.
                    pad_size = max_seq_len - seq_len
                    k = torch.nn.functional.pad(k, (0, 0, 0, pad_size), "constant", 0)
                    v = torch.nn.functional.pad(v, (0, 0, 0, pad_size), "constant", 0)
                layer_keys.append(k)
                layer_values.append(v)
            # Concatenate along batch dimension.
            batched_keys = torch.cat(layer_keys, dim=0)
            batched_values = torch.cat(layer_values, dim=0)
            batched_past.append((batched_keys, batched_values))
        return tuple(batched_past)

    def run_generation_loop(self):
        """Main generation loop.
        
        Every refill_period steps, check if new prompts can be inserted.
        At each step, process the active stream in lockstep.
        """
        self._refill_active_seqs()
        step_counter = 0

        while self.active_seqs:
            # Every refill_period steps, check the queue to refill active sequences.
            if step_counter % self.refill_period == 0:
                self._refill_active_seqs()

            # Prepare input tokens: take the last token from each active sequence.
            input_tokens = [seq.next_input_token() for seq in self.active_seqs]
            input_ids = torch.tensor(input_tokens, dtype=torch.long, device=self.device).unsqueeze(1)  # [batch, 1]
            attention_mask = torch.ones_like(input_ids, device=self.device)
            past = self._stack_past_kv()

            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                    past_key_values=past,
                )
            logits = outputs.logits  # shape: [batch, 1, vocab_size]
            new_past = outputs.past_key_values

            finished_indices = []
            for i, seq in enumerate(self.active_seqs):
                token_logits = logits[i, -1, :]
                next_token = PromptSampler.sample_token(token_logits)
                seq.append_token(next_token)
                # Update the KV cache for this sequence from the batched output.
                updated_kv = []
                for layer in range(len(new_past)):
                    k_layer = new_past[layer][0][i].unsqueeze(0)  # shape [1, n_heads, seq_len, head_dim]
                    v_layer = new_past[layer][1][i].unsqueeze(0)
                    updated_kv.append((k_layer, v_layer))
                seq.kv_cache = updated_kv

                if seq.is_finished():
                    finished_indices.append(i)

            # Process finished sequences.
            finished_seqs = [self.active_seqs[i] for i in finished_indices]
            for seq in finished_seqs:
                generated_text = self.tokenizer.decode(seq.generated_tokens, skip_special_tokens=True)
                print("Prompt:", seq.prompt_text)
                print("Generated:", generated_text)
                print("------")

            # Remove finished sequences (delete in reverse order for safety).
            for idx in sorted(finished_indices, reverse=True):
                del self.active_seqs[idx]

            step_counter += 1