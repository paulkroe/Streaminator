# stream_manager.py
import torch
import json
from collections import deque
from torch.nn.utils.rnn import pad_sequence
from .prompt_sampler import PromptSampler
from .kv_cache_manager import KVCacheManager
from .kv_cache_wrapper import KVCacheWrapper
from .sequence import Sequence
import pynvml
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
        self.debug = debug
        self.logger = logger

        self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Profiling GPU 0

        self.prompt_deque = deque()
        self.active_seqs = []
        # This dictionary will store results keyed by prompt text.
        # (In Python 3.7+ dicts are insertion ordered.)
        self.results = {}
        # Keep a separate list to record the order in which prompts were enqueued.
        self.prompt_order = []
        self.device = next(model.parameters()).device
        
        # Cache prompt KV caches for re-use if available.
        if self.use_kv_cache:
            self.prompt_kv_cache = {}

    def _log_gpu_stats(self, step):
        if not self.logger:
            return
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
        except pynvml.NVMLError as e:
            self._debug_print(f"Failed to log GPU stats: {e}")

        stream_utilization = sum([0 if seq.is_finished() else 1 for seq in self.active_seqs])/self.stream_width * 100

        self.logger.log({
                "gpu SM utilization (%)": util.gpu,
                "gpu memory (MB)": mem_info.used / 1024**2,
                "gpu memory usage (%)": mem_info.used / mem_info.total * 100,
                "generation step": step,
                "steam utilization (%)": stream_utilization,
        })



    def _debug_print(self, msg):
        if self.debug:
            print(f"[DEBUG] {msg}")

    def enqueue_prompt(self, prompt_text, num_completions=1):
        self._debug_print(f"Enqueue prompt: {prompt_text[:60]!r}, num_completions={num_completions}")
        # Record the order as a separate list.
        self.prompt_order.append(prompt_text)
        self.prompt_deque.append((prompt_text, num_completions))

    def _prefill_prompt(self, prompt_text, num_completions):
        # (Same as before)
        self._debug_print(f"Prefilling prompt: {prompt_text[:60]!r}")
        inputs = self.tokenizer(prompt_text, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        prompt_tokens = inputs["input_ids"][0]

        with torch.no_grad():
            outputs = self.model(**inputs, use_cache=self.use_kv_cache)

        seq = Sequence(
            prompt_text=prompt_text,
            max_length=self.max_length,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        seq.set_prompt_tokens(prompt_tokens.clone())

        if self.use_kv_cache:
            if prompt_text in self.prompt_kv_cache:
                seq.kv_cache = self.prompt_kv_cache[prompt_text]
            else:
                seq.kv_cache = KVCacheManager.clone(outputs.past_key_values)
                self.prompt_kv_cache[prompt_text] = seq.kv_cache

        prefix_logits = outputs.logits[0, -1, :]
        token_id = PromptSampler.sample_token(prefix_logits)

        single_tok = torch.tensor([[token_id]], device=self.device)
        if self.use_kv_cache:
            with torch.no_grad():
                out2 = self.model(
                    input_ids=single_tok,
                    past_key_values=KVCacheWrapper.wrap(seq.kv_cache, self.model),
                    use_cache=True
                )
            seq.kv_cache = KVCacheManager.clone(out2.past_key_values)

        seq.append_token(token_id)
        return [seq]

    def _refill_active_seqs(self):
        # Define effective valid length: add +1 if sequence is finished.
        effective_length = lambda seq: seq.get_valid_length() + (1 if seq.is_finished() else 0)

        # Step 1: Determine how many new sequences are needed.
        num_active = len(self.active_seqs)
        num_new_needed = self.stream_width - num_active

        # Step 2: Trim each active sequence's KV cache to its effective valid tokens.
        for seq in self.active_seqs:
            eff_len = effective_length(seq)
            trimmed_kv = []
            for (k, v) in seq.kv_cache:
                trimmed_k = k[..., -eff_len:, :]
                trimmed_v = v[..., -eff_len:, :]
                trimmed_kv.append((trimmed_k, trimmed_v))
            seq.kv_cache = trimmed_kv

        # Step 3: Add new sequences from the prompt deque.
        for _ in range(num_new_needed):
            if not self.prompt_deque:
                break
            prompt_text, count = self.prompt_deque.popleft()
            new_seqs = self._prefill_prompt(prompt_text, 1)
            for new_seq in new_seqs:
                self.active_seqs.append(new_seq)
            if count > 1:
                self.prompt_deque.append((prompt_text, count - 1))
        
        # Step 4: Compute the maximum effective valid length among active sequences.
        if self.active_seqs:
            max_eff_len = max(effective_length(seq) for seq in self.active_seqs)
        else:
            max_eff_len = 0

        # Step 5: Left-pad KV caches so that every sequence has the same time dimension.
        for seq in self.active_seqs:
            cur_eff_len = effective_length(seq)
            pad_size = max_eff_len - cur_eff_len
            if pad_size > 0:
                padded_kv = []
                for (k, v) in seq.kv_cache:
                    padded_k = torch.nn.functional.pad(k, (0, 0, pad_size, 0), mode="constant", value=0)
                    padded_v = torch.nn.functional.pad(v, (0, 0, pad_size, 0), mode="constant", value=0)
                    padded_kv.append((padded_k, padded_v))
                seq.kv_cache = padded_kv

    def _get_batched_kv(self):
        if not self.active_seqs:
            return None

        num_layers = len(self.active_seqs[0].kv_cache)
        for seq in self.active_seqs:
            if len(seq.kv_cache) != num_layers:
                raise ValueError("Mismatched number of layers in KV caches among sequences.")

        batched = []
        for layer_idx in range(num_layers):
            layer_keys = []
            layer_values = []
            for seq in self.active_seqs:
                (k, v) = seq.kv_cache[layer_idx]
                layer_keys.append(k)
                layer_values.append(v)
            batched_keys = torch.cat(layer_keys, dim=0)
            batched_values = torch.cat(layer_values, dim=0)
            batched.append((batched_keys, batched_values))
        return tuple(batched)

    def _build_leftpad_attention_mask(self):
        """
        Build an attention mask based on the left-padded KV cache time dimension.
        We assume that all active sequences have been padded so that their KV caches
        have the same time dimension T. We then create a mask with shape [B, T+1],
        where the last column corresponds to the new token to be generated.
        
        For each sequence, we mark as valid (1) the positions corresponding to real tokens.
        Since the KV cache was left-padded, each sequence's valid tokens appear in the
        rightmost positions. We compute:
        
            valid = seq.get_valid_length()  # (from prompt plus generated tokens)
            left_padding = T - valid
            Then, for that sequence, valid positions are indices [left_padding, T),
            and we additionally mark the extra column (index T) as valid.
        """
        if not self.active_seqs:
            return None

        # All active sequences have been left-padded to the same time dimension T.
        T = self.active_seqs[0].kv_cache[0][0].shape[2]  # time dimension of keys in KV cache
        B = len(self.active_seqs)
        mask = torch.zeros(B, T + 1, dtype=torch.long, device=self.device)
        for i, seq in enumerate(self.active_seqs):
            valid = seq.get_valid_length()  # count of valid tokens (prompt+generated)
            # In left-padded KV caches, the last 'valid' columns are valid.
            # Compute the starting index of valid tokens.
            start = T - valid
            # Set the columns for valid tokens and also the extra new token column to 1.
            mask[i, start:] = 1
        # For debugging, you can print the mask shape:
        # print("Attention mask shape:", mask.shape)
        return mask


    def _generate_step_with_kv(self):
        if not self.active_seqs:
            return None, None

        effective_length = lambda seq: seq.get_valid_length() + (1 if seq.is_finished() else 0)
        input_tokens = []
        positions = []
        num_idle_seqs = 0
        for seq in self.active_seqs:
            last_tok = seq.next_input_token()
            # Every sequence should have been prefilled; last_tok should not be None.
            if last_tok is None:
                raise ValueError("Sequence has no generated token; it should have been prefilled with a first token.")
            # For sequences that have finished generation, supply the EOS token as input.
            if seq.is_finished():
                last_tok = self.tokenizer.eos_token_id
                num_idle_seqs += 1
            input_tokens.append(last_tok)
            # Use the effective valid length (i.e. add one for finished sequences).
            positions.append(effective_length(seq))
        
        if num_idle_seqs == len(self.active_seqs):
            return None, None

        input_ids = torch.tensor(input_tokens, dtype=torch.long, device=self.device).unsqueeze(1)
        position_ids = torch.tensor(positions, dtype=torch.long, device=self.device).unsqueeze(1)

        batched_kv = self._get_batched_kv()
        if batched_kv is None:
            return None, None

        past_for_model = KVCacheWrapper.wrap(batched_kv, self.model)
        attention_mask = self._build_leftpad_attention_mask()
        if attention_mask is None:
            return None, None
        token_mask = torch.ones(attention_mask.size(0), 1, dtype=attention_mask.dtype, device=attention_mask.device)
        attention_mask = torch.cat([attention_mask, token_mask], dim=1)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                position_ids=position_ids,
                past_key_values=past_for_model,
                use_cache=True,
                attention_mask=attention_mask,
            )
        logits_full = outputs.logits  # Shape: [B, 1, vocab_size]
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
        # Initially, refill active sequences.
        self._refill_active_seqs()
        step_counter = 0

        # Continue until no active sequences or prompts remain.
        while self.active_seqs or self.prompt_deque:
            if step_counter % self.refill_period == 0:
                still_active = []
                for seq in self.active_seqs:
                    # If the sequence has generated an EOS (or is finished), process its valid output.
                    if self.tokenizer.eos_token_id in seq.generated_tokens or seq.is_finished():
                        trimmed_tokens = seq.get_final_generation()
                        gen_text = self.tokenizer.decode(trimmed_tokens, skip_special_tokens=True)
                        if seq.prompt_text not in self.results:
                            self.results[seq.prompt_text] = []
                        self.results[seq.prompt_text].append(gen_text)
                    else:
                        still_active.append(seq)
                self.active_seqs = still_active
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
                token_logits = logits[i]
                next_token = PromptSampler.sample_token(token_logits)
                seq.append_token(next_token)
                if self.use_kv_cache and new_past_list is not None:
                    seq.kv_cache = new_past_list[i]

            step_counter += 1

            self._log_gpu_stats(step_counter)

        # Finalize any remaining active sequences.
        for seq in self.active_seqs:
            trimmed_tokens = seq.get_final_generation()
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

                self._log_gpu_stats(step_counter)


    def save_results(self, filename):
        # Create an ordered dictionary based on the prompt order.
        ordered_results = {}
        # Use the recorded prompt order to output results.
        for prompt in self.prompt_order:
            # In case there was no result generated for a prompt, default to an empty list.
            ordered_results[prompt] = self.results.get(prompt, [])
        with open(filename, "w") as f:
            json.dump(ordered_results, f, indent=2)
        print(f"Results saved to {filename}")