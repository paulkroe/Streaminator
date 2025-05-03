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
        import gc
        self.model = model
        self.tokenizer = tokenizer
        self.stream_width = stream_width
        self.max_length = max_length
        self.refill_period = refill_period
        self.use_kv_cache = use_kv_cache
        self.continuous_batching = continuous_batching
        self.debug = debug
        self.logger = logger

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

    def _log_memory(self, context):
        if self.debug:
            alloc = torch.cuda.memory_allocated(self.device) / 1e6
            reserved = torch.cuda.memory_reserved(self.device) / 1e6
            print(f"[MEM] {context}: allocated={alloc:.1f}MB reserved={reserved:.1f}MB")

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
        self._debug_print(f"Enqueue prompt: {prompt_text[:60]!r}, num_completions={num_completions}")
        self.prompt_order.append(prompt_text)
        self.prompt_deque.append((prompt_text, num_completions))

    def _prefill_prompt(self, prompt_text, num_completions):
        self._debug_print(f"Prefilling prompt: {prompt_text[:60]!r}")
        self._log_memory("before prefill model call")
        inputs = self.tokenizer(prompt_text, return_tensors='pt').to(self.device)
        prompt_tokens = inputs.input_ids[0]
        with torch.no_grad():
            outputs = self.model(**inputs, use_cache=self.use_kv_cache)
        self._log_memory("after prefill model call")

        seq = Sequence(prompt_text, self.max_length, self.tokenizer.eos_token_id)
        seq.set_prompt_tokens(prompt_tokens.clone())
        if self.use_kv_cache:
            cache = KVCacheManager.clone(outputs.past_key_values)
            self.prompt_kv_cache.setdefault(prompt_text, cache)
            seq.kv_cache = cache

        first_token = PromptSampler.sample_token(outputs.logits[0, -1])
        single_tok = torch.tensor([[first_token]], device=self.device)
        with torch.no_grad():
            out2 = self.model(input_ids=single_tok,
                              past_key_values=KVCacheWrapper.wrap(seq.kv_cache, self.model),
                              use_cache=True)
        seq.kv_cache = KVCacheManager.clone(out2.past_key_values)
        seq.append_token(first_token)
        return [seq]

    def _refill_active_seqs(self):
        self._log_memory("refill start")
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
            p, c = self.prompt_deque.popleft()
            new = self._prefill_prompt(p, 1)
            self.active_seqs += new
            if c > 1:
                self.prompt_deque.append((p, c-1))
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
        self._log_memory("refill end")

    def _get_batched_kv(self):
        self._log_memory("before batching KV")
        if not self.active_seqs:
            return None
        layers = len(self.active_seqs[0].kv_cache)
        batched=[]
        for i in range(layers):
            ks = [s.kv_cache[i][0] for s in self.active_seqs]
            vs = [s.kv_cache[i][1] for s in self.active_seqs]
            batched.append((torch.cat(ks,0), torch.cat(vs,0)))
        self._log_memory("after batching KV")
        return tuple(batched)

    def _build_leftpad_attention_mask(self):
        self._log_memory("build mask start")
        if not self.active_seqs:
            return None
        T = self.active_seqs[0].kv_cache[0][0].shape[2]
        B = len(self.active_seqs)
        mask = torch.zeros(B, T+1, device=self.device)
        for i,seq in enumerate(self.active_seqs):
            v = seq.get_valid_length()
            mask[i, T-v:]=1
        self._log_memory("build mask end")
        return mask.long()

    def _generate_step_with_kv(self):
        self._log_memory("gen step start")
        if not self.active_seqs:
            return None, None
        toks, pos, idle = [], [], 0
        eff_len = lambda s: s.get_valid_length() + (1 if s.is_finished() else 0)
        for s in self.active_seqs:
            t = s.next_input_token() or self.tokenizer.eos_token_id
            if s.is_finished(): idle+=1
            toks.append(t); pos.append(eff_len(s))
        if idle==len(self.active_seqs): return None, None
        input_ids = torch.tensor(toks, device=self.device).unsqueeze(1)
        position_ids = torch.tensor(pos, device=self.device).unsqueeze(1)
        batched = self._get_batched_kv()
        mask = self._build_leftpad_attention_mask()
        token_mask = torch.ones(len(self.active_seqs),1,device=self.device)
        mask = torch.cat([mask, token_mask],1)
        with torch.no_grad():
            out = self.model(input_ids=input_ids,
                             position_ids=position_ids,
                             past_key_values=KVCacheWrapper.wrap(batched,self.model),
                             use_cache=True,
                             attention_mask=mask)
        logits = out.logits[:,-1]
        new_past = out.past_key_values
        # split per-seq
        per=[]
        for i in range(len(self.active_seqs)):
            seq_kv=[]
            for layer in new_past:
                seq_kv.append((layer[0][i:i+1], layer[1][i:i+1]))
            per.append(seq_kv)
        self._log_memory("gen step end")
        return logits, per

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


    def _run_generation_continuous(self):
        self._refill_active_seqs()
        step=0
        while self.active_seqs or self.prompt_deque:
            if step % self.refill_period==0:
                still=[]
                for seq in self.active_seqs:
                    if seq.is_finished() or self.tokenizer.eos_token_id in seq.generated_tokens:
                        txt = self.tokenizer.decode(seq.get_final_generation(), skip_special_tokens=True)
                        self.results.setdefault(seq.prompt_text, []).append(txt)
                        # free KV cache
                        seq.kv_cache=None
                        del seq
                    else:
                        still.append(seq)
                # force collect
                gc.collect(); torch.cuda.empty_cache()
                self.active_seqs=still
                self._refill_active_seqs()
            # generation step
            logits, new_past = self._generate_step_with_kv()
            if logits is None:
                step+=1; continue
            for i,seq in enumerate(self.active_seqs):
                tok = PromptSampler.sample_token(logits[i])
                seq.append_token(tok)
                seq.kv_cache = new_past[i]
            step+=1
            self._log_gpu_stats(step)
        # finalize remaining
        for seq in self.active_seqs:
            txt=self.tokenizer.decode(seq.get_final_generation(), skip_special_tokens=True)
            self.results.setdefault(seq.prompt_text,[]).append(txt)

    def run_generation_loop(self):
        if self.continuous_batching: self._run_generation_continuous()
        else: self._run_generation_static()
        self.save_results("generation_results.json")

    def save_results(self, filename):
        ordered={}
        for p in self.prompt_order:
            ordered[p]=self.results.get(p,[])
        with open(filename,'w') as f:
            json.dump(ordered,f,indent=2)
        print(f"Results saved to {filename}")
