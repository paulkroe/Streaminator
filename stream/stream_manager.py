# stream_manager.py
import torch
import torch.nn.functional as F
import json
import gc
from collections import deque, defaultdict
from torch.nn.utils.rnn import pad_sequence
from .utils.prompt_sampler import PromptSampler
from .kv_cache.kv_cache_manager import KVCacheManager
from .kv_cache.kv_cache_wrapper import KVCacheWrapper
from .utils.sequence import Sequence
from .utils.profiler import Profiler
import pynvml
import gc
import types

pynvml.nvmlInit()

class StreamManager:
    def __init__(
        self,
        model,
        tokenizer,
        stream_width=8,
        max_length=50,
        use_kv_cache=True,
        continuous_batching=True,
        no_prompt_training=False,
        no_generation_training=False,
        spec_decoding=True,
        ngram_order=3,
        gamma=1,
        logger=None,
        debug=False
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.stream_width = stream_width
        self.max_length = max_length
        self.use_kv_cache = use_kv_cache
        self.continuous_batching = continuous_batching
        self.spec_decoding = spec_decoding
        self.no_prompt_training = no_prompt_training
        self.no_generation_training = no_generation_training
        self.debug = debug
        self.logger = logger

        self.ngram_registry = {}
        self.next_qid = 0
        self.ngram_order = ngram_order
        self.gamma = gamma
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
            self.acceptance_dict = defaultdict(int)
            self.completion_level_acceptance = defaultdict(int)
            self.completion_level_count = defaultdict(int)

        self.len_queue = 0

        self.gamma = 1
        self.profiler = Profiler()
        self.vocab_size = len(self.tokenizer)
        self.uniform_prob_threshold = 2.0 / self.vocab_size
   
    from .generation.generation_loop import run_generation_loop
    from .generation.static_batching import _run_generation_static
    from .generation.continuous_batching import _run_generation_continuous 
   
    from .generation.generate_step_with_kv import _generate_step_with_kv
    from .generation.generate_step_without_kv import _generate_step_without_kv

    from .speculator_models.accept_speculation import _accept_speculative

    from .utils.logging import _log_gpu_stats
    from .utils.logging import save_results
   
    from .utils.fill_stream import enqueue_prompt
    from .utils.fill_stream import  _prefill_prompt
    from .utils.fill_stream import  _refill_active_seqs
    from .utils.fill_stream import  _cleanup_and_refill
    
    from .utils.padding import _get_batched_kv
    from .utils.padding import _build_input_ids
    from .utils.padding import _build_leftpad_attention_mask