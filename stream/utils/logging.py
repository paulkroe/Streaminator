import pynvml
import json

def _log_gpu_stats(self, step):
    if not self.logger:
        return
    try:
        util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
    except pynvml.NVMLError as e:
        self._debug_print(f"Failed to log GPU stats: {e}")
    
    stream_utilization = sum(not seq.is_finished() for seq in self.active_seqs) / self.stream_width * 100
    n_active_seqs = len(self.active_seqs)
    
    acceptance_rate = 0
    if self.spec_decoding:

        if n_active_seqs > 0:
            accepted_proposals = sum(1 for i in self.log_accepted_token if i is not None)
            acceptance_rate = (accepted_proposals / n_active_seqs) * 100
        
        for i, (level, token) in enumerate(zip(self.log_q_level, self.log_accepted_token)):
            self.completion_level_count[level] += 1
            if token is not None:
                self.completion_level_acceptance[level] += 1
                self.acceptance_dict[token] += 1


    self.logger.log({
        "gpu SM utilization (%)": util.gpu,
        "gpu memory (MB)": mem_info.used / 1024**2,
        "gpu memory usage (%)": mem_info.used / mem_info.total * 100,
        "generation step": step,
        "stream utilization (%)": stream_utilization,
        "acceptance rate (%)": acceptance_rate,
    })

def save_results(self, filename):
    ordered={}
    for p, _ in self.prompt_order:
        ordered[p]=self.results.get(p,[])
    with open(filename,'w') as f:
        json.dump(ordered,f,indent=2)
    print(f"Results saved to {filename}")