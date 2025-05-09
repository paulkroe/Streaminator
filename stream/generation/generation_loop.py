from tqdm import tqdm

def run_generation_loop(self):
    """
    Run the generation loop.
    """
    self.pbar = tqdm(total=self.len_queue, desc="Generating...")
    if self.continuous_batching: self._run_generation_continuous()
    else: self._run_generation_static()
    self.save_results("generation_results.json")
    self.pbar.close()
    self.len_queue = 0