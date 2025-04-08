# sequence.py

import torch

class Sequence:
    def __init__(self, prompt_text, max_length=50, eos_token_id=None):
        self.prompt_text = prompt_text
        self.generated_tokens = []  # token ids generated after the prompt
        self.kv_cache = None        # list of (key, value) pairs per layer
        self.finished = False
        self.max_length = max_length
        self.eos_token_id = eos_token_id

    def next_input_token(self):
        # Returns the last generated token (used as input for next step)
        if self.generated_tokens:
            return self.generated_tokens[-1]
        else:
            # Ideally, a Sequence is only added after a token has been generated.
            raise ValueError("No token available in sequence.")

    def append_token(self, token_id):
        self.generated_tokens.append(token_id)
        # Check if token is EOS or if maximum length is reached
        if token_id == self.eos_token_id or len(self.generated_tokens) >= self.max_length:
            self.finished = True

    def is_finished(self):
        return self.finished