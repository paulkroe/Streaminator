# sequence.py

class Sequence:
    def __init__(self, prompt_text, max_length, eos_token_id):
        self.prompt_text = prompt_text
        self.max_length = max_length
        self.eos_token_id = eos_token_id

        self.prompt_tokens = None  # set later
        self.kv_cache = []
        self.generated_tokens = []
        
        # If finished_pos is not None, it means we saw EOS at that index in generated_tokens
        self.finished_pos = None
        
        # length_mask mirrors (prompt_tokens + generated_tokens). 
        # For each token, 1 if valid/unfinished, 0 if padded or after eos.
        self.length_mask = []

    def is_finished(self):
        return self.finished_pos is not None or len(self.generated_tokens) >= self.max_length

    def next_input_token(self):
        """
        For the next forward pass, we feed exactly the last token we generated
        """
        if len(self.generated_tokens) == 0:
            # e.g. if we just created the seq but haven't appended anything
            return None
        # Return the *last* token, ignoring whether it's finished or not.
        return self.generated_tokens[-1]
    
    def current_total_length(self):
        """
        Returns how many tokens are in (prompt_tokens + generated_tokens).
        """
        return len(self.prompt_tokens) + len(self.generated_tokens)
    
    def get_valid_length(self):
        """
        The count of tokens up to (and including) the last real token.
        That is just sum of self.length_mask.
        """
        return sum(self.length_mask)

    def get_final_generation(self):
        """
        Decode the generation up to EOS (if present) or up to the final token.
        """
        if self.finished_pos is not None:
            # decode only up to the eos token (exclusive or inclusive as you prefer)
            valid_gens = self.generated_tokens[: self.finished_pos + 1]
        else:
            valid_gens = self.generated_tokens
        return valid_gens

    def set_prompt_tokens(self, tokens):
        """
        We call this right after we have the prompt tokens from tokenizer.
        We fill 'length_mask' for them so they count as valid steps.
        """
        self.prompt_tokens = tokens
        # Mark the prompt tokens as valid in the length_mask
        self.length_mask = [1] * len(tokens)  

    def append_token(self, token_id):
        if self.is_finished():
            # If the sequence is already finished, keep appending 0 so shapes stay consistent
            self.generated_tokens.append(token_id)
            self.length_mask.append(0)
            return

        self.generated_tokens.append(token_id)
        self.length_mask.append(1)

        if token_id == self.eos_token_id:
            self.finished_pos = len(self.generated_tokens) - 1
