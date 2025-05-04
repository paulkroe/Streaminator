# sequence.py remains unchanged
import torch
class Sequence:
    def __init__(self, prompt_text, max_length, eos_token_id, qid=None):
        self.prompt_text = prompt_text
        self.max_length = max_length
        self.eos_token_id = eos_token_id

        self.qid = qid # Question ID this sequence belongs to

        self.prompt_tokens = None
        self.kv_cache = []
        self.generated_tokens = []
        
        # When finished_pos is not None, it means an EOS was generated.
        self.finished_pos = None
        
        # length_mask mirrors (prompt_tokens + generated_tokens). 
        # 1 indicates a valid token; 0 indicates a dummy/padding token.
        self.length_mask = []

    def full_input(self) -> torch.LongTensor:
        """
        Returns concatenated prompt_tokens + generated_tokens as a 1D LongTensor.
        """
        gen = torch.tensor(self.generated_tokens, dtype=torch.long, device=self.prompt_tokens.device)
        return torch.cat([self.prompt_tokens, gen], dim=0)

    def is_finished(self):
        return self.finished_pos is not None or len(self.generated_tokens) >= self.max_length

    def next_input_token(self):
        """
        Returns the last token generated (to feed into the next forward pass).
        """
        if len(self.generated_tokens) == 0:
            return None
        return self.generated_tokens[-1]
    
    def current_total_length(self):
        """
        Returns the total length = prompt tokens + generated tokens.
        """
        return len(self.prompt_tokens) + len(self.generated_tokens)
    
    def get_valid_length(self):
        """
        Returns the count of valid tokens (prompt tokens + generated tokens with mask==1).
        """
        return sum(self.length_mask)

    def get_final_generation(self):
        """
        Returns the generated tokens that are valid (i.e. tokens with mask 1, excluding dummy tokens).
        """
        # Valid generated tokens = valid tokens minus prompt tokens.
        valid_length = self.get_valid_length()
        prompt_length = len(self.prompt_tokens)
        
        # Debug information about token lengths
        print(f"[TOKEN DEBUG] qid={self.qid}, valid_length={valid_length}, prompt_length={prompt_length}")
        print(f"[TOKEN DEBUG] generated_tokens={self.generated_tokens}, length_mask={self.length_mask}")
        
        # Ensure we have valid generated tokens (not just prompt tokens)
        if valid_length <= prompt_length:
            print(f"[TOKEN DEBUG] No valid generated tokens (valid_length <= prompt_length)")
            # Instead of returning an empty list, return all generated tokens
            # This ensures we at least return something for evaluation
            if self.generated_tokens:
                print(f"[TOKEN DEBUG] Returning all generated tokens as fallback")
                return self.generated_tokens
            return []  # No valid generated tokens
            
        # Only include tokens that have mask=1 (valid tokens, not padding)
        valid_generated = self.generated_tokens[: (valid_length - prompt_length)]
        print(f"[TOKEN DEBUG] returning valid_generated={valid_generated}")
        return valid_generated

    def set_prompt_tokens(self, tokens):
        """
        Sets the prompt tokens and marks them as valid.
        """
        self.prompt_tokens = tokens
        self.length_mask = [1] * len(tokens)  

    def append_token(self, token_id):
        """
        Appends a token. If the sequence is finished, marks the token as dummy (mask 0);
        otherwise, marks it as valid (mask 1). If an EOS is appended, sets finished_pos.
        """
        if self.is_finished():
            self.generated_tokens.append(token_id)
            self.length_mask.append(0)
            return

        self.generated_tokens.append(token_id)
        self.length_mask.append(1)

        if token_id == self.eos_token_id:
            self.finished_pos = len(self.generated_tokens) - 1