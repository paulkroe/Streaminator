# prompt_sampler.py

import torch

class PromptSampler:
    @staticmethod
    def sample_token(logits, temperature=1.0, top_k=1, top_p=0.97):
        # logits: torch.Tensor of shape [vocab_size]
        if temperature != 1.0:
            logits = logits / temperature
        probs = torch.softmax(logits, dim=-1)
        
        # Apply top-p (nucleus) sampling filtering.
        if top_p is not None:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            # Remove tokens where cumulative probability exceeds top_p
            sorted_indices_to_remove = cumulative_probs > top_p
            if sorted_indices_to_remove.any():
                # Ensure at least one token remains by shifting the mask right
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                probs[sorted_indices[sorted_indices_to_remove]] = 0
                probs = probs / probs.sum()
        elif top_k is not None:
            topk_probs, topk_indices = torch.topk(probs, top_k)
            mask = torch.ones_like(probs, dtype=torch.bool)
            mask[topk_indices] = False
            probs[mask] = 0
            probs = probs / probs.sum()
        next_token_id = torch.multinomial(probs, num_samples=1)
        return int(next_token_id.item())