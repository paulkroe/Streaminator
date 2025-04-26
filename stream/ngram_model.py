from collections import defaultdict
import random
import torch

class NGramModel:
    def __init__(self, n):
        """Initialize NGramModel with n-gram size n.
        
        Args:
            n: The n in n-gram (context size + 1)
        """
        self.n = n
        # Nested dictionaries to store counts: {context_tuple: {token_id: count}}
        self.counts = defaultdict(lambda: defaultdict(int))
        # Total counts for each context
        self.totals = defaultdict(int)
    
    def _get_context(self, tokens):
        """Extract the last n-1 tokens as a context tuple."""
        # If we have fewer than n-1 tokens, use what we have
        if len(tokens) < self.n - 1:
            return tuple(tokens)
        return tuple(tokens[-(self.n - 1):])
    
    def update(self, tokens):
        """Update the n-gram model with a sequence of tokens."""
        # Need at least n tokens to form an n-gram
        if len(tokens) < self.n:
            return
        
        # Iterate through the sequence
        for i in range(len(tokens) - self.n + 1):
            # Extract context and next token
            context = tuple(tokens[i:i+self.n-1])
            next_token = tokens[i+self.n-1]
            
            # Update counts
            self.counts[context][next_token] += 1
            self.totals[context] += 1
    
    def get_probabilities(self, context_tokens):
        """Calculate probability distribution for the next token."""
        context = self._get_context(context_tokens)
        
        # If context is not in our counts, return empty dict
        if context not in self.counts:
            return {}
        
        # Calculate probabilities with smoothing
        probs = {}
        total = self.totals[context]
        vocab_size = len(self.counts[context]) + 1  # +1 for smoothing
        
        # Simple add-1 smoothing
        for token_id, count in self.counts[context].items():
            probs[token_id] = (count + 1) / (total + vocab_size)
            
        return probs
    
    def sample(self, context_tokens):
        """Sample the next token based on the n-gram model."""
        probs = self.get_probabilities(context_tokens)
        
        # If no probabilities (unseen context), return None
        if not probs:
            return None
            
        # Convert to list of (token_id, prob) tuples
        items = list(probs.items())
        tokens, weights = zip(*items)
        
        # Sample from the distribution
        return random.choices(tokens, weights=weights, k=1)[0] 