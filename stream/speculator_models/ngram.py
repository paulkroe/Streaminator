import torch
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence

class NGram:
    """
    N-gram language model: stores counts and provides probability distributions.
    Filters out any token IDs >= vocab_size to avoid OOB errors.
    """
    def __init__(self, tokenizer, order=3, profiler=None):
        self.tokenizer = tokenizer
        self.order = order
        self.vocab_size = len(tokenizer) # .vocab_size
        # context -> next-token counts
        self.counts = defaultdict(lambda: defaultdict(int))
        # context -> total count
        self.context_totals = defaultdict(int)
        # unigram fallback counts
        self.unigram = defaultdict(int)
        self.total_unigram = 0
        self.profiler = profiler

    def train(self, texts, tokenized=False):
        """
        Train model on an iterable of text strings.
        Only counts token IDs < vocab_size.
        """
        self.profiler.start("ngram_train")
        for text in texts:
            if not tokenized:
                tokens = self.tokenizer(text, return_tensors='pt').input_ids[0].tolist()
            else:
                tokens = text
            # unigram counts
            for t in tokens:
                if 0 <= t < self.vocab_size:
                    self.unigram[t] += 1
            self.total_unigram += sum(1 for t in tokens if 0 <= t < self.vocab_size)
            # n-gram counts
            for i in range(len(tokens) - self.order + 1):
                context = []
                valid = True
                for x in tokens[i:i + self.order - 1]:
                    if 0 <= x < self.vocab_size:
                        context.append(x)
                    else:
                        valid = False
                        break
                nxt = tokens[i + self.order - 1]
                if not valid or not (0 <= nxt < self.vocab_size):
                    continue
                context = tuple(context)
                self.counts[context][nxt] += 1
                self.context_totals[context] += 1
            self.profiler.stop("ngram_train")

    def __call__(self, context_tensor):
        """
        Given a context tensor of token IDs (1D), return a full probability distribution
        over the vocabulary as a 1D tensor of length vocab_size.
        Only uses counts for token IDs < vocab_size.
        """
        self.profiler.start("ngram_call")
        dist = torch.zeros(self.vocab_size, device=context_tensor.device)
        ctx = context_tensor.tolist()
        if len(ctx) >= self.order - 1:
            suffix = []
            valid = True
            for x in ctx[-(self.order - 1):]:
                if 0 <= x < self.vocab_size:
                    suffix.append(x)
                else:
                    valid = False
                    break
            if valid:
                suffix = tuple(suffix)
            else:
                suffix = None

            if suffix and suffix in self.counts and self.context_totals[suffix] > 0:
                total = self.context_totals[suffix]
                for tok, cnt in self.counts[suffix].items():
                    if 0 <= tok < self.vocab_size:
                        dist[tok] = cnt / total
            else:
                # fallback to unigram
                if self.total_unigram > 0:
                    for tok, cnt in self.unigram.items():
                        dist[tok] = cnt / self.total_unigram
        else:
            # fallback when context too short
            if self.total_unigram > 0:
                for tok, cnt in self.unigram.items():
                    dist[tok] = cnt / self.total_unigram
        # ensure normalization; avoid zero-sum
        total_prob = dist.sum()
        if total_prob > 0:
            dist = dist / total_prob
        else:
            # uniform over vocab if no data
            dist.fill_(1.0 / self.vocab_size)
        self.profiler.stop("ngram_call")
        return dist