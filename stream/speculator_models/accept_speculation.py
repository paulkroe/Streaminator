import torch

def _accept_speculative(self, q_probs: torch.Tensor, p_probs: torch.Tensor) -> torch.BoolTensor:
    """
    Given draft-model probabilities q_i and full-model probabilities p_i for
    i=1..γ, return a boolean mask of length γ indicating which proposed tokens
    to accept contiguously.

    q_probs.shape = (γ,)
    p_probs.shape = (γ,)  or  (γ+1,)
    """
    self.profiler.start("_accept_speculative")
    # both must be 1D
    assert q_probs.ndim == 1 and p_probs.ndim == 1

    gamma = q_probs.size(0)

    # if p_probs has one extra entry (the correction step), drop it
    if p_probs.size(0) == gamma + 1:
        p_slice = p_probs[:gamma]
    else:
        # must match exactly γ
        assert p_probs.size(0) == gamma, f"p_probs must be length γ or γ+1, got {p_probs.size(0)}"
        p_slice = p_probs

    # draw uniforms
    r = torch.rand_like(q_probs)

    # Only consider q_probs above certainty threshold (ow uniform distribution gets high acceptance)
    confident = q_probs > self.uniform_prob_threshold

    # accept_i ~ Bernoulli(p_i / q_i) only if confident
    accept = (r < (p_slice / q_probs)) & confident

    # enforce contiguous acceptance
    first_false = (~accept).nonzero(as_tuple=False)
    if first_false.numel() > 0:
        idx = first_false[0].item()
        accept[idx:] = False

    self.profiler.stop("_accept_speculative")
    return accept
