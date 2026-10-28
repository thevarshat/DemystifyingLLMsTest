"""Nucleus top-p sampling."""
import torch
import torch.nn.functional as F


def apply_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    """
    Nucleus sampling.
    """
    if p >= 1.0 or p <= 0.0:
        return logits

    sorted_logits, indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
    remove_mask = cumulative_probs > p
    remove_mask[..., 1:] = remove_mask[..., :-1].clone()
    remove_mask[..., 0] = False
    remove_mask = torch.zeros_like(remove_mask).scatter(-1, indices, remove_mask)
    return logits.masked_fill(remove_mask, float("-inf"))
