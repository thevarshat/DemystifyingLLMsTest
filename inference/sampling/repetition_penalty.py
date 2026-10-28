"""Repetition penalty."""
import torch
from typing import List, Union


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_tokens: Union[List[int], torch.Tensor],
    penalty: float = 1.2
) -> torch.Tensor:
    """
    Apply repetition penalty to logits for generated tokens.
    DIVISIVELY (logits / penalty for pos, logits * penalty for neg).
    """
    if penalty <= 0:
        raise ValueError("penalty must be positive")
    if penalty == 1.0:
        return logits
    result = logits.clone()
    tokens = torch.as_tensor(generated_tokens, device=logits.device, dtype=torch.long)
    if tokens.ndim == 2 and logits.ndim == 2:
        if tokens.size(0) != logits.size(0):
            raise ValueError("Token history batch size must match logits")
        for row in range(logits.size(0)):
            ids = tokens[row].unique()
            values = result[row, ids]
            result[row, ids] = torch.where(values < 0, values * penalty, values / penalty)
    else:
        ids = tokens.unique()
        values = result[..., ids]
        result[..., ids] = torch.where(values < 0, values * penalty, values / penalty)
    return result
