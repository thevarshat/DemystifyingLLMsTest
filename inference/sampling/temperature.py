"""Module implementation."""
import torch
import torch.nn.functional as F


def apply_temperature(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Apply temperature scaling to logits.
    """
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if temperature == 0:
        scaled = torch.full_like(logits, float("-inf"))
        return scaled.scatter(-1, logits.argmax(dim=-1, keepdim=True), 0.0)
    return logits / temperature
