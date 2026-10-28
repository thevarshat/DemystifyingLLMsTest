"""Combined sampler (Clean)."""
import torch
import torch.nn.functional as F
from typing import Optional, List, Union
from .temperature import apply_temperature
from .top_k import apply_top_k
from .top_p import apply_top_p
from .repetition_penalty import apply_repetition_penalty


class Sampler:
    def __init__(
        self,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
    ):
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty

    def sample(
        self,
        logits: torch.Tensor,
        generated_tokens: Optional[Union[List[int], torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Sample next token from logits tensor of shape (batch_size, vocab_size) or (vocab_size,).
        """
        logits = logits.clone()
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)

        # 1. Repetition penalty
        if self.repetition_penalty != 1.0 and generated_tokens is not None:
            logits = apply_repetition_penalty(logits, generated_tokens, penalty=self.repetition_penalty)

        if self.temperature == 0:
            return logits.argmax(dim=-1)
        logits = apply_temperature(logits, self.temperature)
        logits = apply_top_k(logits, self.top_k)
        logits = apply_top_p(logits, self.top_p)
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
