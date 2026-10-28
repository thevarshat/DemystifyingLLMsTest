"""
Language model output head.
"""

from typing import Optional, Any
import torch
import torch.nn as nn


def _resolve_config(explicit_config: Optional[Any] = None) -> Any:
    if explicit_config is not None:
        return explicit_config
    from config.model_config import ModelConfig
    return ModelConfig()


class LMHead(nn.Module):
    """
    Language Model output head that projects hidden representations into vocabulary logit space.
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        d_model: Optional[int] = None,
        vocab_size: Optional[int] = None,
        bias: bool = False
    ):
        super().__init__()
        cfg = _resolve_config(config)
        self.d_model = d_model or getattr(cfg, "d_model", 768)
        self.vocab_size = vocab_size or getattr(cfg, "vocab_size", 50257)

        self.proj = nn.Linear(self.d_model, self.vocab_size, bias=bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute vocabulary logits from transformer backbone hidden states.

        Args:
            hidden_states: Tensor of shape (batch_size, seq_len, d_model).

        Returns:
            Logits of shape (batch_size, seq_len, vocab_size).
        """
        return self.proj(hidden_states)

    def tie_weights(self, embedding_weight: torch.Tensor) -> None:
        """
        Tie the weights of the LM head to the input token embedding matrix (standard GPT-2).
        """
        self.proj.weight = embedding_weight
