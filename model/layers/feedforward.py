"""
Feed-Forward Network (FFN / MLP) layer for transformer block.

"""

from typing import Optional, Any
import torch
import torch.nn as nn


def _resolve_config(explicit_config: Optional[Any] = None) -> Any:
    if explicit_config is not None:
        return explicit_config
    from config.model_config import ModelConfig
    return ModelConfig()


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network:
    FFN(x) = GELU(x W_1 + b_1) W_2 + b_2
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        d_model: Optional[int] = None,
        d_ffn: Optional[int] = None,
        dropout: Optional[float] = None,
        bias: bool = True
    ):
        super().__init__()
        cfg = _resolve_config(config)

        self.d_model = d_model or getattr(cfg, "d_model", 768)
        
        if d_ffn is not None:
            self.d_ffn = d_ffn
        else:
            self.d_ffn = self.d_model * getattr(cfg, "ffn_multiplier", 4)

        dropout_p = dropout if dropout is not None else getattr(cfg, "dropout", 0.1)

        self.linear1 = nn.Linear(self.d_model, self.d_ffn, bias=bias)
        self.activation = nn.GELU(approximate="tanh")
        self.linear2 = nn.Linear(self.d_ffn, self.d_model, bias=bias)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for feed-forward network.

        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Tensor of shape (batch_size, seq_len, d_model)
        """
        hidden = self.activation(self.linear1(x))
        out = self.linear2(self.dropout(hidden))
        return out
