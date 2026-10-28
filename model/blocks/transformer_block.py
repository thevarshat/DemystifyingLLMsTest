"""
Single Transformer Block module.

"""

from typing import Optional, Any
import torch
import torch.nn as nn

from model.attention.multi_head import MultiHeadAttention
from model.layers.feedforward import FeedForward
from model.layers.layer_norm import LayerNorm
from model.layers.residual import ResidualConnection


def _resolve_config(explicit_config: Optional[Any] = None) -> Any:
    if explicit_config is not None:
        return explicit_config
    from config.model_config import ModelConfig
    return ModelConfig()


class TransformerBlock(nn.Module):
    """
    Standard GPT-2 style pre-layer-norm transformer decoder block.
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        layer_idx: int = 0
    ):
        super().__init__()
        cfg = _resolve_config(config)
        self.layer_idx = layer_idx
        self.d_model = getattr(cfg, "d_model", 768)

        self.ln1 = LayerNorm(config=cfg)
        self.attn = MultiHeadAttention(config=cfg)
        self.ln2 = LayerNorm(config=cfg)
        self.ffn = FeedForward(config=cfg)

        dropout_p = getattr(cfg, "dropout", 0.1)
        self.resid1 = ResidualConnection(dropout=dropout_p)
        self.resid2 = ResidualConnection(dropout=dropout_p)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Any] = None
    ) -> torch.Tensor:
        """
        Forward pass for transformer block.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask
            kv_cache: Optional key-value cache

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        h = self.resid1(x, self.attn(self.ln1(x), mask=mask, kv_cache=kv_cache, layer_idx=self.layer_idx))
        out = self.resid2(h, self.ffn(self.ln2(h)))
        return out
