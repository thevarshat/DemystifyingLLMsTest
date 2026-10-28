"""
Scaled Dot-Product Attention implementation.

"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout: Optional[nn.Dropout] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute scaled dot-product attention:
        Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k) + M) V

    Args:
        q: Query tensor of shape (batch, n_heads, seq_len_q, d_k)
        k: Key tensor of shape (batch, n_heads, seq_len_k, d_k)
        v: Value tensor of shape (batch, n_heads, seq_len_k, d_v)
        mask: Optional additive attention mask
        dropout: Optional dropout module

    Returns:
        Tuple of (output tensor, attention weights)
    """
    d_k = q.size(-1)

    # Compute raw query-key dot products: (batch, n_heads, seq_len_q, seq_len_k)
    scores = torch.matmul(q, k.transpose(-2, -1))

    scores = scores / math.sqrt(d_k)

    if mask is not None:
        scores = scores + mask

    attn_weights = F.softmax(scores, dim=-1)

    if dropout is not None:
        attn_weights = dropout(attn_weights)

    output = torch.matmul(attn_weights, v)
    return output, attn_weights


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention layer wrapper.
    """

    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else None

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return scaled_dot_product_attention(q, k, v, mask=mask, dropout=self.dropout)
