"""
Positional embedding modules for transformer architectures.
"""

import math
from typing import Optional
import torch
import torch.nn as nn


class SinusoidalPositionEmbedding(nn.Module):
    """
    Sinusoidal Positional Encoding (Vaswani et al., 2017).
    Precomputes fixed sin/cos frequencies across sequence positions and hidden dimensions.
    """

    def __init__(self, d_model: int = 768, max_seq_len: int = 1024):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])

        # Shape: (1, max_seq_len, d_model) for automatic batch broadcasting
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(
        self,
        seq_len: int,
        offset: int = 0,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Extract positional encodings for the given sequence length and starting offset.

        Args:
            seq_len: Number of tokens in sequence.
            offset: Position offset (used when KV-cache is active).
            device: Target device placement.

        Returns:
            Tensor of shape (1, seq_len, d_model).
        """
        if offset < 0 or seq_len < 0 or offset + seq_len > self.max_seq_len:
            raise ValueError(
                f"Requested position index {offset + seq_len} exceeds max_seq_len {self.max_seq_len}"
            )

        encoding = self.pe[:, offset : offset + seq_len, :]
        if device is not None:
            encoding = encoding.to(device)
        return encoding


class LearnedPositionEmbedding(nn.Module):
    """
    Learned positional embedding table (GPT-2 standard approach).
    """

    def __init__(self, max_seq_len: int = 1024, d_model: int = 768):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.wpe = nn.Embedding(max_seq_len, d_model)
        nn.init.normal_(self.wpe.weight, std=0.02)

    def forward(
        self,
        seq_len: int,
        offset: int = 0,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        if offset < 0 or seq_len < 0 or offset + seq_len > self.max_seq_len:
            raise ValueError("Requested positions exceed the embedding context window")
        positions = torch.arange(offset, offset + seq_len, dtype=torch.long,
                                 device=device if device is not None else self.wpe.weight.device)
        return self.wpe(positions).unsqueeze(0)
