"""
Token embedding module.

Status: Clean
"""

import math
from typing import Optional
import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """
    Standard vocabulary embedding layer projecting discrete token IDs into continuous latent space.
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 768,
        scale_embeddings: bool = False,
        padding_idx: Optional[int] = None
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.scale_embeddings = scale_embeddings

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=padding_idx
        )
        # Initialize embeddings with small Gaussian variance (GPT-2 standard)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        if self.embedding.padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[self.embedding.padding_idx].zero_()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Embed input token indices.

        Args:
            input_ids: LongTensor of shape (batch_size, seq_len).

        Returns:
            Tensor of shape (batch_size, seq_len, d_model).
        """
        x = self.embedding(input_ids)
        if self.scale_embeddings:
            x = x * math.sqrt(self.d_model)
        return x
