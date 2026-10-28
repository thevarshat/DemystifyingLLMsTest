"""
Combined embedding layer fusing token and positional embeddings.
"""

from typing import Optional, Union, Any
import torch
import torch.nn as nn

from .token_embedding import TokenEmbedding
from .position_embedding import SinusoidalPositionEmbedding, LearnedPositionEmbedding


def _resolve_config(explicit_config: Optional[Any] = None) -> Any:
    if explicit_config is not None:
        return explicit_config
    from config.model_config import ModelConfig
    return ModelConfig()


class EmbeddingLayer(nn.Module):
    """
    Fuses discrete token IDs with positional representations, followed by layer dropout.
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        vocab_size: Optional[int] = None,
        d_model: Optional[int] = None,
        max_seq_len: Optional[int] = None,
        dropout: Optional[float] = None,
        use_learned_pe: bool = False
    ):
        super().__init__()
        cfg = _resolve_config(config)

        self.vocab_size = vocab_size or getattr(cfg, "vocab_size", 50257)
        self.d_model = d_model or getattr(cfg, "d_model", 768)
        self.max_seq_len = max_seq_len or getattr(cfg, "max_seq_len", 1024)
        dropout_p = dropout if dropout is not None else getattr(cfg, "dropout", 0.1)

        self.token_embedding = TokenEmbedding(
            vocab_size=self.vocab_size,
            d_model=self.d_model
        )

        if use_learned_pe:
            self.position_embedding = LearnedPositionEmbedding(
                max_seq_len=self.max_seq_len,
                d_model=self.d_model
            )
        else:
            self.position_embedding = SinusoidalPositionEmbedding(
                d_model=self.d_model,
                max_seq_len=self.max_seq_len
            )

        self.use_rotary = getattr(cfg, "use_rotary", False)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(
        self,
        input_ids: torch.Tensor,
        offset: int = 0
    ) -> torch.Tensor:
        """
        Embed token sequences and combine with positional encodings.

        Args:
            input_ids: LongTensor of shape (batch_size, seq_len).
            offset: Position offset (for incremental decoding).

        Returns:
            Tensor of shape (batch_size, seq_len, d_model).
        """
        if not isinstance(input_ids, torch.Tensor):
            raise TypeError(f"EmbeddingLayer expected torch.Tensor, got {type(input_ids)}")

        batch_size, seq_len = input_ids.shape
        tok_emb = self.token_embedding(input_ids)
        if self.use_rotary:
            return self.dropout(tok_emb)
        pos_emb = self.position_embedding(seq_len=seq_len, offset=offset, device=input_ids.device)

        embeddings = tok_emb + pos_emb.to(dtype=tok_emb.dtype)

        return self.dropout(embeddings)
