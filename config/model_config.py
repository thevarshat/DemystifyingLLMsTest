"""
Model architecture configuration.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional
import json


@dataclass
class ModelConfig:
    d_model: int = 768          # Hidden dimension (embedding size)
    n_heads: int = 12           # Number of attention heads
    n_layers: int = 12          # Number of transformer blocks
    vocab_size: int = 50257     # Vocabulary size (GPT-2 BPE)
    max_seq_len: int = 1024     # Maximum sequence context window
    ffn_multiplier: int = 4     # Expansion ratio for FFN hidden dimension
    dropout: float = 0.1        # Dropout rate for residual & attention connections
    layer_norm_eps: float = 1e-5
    initializer_range: float = 0.02
    use_rotary: bool = False    # Rotary Position Embedding (RoPE) toggle
    tie_weights: bool = True     # Tie embedding weights with LM head output projection

    d_head: int = field(init=False)
    d_ffn: int = field(init=False)

    def __post_init__(self):
        for name in ("d_model", "n_heads", "n_layers", "vocab_size", "max_seq_len", "ffn_multiplier"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 <= self.dropout <= 1.0:
            raise ValueError("dropout must be in [0, 1]")
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        self.d_head = self.d_model // self.n_heads
        self.d_ffn = self.d_model * self.ffn_multiplier

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        """Instantiate configuration from dictionary values."""
        clean_data = {k: v for k, v in data.items() if k in cls.__dataclass_fields__ and k not in ("d_head", "d_ffn")}
        return cls(**clean_data)

    def to_json(self, indent: int = 2) -> str:
        """Serialize configuration to JSON formatted string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "ModelConfig":
        """Load configuration from JSON string."""
        return cls.from_dict(json.loads(json_str))
