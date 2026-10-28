"""
Inference and text generation configuration.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
import json


@dataclass
class InferenceConfig:
    max_new_tokens: int = 20
    use_kv_cache: bool = True
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    stop_token_ids: List[int] = field(default_factory=list)
    early_stopping: bool = True
    beam_width: int = 1         # 1 indicates greedy / stochastic sampling (non-beam)
    seed: Optional[int] = 42

    def validate(self) -> None:
        """Validate inference parameters."""
        if self.beam_width < 1:
            raise ValueError("beam_width must be at least 1")
        if self.max_new_tokens < 0:
            raise ValueError(f"max_new_tokens must be non-negative, got {self.max_new_tokens}")
        if self.temperature < 0.0:
            raise ValueError(f"temperature must be non-negative, got {self.temperature}")
        if not (0.0 <= self.top_p <= 1.0):
            raise ValueError(f"top_p must be in [0.0, 1.0], got {self.top_p}")
        if self.top_k < 0:
            raise ValueError(f"top_k must be non-negative, got {self.top_k}")
        if self.repetition_penalty <= 0.0:
            raise ValueError(f"repetition_penalty must be positive, got {self.repetition_penalty}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InferenceConfig":
        """Load configuration from dictionary."""
        valid_keys = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid_keys)

    def to_json(self, indent: int = 2) -> str:
        """Serialize configuration to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "InferenceConfig":
        """Instantiate configuration from JSON string."""
        return cls.from_dict(json.loads(json_str))

