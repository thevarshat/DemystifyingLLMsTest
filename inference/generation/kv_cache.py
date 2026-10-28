"""KV cache management."""
import torch
from typing import Optional, Tuple, List


class KVCache:
    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.k_cache: List[Optional[torch.Tensor]] = [None] * num_layers
        self.v_cache: List[Optional[torch.Tensor]] = [None] * num_layers

    def update(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update the key and value cache for a given layer.
        key, value shape: (batch_size, num_heads, seq_len, head_dim)
        """
        if self.k_cache[layer_idx] is None:
            self.k_cache[layer_idx] = key
            self.v_cache[layer_idx] = value
        else:
            self.k_cache[layer_idx] = torch.cat([self.k_cache[layer_idx], key], dim=-2)
            self.v_cache[layer_idx] = torch.cat([self.v_cache[layer_idx], value], dim=-2)

        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def reset(self):
        self.k_cache = [None] * self.num_layers
        self.v_cache = [None] * self.num_layers

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if self.k_cache[layer_idx] is None:
            return 0
        return self.k_cache[layer_idx].shape[-2]
