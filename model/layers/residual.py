"""
Residual connection module.

"""

from typing import Optional, Callable, Union, Any
import torch
import torch.nn as nn


class ResidualConnection(nn.Module):
    """
    Residual connection wrapper with optional dropout:
    y = x + Dropout(sublayer(x))
    """

    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        sublayer: Union[torch.Tensor, Callable[..., torch.Tensor]],
        *args,
        **kwargs
    ) -> torch.Tensor:
        """
        Apply residual skip connection.

        Args:
            x: Input residual stream tensor.
            sublayer: Sublayer output tensor OR callable sublayer module/function.

        Returns:
            Residual stream output tensor.
        """
        if callable(sublayer):
            sublayer_out = sublayer(x, *args, **kwargs)
        else:
            sublayer_out = sublayer

        return x + self.dropout(sublayer_out)
