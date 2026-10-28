"""
Tensor manipulation and device placement utilities.
"""

from typing import Union, List, Optional, Any
import torch


def to_device(tensor: torch.Tensor, device: Union[str, torch.device]) -> torch.Tensor:
    """
    Transfer a tensor to the specified compute device.

    Args:
        tensor: The PyTorch tensor to be transferred.
        device: Target execution device (e.g., 'cpu', 'cuda', 'meta', or torch.device).

    Returns:
        torch.Tensor placed on the target device.
    """
    if not isinstance(tensor, torch.Tensor):
        return tensor

    return tensor.to(device)


def get_device(device_str: Optional[str] = None) -> torch.device:
    """
    Resolve and return a valid torch.device instance.

    Args:
        device_str: Optional device string ('cuda', 'cpu', 'mps').

    Returns:
        torch.device instance.
    """
    if device_str is not None:
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pad_sequence(
    sequences: List[torch.Tensor],
    padding_value: int = 0,
    side: str = "right"
) -> torch.Tensor:
    """
    Pad a list of 1D or 2D tensors to equal length.

    Args:
        sequences: List of tensors to pad.
        padding_value: Integer pad token value.
        side: 'left' for left-padding (used in autoregressive generation) or 'right'.

    Returns:
        Padded batch tensor of shape (batch_size, max_len).
    """
    if not sequences:
        return torch.empty(0, dtype=torch.long)

    max_len = max(seq.size(-1) for seq in sequences)
    padded = []

    for seq in sequences:
        pad_len = max_len - seq.size(-1)
        if pad_len == 0:
            padded.append(seq)
        elif side == "right":
            pad = torch.full((*seq.shape[:-1], pad_len), padding_value, dtype=seq.dtype, device=seq.device)
            padded.append(torch.cat([seq, pad], dim=-1))
        elif side == "left":
            pad = torch.full((*seq.shape[:-1], pad_len), padding_value, dtype=seq.dtype, device=seq.device)
            padded.append(torch.cat([pad, seq], dim=-1))
        else:
            raise ValueError(f"Unsupported padding side '{side}', must be 'left' or 'right'")

    return torch.stack(padded, dim=0)


def ensure_tensor(
    data: Any,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[str, torch.device]] = None
) -> torch.Tensor:
    """
    Ensure input is converted to a torch.Tensor with optional dtype and device placement.
    """
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data, dtype=dtype)
    elif dtype is not None and data.dtype != dtype:
        data = data.to(dtype=dtype)

    if device is not None:
        data = data.to(device=device)

    return data
