"""
Evaluation metrics for transformer language model output.

Includes perplexity, cross-entropy loss, token prediction accuracy,
and n-gram repetition rate analysis.
"""

import math
from typing import List, Sequence
import torch
import torch.nn.functional as F


def calculate_perplexity(loss: float) -> float:
    """
    Compute perplexity from average cross-entropy loss (natural log base).

    Args:
        loss: Average cross-entropy loss per token.

    Returns:
        Perplexity value (exp(loss)). Clamped to prevent float overflow.
    """
    try:
        return math.exp(min(loss, 100.0))
    except OverflowError:
        return float("inf")


def cross_entropy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100
) -> torch.Tensor:
    """
    Compute cross entropy loss between logits and target token IDs.

    Args:
        logits: Tensor of shape (batch_size, seq_len, vocab_size)
        targets: Tensor of shape (batch_size, seq_len)
        ignore_index: Target token ID to ignore in loss calculation.

    Returns:
        Scalar loss tensor.
    """
    vocab_size = logits.size(-1)
    flat_logits = logits.reshape(-1, vocab_size)
    flat_targets = targets.reshape(-1)
    return F.cross_entropy(flat_logits, flat_targets, ignore_index=ignore_index)


def token_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100
) -> float:
    """
    Calculate greedy token-level accuracy.

    Args:
        predictions: Logits of shape (batch, seq_len, vocab_size) or argmax IDs.
        targets: Target IDs of shape (batch, seq_len).
        ignore_index: Index to ignore.

    Returns:
        Accuracy fraction in range [0.0, 1.0].
    """
    if predictions.dim() == 3:
        pred_ids = torch.argmax(predictions, dim=-1)
    else:
        pred_ids = predictions

    mask = targets != ignore_index
    if mask.sum() == 0:
        return 0.0

    correct = (pred_ids == targets) & mask
    return (correct.sum().float() / mask.sum().float()).item()


def repetition_rate(token_ids: Sequence[int], n: int = 3) -> float:
    """
    Calculate the fraction of repeated n-grams in generated token sequence.
    High repetition rate indicates degeneration or attention collapse.

    Args:
        token_ids: Sequence of token integers.
        n: n-gram size (default: 3).

    Returns:
        Ratio of duplicate n-grams to total n-grams.
    """
    if len(token_ids) < n:
        return 0.0

    ngrams = [tuple(token_ids[i:i + n]) for i in range(len(token_ids) - n + 1)]
    unique_ngrams = set(ngrams)

    total = len(ngrams)
    duplicates = total - len(unique_ngrams)
    return duplicates / total if total > 0 else 0.0
