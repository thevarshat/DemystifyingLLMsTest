"""
Performance and latency benchmarking utilities for transformer inference.
"""

import time
from typing import Dict, Any, Callable
import torch


def benchmark_forward_latency(
    model: torch.nn.Module,
    sample_input: torch.Tensor,
    warmup_runs: int = 3,
    benchmark_runs: int = 10
) -> Dict[str, float]:
    """
    Measure average forward-pass latency in milliseconds.

    Args:
        model: PyTorch model module.
        sample_input: Input tensor of token IDs.
        warmup_runs: Number of unmeasured warmup forward passes.
        benchmark_runs: Number of timed forward passes.

    Returns:
        Dictionary with 'mean_ms', 'min_ms', 'max_ms', and 'throughput_tokens_per_sec'.
    """
    if benchmark_runs <= 0 or warmup_runs < 0:
        raise ValueError("benchmark_runs must be positive and warmup_runs non-negative")
    model.eval()
    batch_size, seq_len = sample_input.shape

    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(sample_input)

        latencies = []
        for _ in range(benchmark_runs):
            if sample_input.is_cuda:
                torch.cuda.synchronize(sample_input.device)
            start = time.perf_counter()
            _ = model(sample_input)
            if sample_input.is_cuda:
                torch.cuda.synchronize(sample_input.device)
            end = time.perf_counter()
            latencies.append((end - start) * 1000.0)

    mean_ms = sum(latencies) / len(latencies)
    tokens_processed = batch_size * seq_len
    tokens_per_sec = (tokens_processed / (mean_ms / 1000.0)) if mean_ms > 0 else 0.0

    return {
        "mean_ms": round(mean_ms, 3),
        "min_ms": round(min(latencies), 3),
        "max_ms": round(max(latencies), 3),
        "tokens_per_sec": round(tokens_per_sec, 1),
    }


def benchmark_generation_throughput(
    generate_fn: Callable[[torch.Tensor, int], torch.Tensor],
    prompt_ids: torch.Tensor,
    max_new_tokens: int = 32
) -> Dict[str, float]:
    """
    Measure autoregressive token generation speed.

    Args:
        generate_fn: Callable that takes (prompt_ids, max_new_tokens) and returns full sequence.
        prompt_ids: Initial prompt tensor of shape (1, prompt_len).
        max_new_tokens: Number of tokens to generate.

    Returns:
        Dictionary containing total_time_sec, tokens_generated, and tokens_per_sec.
    """
    if prompt_ids.is_cuda:
        torch.cuda.synchronize(prompt_ids.device)
    start = time.perf_counter()
    output_ids = generate_fn(prompt_ids, max_new_tokens)
    if output_ids.is_cuda:
        torch.cuda.synchronize(output_ids.device)
    elapsed = time.perf_counter() - start

    prompt_len = prompt_ids.size(1)
    total_len = output_ids.size(1)
    new_tokens = output_ids.size(0) * max(0, total_len - prompt_len)

    tokens_per_sec = (new_tokens / elapsed) if elapsed > 0 else 0.0

    return {
        "total_time_sec": round(elapsed, 4),
        "tokens_generated": new_tokens,
        "tokens_per_sec": round(tokens_per_sec, 2),
    }
