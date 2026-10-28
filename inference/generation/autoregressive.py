"""Module implementation."""
import torch
from typing import Optional, List, Callable
from .kv_cache import KVCache
from .stop_criteria import StopCriteria


@torch.no_grad()
def autoregressive_generate(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int = 20,
    eos_token_id: Optional[int] = None,
    sampler_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    use_cache: bool = False,
) -> torch.Tensor:
    """
    Generate tokens autoregressively.
    input_ids shape: (batch_size, seq_len)
    """
    if input_ids.ndim != 2 or input_ids.size(1) == 0:
        raise ValueError("input_ids must be a nonempty batch of token sequences")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    model.eval()
    generated = input_ids.clone()
    batch_size = generated.size(0)
    stop_criteria = StopCriteria(max_length=generated.size(1) + max_new_tokens, eos_token_id=eos_token_id)

    num_layers = getattr(model, "num_layers", getattr(getattr(model, "config", None), "n_layers", 12))
    kv_cache = KVCache(num_layers=num_layers) if use_cache else None
    finished = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)

    for step in range(max_new_tokens):
        if kv_cache is not None:
            model_inputs = generated if step == 0 else generated[:, -1:]
            logits = model(model_inputs, kv_cache=kv_cache)
        else:
            logits = model(generated)

        # Take logits for the last token position: (batch_size, vocab_size)
        if isinstance(logits, tuple):
            logits = logits[0]
        next_token_logits = logits[:, -1, :]

        if sampler_fn is not None:
            next_token = sampler_fn(next_token_logits)
        else:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        if next_token.dim() == 1:
            next_token = next_token.unsqueeze(-1)

        if eos_token_id is not None:
            next_token = torch.where(finished[:, None], eos_token_id, next_token)
            finished |= next_token.squeeze(-1).eq(eos_token_id)
        generated = torch.cat([generated, next_token], dim=-1)
        if finished.all():
            break

        last_id = next_token[0].item() if batch_size == 1 else None
        if stop_criteria.should_stop(generated.size(1), last_id):
            break

    return generated
