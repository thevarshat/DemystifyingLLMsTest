"""
High-level inference engine coordinating model execution, caching, and sampling.
"""

import inspect
from typing import Optional, List, Union, Any
import torch
import torch.nn as nn

from config.model_config import ModelConfig
from config.inference_config import InferenceConfig
from utils.logging_utils import get_logger

logger = get_logger("inference_engine")


class FullTransformerModel(nn.Module):
    """Fallback modular transformer wrapper combining embeddings, blocks, and lm_head."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        from model.embeddings import EmbeddingLayer
        from model.blocks import BlockStack
        from model.heads import LMHead

        self.embeddings = EmbeddingLayer(config=config)
        self.blocks = BlockStack(config=config)
        self.lm_head = LMHead(config=config)
        self.num_layers = config.n_layers
        if config.tie_weights:
            from model.heads.weight_tying import tie_weights
            tie_weights(self.lm_head, self.embeddings.token_embedding)

    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        cache = kwargs.get("kv_cache")
        offset = cache.get_seq_length() if cache is not None else 0
        h = self.embeddings(input_ids, offset=offset)
        h = self.blocks(h, **kwargs)
        logits = self.lm_head(h)
        return logits


class InferenceEngine:
    """
    End-to-end inference engine managing model state, token generation loop,
    and sampling orchestration.
    """

    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        inference_config: Optional[InferenceConfig] = None,
        model: Optional[Any] = None,
        device: Optional[Union[str, torch.device]] = None
    ):
        self.model_config = model_config or getattr(model, "config", None) or ModelConfig()
        self.inference_config = inference_config or InferenceConfig()
        self.inference_config.validate()
        self.device = torch.device(device) if device else torch.device("cpu")
        self.model = model if model is not None else FullTransformerModel(self.model_config)
        self._prepare_model()

    def _prepare_model(self) -> None:
        if hasattr(self.model, "to"):
            self.model.to(self.device)
        if hasattr(self.model, "eval"):
            self.model.eval()

    def generate_tokens(
        self,
        prompt_tokens: List[int],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> List[int]:
        """
        Generate a list of token IDs autoregressively from input token sequence.
        """
        from dataclasses import replace
        from inference.sampling.sampler import Sampler
        from inference.generation.kv_cache import KVCache
        from config.constants import BOS_TOKEN_ID, EOS_TOKEN_ID

        self._prepare_model()
        cfg = replace(
            self.inference_config,
            max_new_tokens=max_new_tokens if max_new_tokens is not None else self.inference_config.max_new_tokens,
            temperature=temperature if temperature is not None else self.inference_config.temperature,
            top_k=top_k if top_k is not None else self.inference_config.top_k,
            top_p=top_p if top_p is not None else self.inference_config.top_p,
        )
        cfg.validate()
        tokens = list(prompt_tokens)
        if cfg.max_new_tokens == 0:
            return tokens
        if not tokens:
            if BOS_TOKEN_ID >= self.model_config.vocab_size:
                raise ValueError("An empty prompt requires a valid BOS token")
            tokens = [BOS_TOKEN_ID]
        sampler = Sampler(cfg.temperature, cfg.top_k, cfg.top_p, cfg.repetition_penalty)
        parameters = inspect.signature(self.model.forward if hasattr(self.model, "forward") else self.model).parameters
        supports_cache = "kv_cache" in parameters or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()
        )
        cache = KVCache(self.model_config.n_layers) if cfg.use_kv_cache and supports_cache else None
        stop_ids = set(cfg.stop_token_ids) | {EOS_TOKEN_ID}
        if cfg.seed is not None:
            torch.manual_seed(cfg.seed)
        with torch.no_grad():
            for step in range(cfg.max_new_tokens):
                if len(tokens) >= self.model_config.max_seq_len:
                    break
                model_tokens = tokens if cache is None or step == 0 else tokens[-1:]
                input_tensor = torch.tensor([model_tokens], dtype=torch.long, device=self.device)
                logits = self.model(input_tensor, kv_cache=cache) if cache is not None else self.model(input_tensor)
                if isinstance(logits, tuple):
                    logits = logits[0]
                next_token = sampler.sample(logits[:, -1, :], tokens).item()
                tokens.append(int(next_token))
                if cfg.early_stopping and next_token in stop_ids:
                    break
        return tokens

    def generate(
        self,
        prompt: str,
        tokenizer: Optional[Any] = None,
        max_new_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        End-to-end string-to-string text generation.
        """
        if tokenizer is None:
            from tokenizer import BPETokenizer
            tokenizer = BPETokenizer()
        prompt_ids = tokenizer.encode(prompt)
        output_ids = self.generate_tokens(prompt_ids, max_new_tokens=max_new_tokens, **kwargs)
        return tokenizer.decode(output_ids)
