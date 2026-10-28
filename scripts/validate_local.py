"""Local mathematical and integration checks. Run: python scripts/validate_local.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from config import ModelConfig, InferenceConfig
from inference.engine.inference_engine import FullTransformerModel, InferenceEngine
from inference.generation.autoregressive import autoregressive_generate
from inference.generation.kv_cache import KVCache
from inference.generation.stop_criteria import StopCriteria
from inference.sampling.sampler import Sampler
from inference.sampling.temperature import apply_temperature
from inference.sampling.top_k import apply_top_k
from inference.sampling.top_p import apply_top_p
from inference.sampling.repetition_penalty import apply_repetition_penalty
from model.attention.scaled_dot_product import scaled_dot_product_attention
from model.attention.causal_mask import create_causal_mask
from model.layers.layer_norm import LayerNorm
from model.layers.residual import ResidualConnection
from serving.api.request_handler import RequestHandler
from serving.pipeline import ServingPipeline
from serving.model_loader import ModelLoader
from tokenizer import BPETokenizer
from utils.checkpoint import save_checkpoint
from tempfile import TemporaryDirectory


def main():
    torch.manual_seed(7)
    torch.set_num_threads(1)
    q, k, v = [torch.randn(2, 4, 5, 8) for _ in range(3)]
    mask = create_causal_mask(5)
    actual, weights = scaled_dot_product_attention(q, k, v, mask)
    expected = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(weights.sum(-1), torch.ones(2, 4, 5))
    assert torch.count_nonzero(weights.triu(1)) == 0
    x = torch.randn(2, 5, 32)
    torch.testing.assert_close(LayerNorm(32)(x), F.layer_norm(x, (32,)))
    torch.testing.assert_close(ResidualConnection(0)(x, x * 3), x * 4)

    logits = torch.tensor([[1.0, -2.0, 3.0, 0.0]])
    torch.testing.assert_close(apply_temperature(logits, 2), logits / 2)
    assert Sampler(temperature=0).sample(logits).item() == 2
    assert torch.isfinite(apply_top_k(logits, 2)).tolist() == [[True, False, True, False]]
    assert torch.isfinite(apply_top_p(logits, 0.7)).tolist() == [[False, False, True, False]]
    penalized = apply_repetition_penalty(logits, [0, 1, 0], 2)
    torch.testing.assert_close(penalized, torch.tensor([[0.5, -4.0, 3.0, 0.0]]))
    assert StopCriteria(5).should_stop(5)

    for rotary in (False, True):
        cfg = ModelConfig(d_model=32, n_heads=4, n_layers=2, vocab_size=64,
                          max_seq_len=32, dropout=0.0, use_rotary=rotary)
        model = FullTransformerModel(cfg).eval()
        assert model.lm_head.proj.weight is model.embeddings.token_embedding.embedding.weight
        assert model.blocks[0].attn.q_proj.weight is not model.blocks[0].attn.k_proj.weight
        ids = torch.tensor([[2, 3, 4, 5], [6, 7, 8, 9]])
        with torch.no_grad():
            full = model(ids)
            cache = KVCache(cfg.n_layers)
            chunks = [model(ids[:, :2], kv_cache=cache), model(ids[:, 2:], kv_cache=cache)]
            torch.testing.assert_close(torch.cat(chunks, dim=1), full, atol=1e-5, rtol=1e-4)
            assert cache.get_seq_length() == 4
            altered = ids.clone()
            altered[:, -1] = 10
            torch.testing.assert_close(model(altered)[:, :-1], full[:, :-1])
        cached = autoregressive_generate(model, ids, max_new_tokens=4, use_cache=True)
        uncached = autoregressive_generate(model, ids, max_new_tokens=4, use_cache=False)
        torch.testing.assert_close(cached, uncached)
        torch.testing.assert_close(cached[:, :4], ids)
        assert cached.shape == (2, 8)
        inf_cfg = InferenceConfig(max_new_tokens=4, temperature=0, repetition_penalty=1)
        engine = InferenceEngine(cfg, inf_cfg, model=model)
        assert engine.generate_tokens(ids[0].tolist()) == cached[0].tolist()
        with TemporaryDirectory() as temp:
            path = str(Path(temp) / 'weights.pt')
            save_checkpoint(model.state_dict(), path, use_pickle=False)
            restored = FullTransformerModel(cfg).eval()
            ModelLoader(path).load_weights(restored, strict=True)
            with torch.no_grad():
                torch.testing.assert_close(restored(ids), full)

    # A real small transformer exercises text generation without a pretrained checkpoint.
    cfg = ModelConfig(d_model=16, n_heads=2, n_layers=1, dropout=0, max_seq_len=32)
    model = FullTransformerModel(cfg)
    tok = BPETokenizer()
    inf_cfg = InferenceConfig(max_new_tokens=2, temperature=0, early_stopping=False)
    engine = InferenceEngine(cfg, inf_cfg, model=model)
    prompt = 'hello world'
    result = engine.generate(prompt, tokenizer=tok)
    assert isinstance(result, str) and result.startswith(prompt)
    assert ServingPipeline(model, tok, cfg, inf_cfg).process(prompt) == result
    response = RequestHandler(engine, tok).handle_completion(
        {'prompt': prompt, 'max_tokens': 2, 'temperature': 0})
    assert isinstance(response['choices'][0]['text'], str)
    assert response['usage']['completion_tokens'] == 2
    print('All local mathematical and integration checks passed.')


if __name__ == '__main__':
    main()
