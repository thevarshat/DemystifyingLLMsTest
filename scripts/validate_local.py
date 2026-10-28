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
from tokenizer.vocab import Vocabulary
from model.embeddings.token_embedding import TokenEmbedding
from inference.generation.beam_search import BeamSearch
from serving.api.routes import RouteHandler
from evaluation.benchmarks import benchmark_generation_throughput
from utils.logging_utils import ColoredFormatter
import logging
import itertools


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
    torch.testing.assert_close(LayerNorm([5, 32])(x), F.layer_norm(x, (5, 32)))
    blocked = torch.full_like(mask, float("-inf"))
    zero_output, zero_weights = scaled_dot_product_attention(q, k, v, blocked)
    torch.testing.assert_close(zero_output, torch.zeros_like(zero_output))
    torch.testing.assert_close(zero_weights, torch.zeros_like(zero_weights))
    q_grad = q.clone().requires_grad_()
    scaled_dot_product_attention(q_grad, k, v, blocked)[0].sum().backward()
    assert torch.isfinite(q_grad.grad).all()

    logits = torch.tensor([[1.0, -2.0, 3.0, 0.0]])
    torch.testing.assert_close(apply_temperature(logits, 2), logits / 2)
    assert Sampler(temperature=0).sample(logits).item() == 2
    assert torch.isfinite(apply_top_k(logits, 2)).tolist() == [[True, False, True, False]]
    assert torch.isfinite(apply_top_p(logits, 0.7)).tolist() == [[False, False, True, False]]
    penalized = apply_repetition_penalty(logits, [0, 1, 0], 2)
    torch.testing.assert_close(penalized, torch.tensor([[0.5, -4.0, 3.0, 0.0]]))
    assert StopCriteria(5).should_stop(5)

    for options in ({"temperature": float("nan")}, {"repetition_penalty": float("inf")},
                    {"max_new_tokens": 1.5}, {"beam_width": 0}):
        try:
            InferenceConfig(**options)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid configuration accepted: {options}")
    first = Vocabulary({"alpha": 0})
    second = Vocabulary({"beta": 0})
    assert "alpha" not in second and "beta" not in first
    assert first.id_to_token(first.token_to_id("alpha")) == "alpha"
    assert second.id_to_token(second.token_to_id("beta")) == "beta"
    ranked = BPETokenizer(Vocabulary({"abc": 10}), [("b", "c"), ("a", "b")])
    assert ranked.bpe("abc") == ["a", "bc"]
    embedding = TokenEmbedding(8, 4, padding_idx=0)
    assert torch.count_nonzero(embedding(torch.tensor([0]))) == 0
    record = logging.LogRecord("check", logging.INFO, __file__, 1, "message", (), None)
    ColoredFormatter().format(record)
    assert record.levelname == "INFO"

    # Exhaustive two-step search is an independent oracle for a tiny real model.
    beam_cfg = ModelConfig(d_model=8, n_heads=2, n_layers=1, vocab_size=3,
                           max_seq_len=8, dropout=0, tie_weights=False)
    beam_model = FullTransformerModel(beam_cfg).eval()
    prompt_ids = torch.tensor([[0]])
    candidates = []
    with torch.no_grad():
        for suffix in itertools.product(range(3), repeat=2):
            seq = prompt_ids.clone()
            score = 0.0
            for token in suffix:
                score += beam_model(seq)[0, -1].log_softmax(-1)[token].item()
                seq = torch.cat([seq, torch.tensor([[token]])], dim=1)
            candidates.append((score, seq))
    expected_beam = max(candidates, key=lambda item: item[0])[1]
    actual_beam = BeamSearch(beam_width=9, max_length=2).search(beam_model, prompt_ids)
    torch.testing.assert_close(actual_beam, expected_beam)
    beam_engine = InferenceEngine(beam_cfg, InferenceConfig(
        beam_width=9, max_new_tokens=2, repetition_penalty=1), beam_model)
    assert beam_engine.generate_tokens([0]) == expected_beam[0].tolist()
    # BOS and EOS can be the same token; the prompt must still be expanded.
    assert BeamSearch(2, 1, eos_token_id=0).search(beam_model, prompt_ids).size(1) == 2
    throughput = benchmark_generation_throughput(
        lambda ids, count: autoregressive_generate(beam_model, ids, count),
        torch.tensor([[0], [1]]), max_new_tokens=2)
    assert throughput["tokens_generated"] == 4

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
        full_prompt = [2] * cfg.max_seq_len
        assert len(engine.generate_tokens(full_prompt, max_new_tokens=1)) == cfg.max_seq_len + 1
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
    inferred_pipeline = ServingPipeline(model=model, tokenizer=tok, inference_config=inf_cfg)
    assert inferred_pipeline.config is cfg
    assert inferred_pipeline.process(prompt) == result
    response = RequestHandler(engine, tok).handle_completion(
        {'prompt': prompt, 'max_tokens': 2, 'temperature': 0})
    assert isinstance(response['choices'][0]['text'], str)
    assert response['usage']['completion_tokens'] == 2
    router = RouteHandler(RequestHandler(engine, tok))
    for invalid_body in ([1], {"prompt": 12}, {"max_tokens": -1}, {"temperature": float("nan")}):
        assert router.dispatch("POST", "/v1/completions", invalid_body)[0] == 400
    print('All local mathematical and integration checks passed.')


if __name__ == '__main__':
    main()
