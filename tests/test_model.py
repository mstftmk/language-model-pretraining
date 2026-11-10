import math

import numpy as np
import pytest
import torch

from tinylm.data import sample_batch
from tinylm.model import GPT, GPTConfig, KVCache, RMSNorm, apply_rope, rope_tables

TINY = dict(vocab_size=64, context=32, d_model=32, n_layers=2, n_heads=4)


@pytest.fixture
def model() -> GPT:
    torch.manual_seed(0)
    return GPT(GPTConfig(**TINY)).eval()


def test_default_config_size():
    cfg = GPTConfig()
    assert cfg.d_ff == 1024
    n = GPT(cfg).num_params()
    assert 10e6 < n < 20e6


def test_tied_embeddings(model):
    assert model.head.weight is model.embed.weight


def test_output_shapes_and_initial_loss(model):
    idx = torch.randint(0, 64, (3, 16))
    targets = torch.randint(0, 64, (3, 16))  # not idx: tied weights favour the input token
    logits, loss = model(idx, targets)
    assert logits.shape == (3, 16, 64)
    # an untrained model should be close to uniform: loss about ln(vocab)
    assert abs(loss.item() - math.log(64)) < 0.3


def test_causal_future_tokens_do_not_change_past_logits(model):
    idx = torch.randint(0, 64, (1, 20))
    changed = idx.clone()
    changed[0, 12:] = (changed[0, 12:] + 1) % 64
    a, _ = model(idx)
    b, _ = model(changed)
    assert torch.allclose(a[:, :12], b[:, :12], atol=1e-5)
    assert not torch.allclose(a[:, 12:], b[:, 12:], atol=1e-5)


def test_rope_score_depends_only_on_relative_position():
    torch.manual_seed(0)
    d = 16
    cos, sin = rope_tables(d, 64, 10000.0)
    q = torch.randn(1, 1, 1, d)
    k = torch.randn(1, 1, 1, d)

    def score(i, j):
        qi = apply_rope(q, cos[i : i + 1], sin[i : i + 1])
        kj = apply_rope(k, cos[j : j + 1], sin[j : j + 1])
        return (qi * kj).sum().item()

    assert score(5, 2) == pytest.approx(score(40, 37), abs=1e-4)
    assert score(5, 2) != pytest.approx(score(5, 4), abs=1e-4)


def test_rope_preserves_norm():
    cos, sin = rope_tables(8, 10, 10000.0)
    x = torch.randn(2, 3, 10, 8)
    assert torch.allclose(apply_rope(x, cos, sin).norm(dim=-1), x.norm(dim=-1), atol=1e-5)


def test_rmsnorm_unit_rms():
    x = torch.randn(4, 7, 32) * 5 + 3
    y = RMSNorm(32)(x)
    assert torch.allclose(y.pow(2).mean(-1), torch.ones(4, 7), atol=1e-3)


def test_kv_cache_logits_match_full_forward(model):
    idx = torch.randint(0, 64, (2, 12))
    full, _ = model(idx)
    cache = KVCache(model.cfg.n_layers)
    first, _ = model(idx[:, :5], cache=cache)  # prompt in one go
    steps = [model(idx[:, t : t + 1], cache=cache)[0] for t in range(5, 12)]
    incremental = torch.cat([first] + steps, dim=1)
    assert torch.allclose(full, incremental, atol=1e-5)


def test_greedy_generation_same_with_and_without_cache(model):
    prompt = torch.randint(0, 64, (2, 4))
    a = model.generate(prompt, 20, temperature=0, use_cache=True)
    b = model.generate(prompt, 20, temperature=0, use_cache=False)
    assert torch.equal(a, b)
    assert a.shape == (2, 24)


def test_generation_stops_at_context(model):
    out = model.generate(torch.zeros(1, 30, dtype=torch.long), 50, temperature=0)
    assert out.shape[1] == 32


def test_too_long_input_raises(model):
    with pytest.raises(ValueError):
        model(torch.zeros(1, 33, dtype=torch.long))


def test_tiny_model_memorises_a_batch():
    torch.manual_seed(0)
    model = GPT(GPTConfig(**TINY))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    x = torch.randint(0, 64, (4, 32))
    y = torch.roll(x, -1, dims=1)
    for _ in range(300):
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1


def test_sample_batch_targets_are_shifted_inputs():
    tokens = np.arange(1000, dtype=np.uint16)
    x, y = sample_batch(tokens, 8, 16, np.random.default_rng(0))
    assert x.shape == y.shape == (8, 16)
    assert np.array_equal(x[:, 1:], y[:, :-1])
    assert np.array_equal(y - x, np.ones_like(x))


def test_sample_batch_never_reads_past_the_end():
    tokens = np.arange(20, dtype=np.uint16)
    for seed in range(50):
        x, y = sample_batch(tokens, 4, 19, np.random.default_rng(seed))
        assert y.max() <= 19


def test_checkpoint_round_trip(model, tmp_path):
    path = tmp_path / "ckpt.pt"
    torch.save({"config": model.cfg.to_dict(), "model": model.state_dict()}, path)
    loaded = GPT.from_checkpoint(path)
    idx = torch.randint(0, 64, (1, 10))
    assert torch.allclose(model(idx)[0], loaded(idx)[0])
    assert loaded.head.weight is loaded.embed.weight  # tying survives loading
