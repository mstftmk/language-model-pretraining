"""A small decoder-only transformer (GPT-style) in PyTorch.

Components, with the paper each comes from:

* RMSNorm (Zhang and Sennrich, 2019): scales by the root mean square only,
  no mean subtraction and no bias. Applied before each sub-layer (pre-norm).
* Rotary position embeddings, RoPE (Su et al., 2021): rotates query and key
  vectors by an angle proportional to their position, so the attention score
  between two tokens depends on their distance, not their absolute positions.
* SwiGLU feed-forward (Shazeer, 2020): ``W2(silu(W1 x) * W3 x)``.
* Causal self-attention through ``F.scaled_dot_product_attention``.
* Tied input and output embeddings (Press and Wolf, 2017).
* A key/value cache for generation: each new token attends to cached keys
  and values instead of recomputing the whole prefix.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 4096
    context: int = 256
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    d_ff: int | None = None  # SwiGLU hidden size; default about 8/3 * d_model
    rope_base: float = 10000.0
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if (self.d_model // self.n_heads) % 2:
            raise ValueError("head dimension must be even for RoPE")
        if self.d_ff is None:
            # 8/3 * d keeps the parameter count of a 4*d two-matrix MLP,
            # rounded up to a multiple of 64 for efficient matmuls
            self.d_ff = 64 * math.ceil(8 * self.d_model / 3 / 64)

    def to_dict(self) -> dict:
        return asdict(self)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rms).type_as(x) * self.weight


def rope_tables(head_dim: int, length: int, base: float, device=None):
    """cos and sin tables of shape (length, head_dim // 2)."""
    inv_freq = 1.0 / base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    angles = torch.outer(torch.arange(length, device=device).float(), inv_freq)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate pairs (x[2i], x[2i+1]) of ``x`` (B, H, T, D) by the position angle."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.flatten(-2).type_as(x)


class KVCache:
    """Per-layer cache of keys and values, filled one step at a time."""

    def __init__(self, n_layers: int) -> None:
        self.k: list[torch.Tensor | None] = [None] * n_layers
        self.v: list[torch.Tensor | None] = [None] * n_layers

    def layer_length(self, layer: int) -> int:
        return 0 if self.k[layer] is None else self.k[layer].shape[2]

    @property
    def length(self) -> int:
        """Tokens already cached (read before a forward pass updates layer 0)."""
        return self.layer_length(0)

    def update(self, layer: int, k: torch.Tensor, v: torch.Tensor):
        if self.k[layer] is not None:
            k = torch.cat([self.k[layer], k], dim=2)
            v = torch.cat([self.v[layer], v], dim=2)
        self.k[layer], self.v[layer] = k, v
        return k, v


class Attention(nn.Module):
    def __init__(self, cfg: GPTConfig, layer: int) -> None:
        super().__init__()
        self.layer = layer
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x, cos, sin, cache: KVCache | None = None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        past = 0
        if cache is not None:
            # this layer's own length: layer 0 has already been updated
            # by the time later layers run in the same forward pass
            past = cache.layer_length(self.layer)
            k, v = cache.update(self.layer, k, v)
        if past == 0:
            mask, causal = None, True
        else:
            # new queries sit at positions past..past+T-1 and may see keys 0..their own position
            q_pos = torch.arange(past, past + T, device=x.device)[:, None]
            k_pos = torch.arange(k.shape[2], device=x.device)[None, :]
            mask, causal = k_pos <= q_pos, False
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=causal,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


class SwiGLU(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.w1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w3 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig, layer: int) -> None:
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg, layer)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cos, sin, cache=None):
        x = x + self.drop(self.attn(self.norm1(x), cos, sin, cache))
        return x + self.drop(self.mlp(self.norm2(x)))


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg, i) for i in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.embed.weight  # tied embeddings
        cos, sin = rope_tables(cfg.d_model // cfg.n_heads, cfg.context, cfg.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init_weights)
        # scale the residual output projections by depth (GPT-2 style)
        for name, p in self.named_parameters():
            if name.endswith(("attn.proj.weight", "mlp.w2.weight")):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    @classmethod
    def from_checkpoint(cls, path, device="cpu") -> "GPT":
        """Load a model saved by scripts/train.py (``config`` + ``model`` state)."""
        state = torch.load(path, map_location=device)
        model = cls(GPTConfig(**state["config"]))
        model.load_state_dict(state["model"])
        return model.to(device).eval()

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())  # tied weights count once

    def forward(self, idx, targets=None, cache: KVCache | None = None):
        B, T = idx.shape
        start = 0 if cache is None else cache.length
        if start + T > self.cfg.context:
            raise ValueError(f"sequence length {start + T} exceeds context {self.cfg.context}")
        cos = self.rope_cos[start : start + T]
        sin = self.rope_sin[start : start + T]
        x = self.embed(idx)
        for block in self.blocks:
            x = block(x, cos, sin, cache)
        logits = self.head(self.norm(x))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        use_cache: bool = True,
        stop_token: int | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample tokens; with ``temperature == 0`` decoding is greedy.

        With ``use_cache`` the prompt is processed once and every later step
        feeds only the newest token. Without it, every step recomputes the
        whole sequence (the slow reference, kept for tests and benchmarks).
        """
        self.eval()
        max_new_tokens = min(max_new_tokens, self.cfg.context - idx.shape[1])
        cache = KVCache(self.cfg.n_layers) if use_cache else None
        out = idx
        step_input = idx
        for _ in range(max_new_tokens):
            logits, _ = self(step_input if use_cache else out, cache=cache)
            logits = logits[:, -1, :].float()
            if temperature == 0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1:]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                probs = F.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, 1, generator=generator)
            out = torch.cat([out, nxt], dim=1)
            step_input = nxt
            if stop_token is not None and bool((nxt == stop_token).all()):
                break
        return out
