"""A minimal pre-norm transformer with Qwen3-style grouped-query attention.

Each attention block projects `x` to a fused `qkv` buffer, splits it into `q (B, S, Hq, D)`,
`k`, `v (B, S, Hk, D)` views, and runs `qk_prep` on q and k: per-head RMSNorm (one weight row
per head), interleaved rotary embedding, and the transpose to `(B, H, S, D)` that the attention
kernel wants. `qk_prep` is the function the fused kernel replaces.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


def rmsnorm_per_head(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """``x: (B, S, H, D)``, ``weight: (H, D)``: RMSNorm over ``D`` with head ``h`` scaled by ``weight[h]``."""
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (y * weight.float()).to(x.dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, interleaved: bool = True) -> torch.Tensor:
    """Rotary embedding at position ``s``; ``cos, sin: (S, D/2)`` fp32. Interleaved pairs ``(x[2i], x[2i+1])``."""
    S = x.shape[1]
    c, s = cos[None, :S, None, :], sin[None, :S, None, :]
    xf = x.float()
    if interleaved:
        x1, x2 = xf[..., 0::2], xf[..., 1::2]
        return torch.stack((x1 * c - x2 * s, x2 * c + x1 * s), dim=-1).flatten(-2).to(x.dtype)
    x1, x2 = xf.chunk(2, dim=-1)
    return torch.cat((x1 * c - x2 * s, x2 * c + x1 * s), dim=-1).to(x.dtype)


def qk_prep(
    q: torch.Tensor, k: torch.Tensor, q_norm_w: torch.Tensor, k_norm_w: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, eps: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``q: (B, S, Hq, D)``, ``k: (B, S, Hk, D)`` -> both as ``(B, H, S, D)``, normalised per head and rotated."""
    q = rmsnorm_per_head(q, q_norm_w, eps)
    k = rmsnorm_per_head(k, k_norm_w, eps)
    q = apply_rope(q, cos, sin, interleaved=True)
    k = apply_rope(k, cos, sin, interleaved=True)
    return q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous()


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (y * weight.float()).to(x.dtype)


def rope_tables(seq_len: int, head_dim: int, base: float, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    freqs = torch.arange(seq_len, device=device, dtype=torch.float32)[:, None] * inv_freq[None, :]
    return freqs.cos(), freqs.sin()


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.hq, self.hk, self.d = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        self.eps = cfg.norm_eps
        self.qkv = nn.Linear(cfg.d_model, (self.hq + 2 * self.hk) * self.d, bias=False)
        self.q_norm_w = nn.Parameter(torch.ones(self.hq, self.d))
        self.k_norm_w = nn.Parameter(torch.ones(self.hk, self.d))
        self.proj = nn.Linear(self.hq * self.d, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        qkv = self.qkv(x)
        # views into the fused buffer: each has row stride (Hq + 2 Hk) * D, not H * D
        q = qkv[..., : self.hq * self.d].view(b, s, self.hq, self.d)
        k = qkv[..., self.hq * self.d : (self.hq + self.hk) * self.d].view(b, s, self.hk, self.d)
        v = qkv[..., (self.hq + self.hk) * self.d :].view(b, s, self.hk, self.d)
        q, k = qk_prep(q, k, self.q_norm_w, self.k_norm_w, cos, sin, self.eps)
        o = F.scaled_dot_product_attention(q, k, v.transpose(1, 2), is_causal=True, enable_gqa=True)
        return self.proj(o.transpose(1, 2).reshape(b, s, self.hq * self.d))


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden = cfg.mlp_ratio * cfg.d_model
        self.fc1 = nn.Linear(cfg.d_model, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.silu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.eps = cfg.norm_eps
        self.norm1_weight = nn.Parameter(torch.ones(cfg.d_model))
        self.norm2_weight = nn.Parameter(torch.ones(cfg.d_model))
        self.attn = Attention(cfg)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(rmsnorm(x, self.norm1_weight, self.eps), cos, sin)
        return x + self.mlp(rmsnorm(x, self.norm2_weight, self.eps))


class MiniLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm_f_weight = nn.Parameter(torch.ones(cfg.d_model))
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        cos, sin = rope_tables(cfg.seq_len, cfg.head_dim, cfg.rope_base, torch.device("cpu"))
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor, compute_dtype: torch.dtype) -> torch.Tensor:
        x = self.embed(tokens).to(compute_dtype)
        for block in self.blocks:
            x = block(x, self.rope_cos, self.rope_sin)
        logits = self.lm_head(rmsnorm(x, self.norm_f_weight, self.cfg.norm_eps)).float()
        return F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.reshape(-1))
