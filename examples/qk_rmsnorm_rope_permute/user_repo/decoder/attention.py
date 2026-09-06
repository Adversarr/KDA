"""attention components for the local training model."""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from decoder.config import ModelConfig


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
    """``q: (B, S, Hq, D)``, ``k: (B, S, Hk, D)`` -> both as ``(B, H, S, D)``, shared-weight RMS-normalized and rotate-half embedded."""
    q = rmsnorm(q, q_norm_w, eps)
    k = rmsnorm(k, k_norm_w, eps)
    q = apply_rope(q, cos, sin, interleaved=False)
    k = apply_rope(k, cos, sin, interleaved=False)
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
        self.q_proj = nn.Linear(cfg.d_model, self.hq * self.d, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, self.hk * self.d, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, self.hk * self.d, bias=False)
        self.q_norm_w = nn.Parameter(torch.ones(self.d))
        self.k_norm_w = nn.Parameter(torch.ones(self.d))
        self.proj = nn.Linear(self.hq * self.d, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, self.hq, self.d)
        k = self.k_proj(x).view(b, s, self.hk, self.d)
        v = self.v_proj(x).view(b, s, self.hk, self.d)
        q, k = qk_prep(q, k, self.q_norm_w, self.k_norm_w, cos, sin, self.eps)
        o = F.scaled_dot_product_attention(q, k, v.transpose(1, 2), is_causal=True)
        return self.proj(o.transpose(1, 2).reshape(b, s, self.hq * self.d))
