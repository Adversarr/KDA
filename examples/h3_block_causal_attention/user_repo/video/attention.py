"""attention components for the local training model."""

from typing import Tuple

import torch
import torch.nn as nn

from video.settings import ModelConfig


def rmsnorm_heads(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """``x: (B, S, H, D)``, ``weight: (D,)``: RMSNorm over the last dim of every head, in fp32."""
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (y * weight.float()).to(x.dtype)


def apply_h3_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate-half RoPE on the leading ``rotary_dim = cos.shape[-1]`` channels of ``x: (B, S, H, D)``; the rest pass through."""
    rotary_dim = cos.shape[-1]
    x_rot, x_pass = x[..., :rotary_dim].float(), x[..., rotary_dim:]
    c, s = cos[None, :, None, :], sin[None, :, None, :]
    x1, x2 = x_rot.chunk(2, dim=-1)
    rotate_half = torch.cat([-x2, x1], dim=-1)
    return torch.cat([(x_rot * c + rotate_half * s).to(x.dtype), x_pass], dim=-1)


def qk_prep(
    q: torch.Tensor, k: torch.Tensor, q_norm_w: torch.Tensor, k_norm_w: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, eps: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``q, k: (B, S, H, D)`` -> both as ``(B, H, S, D)``: per-head RMSNorm, partial 3D RoPE, transpose."""
    q = apply_h3_rope(rmsnorm_heads(q, q_norm_w, eps), cos, sin)
    k = apply_h3_rope(rmsnorm_heads(k, k_norm_w, eps), cos, sin)
    return q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous()


def block_causal_mask(seq_len: int, block_size: int, device: torch.device) -> torch.Tensor:
    """``(S, S)`` bool: ``mask[i, j]`` is True when key ``j`` is in the same or an earlier chunk than query ``i``."""
    chunk = torch.arange(seq_len, device=device) // block_size
    return chunk[:, None] >= chunk[None, :]


def block_causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, block_size: int) -> torch.Tensor:
    """``q, k, v: (B, H, S, D)`` -> ``(B, H, S, D)``: softmax(q k^T / sqrt(D), block-causal) v, softmax in fp32.

    The mask is a function of ``S`` and ``block_size`` only (built here from indices, never
    stored or passed in). The materialised ``(B, H, S, S)`` fp32 score plane is what makes this
    the memory and time bottleneck at video lengths: 56 heads x 21840 tokens is 107 GB.
    """
    d = q.shape[-1]
    scores = (q @ k.transpose(-2, -1)).float() * (d**-0.5)
    scores = scores.masked_fill(~block_causal_mask(q.shape[-2], block_size, q.device), float("-inf"))
    probs = torch.softmax(scores, dim=-1).to(v.dtype)
    return probs @ v


class H3Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.h, self.d = cfg.num_heads, cfg.head_dim
        self.eps = cfg.norm_eps
        self.block_size = cfg.block_size
        inner = self.h * self.d
        self.to_q = nn.Linear(cfg.hidden_size, inner, bias=False)
        self.to_k = nn.Linear(cfg.hidden_size, inner, bias=False)
        self.to_v = nn.Linear(cfg.hidden_size, inner, bias=False)
        self.q_norm_w = nn.Parameter(torch.ones(self.d))
        self.k_norm_w = nn.Parameter(torch.ones(self.d))
        self.to_out = nn.Linear(inner, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        q = self.to_q(x).view(b, s, self.h, self.d)
        k = self.to_k(x).view(b, s, self.h, self.d)
        v = self.to_v(x).view(b, s, self.h, self.d)
        q, k = qk_prep(q, k, self.q_norm_w, self.k_norm_w, cos, sin, self.eps)
        v = v.transpose(1, 2)
        o = block_causal_attention(q, k, v, self.block_size)
        return self.to_out(o.transpose(1, 2).reshape(b, s, self.h * self.d))
