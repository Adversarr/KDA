"""A MiniMax-H3-style video transformer block on a packed (t, h, w) token sequence.

The attention path follows the public H3 implementation (`to_q/to_k/to_v` expand the 5376-wide
residual to 56 heads x 128, per-head RMSNorm and 3D MM-RoPE on q and k, fp32 softmax) with the
mask of the video-generation setting: **block-causal over frames**. The packed sequence is
`t` frames of `h * w` latent tokens each, in frame order; every token attends to all tokens of
its own frame and of every earlier frame, and to none of the later ones
(`frame(q) >= frame(k)` with `frame(i) = i // block_size`, `block_size = h * w`). Inside a frame
the attention is dense (non-causal); across frames it is causal at frame granularity.
`block_causal_attention` is the function the fused kernel replaces; `qk_prep` stays eager.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


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


def mm_rope_tables(grid: Tuple[int, int, int], rope_freq_dim: int, theta: float, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """``cos, sin: (S, 3 * rope_freq_dim * 2)`` for the packed ``(t, h, w)`` grid, rotate-half layout (angles duplicated)."""
    t, h, w = grid
    tt, hh, ww = torch.meshgrid(torch.arange(t), torch.arange(h), torch.arange(w), indexing="ij")
    position_ids = torch.stack([tt.flatten(), hh.flatten(), ww.flatten()], dim=-1).to(device)  # (S, 3)
    inv_freq = 1.0 / (theta ** (torch.arange(0, 2 * rope_freq_dim, 2, dtype=torch.float32, device=device) / (2 * rope_freq_dim)))
    freqs = position_ids.float().unsqueeze(-1) * inv_freq[None, None, :]  # (S, 3, 16)
    freqs = torch.cat(freqs.unbind(dim=1), dim=-1)  # (S, 48)
    freqs = torch.cat([freqs, freqs], dim=-1)  # (S, 96)
    return freqs.cos(), freqs.sin()


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


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden = cfg.mlp_ratio * cfg.hidden_size
        self.fc1 = nn.Linear(cfg.hidden_size, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (y * weight.float()).to(x.dtype)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.eps = cfg.norm_eps
        self.norm1_weight = nn.Parameter(torch.ones(cfg.hidden_size))
        self.norm2_weight = nn.Parameter(torch.ones(cfg.hidden_size))
        self.attn = H3Attention(cfg)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(rmsnorm(x, self.norm1_weight, self.eps), cos, sin)
        return x + self.mlp(rmsnorm(x, self.norm2_weight, self.eps))


class MiniH3(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Linear(cfg.in_channels, cfg.hidden_size)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm_f_weight = nn.Parameter(torch.ones(cfg.hidden_size))
        self.out = nn.Linear(cfg.hidden_size, cfg.in_channels)
        cos, sin = mm_rope_tables(cfg.grid, cfg.rope_freq_dim, cfg.rope_theta, torch.device("cpu"))
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, latents: torch.Tensor, target: torch.Tensor, compute_dtype: torch.dtype) -> torch.Tensor:
        """``latents, target: (B, S, C)`` with ``S = t * h * w`` packed tokens -> MSE loss."""
        x = self.embed(latents).to(compute_dtype)
        for block in self.blocks:
            x = block(x, self.rope_cos, self.rope_sin)
        pred = self.out(rmsnorm(x, self.norm_f_weight, self.cfg.norm_eps)).float()
        return F.mse_loss(pred, target.float())
