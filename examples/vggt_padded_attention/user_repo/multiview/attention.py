"""attention components for the local training model."""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from multiview.config import ModelConfig


def layernorm_heads(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
    """``x: (B, H, N, d)``, ``weight, bias: (d,)``: affine LayerNorm over the head dim of every head, in fp32."""
    xf = x.float()
    mean = xf.mean(dim=-1, keepdim=True)
    var = xf.var(dim=-1, unbiased=False, keepdim=True)
    y = (xf - mean) * torch.rsqrt(var + eps)
    return (y * weight.float() + bias.float()).to(x.dtype)


def rope2d_inv_freq(head_dim: int, base: float) -> torch.Tensor:
    """``(d / 4,)`` fp32: one frequency per rotated pair of one axis half (``axis_dim = d / 2`` channels)."""
    axis_dim = head_dim // 2
    exponent = torch.arange(0, axis_dim, 2, dtype=torch.float32) / axis_dim
    return 1.0 / (base**exponent)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope2d(x: torch.Tensor, positions: torch.Tensor, inv_freq: torch.Tensor) -> torch.Tensor:
    """2-D rotary embedding on ``x: (B, H, N, d)`` from integer ``positions: (B, N, 2)`` = (y, x).

    Channels ``[0, d/2)`` of every head are rotated by the token's y coordinate and channels
    ``[d/2, d)`` by its x coordinate, rotate-half inside each half (``x1, x2 = half.chunk(2)``).
    Angles, cos and sin are computed in fp32 (``angle = pos * inv_freq``, duplicated for the two
    rotate-half quarters); special tokens sit at position ``(0, 0)`` and are left unrotated.
    """
    xf = x.float()
    parts = []
    for part, pos in zip(xf.chunk(2, dim=-1), (positions[..., 0], positions[..., 1])):
        angle = pos.float()[:, None, :, None] * inv_freq  # (B, 1, N, d/4)
        angle = torch.cat((angle, angle), dim=-1)  # (B, 1, N, d/2)
        parts.append(part * angle.cos() + rotate_half(part) * angle.sin())
    return torch.cat(parts, dim=-1).to(x.dtype)


def qkv_prep(
    qkv: torch.Tensor,
    q_norm_w: torch.Tensor,
    q_norm_b: torch.Tensor,
    k_norm_w: torch.Tensor,
    k_norm_b: torch.Tensor,
    positions: torch.Tensor,
    inv_freq: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``qkv: (B, N, 3, H, d)`` (the projection output viewed) -> ``q, k, v`` each ``(B, H, N, d)``.

    q and k get the per-head affine LayerNorm (one ``(d,)`` weight and bias each, shared by all
    heads) followed by the 2-D RoPE; v is only re-laid out. All three come out contiguous and
    head-major, the layout the attention consumes.
    """
    q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
    q = apply_rope2d(layernorm_heads(q, q_norm_w, q_norm_b, eps), positions, inv_freq)
    k = apply_rope2d(layernorm_heads(k, k_norm_w, k_norm_b, eps), positions, inv_freq)
    return q.contiguous(), k.contiguous(), v.contiguous()


def key_valid_from_token_valid(valid: torch.Tensor) -> torch.Tensor:
    """``valid: (B, N)`` bool -> the keys attention may use; a view with no valid token gets key 0 as a harmless dummy."""
    key_valid = valid.clone()
    has_no_key = ~key_valid.any(dim=-1)
    if has_no_key.any():
        key_valid[has_no_key, 0] = True
    return key_valid


def padded_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, key_valid: torch.Tensor) -> torch.Tensor:
    """``q, k, v: (B, H, N, d)``, ``key_valid: (B, N)`` bool (True = this key may be attended) -> ``(B, H, N, d)``.

    Dense, non-causal. Logits, softmax and the PV accumulation are explicit fp32 (autocast is
    disabled inside, exactly as the model's reference path does it); the result is cast back to
    v's dtype. Every query row attends to the same key set of its batch entry, so the mask is a
    function of ``key_valid`` alone and is never stored as an ``(N, N)`` plane. The ``(B, H, N, N)``
    fp32 logits are what make this the memory and time bottleneck of the global block.
    """
    d = q.shape[-1]
    with torch.autocast(device_type=q.device.type, enabled=False):
        logits = torch.matmul(q.float(), k.float().transpose(-2, -1)) * (d**-0.5)
        logits = logits.masked_fill(~key_valid[:, None, None, :], float("-inf"))
        prob = torch.softmax(logits, dim=-1)
        prob = torch.nan_to_num(prob, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.matmul(prob, v.float()).to(v.dtype)


def sdpa_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, key_valid: torch.Tensor) -> torch.Tensor:
    """The library path: SDPA with the boolean key mask broadcast to ``(B, 1, 1, N)``."""
    return F.scaled_dot_product_attention(q, k, v, attn_mask=key_valid[:, None, None, :], is_causal=False)


class SelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.h, self.d = cfg.num_heads, cfg.head_dim
        self.eps = cfg.norm_eps
        self.impl = cfg.attention_impl
        self.qkv = nn.Linear(cfg.dim, 3 * cfg.dim, bias=True)
        self.q_norm_w = nn.Parameter(torch.ones(self.d))
        self.q_norm_b = nn.Parameter(torch.zeros(self.d))
        self.k_norm_w = nn.Parameter(torch.ones(self.d))
        self.k_norm_b = nn.Parameter(torch.zeros(self.d))
        self.register_buffer("inv_freq", rope2d_inv_freq(self.d, cfg.rope_base), persistent=False)
        self.proj = nn.Linear(cfg.dim, cfg.dim, bias=True)

    def forward(self, y: torch.Tensor, positions: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """``y: (B, N, D)`` normalized tokens, ``positions: (B, N, 2)`` int64, ``valid: (B, N)`` bool -> ``(B, N, D)``."""
        b, n, _ = y.shape
        qkv = self.qkv(y).view(b, n, 3, self.h, self.d)
        q, k, v = qkv_prep(qkv, self.q_norm_w, self.q_norm_b, self.k_norm_w, self.k_norm_b, positions, self.inv_freq, self.eps)
        key_valid = key_valid_from_token_valid(valid)
        if self.impl == "eager":
            o = padded_attention(q, k, v, key_valid)
        elif self.impl == "sdpa":
            o = sdpa_attention(q, k, v, key_valid)
        else:
            raise ValueError(f"unknown attention_impl {self.impl!r}")
        out = self.proj(o.transpose(1, 2).reshape(b, n, self.h * self.d))
        # Padded queries produce finite garbage; zero them after the projection.
        return out.masked_fill(~valid.unsqueeze(-1), 0)
