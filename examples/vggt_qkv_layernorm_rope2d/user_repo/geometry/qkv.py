"""qkv components for the local training model."""

from typing import Tuple

import torch



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
