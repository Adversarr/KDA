"""Eager reference: the three ops as a user would write them (Qwen3 q_norm -> rope -> transpose)."""

import torch


def rmsnorm_rope_permute_ref(x: torch.Tensor, w: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    B, S, H, D = x.shape
    xf = x.float()
    rstd = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    n = (xf * rstd * w.float()).to(x.dtype).float()  # norm output materialised in x.dtype, as eager does
    c, s = cos[None, :S, None, :], sin[None, :S, None, :]
    n1, n2 = n.chunk(2, dim=-1)
    y = torch.cat([n1 * c - n2 * s, n2 * c + n1 * s], dim=-1).to(x.dtype)
    return y.permute(0, 2, 1, 3).contiguous()


__all__ = ["rmsnorm_rope_permute_ref"]


def qk_prep(q, k, q_norm_w, k_norm_w, cos, sin, eps=1e-6):
    """Prepare q and k with shared head normalization and half-pairing RoPE."""
    return rmsnorm_rope_permute_ref(q, q_norm_w, cos, sin, eps), rmsnorm_rope_permute_ref(k, k_norm_w, cos, sin, eps)
