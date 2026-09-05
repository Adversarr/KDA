"""Eager reference: verbatim ``minilm/model.py`` (``rmsnorm_per_head``, ``apply_rope``, ``qk_prep``) from the user repo."""

from typing import Tuple

import torch


def rmsnorm_per_head(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (y * weight.float()).to(x.dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, interleaved: bool = True) -> torch.Tensor:
    S = x.shape[1]
    c, s = cos[None, :S, None, :], sin[None, :S, None, :]
    xf = x.float()
    if interleaved:
        x1, x2 = xf[..., 0::2], xf[..., 1::2]
        return torch.stack((x1 * c - x2 * s, x2 * c + x1 * s), dim=-1).flatten(-2).to(x.dtype)
    x1, x2 = xf.chunk(2, dim=-1)
    return torch.cat((x1 * c - x2 * s, x2 * c + x1 * s), dim=-1).to(x.dtype)


def qk_prep_ref(
    q: torch.Tensor, k: torch.Tensor, q_norm_w: torch.Tensor, k_norm_w: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, eps: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    q = rmsnorm_per_head(q, q_norm_w, eps)
    k = rmsnorm_per_head(k, k_norm_w, eps)
    q = apply_rope(q, cos, sin, interleaved=True)
    k = apply_rope(k, cos, sin, interleaved=True)
    return q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous()
