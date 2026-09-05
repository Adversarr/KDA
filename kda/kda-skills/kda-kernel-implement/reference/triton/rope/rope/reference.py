"""Eager RoPE reference for both pairings; fp32 math, one cast back to ``x.dtype``.

HF-style code passes ``cos``/``sin`` of shape ``(S, D)`` built as ``cat(freqs, freqs)``; the
``(S, D/2)`` tables here are their first half. Match the user's convention in ``_eager.py``.
"""

from typing import Optional

import torch


def rope_ref(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, positions: Optional[torch.Tensor] = None, interleaved: bool = False) -> torch.Tensor:
    B, S, H, D = x.shape
    if positions is not None:
        c, s = cos[positions], sin[positions]  # (B, S, D/2)
        c, s = c[:, :, None, :], s[:, :, None, :]
    else:
        c, s = cos[None, :S, None, :], sin[None, :S, None, :]
    xf = x.float()
    if interleaved:
        x1, x2 = xf[..., 0::2], xf[..., 1::2]
        y = torch.stack([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1).flatten(-2)
    else:
        x1, x2 = xf.chunk(2, dim=-1)
        y = torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)
    return y.to(x.dtype)


__all__ = ["rope_ref"]
