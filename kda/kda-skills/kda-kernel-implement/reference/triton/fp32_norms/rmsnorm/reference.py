"""Eager RMSNorm reference: all math in fp32, one cast back to ``x.dtype`` at the end.

Frameworks differ in where the cast happens. HF Qwen/LLaMA cast ``xhat`` back to the input
dtype *before* multiplying by the weight (``weight * xhat.to(input_dtype)``); this reference
multiplies in fp32. When extracting a user's op into ``_eager.py`` keep their order exactly.
"""

import torch


def rmsnorm_ref(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    xf = x.float()
    rstd = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (xf * rstd * w.float()).to(x.dtype)


__all__ = ["rmsnorm_ref"]
