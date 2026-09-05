"""Eager residual add: fp32 accumulation, one cast to the output dtype."""

from typing import Optional

import torch


def residual_add_ref(x: torch.Tensor, residual: torch.Tensor, out_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    return (x.float() + residual.float()).to(out_dtype or x.dtype)


__all__ = ["residual_add_ref"]
