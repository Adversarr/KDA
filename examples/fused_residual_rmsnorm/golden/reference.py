"""Eager reference: verbatim ``minilm/model.py::add_rmsnorm`` from the example's user repo."""

from typing import Tuple

import torch


def add_rmsnorm_ref(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    h = residual + x.float()
    rstd = torch.rsqrt(h.pow(2).mean(dim=-1, keepdim=True) + eps)
    y = (h * rstd) * weight.float()
    return h, y.to(x.dtype)
