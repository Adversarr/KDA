"""Eager reference: verbatim ``language/feedforward.py::fc1_gelu`` from the example's user repo."""

import torch
import torch.nn.functional as F


def fc1_gelu_ref(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    z = F.linear(x, weight.to(x.dtype), bias.to(x.dtype))
    return F.gelu(z, approximate="tanh").to(x.dtype)
