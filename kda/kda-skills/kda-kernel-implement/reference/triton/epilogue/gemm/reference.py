"""Eager reference for the GEMM + epilogue snippet: what a user's MLP block computes."""

from typing import Optional

import torch
import torch.nn.functional as F


def gemm_epilogue_ref(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    act: str = "none",
) -> torch.Tensor:
    """``act(x @ w.T + bias)`` the way ``nn.Linear`` + activation runs in eager.

    Every op rounds to the storage dtype (cuBLAS accumulates in fp32 and rounds once; bias and
    activation each round again). The fused kernel rounds once at the store, so it is *closer*
    to the fp32 value than this reference; the test tolerances allow for that.

    ``gelu_tanh`` is what transformer code means by GELU today (GPT-2, BERT, ViT, HunyuanVideo,
    ``nn.GELU(approximate="tanh")``) and the only GELU cuBLASLt / nvmath fuse; ``gelu`` is the
    erf form, kept because ``F.gelu`` defaults to it. They differ by up to 2e-3 absolute, which
    is above a bf16 ulp near 1, so a kernel must implement the one the user's code calls.
    """
    z = F.linear(x, w, bias)
    if act == "gelu":
        z = F.gelu(z)
    elif act == "gelu_tanh":
        z = F.gelu(z, approximate="tanh")
    elif act == "silu":
        z = F.silu(z)
    elif act == "relu":
        z = F.relu(z)
    elif act != "none":
        raise ValueError(f"unknown act {act!r}")
    return z
