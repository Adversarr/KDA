"""Eager reference for causal multi-head attention: what a user's attention block computes."""

from typing import Optional

import torch


def attention_ref(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: Optional[float] = None) -> torch.Tensor:
    """``softmax(q k^T * scale + causal_mask) v`` with ``q, k, v: (B, H, S, D)``, softmax in fp32.

    The materialised ``(B, H, S, S)`` score plane is the reference's cost: ``B*H*S*S*4`` bytes
    (4 GiB at ``H = 32, S = 8192``), which is why the fused kernel exists. For long rows the
    KDA ``_eager.py`` may loop this function over heads (same math, ``torch.cat`` of the
    results) so the reference fits next to the kernel.
    """
    D = q.shape[-1]
    scale = D**-0.5 if scale is None else scale
    S = q.shape[-2]
    scores = (q @ k.transpose(-2, -1)).float() * scale
    causal = torch.ones(S, S, dtype=torch.bool, device=q.device).tril()
    scores = scores.masked_fill(~causal, float("-inf"))
    probs = torch.softmax(scores, dim=-1).to(v.dtype)
    return probs @ v


def attention_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: Optional[float] = None) -> torch.Tensor:
    """``F.scaled_dot_product_attention(is_causal=True)``: the flash/mem-efficient kernel torch ships, the baseline to beat."""
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)
