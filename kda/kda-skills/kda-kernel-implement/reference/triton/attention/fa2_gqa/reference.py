"""Eager reference for full (non-causal) grouped-query attention: what a user's attention block computes."""

from typing import Optional

import torch


def attention_ref(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: Optional[float] = None) -> torch.Tensor:
    """``softmax(q k^T * scale) v`` with ``q: (B, H, S_q, D)``, ``k, v: (B, H_kv, S_kv, D)``, softmax in fp32.

    K and V are expanded to the query heads with ``repeat_interleave`` (query head ``h`` uses
    K,V head ``h // (H // H_kv)``), which is what the model does before ``matmul``. The
    materialised ``(B, H, S_q, S_kv)`` fp32 score plane is the reference's memory cost; a KDA
    ``_eager.py`` may loop this function over heads for long rows (same math).
    """
    D = q.shape[-1]
    scale = D**-0.5 if scale is None else scale
    group = q.shape[1] // k.shape[1]
    if group > 1:
        k = k.repeat_interleave(group, dim=1)
        v = v.repeat_interleave(group, dim=1)
    scores = (q @ k.transpose(-2, -1)).float() * scale
    probs = torch.softmax(scores, dim=-1).to(v.dtype)
    return probs @ v


def attention_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: Optional[float] = None) -> torch.Tensor:
    """``F.scaled_dot_product_attention(enable_gqa=True)``: torch's flash/mem-efficient kernel, the baseline to beat."""
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False, scale=scale, enable_gqa=q.shape[1] != k.shape[1])
