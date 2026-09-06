"""attention components for the local training model."""


import torch
import torch.nn as nn
import torch.nn.functional as F

from geometry.config import ModelConfig
from geometry.qkv import qkv_prep
from geometry.qkv import rope2d_inv_freq


def key_valid_from_token_valid(valid: torch.Tensor) -> torch.Tensor:
    """``valid: (B, N)`` bool -> the keys attention may use; a view with no valid token gets key 0 as a harmless dummy."""
    key_valid = valid.clone()
    has_no_key = ~key_valid.any(dim=-1)
    if has_no_key.any():
        key_valid[has_no_key, 0] = True
    return key_valid


def padded_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, key_valid: torch.Tensor) -> torch.Tensor:
    """``q, k, v: (B, H, N, d)``, ``key_valid: (B, N)`` bool (True = this key may be attended) -> ``(B, H, N, d)``.

    Dense, non-causal. Logits, softmax and the PV accumulation are explicit fp32 (autocast is
    disabled inside, exactly as the model's reference path does it); the result is cast back to
    v's dtype. Every query row attends to the same key set of its batch entry, so the mask is a
    function of ``key_valid`` alone and is never stored as an ``(N, N)`` plane. The ``(B, H, N, N)``
    fp32 logits are what make this the memory and time bottleneck of the global block.
    """
    d = q.shape[-1]
    with torch.autocast(device_type=q.device.type, enabled=False):
        logits = torch.matmul(q.float(), k.float().transpose(-2, -1)) * (d**-0.5)
        logits = logits.masked_fill(~key_valid[:, None, None, :], float("-inf"))
        prob = torch.softmax(logits, dim=-1)
        prob = torch.nan_to_num(prob, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.matmul(prob, v.float()).to(v.dtype)


def sdpa_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, key_valid: torch.Tensor) -> torch.Tensor:
    """The library path: SDPA with the boolean key mask broadcast to ``(B, 1, 1, N)``."""
    return F.scaled_dot_product_attention(q, k, v, attn_mask=key_valid[:, None, None, :], is_causal=False)


class SelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.h, self.d = cfg.num_heads, cfg.head_dim
        self.eps = cfg.norm_eps
        self.impl = cfg.attention_impl
        self.qkv = nn.Linear(cfg.dim, 3 * cfg.dim, bias=True)
        self.q_norm_w = nn.Parameter(torch.ones(self.d))
        self.q_norm_b = nn.Parameter(torch.zeros(self.d))
        self.k_norm_w = nn.Parameter(torch.ones(self.d))
        self.k_norm_b = nn.Parameter(torch.zeros(self.d))
        self.register_buffer("inv_freq", rope2d_inv_freq(self.d, cfg.rope_base), persistent=False)
        self.proj = nn.Linear(cfg.dim, cfg.dim, bias=True)

    def forward(self, y: torch.Tensor, positions: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """``y: (B, N, D)`` normalized tokens, ``positions: (B, N, 2)`` int64, ``valid: (B, N)`` bool -> ``(B, N, D)``."""
        b, n, _ = y.shape
        qkv = self.qkv(y).view(b, n, 3, self.h, self.d)
        q, k, v = qkv_prep(qkv, self.q_norm_w, self.q_norm_b, self.k_norm_w, self.k_norm_b, positions, self.inv_freq, self.eps)
        key_valid = key_valid_from_token_valid(valid)
        if self.impl == "eager":
            o = padded_attention(q, k, v, key_valid)
        elif self.impl == "sdpa":
            o = sdpa_attention(q, k, v, key_valid)
        else:
            raise ValueError(f"unknown attention_impl {self.impl!r}")
        out = self.proj(o.transpose(1, 2).reshape(b, n, self.h * self.d))
        # Padded queries produce finite garbage; zero them after the projection.
        return out.masked_fill(~valid.unsqueeze(-1), 0)
