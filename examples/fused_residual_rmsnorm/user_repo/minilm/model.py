"""A minimal pre-norm transformer language model.

The residual stream is kept in fp32 for the whole depth of the network; every sublayer reads an
RMS-normalised, low-precision view of it. ``add_rmsnorm`` is the function each block calls
twice (before attention and before the MLP) and the final norm calls once.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


def add_rmsnorm(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Residual add in fp32 followed by RMSNorm.

    Returns the new fp32 residual stream ``h = residual + x`` and its normalised view
    ``y = rmsnorm(h) * weight`` cast back to ``x.dtype``, which feeds the next sublayer.
    """
    h = residual + x.float()
    rstd = torch.rsqrt(h.pow(2).mean(dim=-1, keepdim=True) + eps)
    y = (h * rstd) * weight.float()
    return h, y.to(x.dtype)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, d = x.shape
        q, k, v = self.qkv(x).view(b, s, 3, self.n_heads, d // self.n_heads).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(o.transpose(1, 2).reshape(b, s, d))


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden = cfg.mlp_ratio * cfg.d_model
        self.fc1 = nn.Linear(cfg.d_model, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.silu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.eps = cfg.norm_eps
        self.norm1_weight = nn.Parameter(torch.ones(cfg.d_model))
        self.norm2_weight = nn.Parameter(torch.ones(cfg.d_model))
        self.attn = Attention(cfg)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """``x`` is the previous sublayer's output (low precision), ``residual`` the fp32 stream."""
        residual, y = add_rmsnorm(x, residual, self.norm1_weight, self.eps)
        x = self.attn(y)
        residual, y = add_rmsnorm(x, residual, self.norm2_weight, self.eps)
        x = self.mlp(y)
        return x, residual


class MiniLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm_f_weight = nn.Parameter(torch.ones(cfg.d_model))
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor, compute_dtype: torch.dtype) -> torch.Tensor:
        x = self.embed(tokens).to(compute_dtype)
        residual = torch.zeros_like(x, dtype=torch.float32)
        for block in self.blocks:
            x, residual = block(x, residual)
        _, y = add_rmsnorm(x, residual, self.norm_f_weight, self.cfg.norm_eps)
        logits = self.lm_head(y).float()
        return F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.reshape(-1))
