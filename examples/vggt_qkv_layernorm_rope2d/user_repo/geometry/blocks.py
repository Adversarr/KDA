"""blocks components for the local training model."""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from geometry.config import ModelConfig
from geometry.mlp import MLP
from geometry.attention import SelfAttention


def layerscale_residual_layernorm(
    x: torch.Tensor,
    branch: torch.Tensor,
    gamma: torch.Tensor,
    norm_w: torch.Tensor,
    norm_b: torch.Tensor,
    eps: float,
    valid: torch.Tensor,
    out_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``x: (B, N, D)`` fp32 residual stream, ``branch: (B, N, D)`` (attention or MLP output, the
    autocast dtype), ``gamma: (D,)`` fp32 LayerScale, ``norm_w, norm_b: (D,)``, ``valid: (B, N)`` bool.

    Returns the new residual ``x + gamma * branch`` (fp32, padded rows set to zero so biases
    never revive them) and its affine LayerNorm in ``out_dtype``, the next branch's input.
    """
    x_new = x + branch.float() * gamma
    x_new = x_new.masked_fill(~valid.unsqueeze(-1), 0.0)
    y = F.layer_norm(x_new, (x_new.shape[-1],), norm_w, norm_b, eps).to(out_dtype)
    return x_new, y


class Block(nn.Module):
    """One pre-norm block, residual-first: ``(x, y) -> (x', y')``.

        x  <- zero_padded(x + ls1 * Attention(y));   y  <- LN2(x)
        x' <- zero_padded(x + ls2 * MLP(y));         y' <- LN_out(x')     (the next block's LN1)
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.eps = cfg.norm_eps
        self.attn = SelfAttention(cfg)
        self.mlp = MLP(cfg)
        self.ls1 = nn.Parameter(torch.full((cfg.dim,), cfg.layer_scale_init))
        self.ls2 = nn.Parameter(torch.full((cfg.dim,), cfg.layer_scale_init))
        self.norm2_w = nn.Parameter(torch.ones(cfg.dim))
        self.norm2_b = nn.Parameter(torch.zeros(cfg.dim))
        self.norm_out_w = nn.Parameter(torch.ones(cfg.dim))
        self.norm_out_b = nn.Parameter(torch.zeros(cfg.dim))

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, positions: torch.Tensor, valid: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        a = self.attn(y, positions, valid)
        x, y = layerscale_residual_layernorm(x, a, self.ls1, self.norm2_w, self.norm2_b, self.eps, valid, y.dtype)
        m = self.mlp(y)
        x, y = layerscale_residual_layernorm(x, m, self.ls2, self.norm_out_w, self.norm_out_b, self.eps, valid, y.dtype)
        return x, y
