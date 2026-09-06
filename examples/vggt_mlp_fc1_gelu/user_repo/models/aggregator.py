"""aggregator components for the local training model."""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.config import ModelConfig
from models.mlp import MLP
from models.attention import SelfAttention
from models.embedding import expand_first_vs_other
from models.embedding import make_positions


class MiniVGGT(nn.Module):
    """Camera + register tokens, ``depth`` (frame, global) block pairs, a linear head on the patch tokens."""

    def __init__(self, cfg: ModelConfig, out_channels: int = 64):
        super().__init__()
        self.cfg = cfg
        self.camera_token = nn.Parameter(torch.empty(1, 2, 1, cfg.dim))
        self.register_token = nn.Parameter(torch.empty(1, 2, cfg.num_register_tokens, cfg.dim))
        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)
        self.norm_in_w = nn.Parameter(torch.ones(cfg.dim))
        self.norm_in_b = nn.Parameter(torch.zeros(cfg.dim))
        self.frame_blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.depth))
        self.global_blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.depth))
        self.out = nn.Linear(cfg.dim, out_channels)
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(
        self,
        patch_tokens: torch.Tensor,
        target: torch.Tensor,
        patch_valid: torch.Tensor,
        frame_valid: torch.Tensor,
        compute_dtype: torch.dtype,
    ) -> torch.Tensor:
        """``patch_tokens: (B, S, P, D)`` fp32, ``target: (B, S, P, C)``, ``patch_valid: (B, S, P)``, ``frame_valid: (B, S)`` -> masked MSE."""
        cfg = self.cfg
        b, s, p, d = patch_tokens.shape
        p0 = cfg.patch_start_index
        camera = expand_first_vs_other(self.camera_token, b, s)
        registers = expand_first_vs_other(self.register_token, b, s)
        x = torch.cat((camera, registers, patch_tokens), dim=2)  # (B, S, T, D) fp32 residual stream
        t = x.shape[2]

        positions = make_positions(b, s, cfg.grid, p0, x.device)
        special_valid = frame_valid[:, :, None].expand(b, s, p0)
        patch_valid = patch_valid & frame_valid[:, :, None]
        valid = torch.cat((special_valid, patch_valid), dim=2)  # (B, S, T)

        x = x.masked_fill(~valid.unsqueeze(-1), 0.0)
        y = F.layer_norm(x, (d,), self.norm_in_w, self.norm_in_b, cfg.norm_eps).to(compute_dtype)
        for frame_block, global_block in zip(self.frame_blocks, self.global_blocks):
            # Frame block: views fold into the batch, every view attends inside itself.
            x, y = frame_block(x.view(b * s, t, d), y.view(b * s, t, d), positions.view(b * s, t, 2), valid.view(b * s, t))
            # Global block: every token of every view in one sequence.
            x, y = global_block(x.view(b, s * t, d), y.view(b, s * t, d), positions.view(b, s * t, 2), valid.view(b, s * t))
        y = y.view(b, s, t, d)

        pred = self.out(y[:, :, p0:]).float()
        w = patch_valid.unsqueeze(-1).float()
        return ((pred - target.float()) ** 2 * w).sum() / (w.sum() * pred.shape[-1])


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
