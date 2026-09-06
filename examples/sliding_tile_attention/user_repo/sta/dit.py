"""dit components for the local training model."""


import torch
import torch.nn as nn
import torch.nn.functional as F

from sta.config import ModelConfig
from sta.attention import STAttention
from sta.layout import tile_order


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden = cfg.mlp_ratio * cfg.hidden_size
        self.fc1 = nn.Linear(cfg.hidden_size, hidden)
        self.fc2 = nn.Linear(hidden, cfg.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (y * weight.float()).to(x.dtype)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.eps = cfg.norm_eps
        self.norm1_weight = nn.Parameter(torch.ones(cfg.hidden_size))
        self.norm2_weight = nn.Parameter(torch.ones(cfg.hidden_size))
        self.attn = STAttention(cfg)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(rmsnorm(x, self.norm1_weight, self.eps))
        return x + self.mlp(rmsnorm(x, self.norm2_weight, self.eps))


class MiniSTA(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Linear(cfg.in_channels, cfg.hidden_size)
        self.text_embed = nn.Linear(cfg.text_channels, cfg.hidden_size)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm_f_weight = nn.Parameter(torch.ones(cfg.hidden_size))
        self.out = nn.Linear(cfg.hidden_size, cfg.in_channels)
        perm = tile_order(cfg.canvas, cfg.tile, torch.device("cpu"))
        self.register_buffer("perm", perm, persistent=False)
        self.register_buffer("inv_perm", torch.argsort(perm), persistent=False)
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, latents: torch.Tensor, text: torch.Tensor, target: torch.Tensor, compute_dtype: torch.dtype) -> torch.Tensor:
        """``latents, target: (B, T*H*W, C)`` in raster order, ``text: (B, text_len, C_text)`` -> MSE loss on the video tokens."""
        vid = self.embed(latents)[:, self.perm]  # raster -> tile order
        x = torch.cat([vid, self.text_embed(text)], dim=1).to(compute_dtype)
        for block in self.blocks:
            x = block(x)
        x = x[:, : self.cfg.video_len][:, self.inv_perm]  # back to raster order
        pred = self.out(rmsnorm(x, self.norm_f_weight, self.cfg.norm_eps)).float()
        return F.mse_loss(pred, target.float())
