"""model components for the local training model."""


import torch
import torch.nn as nn
import torch.nn.functional as F

from h3.config import ModelConfig
from h3.attention import H3Attention
from h3.positions import mm_rope_tables


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden = cfg.mlp_ratio * cfg.hidden_size
        self.fc1 = nn.Linear(cfg.hidden_size, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, cfg.hidden_size, bias=False)

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
        self.attn = H3Attention(cfg)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(rmsnorm(x, self.norm1_weight, self.eps), cos, sin)
        return x + self.mlp(rmsnorm(x, self.norm2_weight, self.eps))


class MiniH3(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Linear(cfg.in_channels, cfg.hidden_size)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm_f_weight = nn.Parameter(torch.ones(cfg.hidden_size))
        self.out = nn.Linear(cfg.hidden_size, cfg.in_channels)
        cos, sin = mm_rope_tables(cfg.grid, cfg.rope_freq_dim, cfg.rope_theta, torch.device("cpu"))
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, latents: torch.Tensor, target: torch.Tensor, compute_dtype: torch.dtype) -> torch.Tensor:
        """``latents, target: (B, S, C)`` with ``S = t * h * w`` packed tokens -> MSE loss."""
        x = self.embed(latents).to(compute_dtype)
        for block in self.blocks:
            x = block(x, self.rope_cos, self.rope_sin)
        pred = self.out(rmsnorm(x, self.norm_f_weight, self.cfg.norm_eps)).float()
        return F.mse_loss(pred, target.float())
