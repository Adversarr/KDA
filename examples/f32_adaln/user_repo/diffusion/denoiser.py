"""denoiser components for the local training model."""


import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.settings import ModelConfig
from diffusion.layers import Block
from diffusion.layers import FinalLayer


class MiniDiT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.patch_embed = nn.Linear(cfg.in_channels, cfg.d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, cfg.n_tokens, cfg.d_model))
        self.t_embed = nn.Sequential(nn.Linear(256, cfg.d_cond), nn.SiLU(), nn.Linear(cfg.d_cond, cfg.d_cond))
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.final = FinalLayer(cfg)
        self.apply(self._init)
        nn.init.normal_(self.pos_embed, std=0.02)
        for block in self.blocks:  # adaLN-Zero: modulation starts at zero, blocks start as identity
            nn.init.zeros_(block.modulation.weight)
            nn.init.zeros_(block.modulation.bias)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int = 256) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(-torch.log(torch.tensor(10000.0, device=t.device)) * torch.arange(half, device=t.device) / half)
        args = t.float()[:, None] * freqs[None, :]
        return torch.cat([args.cos(), args.sin()], dim=-1)

    def forward(self, latents: torch.Tensor, t: torch.Tensor, target: torch.Tensor, compute_dtype: torch.dtype) -> torch.Tensor:
        """``latents, target: (B, N, C)``, ``t: (B,)`` -> MSE between the predicted and the target noise."""
        with torch.autocast("cuda", enabled=False):
            c = self.t_embed(self.timestep_embedding(t))  # (B, d_cond) fp32
        x = (self.patch_embed(latents) + self.pos_embed).to(compute_dtype)
        for block in self.blocks:
            x = block(x, c)
        pred = self.final(x, c).float()
        return F.mse_loss(pred, target.float())
