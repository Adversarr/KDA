"""mlp components for the local training model."""


import torch
import torch.nn as nn
import torch.nn.functional as F

from models.config import ModelConfig


def mlp_fc1_gelu(x: torch.Tensor, w1: torch.Tensor, b1: torch.Tensor) -> torch.Tensor:
    """``x: (B, N, D)`` -> ``(B, N, M)``: the first MLP GEMM with bias and the exact (erf) GELU."""
    return F.gelu(F.linear(x, w1, b1))


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden = int(cfg.dim * cfg.mlp_ratio)
        self.fc1 = nn.Linear(cfg.dim, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, cfg.dim, bias=True)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.fc2(mlp_fc1_gelu(y, self.fc1.weight, self.fc1.bias))
