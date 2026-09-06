"""feedforward components for the local training model."""


import torch
import torch.nn as nn
import torch.nn.functional as F

from language.config import ModelConfig


def fc1_gelu(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """``gelu_tanh(x @ weight.T + bias)`` in ``x.dtype``.

    ``x: (..., K)`` is an activation (bf16 under autocast); ``weight: (N, K)`` is the
    ``nn.Linear`` parameter and ``bias: (N,)`` is kept in fp32.
    """
    z = F.linear(x, weight.to(x.dtype), bias.to(x.dtype))
    return F.gelu(z, approximate="tanh").to(x.dtype)


class MLP(nn.Module):
    """``fc2(gelu(fc1(y) + b1))``: the GPT-2 MLP."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden = cfg.mlp_ratio * cfg.d_model
        self.fc1 = nn.Linear(cfg.d_model, hidden, bias=False)
        self.fc1_bias = nn.Parameter(torch.zeros(hidden))  # fp32 even under autocast
        self.fc2 = nn.Linear(hidden, cfg.d_model, bias=False)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.fc2(fc1_gelu(y, self.fc1.weight, self.fc1_bias))
