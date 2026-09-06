"""A normalized encoder with both wide and per-head normalization sites."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .ops import l2_norm_scale


class NormalizedEncoder(nn.Module):
    """Predict continuous token features after normalized self-attention."""
    def __init__(self, width: int = 2048, head_dim: int = 128, eps: float = 1e-6):
        super().__init__()
        self.width, self.head_dim, self.eps = width, head_dim, eps
        self.input = nn.Linear(64, width)
        self.hidden_scale = nn.Parameter(torch.full((width,), math.sqrt(width)))
        self.query_scale = nn.Parameter(torch.full((head_dim,), math.sqrt(head_dim)))
        self.key_scale = nn.Parameter(torch.full((head_dim,), math.sqrt(head_dim)))
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.output = nn.Linear(width, 32)

    def forward(self, features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        h = l2_norm_scale(self.input(features), self.hidden_scale, self.eps)
        b, s, d = h.shape
        q, k, v = self.qkv(h).view(b, s, 3, d // self.head_dim, self.head_dim).unbind(2)
        q = l2_norm_scale(q, self.query_scale, self.eps).transpose(1, 2)
        k = l2_norm_scale(k, self.key_scale, self.eps).transpose(1, 2)
        v = v.transpose(1, 2)
        mixed = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, s, d)
        return F.mse_loss(self.output(h + mixed).float(), targets)
