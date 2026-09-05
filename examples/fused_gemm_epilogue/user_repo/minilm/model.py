"""A minimal pre-norm transformer language model.

The MLP's first projection is followed by a bias and a tanh-GELU (``hidden = gelu(fc1(y) +
b1)``, the GPT-2 MLP). ``fc1_gelu`` is that GEMM + epilogue; it is the function the fused kernel
replaces, and the largest tensor the model materialises (``4 * d_model`` wide).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


def fc1_gelu(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """``gelu_tanh(x @ weight.T + bias)`` in ``x.dtype``.

    ``x: (..., K)`` is an activation (bf16 under autocast); ``weight: (N, K)`` is the
    ``nn.Linear`` parameter and ``bias: (N,)`` is kept in fp32.
    """
    z = F.linear(x, weight.to(x.dtype), bias.to(x.dtype))
    return F.gelu(z, approximate="tanh").to(x.dtype)


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (y * weight.float()).to(x.dtype)


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
    """``fc2(gelu(fc1(y) + b1))``: the GPT-2 MLP."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden = cfg.mlp_ratio * cfg.d_model
        self.fc1 = nn.Linear(cfg.d_model, hidden, bias=False)
        self.fc1_bias = nn.Parameter(torch.zeros(hidden))  # fp32 even under autocast
        self.fc2 = nn.Linear(hidden, cfg.d_model, bias=False)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.fc2(fc1_gelu(y, self.fc1.weight, self.fc1_bias))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.eps = cfg.norm_eps
        self.norm1_weight = nn.Parameter(torch.ones(cfg.d_model))
        self.norm2_weight = nn.Parameter(torch.ones(cfg.d_model))
        self.attn = Attention(cfg)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(rmsnorm(x, self.norm1_weight, self.eps))
        return x + self.mlp(rmsnorm(x, self.norm2_weight, self.eps))


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
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(rmsnorm(x, self.norm_f_weight, self.cfg.norm_eps)).float()
        return F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.reshape(-1))
