"""layers components for the local training model."""


import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.settings import ModelConfig


def adaln(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor, eps: float) -> torch.Tensor:
    """``x: (B, N, D)`` bf16; ``scale, shift: (B, D)`` fp32 from the conditioning MLP. Everything in fp32 until the cast."""
    h = x.float()
    mean = h.mean(-1, keepdim=True)
    var = (h - mean).pow(2).mean(-1, keepdim=True)
    n = (h - mean) * torch.rsqrt(var + eps)
    return (n * (1 + scale[:, None, :]) + shift[:, None, :]).to(x.dtype)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.h, self.d = cfg.n_heads, cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        q, k, v = self.qkv(x).view(b, n, 3, self.h, self.d).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(q, k, v)  # dense, non-causal
        return self.proj(o.transpose(1, 2).reshape(b, n, self.h * self.d))


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden = cfg.mlp_ratio * cfg.d_model
        self.fc1 = nn.Linear(cfg.d_model, hidden)
        self.fc2 = nn.Linear(hidden, cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class Block(nn.Module):
    """adaLN-Zero block: `x += gate * branch(adaln(x, scale, shift))` for attention and MLP."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.eps = cfg.norm_eps
        self.attn = Attention(cfg)
        self.mlp = MLP(cfg)
        self.modulation = nn.Linear(cfg.d_cond, 6 * cfg.d_model)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # the conditioning path stays fp32: the modulation runs with autocast off, so
        # shift/scale/gate are fp32 (B, D) tensors whatever the activation dtype is
        with torch.autocast("cuda", enabled=False):
            shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.modulation(F.silu(c.float())).chunk(6, dim=-1)
        x = x + gate_a[:, None, :].to(x.dtype) * self.attn(adaln(x, scale_a, shift_a, self.eps))
        return x + gate_m[:, None, :].to(x.dtype) * self.mlp(adaln(x, scale_m, shift_m, self.eps))


class FinalLayer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.eps = cfg.norm_eps
        self.modulation = nn.Linear(cfg.d_cond, 2 * cfg.d_model)
        self.out = nn.Linear(cfg.d_model, cfg.in_channels)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        with torch.autocast("cuda", enabled=False):
            shift, scale = self.modulation(F.silu(c.float())).chunk(2, dim=-1)
        # the SiLU after the final adaLN is the optional fused epilogue the request mentions
        return self.out(F.silu(adaln(x, scale, shift, self.eps)))
