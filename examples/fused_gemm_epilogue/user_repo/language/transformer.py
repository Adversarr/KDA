"""transformer components for the local training model."""


import torch
import torch.nn as nn
import torch.nn.functional as F

from language.config import ModelConfig
from language.feedforward import MLP


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
