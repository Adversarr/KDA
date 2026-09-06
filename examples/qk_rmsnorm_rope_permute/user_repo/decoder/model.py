"""model components for the local training model."""


import torch
import torch.nn as nn
import torch.nn.functional as F

from decoder.config import ModelConfig
from decoder.attention import Attention
from decoder.attention import rmsnorm
from decoder.attention import rope_tables


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden = cfg.mlp_ratio * cfg.d_model
        self.fc1 = nn.Linear(cfg.d_model, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.silu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.eps = cfg.norm_eps
        self.norm1_weight = nn.Parameter(torch.ones(cfg.d_model))
        self.norm2_weight = nn.Parameter(torch.ones(cfg.d_model))
        self.attn = Attention(cfg)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(rmsnorm(x, self.norm1_weight, self.eps), cos, sin)
        return x + self.mlp(rmsnorm(x, self.norm2_weight, self.eps))


class MiniLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm_f_weight = nn.Parameter(torch.ones(cfg.d_model))
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        cos, sin = rope_tables(cfg.seq_len, cfg.head_dim, cfg.rope_base, torch.device("cpu"))
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor, compute_dtype: torch.dtype) -> torch.Tensor:
        x = self.embed(tokens).to(compute_dtype)
        for block in self.blocks:
            x = block(x, self.rope_cos, self.rope_sin)
        logits = self.lm_head(rmsnorm(x, self.norm_f_weight, self.cfg.norm_eps)).float()
        return F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.reshape(-1))
