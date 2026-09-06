"""attention components for the local training model."""


import torch
import torch.nn as nn

from sta.config import ModelConfig
from sta.masks import sta_tile_mask
from sta.masks import sta_token_mask


def sliding_tile_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, tile_mask: torch.Tensor, tile_size: int, text_len: int
) -> torch.Tensor:
    """``q, k, v: (B, H, S, D)`` in tile order (video tokens, then ``text_len`` text tokens) -> ``(B, H, S, D)``.

    softmax(q k^T / sqrt(D), masked by ``sta_token_mask``) v with the softmax in fp32. ``tile_mask``
    is ``(H, NT, NT)`` bool with ``S = NT * tile_size + text_len``; nothing else is stored for the mask.
    """
    d = q.shape[-1]
    scores = (q @ k.transpose(-2, -1)).float() * (d**-0.5)
    scores = scores.masked_fill(~sta_token_mask(tile_mask, tile_size, text_len)[None], float("-inf"))
    probs = torch.softmax(scores, dim=-1).to(v.dtype)
    return probs @ v


class STAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.h, self.d = cfg.num_heads, cfg.head_dim
        assert len(cfg.windows) == self.h, "one window per head"
        self.tile_size, self.text_len = cfg.tile_size, cfg.text_len
        inner = self.h * self.d
        self.to_qkv = nn.Linear(cfg.hidden_size, 3 * inner, bias=True)
        self.to_out = nn.Linear(inner, cfg.hidden_size, bias=True)
        self.register_buffer("tile_mask", sta_tile_mask(cfg.tiles, cfg.windows, torch.device("cpu")), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        q, k, v = self.to_qkv(x).view(b, s, 3, self.h, self.d).permute(2, 0, 3, 1, 4)  # each (B, H, S, D)
        o = sliding_tile_attention(q, k, v, self.tile_mask, self.tile_size, self.text_len)
        return self.to_out(o.transpose(1, 2).reshape(b, s, self.h * self.d))
