"""A HunyuanVideo-style DiT block with Sliding Tile Attention (STA) over a tiled video sequence plus text.

STA (Zhang et al., "Fast Video Generation with Sliding Tile Attention") keeps the video tokens
in **tile order**: the `(t, h, w)` latent grid is cut into tiles of `tile = (6, 8, 8)` = 384
tokens, the tokens of one tile are contiguous, and tiles are enumerated t-major over the
`(t // 6, h // 8, w // 8)` tile grid. Text tokens follow the video tokens. Each head has its own
window `(kt, kh, kw)` in tiles: a query tile attends to the `kt x kh x kw` tiles centred on it
(the centre is clamped so the window stays inside the tile grid, which is what makes the window
a fixed number of tiles for every query), plus every text token. Text queries attend to every
token. The mask is therefore **block-sparse at tile granularity and dense inside a tile pair**;
`sta_tile_mask` builds it once per model as an `(H, NT, NT)` bool tensor.

`sliding_tile_attention` is the function the fused kernel replaces: it expands the tile mask
to a token mask and materialises the full `(B, H, S, S)` fp32 score plane, which is the memory
and time bottleneck at video lengths (HunyuanVideo: 115200 video tokens + 256 text, 24 heads).
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


def tile_order(canvas: Tuple[int, int, int], tile: Tuple[int, int, int], device: torch.device) -> torch.Tensor:
    """``(video_len,)`` long: ``perm[i]`` is the raster index of the ``i``-th token in tile order."""
    T, H, W = canvas
    tt, th, tw = tile
    idx = torch.arange(T * H * W, device=device).view(T, H, W)
    idx = idx.view(T // tt, tt, H // th, th, W // tw, tw).permute(0, 2, 4, 1, 3, 5)  # (nT, nH, nW, tt, th, tw)
    return idx.reshape(-1)


def sta_tile_mask(
    tiles: Tuple[int, int, int], windows: List[Tuple[int, int, int]], device: torch.device
) -> torch.Tensor:
    """``(H, NT, NT)`` bool: ``mask[h, qt, kt]`` is True when key tile ``kt`` is inside head ``h``'s window around query tile ``qt``.

    Per axis with ``n`` tiles and window ``k`` (odd, ``k <= n``): ``centre = clamp(q, k // 2, n - 1 - k // 2)``
    and the key tile is allowed when ``|kt - centre| <= k // 2``. Tiles are enumerated t-major.
    """
    coords = torch.stack(torch.meshgrid(*[torch.arange(n, device=device) for n in tiles], indexing="ij"), dim=-1).view(-1, 3)
    masks = []
    for k in windows:
        allowed = torch.ones(coords.shape[0], coords.shape[0], dtype=torch.bool, device=device)
        for axis, (n, kk) in enumerate(zip(tiles, k)):
            assert kk % 2 == 1 and kk <= n, f"window {k} does not fit the tile grid {tiles}"
            q = coords[:, axis]
            centre = q.clamp(kk // 2, n - 1 - kk // 2)
            allowed &= (coords[None, :, axis] - centre[:, None]).abs() <= kk // 2
        masks.append(allowed)
    return torch.stack(masks)


def sta_token_mask(tile_mask: torch.Tensor, tile_size: int, text_len: int) -> torch.Tensor:
    """``(H, S, S)`` bool from the ``(H, NT, NT)`` tile mask: dense inside a tile pair, text keys for every query, text queries see all."""
    H, NT, _ = tile_mask.shape
    video_len = NT * tile_size
    S = video_len + text_len
    m = torch.ones(H, S, S, dtype=torch.bool, device=tile_mask.device)
    vid = tile_mask.repeat_interleave(tile_size, dim=1).repeat_interleave(tile_size, dim=2)
    m[:, :video_len, :video_len] = vid
    return m


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


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden = cfg.mlp_ratio * cfg.hidden_size
        self.fc1 = nn.Linear(cfg.hidden_size, hidden)
        self.fc2 = nn.Linear(hidden, cfg.hidden_size)

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
        self.attn = STAttention(cfg)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(rmsnorm(x, self.norm1_weight, self.eps))
        return x + self.mlp(rmsnorm(x, self.norm2_weight, self.eps))


class MiniSTA(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Linear(cfg.in_channels, cfg.hidden_size)
        self.text_embed = nn.Linear(cfg.text_channels, cfg.hidden_size)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm_f_weight = nn.Parameter(torch.ones(cfg.hidden_size))
        self.out = nn.Linear(cfg.hidden_size, cfg.in_channels)
        perm = tile_order(cfg.canvas, cfg.tile, torch.device("cpu"))
        self.register_buffer("perm", perm, persistent=False)
        self.register_buffer("inv_perm", torch.argsort(perm), persistent=False)
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, latents: torch.Tensor, text: torch.Tensor, target: torch.Tensor, compute_dtype: torch.dtype) -> torch.Tensor:
        """``latents, target: (B, T*H*W, C)`` in raster order, ``text: (B, text_len, C_text)`` -> MSE loss on the video tokens."""
        vid = self.embed(latents)[:, self.perm]  # raster -> tile order
        x = torch.cat([vid, self.text_embed(text)], dim=1).to(compute_dtype)
        for block in self.blocks:
            x = block(x)
        x = x[:, : self.cfg.video_len][:, self.inv_perm]  # back to raster order
        pred = self.out(rmsnorm(x, self.norm_f_weight, self.cfg.norm_eps)).float()
        return F.mse_loss(pred, target.float())
