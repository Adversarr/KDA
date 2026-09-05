"""Eager references: verbatim ``minista/model.py`` mask builders and ``sliding_tile_attention``, plus the SDPA baseline."""

from typing import List, Tuple

import torch
import torch.nn.functional as F


def sta_tile_mask(tiles: Tuple[int, int, int], windows: List[Tuple[int, int, int]], device: torch.device) -> torch.Tensor:
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
    H, NT, _ = tile_mask.shape
    video_len = NT * tile_size
    S = video_len + text_len
    m = torch.ones(H, S, S, dtype=torch.bool, device=tile_mask.device)
    vid = tile_mask.repeat_interleave(tile_size, dim=1).repeat_interleave(tile_size, dim=2)
    m[:, :video_len, :video_len] = vid
    return m


def sliding_tile_attention_ref(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, tile_mask: torch.Tensor, tile_size: int, text_len: int
) -> torch.Tensor:
    """The user's function: materialised fp32 scores, masked softmax, ``probs @ v``."""
    d = q.shape[-1]
    scores = (q @ k.transpose(-2, -1)).float() * (d**-0.5)
    scores = scores.masked_fill(~sta_token_mask(tile_mask, tile_size, text_len)[None], float("-inf"))
    probs = torch.softmax(scores, dim=-1).to(v.dtype)
    return probs @ v


def sliding_tile_attention_sdpa(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, tile_mask: torch.Tensor, tile_size: int, text_len: int
) -> torch.Tensor:
    """The strongest torch baseline: SDPA with the boolean token mask (the efficient-attention kernel; FA2 refuses a mask)."""
    return F.scaled_dot_product_attention(q, k, v, attn_mask=sta_token_mask(tile_mask, tile_size, text_len)[None])
