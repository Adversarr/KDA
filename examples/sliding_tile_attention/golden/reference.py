"""Eager references: verbatim ``sta/attention.py`` mask builders and ``sliding_tile_attention``, plus the SDPA baseline."""

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
    if q.shape[2] > 16384:
        return _BoundedReference.apply(q, k, v, tile_mask, tile_size, text_len,
                                       _chunk_math, _chunk_adjoint)
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


COMPILE_DESCRIPTION = "dynamic 512-query STA forward/adjoint chunks; Python loops; fp32 dK/dV accumulation"


def _chunk_prob(q, k, tile_mask, tile_size, queries):
    """Build only this query chunk's mask and probabilities, in original key order."""
    keys = torch.arange(k.shape[-2], device=q.device)
    nt = tile_mask.shape[-1]
    video = nt * tile_size
    qi = (queries // tile_size).clamp_max(nt - 1)
    ki = (keys // tile_size).clamp_max(nt - 1)
    allowed = tile_mask[:, qi[:, None], ki[None, :]]
    allowed = allowed | (queries[:, None] >= video) | (keys[None, :] >= video)
    score = (q @ k.transpose(-2, -1)).float() * (q.shape[-1] ** -0.5)
    return score.masked_fill(~allowed[None], float("-inf")).softmax(-1)


def _chunk_math(q, k, v, tile_mask, tile_size, queries):
    with torch.autocast("cuda", enabled=False):
        return _chunk_prob(q, k, tile_mask, tile_size, queries).to(v.dtype) @ v


def _chunk_adjoint(q, k, v, tile_mask, tile_size, queries, upstream):
    """Keep both bf16 adjoint boundaries and fp32 cross-chunk contributions."""
    with torch.autocast("cuda", enabled=False):
        prob = _chunk_prob(q, k, tile_mask, tile_size, queries)
        low = prob.to(v.dtype)
        dp = (upstream @ v.transpose(-2, -1)).float()
        ds = prob * (dp - (prob * dp).sum(-1, keepdim=True))
        ds = (ds * (q.shape[-1] ** -0.5)).to(q.dtype)
        return (ds @ k, ds.float().transpose(-2, -1) @ q.float(),
                low.float().transpose(-2, -1) @ upstream.float())


class _BoundedReference(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, mask, tile_size, text_len, chunk_fn, adjoint_fn):
        ctx.save_for_backward(q, k, v, mask)
        ctx.tile_size = tile_size
        ctx.adjoint_fn = adjoint_fn
        out = torch.empty_like(q)
        queries = torch.arange(q.shape[2], device=q.device)
        for b in range(q.shape[0]):
            for h in range(q.shape[1]):
                for start in range(0, q.shape[2], 512):
                    out[b:b+1, h:h+1, start:start+512] = chunk_fn(
                        q[b:b+1, h:h+1, start:start+512], k[b:b+1, h:h+1],
                        v[b:b+1, h:h+1], mask[h:h+1], tile_size, queries[start:start+512])
        return out

    @staticmethod
    def backward(ctx, do):
        q, k, v, mask = ctx.saved_tensors
        dq = torch.empty_like(q)
        dk = torch.zeros_like(k, dtype=torch.float32)
        dv = torch.zeros_like(v, dtype=torch.float32)
        queries = torch.arange(q.shape[2], device=q.device)
        for b in range(q.shape[0]):
            for h in range(q.shape[1]):
                for start in range(0, q.shape[2], 512):
                    a, c, d = ctx.adjoint_fn(
                        q[b:b+1, h:h+1, start:start+512], k[b:b+1, h:h+1],
                        v[b:b+1, h:h+1], mask[h:h+1], ctx.tile_size,
                        queries[start:start+512], do[b:b+1, h:h+1, start:start+512])
                    dq[b:b+1, h:h+1, start:start+512] = a
                    dk[b:b+1, h:h+1] += c
                    dv[b:b+1, h:h+1] += d
        return dq, dk.to(k.dtype), dv.to(v.dtype), None, None, None, None, None


def compiled_reference():
    """Compile bounded chunk math; keep every sequence/head loop outside the graph."""
    forward = torch.compile(_chunk_math, fullgraph=True, dynamic=True,
                            mode="max-autotune-no-cudagraphs")
    adjoint = torch.compile(_chunk_adjoint, fullgraph=True, dynamic=True,
                            mode="max-autotune-no-cudagraphs")

    def run(q, k, v, mask, tile_size, text_len):
        return _BoundedReference.apply(q, k, v, mask, tile_size, text_len, forward, adjoint)

    return run
