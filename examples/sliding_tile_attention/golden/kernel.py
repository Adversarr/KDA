"""Golden for ``sliding_tile_attention``: block-sparse FlashAttention-2 in Triton, forward and backward.

The math is ``softmax(q k^T / sqrt(D), masked) v`` with the STA mask of `sta/attention.py`
(``tile_mask: (H, NT, NT)`` bool over ``tile_size``-token tiles, text keys visible to every
query, text queries see every key, softmax in fp32). The kernels are the FA2 reference
(`kda-kernel-implement/reference/triton/attention/fa2_causal/`: log2-domain online softmax,
deterministic two-kernel backward, ``P`` recomputed from ``lse``) with one change: **the loop
over K,V blocks is driven by a per-(head, Q block) list of allowed blocks** instead of the causal
bounds. The host turns the tile mask into that list once per (mask, shape, tile geometry):

    any[h, i, j]   some (query in Q block i, key in K block j) pair is allowed
    full[h, i, j]  every pair is allowed (no element mask needed)

    idx[h, i, :]   the allowed j's, the full ones first; n_full[h, i], n_any[h, i]

so the kernel runs an unmasked loop over ``idx[:n_full]`` and a masked loop over
``idx[n_full:n_any]``. The masked loop rebuilds the token mask from the tile mask with one
gather per block (``tile_mask[h, m // tile_size, n // tile_size] | m >= V | n >= V``, and
``n < S``); with ``tile_size = 384 = 3 x 128`` and 128-row blocks every video block lies inside
one tile, so in practice only a K block that crosses ``S`` is partial and the masked loop is
empty or one block long. The dK/dV kernel uses the transposed lists (per K block, the allowed Q
blocks). The mask is never materialised at token granularity: the biggest mask-derived tensor
is ``idx``, ``H x NQB x max_nnz`` int32 (24 x 902 x 902 = 75 MB for HunyuanVideo, since text
queries attend to every key block; built once per mask and cached).

Work is proportional to the allowed area: 31% of the plane in the smoke config, 5-15% on
HunyuanVideo. ``B*H*S*S`` is logical only; the ``(b, h)`` base offsets are int64 and one
``(b, h)`` plane must fit int32 (host check), as in the FA2 reference.
"""

from typing import Dict, Optional, Tuple

import torch
import triton
import triton.language as tl

_MMA_DTYPES = (torch.bfloat16, torch.float16)
LOG2E = 1.4426950408889634


# ----------------------------------------------------------------------------------------------
# Block lists from the tile mask (host, cached)
# ----------------------------------------------------------------------------------------------


def _touch(S: int, V: int, ts: int, NT: int, blk: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per ``blk``-token block: which video tiles it touches ``(NB, NT)`` fp32 0/1, whether it holds text tokens, whether it crosses ``S``."""
    nb = -(-S // blk)
    tok = torch.arange(S, device=device)
    blk_of = tok // blk
    vid = tok < V
    touch = torch.zeros(nb, NT, device=device)
    touch[blk_of[vid], (tok[vid] // ts).clamp_(max=NT - 1)] = 1.0
    has_text = torch.zeros(nb, dtype=torch.bool, device=device)
    has_text[blk_of[~vid]] = True
    tail = torch.zeros(nb, dtype=torch.bool, device=device)
    if S % blk:
        tail[-1] = True
    return touch, has_text, tail


def _block_lists(tile_mask: torch.Tensor, ts: int, text_len: int, S: int, bq: int, bk: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(idx, n_full, n_any)`` for Q blocks of ``bq`` rows over K blocks of ``bk`` keys.

    ``idx: (H, NQB, maxn)`` int32, the allowed K blocks of each Q block with the fully-allowed
    ones first; ``n_full, n_any: (H, NQB)`` int32; ``perm: (H * NQB,)`` int32, the (head, Q block)
    rows sorted by ``n_any`` descending (program order).
    """
    H, NT, _ = tile_mask.shape
    V = NT * ts
    assert S == V + text_len, f"S = {S} but NT * tile_size + text_len = {V + text_len}"
    qt, qtext, _ = _touch(S, V, ts, NT, bq, tile_mask.device)
    kt, ktext, ktail = _touch(S, V, ts, NT, bk, tile_mask.device)
    tm = tile_mask.float()
    any_video = torch.einsum("qt,hts,ks->hqk", qt, tm, kt) > 0  # some allowed (video tile, video tile) pair
    denied = torch.einsum("qt,hts,ks->hqk", qt, 1.0 - tm, kt) > 0  # some denied pair
    any_ = any_video | qtext[None, :, None] | ktext[None, None, :]
    full = ~denied & ~ktail[None, None, :]  # text rows/cols are always allowed; a K block past S needs the n < S mask
    score = torch.where(any_, torch.where(full, 0, 1), 2)
    order = torch.argsort(score, dim=-1, stable=True)
    n_any = any_.sum(-1, dtype=torch.int32)
    n_full = (any_ & full).sum(-1, dtype=torch.int32)
    maxn = int(n_any.max()) if n_any.numel() else 0
    # Heavy-first program order: the windows differ per head (27 tiles for (3, 3, 3), 1 for
    # (1, 1, 1) in the smoke config) and a (head, block) grid in natural order leaves the last
    # wave to the dense heads; launching the longest lists first fills the tail.
    perm = torch.argsort(n_any.flatten(), descending=True, stable=True).to(torch.int32)
    return order[:, :, :maxn].to(torch.int32).contiguous(), n_full.contiguous(), n_any.contiguous(), perm.contiguous()


_LISTS: Dict[tuple, Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]] = {}


def block_lists(tile_mask: torch.Tensor, ts: int, text_len: int, S: int, bq: int, bk: int, transpose: bool = False):
    """Cached `_block_lists`; ``transpose`` gives the per-K-block lists of Q blocks (dK/dV kernel)."""
    # Retain the owner and version: pointer reuse and in-place mask edits must not
    # reuse block lists computed for different allowed pairs.
    try:
        version = tile_mask._version
    except RuntimeError:  # inference tensors have no version counter
        version = None
    key = (id(tile_mask), version, tuple(tile_mask.shape), ts, text_len, S, bq, bk, transpose, tile_mask.device.index)
    entry = _LISTS.get(key) if version is not None else None
    hit = entry[1] if entry is not None else None
    if hit is None:
        if transpose:
            hit = _block_lists(tile_mask.transpose(1, 2).contiguous(), ts, text_len, S, bq, bk)
        else:
            hit = _block_lists(tile_mask, ts, text_len, S, bq, bk)
        if len(_LISTS) > 64:
            _LISTS.clear()
        if version is not None:
            _LISTS[key] = (tile_mask, hit)
    return hit


def attended_pairs(tile_mask: torch.Tensor, ts: int, text_len: int) -> float:
    """Exact number of allowed (query, key) pairs per batch element, all heads: the FLOP base for the roofline."""
    H, NT, _ = tile_mask.shape
    V = NT * ts
    S = V + text_len
    video = float(tile_mask.sum()) * ts * ts
    text = H * (text_len * S + V * text_len)  # text queries see all keys; video queries see all text keys
    return video + text


def attention_flops(B: int, D: int, pairs: float, phase: str) -> float:
    per_pair = {"fwd": 4 * D, "infer": 4 * D, "bwd": 10 * D}[phase]  # 2 GEMMs forward, 5 backward, 2 FLOP each
    return B * pairs * per_pair


# ----------------------------------------------------------------------------------------------
# Kernels
# ----------------------------------------------------------------------------------------------


@triton.jit
def _token_mask(tm_base, offs_m, offs_n, V, S, TS, NT):
    """``(BLOCK_M, BLOCK_N)`` bool: allowed pairs of a partial block, rebuilt from the tile mask."""
    qt = tl.minimum(offs_m // TS, NT - 1)
    kt = tl.minimum(offs_n // TS, NT - 1)
    tm = tl.load(tm_base + qt[:, None] * NT + kt[None, :])
    return ((tm != 0) | (offs_m[:, None] >= V) | (offs_n[None, :] >= V)) & (offs_n[None, :] < S)


@triton.jit
def _attn_fwd_inner(
    acc, l_i, m_i, q, k_base, v_base, stride_kn, stride_vn, offs_m, offs_d, mask_d,
    idx_base, lo, hi, tm_base, V, S, TS, NT, scale_log2,
    BLOCK_N: tl.constexpr, MASK: tl.constexpr,
):
    """K,V blocks ``idx_base[lo:hi]``; ``MASK`` rebuilds the token mask for the partial blocks."""
    for it in range(lo, hi):
        start_n = tl.load(idx_base + it) * BLOCK_N
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + tl.arange(0, BLOCK_N)
        if MASK:
            mask_n = offs_n < S
            k = tl.load(k_base + offs_d[:, None] + offs_n[None, :] * stride_kn, mask=mask_d[:, None] & mask_n[None, :], other=0.0)
        else:
            k = tl.load(k_base + offs_d[:, None] + offs_n[None, :] * stride_kn, mask=mask_d[:, None], other=0.0)
        s = tl.dot(q, k).to(q.dtype).to(tl.float32) * scale_log2
        if MASK:
            s = tl.where(_token_mask(tm_base, offs_m, offs_n, V, S, TS, NT), s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        if MASK:
            # A partial block can deny a whole row (unlike the causal diagonal); with no earlier
            # block for that row m_new is -inf and (-inf) - (-inf) would poison the row.
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        else:
            m_safe = m_new
        alpha = tl.math.exp2(m_i - m_safe)
        p = tl.math.exp2(s - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        if MASK:
            v = tl.load(v_base + offs_n[:, None] * stride_vn + offs_d[None, :], mask=mask_n[:, None] & mask_d[None, :], other=0.0)
        else:
            v = tl.load(v_base + offs_n[:, None] * stride_vn + offs_d[None, :], mask=mask_d[None, :], other=0.0)
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_new
    return acc, l_i, m_i


@triton.jit
def _attn_fwd_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, lse_ptr, tm_ptr, idx_ptr, nfull_ptr, nany_ptr, perm_ptr,
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_ob, stride_oh, stride_om,
    H, S, D, V, TS, NT, NQB, MAXN, scale_log2,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr, SAVE_LSE: tl.constexpr,
):
    row = tl.load(perm_ptr + tl.program_id(0))  # (head, Q block) in heavy-first order
    h = (row // NQB).to(tl.int64)
    start_m = row % NQB
    b = tl.program_id(1).to(tl.int64)
    bh = b * H + h
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)  # int32: compared against keys in the loop
    offs_m64 = start_m.to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)  # int64: this tile's own addresses
    offs_d = tl.arange(0, BLOCK_D)
    mask_m = offs_m < S
    mask_d = offs_d < D

    q_base = q_ptr + b * stride_qb + h * stride_qh
    k_base = k_ptr + b * stride_kb + h * stride_kh
    v_base = v_ptr + b * stride_vb + h * stride_vh
    q = tl.load(q_base + offs_m64[:, None] * stride_qm + offs_d[None, :], mask=mask_m[:, None] & mask_d[None, :], other=0.0)

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)

    n_full = tl.load(nfull_ptr + row)
    n_any = tl.load(nany_ptr + row)
    idx_base = idx_ptr + row.to(tl.int64) * MAXN
    tm_base = tm_ptr + h * NT * NT
    acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, k_base, v_base, stride_kn, stride_vn, offs_m, offs_d, mask_d, idx_base, 0, n_full, tm_base, V, S, TS, NT, scale_log2, BLOCK_N, False)
    acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, k_base, v_base, stride_kn, stride_vn, offs_m, offs_d, mask_d, idx_base, n_full, n_any, tm_base, V, S, TS, NT, scale_log2, BLOCK_N, True)

    acc = acc / l_i[:, None]
    o_base = o_ptr + b * stride_ob + h * stride_oh
    tl.store(o_base + offs_m64[:, None] * stride_om + offs_d[None, :], acc.to(o_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_d[None, :])
    if SAVE_LSE:
        tl.store(lse_ptr + bh * S + offs_m, m_i + tl.math.log2(l_i), mask=mask_m)


@triton.jit
def _attn_bwd_preprocess_kernel(
    o_ptr, do_ptr, delta_ptr,
    stride_ob, stride_oh, stride_om,
    stride_dob, stride_doh, stride_dom,
    H, S, D,
    BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr,
):
    start_m = tl.program_id(0)
    bh = tl.program_id(1).to(tl.int64)
    b = bh // H
    h = bh % H
    offs_m = start_m.to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    mask = (offs_m < S)[:, None] & (offs_d < D)[None, :]
    o = tl.load(o_ptr + b * stride_ob + h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :], mask=mask, other=0.0).to(tl.float32)
    do = tl.load(do_ptr + b * stride_dob + h * stride_doh + offs_m[:, None] * stride_dom + offs_d[None, :], mask=mask, other=0.0).to(tl.float32)
    tl.store(delta_ptr + bh * S + offs_m, tl.sum(o * do, axis=1), mask=offs_m < S)


@triton.jit
def _attn_bwd_dkdv_inner(
    dk, dv, k, v, q_base, do_base, lse_base, delta_base, stride_qm, stride_dom, offs_n, offs_d, mask_d,
    idx_base, lo, hi, tm_base, V, S, TS, NT, scale_log2,
    BLOCK_M: tl.constexpr, MASK: tl.constexpr,
):
    """Q blocks ``idx_base[lo:hi]`` for one K,V block. The transposed tile mask is passed, so the gather reads ``[key tile, query tile]``."""
    for it in range(lo, hi):
        start_m = tl.load(idx_base + it) * BLOCK_M
        start_m = tl.multiple_of(start_m, BLOCK_M)
        offs_m = start_m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < S
        q = tl.load(q_base + offs_m[:, None] * stride_qm + offs_d[None, :], mask=mask_m[:, None] & mask_d[None, :], other=0.0)
        do = tl.load(do_base + offs_m[:, None] * stride_dom + offs_d[None, :], mask=mask_m[:, None] & mask_d[None, :], other=0.0)
        lse = tl.load(lse_base + offs_m, mask=mask_m, other=float("inf"))  # tail rows: lse = +inf -> p = 0
        delta = tl.load(delta_base + offs_m, mask=mask_m, other=0.0)
        sT = tl.dot(k, tl.trans(q)).to(q.dtype).to(tl.float32) * scale_log2  # (BLOCK_N, BLOCK_M): keys down, queries across
        pT = tl.math.exp2(sT - lse[None, :])
        if MASK:
            # rows are keys here: the gather runs on the transposed tile mask with (key, query) roles
            pT = tl.where(_token_mask(tm_base, offs_n, offs_m, V, S, TS, NT), pT, 0.0)
        dv += tl.dot(pT.to(do.dtype), do)
        dpT = tl.dot(v, tl.trans(do)).to(do.dtype).to(tl.float32)
        dsT = pT * (dpT - delta[None, :])
        dk += tl.dot(dsT.to(q.dtype), q)
    return dk, dv


@triton.jit
def _attn_bwd_dkdv_kernel(
    q_ptr, k_ptr, v_ptr, do_ptr, lse_ptr, delta_ptr, dk_ptr, dv_ptr, tmT_ptr, idx_ptr, nfull_ptr, nany_ptr, perm_ptr,
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_dob, stride_doh, stride_dom,
    H, S, D, V, TS, NT, NKB, MAXN, scale, scale_log2,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    row = tl.load(perm_ptr + tl.program_id(0))  # (head, K block) in heavy-first order
    h = (row // NKB).to(tl.int64)
    start_n = row % NKB
    b = tl.program_id(1).to(tl.int64)
    bh = b * H + h
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_n64 = start_n.to(tl.int64) * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    mask_n = offs_n < S
    mask_d = offs_d < D

    k = tl.load(k_ptr + b * stride_kb + h * stride_kh + offs_n64[:, None] * stride_kn + offs_d[None, :], mask=mask_n[:, None] & mask_d[None, :], other=0.0)
    v = tl.load(v_ptr + b * stride_vb + h * stride_vh + offs_n64[:, None] * stride_vn + offs_d[None, :], mask=mask_n[:, None] & mask_d[None, :], other=0.0)
    dk = tl.zeros([BLOCK_N, BLOCK_D], tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_D], tl.float32)
    q_base = q_ptr + b * stride_qb + h * stride_qh
    do_base = do_ptr + b * stride_dob + h * stride_doh
    lse_base = lse_ptr + bh * S
    delta_base = delta_ptr + bh * S

    n_full = tl.load(nfull_ptr + row)
    n_any = tl.load(nany_ptr + row)
    idx_base = idx_ptr + row.to(tl.int64) * MAXN
    tm_base = tmT_ptr + h * NT * NT
    dk, dv = _attn_bwd_dkdv_inner(dk, dv, k, v, q_base, do_base, lse_base, delta_base, stride_qm, stride_dom, offs_n, offs_d, mask_d, idx_base, 0, n_full, tm_base, V, S, TS, NT, scale_log2, BLOCK_M, False)
    dk, dv = _attn_bwd_dkdv_inner(dk, dv, k, v, q_base, do_base, lse_base, delta_base, stride_qm, stride_dom, offs_n, offs_d, mask_d, idx_base, n_full, n_any, tm_base, V, S, TS, NT, scale_log2, BLOCK_M, True)

    dk_base = dk_ptr + b * stride_kb + h * stride_kh
    dv_base = dv_ptr + b * stride_vb + h * stride_vh
    tl.store(dk_base + offs_n64[:, None] * stride_kn + offs_d[None, :], (dk * scale).to(dk_ptr.dtype.element_ty), mask=mask_n[:, None] & mask_d[None, :])
    tl.store(dv_base + offs_n64[:, None] * stride_vn + offs_d[None, :], dv.to(dv_ptr.dtype.element_ty), mask=mask_n[:, None] & mask_d[None, :])


@triton.jit
def _attn_bwd_dq_inner(
    dq, q, do, lse, delta, k_base, v_base, stride_kn, stride_vn, offs_m, offs_d, mask_d,
    idx_base, lo, hi, tm_base, V, S, TS, NT, scale_log2,
    BLOCK_N: tl.constexpr, MASK: tl.constexpr,
):
    for it in range(lo, hi):
        start_n = tl.load(idx_base + it) * BLOCK_N
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + tl.arange(0, BLOCK_N)
        if MASK:
            mask_n = offs_n < S
            k = tl.load(k_base + offs_n[:, None] * stride_kn + offs_d[None, :], mask=mask_n[:, None] & mask_d[None, :], other=0.0)
            v = tl.load(v_base + offs_n[:, None] * stride_vn + offs_d[None, :], mask=mask_n[:, None] & mask_d[None, :], other=0.0)
        else:
            k = tl.load(k_base + offs_n[:, None] * stride_kn + offs_d[None, :], mask=mask_d[None, :], other=0.0)
            v = tl.load(v_base + offs_n[:, None] * stride_vn + offs_d[None, :], mask=mask_d[None, :], other=0.0)
        s = tl.dot(q, tl.trans(k)).to(q.dtype).to(tl.float32) * scale_log2
        p = tl.math.exp2(s - lse[:, None])
        if MASK:
            p = tl.where(_token_mask(tm_base, offs_m, offs_n, V, S, TS, NT), p, 0.0)
        dp = tl.dot(do, tl.trans(v)).to(do.dtype).to(tl.float32)
        ds = p * (dp - delta[:, None])
        dq += tl.dot(ds.to(k.dtype), k)
    return dq


@triton.jit
def _attn_bwd_dq_kernel(
    q_ptr, k_ptr, v_ptr, do_ptr, lse_ptr, delta_ptr, dq_ptr, tm_ptr, idx_ptr, nfull_ptr, nany_ptr, perm_ptr,
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_dob, stride_doh, stride_dom,
    H, S, D, V, TS, NT, NQB, MAXN, scale, scale_log2,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    row = tl.load(perm_ptr + tl.program_id(0))  # (head, Q block) in heavy-first order
    h = (row // NQB).to(tl.int64)
    start_m = row % NQB
    b = tl.program_id(1).to(tl.int64)
    bh = b * H + h
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_m64 = start_m.to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    mask_m = offs_m < S
    mask_d = offs_d < D

    q = tl.load(q_ptr + b * stride_qb + h * stride_qh + offs_m64[:, None] * stride_qm + offs_d[None, :], mask=mask_m[:, None] & mask_d[None, :], other=0.0)
    do = tl.load(do_ptr + b * stride_dob + h * stride_doh + offs_m64[:, None] * stride_dom + offs_d[None, :], mask=mask_m[:, None] & mask_d[None, :], other=0.0)
    lse = tl.load(lse_ptr + bh * S + offs_m, mask=mask_m, other=float("inf"))
    delta = tl.load(delta_ptr + bh * S + offs_m, mask=mask_m, other=0.0)
    dq = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)
    k_base = k_ptr + b * stride_kb + h * stride_kh
    v_base = v_ptr + b * stride_vb + h * stride_vh

    n_full = tl.load(nfull_ptr + row)
    n_any = tl.load(nany_ptr + row)
    idx_base = idx_ptr + row.to(tl.int64) * MAXN
    tm_base = tm_ptr + h * NT * NT
    dq = _attn_bwd_dq_inner(dq, q, do, lse, delta, k_base, v_base, stride_kn, stride_vn, offs_m, offs_d, mask_d, idx_base, 0, n_full, tm_base, V, S, TS, NT, scale_log2, BLOCK_N, False)
    dq = _attn_bwd_dq_inner(dq, q, do, lse, delta, k_base, v_base, stride_kn, stride_vn, offs_m, offs_d, mask_d, idx_base, n_full, n_any, tm_base, V, S, TS, NT, scale_log2, BLOCK_N, True)

    tl.store(dq_ptr + b * stride_qb + h * stride_qh + offs_m64[:, None] * stride_qm + offs_d[None, :], (dq * scale).to(dq_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_d[None, :])


# ----------------------------------------------------------------------------------------------
# Host side
# ----------------------------------------------------------------------------------------------


def select_config(D: int, phase: str, sequence: int = 0) -> Dict[str, int]:
    """Same geometry as the FA2 reference (A800): 128 x 128 / 8 warps forward at D = 128, 64 x 64 / 4 warps backward."""
    if 0 < sequence <= 512:
        # Smaller tiles provide enough independent programs for the short sequence.
        if phase == "fwd":
            return dict(BLOCK_M=32, BLOCK_N=32, num_warps=4, num_stages=2)
        return dict(BLOCK_M1=32, BLOCK_N1=32, BLOCK_M2=32, BLOCK_N2=32,
                    num_warps=4, num_stages=2)
    if phase == "fwd":
        if D <= 64:
            return dict(BLOCK_M=128, BLOCK_N=64, num_warps=4, num_stages=3)
        if D <= 128:
            # 3 stages fit the causal reference; the indirect K,V loads here pipeline one more
            # tile and run out of shared memory at 3 (192 KB > 167 KB), so 2.
            return dict(BLOCK_M=128, BLOCK_N=128, num_warps=8, num_stages=2)
        return dict(BLOCK_M=64, BLOCK_N=64, num_warps=8, num_stages=2)
    if D <= 128:
        return dict(BLOCK_M1=64, BLOCK_N1=64, BLOCK_M2=64, BLOCK_N2=64, num_warps=4, num_stages=2)
    return dict(BLOCK_M1=32, BLOCK_N1=128, BLOCK_M2=128, BLOCK_N2=32, num_warps=8, num_stages=2)


def _check(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, tile_mask: torch.Tensor, tile_size: int, text_len: int) -> None:
    if q.dtype not in _MMA_DTYPES or k.dtype != q.dtype or v.dtype != q.dtype:
        raise TypeError(f"sta: q, k, v must share a bf16/fp16 dtype, got {q.dtype}, {k.dtype}, {v.dtype}")
    if q.dim() != 4 or k.shape != q.shape or v.shape != q.shape:
        raise ValueError(f"sta: q, k, v must be (B, H, S, D) of one shape, got {tuple(q.shape)}, {tuple(k.shape)}, {tuple(v.shape)}")
    if q.shape[-1] > 256:
        raise ValueError(f"sta: head dim {q.shape[-1]} > 256")
    H, NT, NT2 = tile_mask.shape
    if tile_mask.dtype != torch.bool or NT != NT2 or H != q.shape[1]:
        raise ValueError(f"sta: tile_mask must be (H, NT, NT) bool with H = {q.shape[1]}, got {tuple(tile_mask.shape)} {tile_mask.dtype}")
    if q.shape[2] != NT * tile_size + text_len:
        raise ValueError(f"sta: S = {q.shape[2]} != NT * tile_size + text_len = {NT * tile_size + text_len}")
    for name, t in (("q", q), ("k", k), ("v", v)):
        if t.stride(-1) != 1:
            raise ValueError(f"sta: {name} must be contiguous in the head dim")
        if t.shape[2] * t.stride(2) >= 2**31:  # in-loop offsets are int32; the (b, h) base is int64
            raise ValueError(f"sta: one (batch, head) plane of {name} spans {t.shape[2] * t.stride(2)} elements >= 2^31")


def _u8(tile_mask: torch.Tensor) -> torch.Tensor:
    return tile_mask.contiguous().view(torch.uint8)


def sta_fwd(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, tile_mask: torch.Tensor, tile_size: int, text_len: int, scale: Optional[float] = None, *, save_lse: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns ``o`` (q's shape and dtype) and ``lse`` (``(B, H, S)`` fp32, log2 domain)."""
    _check(q, k, v, tile_mask, tile_size, text_len)
    B, H, S, D = q.shape
    NT = tile_mask.shape[1]
    V = NT * tile_size
    scale = D**-0.5 if scale is None else scale
    cfg = select_config(D, "fwd", S)
    idx, n_full, n_any, perm = block_lists(tile_mask, tile_size, text_len, S, cfg["BLOCK_M"], cfg["BLOCK_N"])
    o = torch.empty_like(q)
    lse = torch.empty((B, H, S) if save_lse else (0,), dtype=torch.float32, device=q.device)
    NQB = triton.cdiv(S, cfg["BLOCK_M"])
    _attn_fwd_kernel[(H * NQB, B)](
        q, k, v, o, lse, _u8(tile_mask), idx, n_full, n_any, perm,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        H, S, D, V, tile_size, NT, NQB, idx.shape[-1], scale * LOG2E,
        BLOCK_M=cfg["BLOCK_M"], BLOCK_N=cfg["BLOCK_N"], BLOCK_D=triton.next_power_of_2(D), SAVE_LSE=save_lse,
        num_warps=cfg["num_warps"], num_stages=cfg["num_stages"],
    )
    return o, lse


def sta_bwd(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, o: torch.Tensor, do: torch.Tensor, lse: torch.Tensor,
    tile_mask: torch.Tensor, tile_size: int, text_len: int, scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``dq, dk, dv`` in the storage dtype, deterministic (no atomics)."""
    B, H, S, D = q.shape
    NT = tile_mask.shape[1]
    V = NT * tile_size
    scale = D**-0.5 if scale is None else scale
    if do.stride(-1) != 1:
        raise ValueError("sta backward: dO must be contiguous in the head dim (call .contiguous() in the caller)")
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    delta = torch.empty_like(lse)
    BLOCK_D = triton.next_power_of_2(D)
    cfg = select_config(D, "bwd", S)
    pre_block = 128
    _attn_bwd_preprocess_kernel[(triton.cdiv(S, pre_block), B * H)](
        o, do, delta,
        o.stride(0), o.stride(1), o.stride(2),
        do.stride(0), do.stride(1), do.stride(2),
        H, S, D, BLOCK_M=pre_block, BLOCK_D=BLOCK_D, num_warps=4,
    )
    strides = (
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        do.stride(0), do.stride(1), do.stride(2),
    )
    # dK/dV: per K block, the Q blocks that attend to it = the lists of the transposed tile mask.
    tmT = tile_mask.transpose(1, 2).contiguous()
    idxT, n_fullT, n_anyT, permT = block_lists(tile_mask, tile_size, text_len, S, cfg["BLOCK_N1"], cfg["BLOCK_M1"], transpose=True)
    NKB = triton.cdiv(S, cfg["BLOCK_N1"])
    _attn_bwd_dkdv_kernel[(H * NKB, B)](
        q, k, v, do, lse, delta, dk, dv, _u8(tmT), idxT, n_fullT, n_anyT, permT, *strides,
        H, S, D, V, tile_size, NT, NKB, idxT.shape[-1], scale, scale * LOG2E,
        BLOCK_M=cfg["BLOCK_M1"], BLOCK_N=cfg["BLOCK_N1"], BLOCK_D=BLOCK_D,
        num_warps=cfg["num_warps"], num_stages=cfg["num_stages"],
    )
    idx, n_full, n_any, perm = block_lists(tile_mask, tile_size, text_len, S, cfg["BLOCK_M2"], cfg["BLOCK_N2"])
    NQB = triton.cdiv(S, cfg["BLOCK_M2"])
    _attn_bwd_dq_kernel[(H * NQB, B)](
        q, k, v, do, lse, delta, dq, _u8(tile_mask), idx, n_full, n_any, perm, *strides,
        H, S, D, V, tile_size, NT, NQB, idx.shape[-1], scale, scale * LOG2E,
        BLOCK_M=cfg["BLOCK_M2"], BLOCK_N=cfg["BLOCK_N2"], BLOCK_D=BLOCK_D,
        num_warps=cfg["num_warps"], num_stages=cfg["num_stages"],
    )
    return dq, dk, dv


class _STA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, tile_mask, tile_size, text_len, save_lse):
        o, lse = sta_fwd(q, k, v, tile_mask, tile_size, text_len, save_lse=save_lse)
        if save_lse:
            ctx.save_for_backward(q, k, v, o, lse, tile_mask)
        ctx.geometry = (tile_size, text_len)
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse, tile_mask = ctx.saved_tensors
        dq, dk, dv = sta_bwd(q, k, v, o, do.contiguous() if do.stride(-1) != 1 else do, lse, tile_mask, *ctx.geometry)
        return dq, dk, dv, None, None, None, None


def sliding_tile_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, tile_mask: torch.Tensor, tile_size: int, text_len: int
) -> torch.Tensor:
    """Drop-in for ``sta.attention.sliding_tile_attention``: ``(B, H, S, D)`` in, ``(B, H, S, D)`` out."""
    save_lse = torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v))
    return _STA.apply(q, k, v, tile_mask, tile_size, text_len, save_lse)


__all__ = [
    "sliding_tile_attention", "sta_fwd", "sta_bwd", "block_lists", "attended_pairs", "attention_flops", "select_config", "LOG2E",
]
