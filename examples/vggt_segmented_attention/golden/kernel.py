"""Standalone rectangular FA2 with compact per-scene KV; see GOLDEN.md."""

from typing import Dict

import torch
import triton
import triton.language as tl

_MMA_DTYPES = (torch.bfloat16, torch.float16)
LOG2E = 1.4426950408889634


@triton.jit
def _probability_dot(a, b, acc):
    """Flash-style operand rounding; tensor-core products accumulate in fp32."""
    return tl.dot(a.to(b.dtype), b, acc)


@triton.jit
def _attn_fwd_inner(
    acc,
    l_i,
    m_i,
    q,
    k_base,
    v_base,
    stride_kn,
    stride_vn,
    offs_m,
    offs_d,
    mask_d,
    lo,
    hi,
    S,
    scale_log2,
    BLOCK_N: tl.constexpr,
    MASK: tl.constexpr,
):
    """K,V blocks [lo, hi); MASK protects the key-length tail."""
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + tl.arange(0, BLOCK_N)
        if MASK:
            mask_n = offs_n < S
            # K is loaded transposed, (BLOCK_D, BLOCK_N), so tl.dot needs no tl.trans (a layout
            # conversion through shared memory that costs ~10% at D = 128).
            k = tl.load(
                k_base + offs_d[:, None] + offs_n[None, :] * stride_kn,
                mask=mask_d[:, None] & mask_n[None, :],
                other=0.0,
            )
        else:
            k = tl.load(
                k_base + offs_d[:, None] + offs_n[None, :] * stride_kn,
                mask=mask_d[:, None],
                other=0.0,
            )
        s = (
            tl.dot(q, k) * scale_log2
        )  # (BLOCK_M, BLOCK_N) fp32, already in the log2 domain
        if MASK:
            s = tl.where(mask_n[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        m_new = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        if MASK:
            v = tl.load(
                v_base + offs_n[:, None] * stride_vn + offs_d[None, :],
                mask=mask_n[:, None] & mask_d[None, :],
                other=0.0,
            )
        else:
            v = tl.load(
                v_base + offs_n[:, None] * stride_vn + offs_d[None, :],
                mask=mask_d[None, :],
                other=0.0,
            )
        acc = _probability_dot(p, v, acc)
        m_i = m_new
    return acc, l_i, m_i


@triton.jit
def _attn_fwd_kernel(
    q_ptr,
    meta_ptr,
    VIEWS: tl.constexpr,
    k_ptr,
    v_ptr,
    o_ptr,
    lse_ptr,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_ob,
    stride_oh,
    stride_om,
    H,
    SQ,
    SK,
    D: tl.constexpr,
    scale_log2,
    SAVE_AUX: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    start_m = tl.program_id(0)
    bh = tl.program_id(1).to(tl.int64)
    b = bh // H
    h = bh % H
    SK = tl.maximum(1, tl.load(meta_ptr + b * (VIEWS + 1) + VIEWS))
    offs_m = start_m * BLOCK_M + tl.arange(
        0, BLOCK_M
    )  # int32: compared against keys in the loop
    offs_m64 = start_m.to(tl.int64) * BLOCK_M + tl.arange(
        0, BLOCK_M
    )  # int64: this tile's own addresses
    offs_d = tl.arange(0, BLOCK_D)
    mask_m = offs_m < SQ
    mask_d = offs_d < D

    q_base = q_ptr + b * stride_qb + h * stride_qh
    k_base = k_ptr + b * stride_kb + h * stride_kh
    v_base = v_ptr + b * stride_vb + h * stride_vh
    q = tl.load(
        q_base + offs_m64[:, None] * stride_qm + offs_d[None, :],
        mask=mask_m[:, None] & mask_d[None, :],
        other=0.0,
    )

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)

    hi = SK
    full = SK // BLOCK_N * BLOCK_N
    acc, l_i, m_i = _attn_fwd_inner(
        acc,
        l_i,
        m_i,
        q,
        k_base,
        v_base,
        stride_kn,
        stride_vn,
        offs_m,
        offs_d,
        mask_d,
        0,
        full,
        SK,
        scale_log2,
        BLOCK_N,
        False,
    )
    acc, l_i, m_i = _attn_fwd_inner(
        acc,
        l_i,
        m_i,
        q,
        k_base,
        v_base,
        stride_kn,
        stride_vn,
        offs_m,
        offs_d,
        mask_d,
        full,
        hi,
        SK,
        scale_log2,
        BLOCK_N,
        True,
    )

    acc = acc / l_i[:, None]
    o_base = o_ptr + b * stride_ob + h * stride_oh
    tl.store(
        o_base + offs_m64[:, None] * stride_om + offs_d[None, :],
        acc.to(o_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_d[None, :],
    )
    # lse in the log2 domain: the backward recomputes p = 2^(s * scale_log2 - lse).
    if SAVE_AUX:
        tl.store(lse_ptr + bh * SQ + offs_m, m_i + tl.math.log2(l_i), mask=mask_m)


@triton.jit
def _attn_bwd_preprocess_kernel(
    o_ptr,
    do_ptr,
    delta_ptr,
    stride_ob,
    stride_oh,
    stride_om,
    stride_dob,
    stride_doh,
    stride_dom,
    H,
    S,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    start_m = tl.program_id(0)
    bh = tl.program_id(1).to(tl.int64)
    b = bh // H
    h = bh % H
    offs_m = start_m.to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    mask = (offs_m < S)[:, None] & (offs_d < D)[None, :]
    o = tl.load(
        o_ptr
        + b * stride_ob
        + h * stride_oh
        + offs_m[:, None] * stride_om
        + offs_d[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    do = tl.load(
        do_ptr
        + b * stride_dob
        + h * stride_doh
        + offs_m[:, None] * stride_dom
        + offs_d[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    tl.store(delta_ptr + bh * S + offs_m, tl.sum(o * do, axis=1), mask=offs_m < S)


@triton.jit
def _attn_bwd_dkdv_inner(
    dk,
    dv,
    k,
    v,
    q_base,
    do_base,
    lse_base,
    delta_base,
    stride_qm,
    stride_dom,
    offs_n,
    offs_d,
    mask_d,
    lo,
    hi,
    SQ,
    SK,
    scale_log2,
    BLOCK_M: tl.constexpr,
    MASK: tl.constexpr,
):
    """Q blocks [lo, hi) for one K,V block; MASK protects tail keys.

    Tiles are loaded in their natural ``(rows, BLOCK_D)`` layout and transposed only as the
    *B* operand of ``tl.dot`` (``k @ trans(q)``, ``v @ trans(do)``), which Triton folds into
    the MMA operand layout. Loading ``q`` transposed and transposing it back for ``dsT @ q``
    ran 1.3x slower; loading each tile twice in both layouts ran out of shared memory at the
    3-stage pipeline (SNIPPET.md).
    """
    for start_m in range(lo, hi, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        offs_m = start_m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < SQ
        q = tl.load(
            q_base + offs_m[:, None] * stride_qm + offs_d[None, :],
            mask=mask_m[:, None] & mask_d[None, :],
            other=0.0,
        )
        do = tl.load(
            do_base + offs_m[:, None] * stride_dom + offs_d[None, :],
            mask=mask_m[:, None] & mask_d[None, :],
            other=0.0,
        )
        lse = tl.load(
            lse_base + offs_m, mask=mask_m, other=float("inf")
        )  # tail rows: lse = +inf -> p = 0
        delta = tl.load(delta_base + offs_m, mask=mask_m, other=0.0)
        sT = (
            tl.dot(k, tl.trans(q)) * scale_log2
        )  # (BLOCK_N, BLOCK_M): keys down, queries across
        pT = tl.math.exp2(sT - lse[None, :])
        if MASK:
            pT = tl.where((offs_n < SK)[:, None], pT, 0.0)
        dv = _probability_dot(pT, do, dv)
        dpT = tl.dot(v, tl.trans(do))
        dsT = pT * (dpT - delta[None, :])
        dk = _probability_dot(dsT, q, dk)
    return dk, dv


@triton.jit
def _attn_bwd_dkdv_kernel(
    q_ptr,
    meta_ptr,
    VIEWS: tl.constexpr,
    k_ptr,
    v_ptr,
    do_ptr,
    lse_ptr,
    delta_ptr,
    dk_ptr,
    dv_ptr,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_dob,
    stride_doh,
    stride_dom,
    H,
    SQ,
    SK,
    D: tl.constexpr,
    scale,
    scale_log2,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    start_n = tl.program_id(0)
    bh = tl.program_id(1).to(tl.int64)
    b = bh // H
    h = bh % H
    SK = tl.maximum(1, tl.load(meta_ptr + b * (VIEWS + 1) + VIEWS))
    offs_n = start_n * BLOCK_N + tl.arange(
        0, BLOCK_N
    )  # int32: compared against queries in the loop
    offs_n64 = start_n.to(tl.int64) * BLOCK_N + tl.arange(
        0, BLOCK_N
    )  # int64: this tile's own addresses
    offs_d = tl.arange(0, BLOCK_D)
    mask_n = offs_n < SK
    mask_d = offs_d < D

    k = tl.load(
        k_ptr
        + b * stride_kb
        + h * stride_kh
        + offs_n64[:, None] * stride_kn
        + offs_d[None, :],
        mask=mask_n[:, None] & mask_d[None, :],
        other=0.0,
    )
    v = tl.load(
        v_ptr
        + b * stride_vb
        + h * stride_vh
        + offs_n64[:, None] * stride_vn
        + offs_d[None, :],
        mask=mask_n[:, None] & mask_d[None, :],
        other=0.0,
    )
    dk = tl.zeros([BLOCK_N, BLOCK_D], tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_D], tl.float32)
    q_base = q_ptr + b * stride_qb + h * stride_qh
    do_base = do_ptr + b * stride_dob + h * stride_doh
    lse_base = lse_ptr + bh * SQ
    delta_base = delta_ptr + bh * SQ

    lo = 0
    dk, dv = _attn_bwd_dkdv_inner(
        dk,
        dv,
        k,
        v,
        q_base,
        do_base,
        lse_base,
        delta_base,
        stride_qm,
        stride_dom,
        offs_n,
        offs_d,
        mask_d,
        lo,
        SQ,
        SQ,
        SK,
        scale_log2,
        BLOCK_M,
        True,
    )

    dk_base = (
        dk_ptr + b * stride_kb + h * stride_kh
    )  # dk, dv are allocated with k's / v's strides
    dv_base = dv_ptr + b * stride_vb + h * stride_vh
    tl.store(
        dk_base + offs_n64[:, None] * stride_kn + offs_d[None, :],
        (dk * scale).to(dk_ptr.dtype.element_ty),
        mask=mask_n[:, None] & mask_d[None, :],
    )
    tl.store(
        dv_base + offs_n64[:, None] * stride_vn + offs_d[None, :],
        dv.to(dv_ptr.dtype.element_ty),
        mask=mask_n[:, None] & mask_d[None, :],
    )


@triton.jit
def _attn_bwd_dq_inner(
    dq,
    q,
    do,
    lse,
    delta,
    k_base,
    v_base,
    stride_kn,
    stride_vn,
    offs_m,
    offs_d,
    mask_d,
    lo,
    hi,
    S,
    scale_log2,
    BLOCK_N: tl.constexpr,
    MASK: tl.constexpr,
):
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + tl.arange(0, BLOCK_N)
        if MASK:
            mask_n = offs_n < S
            k = tl.load(
                k_base + offs_n[:, None] * stride_kn + offs_d[None, :],
                mask=mask_n[:, None] & mask_d[None, :],
                other=0.0,
            )
            v = tl.load(
                v_base + offs_n[:, None] * stride_vn + offs_d[None, :],
                mask=mask_n[:, None] & mask_d[None, :],
                other=0.0,
            )
        else:
            k = tl.load(
                k_base + offs_n[:, None] * stride_kn + offs_d[None, :],
                mask=mask_d[None, :],
                other=0.0,
            )
            v = tl.load(
                v_base + offs_n[:, None] * stride_vn + offs_d[None, :],
                mask=mask_d[None, :],
                other=0.0,
            )
        s = tl.dot(q, tl.trans(k)) * scale_log2
        p = tl.math.exp2(s - lse[:, None])
        if MASK:
            p = tl.where(mask_n[None, :], p, 0.0)
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None])
        dq = _probability_dot(ds, k, dq)
    return dq


@triton.jit
def _attn_bwd_dq_kernel(
    q_ptr,
    meta_ptr,
    VIEWS: tl.constexpr,
    k_ptr,
    v_ptr,
    do_ptr,
    lse_ptr,
    delta_ptr,
    dq_ptr,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_dob,
    stride_doh,
    stride_dom,
    H,
    SQ,
    SK,
    D: tl.constexpr,
    scale,
    scale_log2,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    start_m = tl.program_id(0)
    bh = tl.program_id(1).to(tl.int64)
    b = bh // H
    h = bh % H
    SK = tl.maximum(1, tl.load(meta_ptr + b * (VIEWS + 1) + VIEWS))
    offs_m = start_m * BLOCK_M + tl.arange(
        0, BLOCK_M
    )  # int32: compared against keys in the loop
    offs_m64 = start_m.to(tl.int64) * BLOCK_M + tl.arange(
        0, BLOCK_M
    )  # int64: this tile's own addresses
    offs_d = tl.arange(0, BLOCK_D)
    mask_m = offs_m < SQ
    mask_d = offs_d < D

    q = tl.load(
        q_ptr
        + b * stride_qb
        + h * stride_qh
        + offs_m64[:, None] * stride_qm
        + offs_d[None, :],
        mask=mask_m[:, None] & mask_d[None, :],
        other=0.0,
    )
    do = tl.load(
        do_ptr
        + b * stride_dob
        + h * stride_doh
        + offs_m64[:, None] * stride_dom
        + offs_d[None, :],
        mask=mask_m[:, None] & mask_d[None, :],
        other=0.0,
    )
    lse = tl.load(lse_ptr + bh * SQ + offs_m, mask=mask_m, other=float("inf"))
    delta = tl.load(delta_ptr + bh * SQ + offs_m, mask=mask_m, other=0.0)
    dq = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)
    k_base = k_ptr + b * stride_kb + h * stride_kh
    v_base = v_ptr + b * stride_vb + h * stride_vh

    hi = SK
    full = SK // BLOCK_N * BLOCK_N
    dq = _attn_bwd_dq_inner(
        dq,
        q,
        do,
        lse,
        delta,
        k_base,
        v_base,
        stride_kn,
        stride_vn,
        offs_m,
        offs_d,
        mask_d,
        0,
        full,
        SK,
        scale_log2,
        BLOCK_N,
        False,
    )
    dq = _attn_bwd_dq_inner(
        dq,
        q,
        do,
        lse,
        delta,
        k_base,
        v_base,
        stride_kn,
        stride_vn,
        offs_m,
        offs_d,
        mask_d,
        full,
        hi,
        SK,
        scale_log2,
        BLOCK_N,
        True,
    )

    tl.store(
        dq_ptr
        + b * stride_qb
        + h * stride_qh
        + offs_m64[:, None] * stride_qm
        + offs_d[None, :],
        (dq * scale).to(dq_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_d[None, :],
    )


def select_config(D: int, phase: str) -> Dict[str, int]:
    """Conservative launch geometry borrowed from the independently tested FA2 primitive.

    Forward: a 128 x 128 tile with 8 warps at ``D = 128`` (``acc`` is ``BLOCK_M x BLOCK_D``
    fp32; the same tile with 4 warps spills and runs 50x slower); 64-row tiles above 128.
    Backward: ``BLOCK_N1`` keys per dK/dV program iterating ``BLOCK_M1``-row Q blocks, and
    ``BLOCK_M2`` queries per dQ program iterating ``BLOCK_N2``-key blocks; the wide side is
    the accumulator's, the narrow side the recomputed ``P`` tile's.
    """
    if phase == "fwd":
        if D <= 64:
            return dict(BLOCK_M=128, BLOCK_N=64, num_warps=4, num_stages=3)
        if D <= 128:
            return dict(BLOCK_M=128, BLOCK_N=128, num_warps=8, num_stages=3)
        return dict(BLOCK_M=64, BLOCK_N=64, num_warps=8, num_stages=2)
    if D <= 128:
        return dict(
            BLOCK_M1=64,
            BLOCK_N1=64,
            BLOCK_M2=128,
            BLOCK_N2=32,
            num_warps=4,
            num_stages=2,
        )
    return dict(
        BLOCK_M1=32, BLOCK_N1=64, BLOCK_M2=64, BLOCK_N2=32, num_warps=8, num_stages=2
    )


def _forward(q, k, v, save_aux, meta, views):
    b, h, sq, d = q.shape
    sk = k.shape[2]
    out = torch.empty_like(q)
    lse = (
        torch.empty((b, h, sq), device=q.device, dtype=torch.float32)
        if save_aux
        else torch.empty(0, device=q.device)
    )
    cfg = select_config(d, "fwd")
    _attn_fwd_kernel[(triton.cdiv(sq, cfg["BLOCK_M"]), b * h)](
        q,
        meta,
        views,
        k,
        v,
        out,
        lse,
        *q.stride()[:3],
        *k.stride()[:3],
        *v.stride()[:3],
        *out.stride()[:3],
        h,
        sq,
        sk,
        d,
        d**-0.5 * LOG2E,
        SAVE_AUX=save_aux,
        BLOCK_D=triton.next_power_of_2(d),
        **cfg
    )
    return out, lse


def _backward(q, k, v, out, do, lse, meta, views):
    b, h, sq, d = q.shape
    sk = k.shape[2]
    if do.stride(-1) != 1:
        do = do.contiguous()  # Autograd may supply a broadcast upstream gradient.
    dq, dk, dv = (
        torch.empty_strided(x.shape, x.stride(), dtype=x.dtype, device=x.device)
        for x in (q, k, v)
    )
    delta = torch.empty_like(lse)
    bd = triton.next_power_of_2(d)
    cfg = select_config(d, "bwd")
    _attn_bwd_preprocess_kernel[(triton.cdiv(sq, 128), b * h)](
        out,
        do,
        delta,
        *out.stride()[:3],
        *do.stride()[:3],
        h,
        sq,
        d,
        BLOCK_M=128,
        BLOCK_D=bd,
        num_warps=4
    )
    common = (
        *q.stride()[:3],
        *k.stride()[:3],
        *v.stride()[:3],
        *do.stride()[:3],
        h,
        sq,
        sk,
        d,
        d**-0.5,
        d**-0.5 * LOG2E,
    )
    _attn_bwd_dkdv_kernel[(triton.cdiv(sk, cfg["BLOCK_N1"]), b * h)](
        q,
        meta,
        views,
        k,
        v,
        do,
        lse,
        delta,
        dk,
        dv,
        *common,
        BLOCK_M=cfg["BLOCK_M1"],
        BLOCK_N=cfg["BLOCK_N1"],
        BLOCK_D=bd,
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"]
    )
    _attn_bwd_dq_kernel[(triton.cdiv(sq, cfg["BLOCK_M2"]), b * h)](
        q,
        meta,
        views,
        k,
        v,
        do,
        lse,
        delta,
        dq,
        *common,
        BLOCK_M=cfg["BLOCK_M2"],
        BLOCK_N=cfg["BLOCK_N2"],
        BLOCK_D=bd,
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"]
    )
    return dq, dk, dv


def _validate(q, k, v, lengths, tokens_per_view):
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q/k/v must have the same (B,H,N,D) shape")
    if any(
        x.dtype != torch.bfloat16
        or x.device != q.device
        or not x.is_cuda
        or x.stride(-1) != 1
        for x in (q, k, v)
    ):
        raise ValueError(
            "q/k/v must be bf16 CUDA tensors on one device with unit last stride"
        )
    if not 1 <= q.shape[-1] <= 256:
        raise ValueError("head width must be in [1,256]")
    if type(tokens_per_view) is not int or tokens_per_view <= 0:
        raise ValueError("tokens_per_view must be a positive integer")
    if not isinstance(lengths, tuple) or len(lengths) != q.shape[0]:
        raise ValueError(
            "lengths must be immutable CPU tuples, one row per batch entry"
        )
    for row in lengths:
        if not isinstance(row, tuple) or len(row) * tokens_per_view != q.shape[2]:
            raise ValueError("each length row must describe every view")
        if any(type(n) is not int or not 0 <= n <= tokens_per_view for n in row):
            raise ValueError("prefix lengths must be integers in [0,tokens_per_view]")
    if any(x.shape[2] * x.stride(2) >= 2**31 for x in (q, k, v)):
        raise ValueError("each head plane must fit int32 offsets")


@triton.jit
def _pack(K, V, KP, VP, META, H: tl.constexpr, T: tl.constexpr, VIEWS: tl.constexpr,
          SK: tl.constexpr, D: tl.constexpr,
          KB: tl.constexpr, KH: tl.constexpr, KN: tl.constexpr,
          VB: tl.constexpr, VH: tl.constexpr, VN: tl.constexpr, BLOCK: tl.constexpr):
    bh = tl.program_id(1).to(tl.int64)
    b, h = bh // H, bh % H
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, d = i // D, i % D
    total = tl.load(META + b * (VIEWS + 1) + VIEWS)
    original = row
    for view in tl.static_range(VIEWS):
        start = tl.load(META + b * (VIEWS + 1) + view)
        end = tl.load(META + b * (VIEWS + 1) + view + 1)
        original = tl.where((row >= start) & (row < end), view * T + row - start, original)
    active = row < tl.maximum(1, total)
    kval = tl.load(K + b.to(tl.int64)*KB + h*KH + original*KN + d, active, 0)
    vval = tl.load(V + b.to(tl.int64)*VB + h*VH + original*VN + d, active, 0)
    tl.store(KP + bh.to(tl.int64)*SK*D + i, kval, active)
    tl.store(VP + bh.to(tl.int64)*SK*D + i, vval, active)


@triton.jit
def _scatter(DKP, DVP, DK, DV, META, H: tl.constexpr, T: tl.constexpr,
             VIEWS: tl.constexpr, SK: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr):
    bh = tl.program_id(1).to(tl.int64)
    b = bh // H
    i = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
    row, d = i // D, i % D
    active = row < VIEWS*T
    view = row // T
    start = tl.load(META + b*(VIEWS+1) + view, active, 0)
    end = tl.load(META + b*(VIEWS+1) + view + 1, active, 0)
    total = tl.load(META + b*(VIEWS+1) + VIEWS)
    valid = active & ((row % T < end - start) | ((total == 0) & (row == 0)))
    packed = start + row % T
    dk = tl.load(DKP + bh.to(tl.int64)*SK*D + packed*D + d, valid, 0)
    dv = tl.load(DVP + bh.to(tl.int64)*SK*D + packed*D + d, valid, 0)
    tl.store(DK + bh.to(tl.int64)*VIEWS*T*D + i, dk, active)
    tl.store(DV + bh.to(tl.int64)*VIEWS*T*D + i, dv, active)


def _compact(k, v, lengths, t):
    b, h, _, d = k.shape
    offsets = []
    for row in lengths:
        values = [0]
        for n in row:
            values.append(values[-1] + n)
        offsets.append(values)
    sk = max(1, max(row[-1] for row in offsets))
    meta = torch.tensor(offsets, dtype=torch.int32, device=k.device)
    kp = torch.empty((b,h,sk,d), dtype=k.dtype, device=k.device)
    vp = torch.empty_like(kp)
    _pack[(triton.cdiv(sk*d,1024), b*h)](k,v,kp,vp,meta,h,t,len(lengths[0]),sk,d,*k.stride()[:3],*v.stride()[:3],BLOCK=1024)
    return kp, vp, meta


class _BatchedAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,v,lengths,t,recompute):
        kp,vp,meta = _compact(k,v,lengths,t)
        views = len(lengths[0])
        out,lse = _forward(q,kp,vp,not recompute,meta,views)
        ctx.save_for_backward(q,kp,vp,out,lse,meta)
        ctx.t,ctx.views,ctx.recompute = t,views,recompute
        return out

    @staticmethod
    def backward(ctx,do):
        q,kp,vp,out,lse,meta = ctx.saved_tensors
        if ctx.recompute:
            out,lse = _forward(q,kp,vp,True,meta,ctx.views)
        dq,dkp,dvp = _backward(q,kp,vp,out,do,lse,meta,ctx.views)
        b,h,n,d = q.shape
        dk = torch.empty(q.shape,device=q.device,dtype=q.dtype)
        dv = torch.empty_like(dk)
        _scatter[(triton.cdiv(n*d,1024),b*h)](dkp,dvp,dk,dv,meta,h,ctx.t,ctx.views,kp.shape[2],d,BLOCK=1024)
        return dq,dk,dv,None,None,None


def segmented_attention(q,k,v,lengths,tokens_per_view,*,recompute=False):
    _validate(q,k,v,lengths,tokens_per_view)
    if q.shape[0] == 0 or q.shape[2] == 0:
        return q+k+v
    if torch.is_grad_enabled() and any(x.requires_grad for x in (q,k,v)):
        return _BatchedAttention.apply(q,k,v,lengths,tokens_per_view,recompute)
    kp,vp,meta = _compact(k,v,lengths,tokens_per_view)
    return _forward(q,kp,vp,False,meta,len(lengths[0]))[0]
