"""Standalone masked FA2 candidate; see GOLDEN.md for semantics and evidence status."""
from typing import Dict

import torch
import triton
import triton.language as tl

_MMA_DTYPES = (torch.bfloat16, torch.float16)
LOG2E = 1.4426950408889634


@triton.jit
def _probability_dot(a, b):
    """Retain the fp32 probability residual with two bf16 tensor-core products.

    The second component keeps about 16 significant bits of a, avoiding the
    single-bf16 probability rounding of ordinary fused attention.
    """
    high = a.to(tl.bfloat16)
    low = (a - high.to(tl.float32)).to(tl.bfloat16)
    return tl.dot(high, b, tl.dot(low, b))


@triton.jit
def _attn_fwd_inner(
    acc, l_i, m_i, q, k_base, v_base, stride_kn, stride_vn, offs_m, offs_d, mask_d,
    lo, hi, S, scale_log2, valid_base,
    BLOCK_N: tl.constexpr, MASK: tl.constexpr,
):
    """K,V blocks ``[lo, hi)``; ``MASK`` only for the blocks that touch the diagonal or the S tail."""
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + tl.arange(0, BLOCK_N)
        if MASK:
            mask_n = offs_n < S
            # K is loaded transposed, (BLOCK_D, BLOCK_N), so tl.dot needs no tl.trans (a layout
            # conversion through shared memory that costs ~10% at D = 128).
            k = tl.load(k_base + offs_d[:, None] + offs_n[None, :] * stride_kn, mask=mask_d[:, None] & mask_n[None, :], other=0.0)
        else:
            k = tl.load(k_base + offs_d[:, None] + offs_n[None, :] * stride_kn, mask=mask_d[:, None], other=0.0)
        s = tl.dot(q, k) * scale_log2  # (BLOCK_M, BLOCK_N) fp32, already in the log2 domain
        if MASK:
            s = tl.where(tl.load(valid_base + offs_n, mask=mask_n, other=False)[None, :] & mask_n[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        m_new = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        if MASK:
            v = tl.load(v_base + offs_n[:, None] * stride_vn + offs_d[None, :], mask=mask_n[:, None] & mask_d[None, :], other=0.0)
        else:
            v = tl.load(v_base + offs_n[:, None] * stride_vn + offs_d[None, :], mask=mask_d[None, :], other=0.0)
        acc += _probability_dot(p, v)
        m_i = m_new
    return acc, l_i, m_i


@triton.jit
def _attn_fwd_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, lse_ptr,
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_ob, stride_oh, stride_om,
    H, S, D, scale_log2, valid_ptr, valid_stride, SAVE_AUX: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    start_m = tl.program_id(0)
    bh = tl.program_id(1).to(tl.int64)
    b = bh // H
    h = bh % H
    valid_base = valid_ptr + b * valid_stride
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

    hi = S
    acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, k_base, v_base, stride_kn, stride_vn, offs_m, offs_d, mask_d, 0, hi, S, scale_log2, valid_base, BLOCK_N, True)

    acc = acc / l_i[:, None]
    o_base = o_ptr + b * stride_ob + h * stride_oh
    tl.store(o_base + offs_m64[:, None] * stride_om + offs_d[None, :], acc.to(o_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_d[None, :])
    # lse in the log2 domain: the backward recomputes p = 2^(s * scale_log2 - lse).
    if SAVE_AUX:
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
    lo, hi, S, scale_log2, valid_base,
    BLOCK_M: tl.constexpr, MASK: tl.constexpr,
):
    """Q blocks ``[lo, hi)`` for one K,V block; ``MASK`` for the blocks crossing the diagonal.

    Tiles are loaded in their natural ``(rows, BLOCK_D)`` layout and transposed only as the
    *B* operand of ``tl.dot`` (``k @ trans(q)``, ``v @ trans(do)``), which Triton folds into
    the MMA operand layout. Loading ``q`` transposed and transposing it back for ``dsT @ q``
    ran 1.3x slower; loading each tile twice in both layouts ran out of shared memory at the
    3-stage pipeline (SNIPPET.md).
    """
    for start_m in range(lo, hi, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        offs_m = start_m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < S
        q = tl.load(q_base + offs_m[:, None] * stride_qm + offs_d[None, :], mask=mask_m[:, None] & mask_d[None, :], other=0.0)
        do = tl.load(do_base + offs_m[:, None] * stride_dom + offs_d[None, :], mask=mask_m[:, None] & mask_d[None, :], other=0.0)
        lse = tl.load(lse_base + offs_m, mask=mask_m, other=float("inf"))  # tail rows: lse = +inf -> p = 0
        delta = tl.load(delta_base + offs_m, mask=mask_m, other=0.0)
        sT = tl.dot(k, tl.trans(q)) * scale_log2  # (BLOCK_N, BLOCK_M): keys down, queries across
        pT = tl.math.exp2(sT - lse[None, :])
        if MASK:
            pT = tl.where(tl.load(valid_base + offs_n, mask=offs_n < S, other=False)[:, None], pT, 0.0)
        dv += _probability_dot(pT, do)
        dpT = tl.dot(v, tl.trans(do))
        dsT = pT * (dpT - delta[None, :])
        dk += _probability_dot(dsT, q)
    return dk, dv


@triton.jit
def _attn_bwd_dkdv_kernel(
    q_ptr, k_ptr, v_ptr, do_ptr, lse_ptr, delta_ptr, dk_ptr, dv_ptr,
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_dob, stride_doh, stride_dom,
    H, S, D, scale, scale_log2, valid_ptr, valid_stride,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    start_n = tl.program_id(0)
    bh = tl.program_id(1).to(tl.int64)
    b = bh // H
    h = bh % H
    valid_base = valid_ptr + b * valid_stride
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)  # int32: compared against queries in the loop
    offs_n64 = start_n.to(tl.int64) * BLOCK_N + tl.arange(0, BLOCK_N)  # int64: this tile's own addresses
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

    lo = 0
    dk, dv = _attn_bwd_dkdv_inner(dk, dv, k, v, q_base, do_base, lse_base, delta_base, stride_qm, stride_dom, offs_n, offs_d, mask_d, lo, S, S, scale_log2, valid_base, BLOCK_M, True)

    dk_base = dk_ptr + b * stride_kb + h * stride_kh  # dk, dv are allocated with k's / v's strides
    dv_base = dv_ptr + b * stride_vb + h * stride_vh
    tl.store(dk_base + offs_n64[:, None] * stride_kn + offs_d[None, :], (dk * scale).to(dk_ptr.dtype.element_ty), mask=mask_n[:, None] & mask_d[None, :])
    tl.store(dv_base + offs_n64[:, None] * stride_vn + offs_d[None, :], dv.to(dv_ptr.dtype.element_ty), mask=mask_n[:, None] & mask_d[None, :])


@triton.jit
def _attn_bwd_dq_inner(
    dq, q, do, lse, delta, k_base, v_base, stride_kn, stride_vn, offs_m, offs_d, mask_d,
    lo, hi, S, scale_log2, valid_base,
    BLOCK_N: tl.constexpr, MASK: tl.constexpr,
):
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + tl.arange(0, BLOCK_N)
        if MASK:
            mask_n = offs_n < S
            k = tl.load(k_base + offs_n[:, None] * stride_kn + offs_d[None, :], mask=mask_n[:, None] & mask_d[None, :], other=0.0)
            v = tl.load(v_base + offs_n[:, None] * stride_vn + offs_d[None, :], mask=mask_n[:, None] & mask_d[None, :], other=0.0)
        else:
            k = tl.load(k_base + offs_n[:, None] * stride_kn + offs_d[None, :], mask=mask_d[None, :], other=0.0)
            v = tl.load(v_base + offs_n[:, None] * stride_vn + offs_d[None, :], mask=mask_d[None, :], other=0.0)
        s = tl.dot(q, tl.trans(k)) * scale_log2
        p = tl.math.exp2(s - lse[:, None])
        if MASK:
            p = tl.where(tl.load(valid_base + offs_n, mask=mask_n, other=False)[None, :] & mask_n[None, :], p, 0.0)
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None])
        dq += _probability_dot(ds, k)
    return dq


@triton.jit
def _attn_bwd_dq_kernel(
    q_ptr, k_ptr, v_ptr, do_ptr, lse_ptr, delta_ptr, dq_ptr,
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_dob, stride_doh, stride_dom,
    H, S, D, scale, scale_log2, valid_ptr, valid_stride,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    start_m = tl.program_id(0)
    bh = tl.program_id(1).to(tl.int64)
    b = bh // H
    h = bh % H
    valid_base = valid_ptr + b * valid_stride
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)  # int32: compared against keys in the loop
    offs_m64 = start_m.to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)  # int64: this tile's own addresses
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

    hi = S
    dq = _attn_bwd_dq_inner(dq, q, do, lse, delta, k_base, v_base, stride_kn, stride_vn, offs_m, offs_d, mask_d, 0, hi, S, scale_log2, valid_base, BLOCK_N, True)

    tl.store(dq_ptr + b * stride_qb + h * stride_qh + offs_m64[:, None] * stride_qm + offs_d[None, :], (dq * scale).to(dq_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_d[None, :])


# ----------------------------------------------------------------------------------------------
# Host side
# ----------------------------------------------------------------------------------------------


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
        return dict(BLOCK_M1=64, BLOCK_N1=64, BLOCK_M2=64, BLOCK_N2=64, num_warps=4, num_stages=2)
    return dict(BLOCK_M1=32, BLOCK_N1=64, BLOCK_M2=64, BLOCK_N2=32, num_warps=8, num_stages=2)


def _check(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    if q.dtype not in _MMA_DTYPES or k.dtype != q.dtype or v.dtype != q.dtype:
        raise TypeError(f"attention: q, k, v must share a bf16/fp16 dtype, got {q.dtype}, {k.dtype}, {v.dtype}")
    if q.dim() != 4 or k.shape != q.shape or v.shape != q.shape:
        raise ValueError(f"attention (causal, MHA): q, k, v must be (B, H, S, D) of one shape, got {tuple(q.shape)}, {tuple(k.shape)}, {tuple(v.shape)}")
    if q.shape[-1] > 256:
        raise ValueError(f"attention: head dim {q.shape[-1]} > 256")
    for name, t in (("q", q), ("k", k), ("v", v)):
        if t.stride(-1) != 1:
            raise ValueError(f"attention: {name} must be contiguous in the head dim")
        # Offsets *inside* the K,V / Q loops are int32 (int64 there costs 11% on the forward);
        # the (b, h) base is int64. So one (b, h) plane must be addressable in int32.
        if t.shape[2] * t.stride(2) >= 2**31:
            raise ValueError(f"attention: one (batch, head) plane of {name} spans {t.shape[2] * t.stride(2)} elements >= 2^31")


def attention_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, key_valid: torch.Tensor, save_aux: bool = True):
    """Return output and optional fp32 log2-normalizer; no score plane is allocated."""
    _check(q, k, v)
    B, H, S, D = q.shape
    if key_valid.shape != (B, S) or key_valid.dtype != torch.bool or key_valid.stride(-1) != 1:
        raise ValueError("key_valid must be a bool (B, S) tensor with unit key stride")
    o = torch.empty_like(q)
    lse = torch.empty((B, H, S) if save_aux else (0,), dtype=torch.float32, device=q.device)
    if S == 0 or B == 0:
        return o, lse
    cfg = select_config(D, "fwd")
    _attn_fwd_kernel[(triton.cdiv(S, cfg["BLOCK_M"]), B * H)](
        q, k, v, o, lse, *q.stride()[:3], *k.stride()[:3], *v.stride()[:3], *o.stride()[:3],
        H, S, D, D**-0.5 * LOG2E, key_valid, key_valid.stride(0), SAVE_AUX=save_aux,
        BLOCK_D=triton.next_power_of_2(D), **cfg)
    return o, lse


def attention_bwd(q, k, v, o, do, lse, key_valid):
    """Deterministic dQ/dK/dV using recomputed probability tiles."""
    B, H, S, D = q.shape
    if do.stride(-1) != 1:
        do = do.contiguous()
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    if S == 0 or B == 0:
        return dq, dk, dv
    delta = torch.empty_like(lse)
    bd = triton.next_power_of_2(D)
    cfg = select_config(D, "bwd")
    _attn_bwd_preprocess_kernel[(triton.cdiv(S, 128), B * H)](
        o, do, delta, *o.stride()[:3], *do.stride()[:3], H, S, D,
        BLOCK_M=128, BLOCK_D=bd, num_warps=4)
    common = (*q.stride()[:3], *k.stride()[:3], *v.stride()[:3], *do.stride()[:3],
              H, S, D, D**-0.5, D**-0.5 * LOG2E, key_valid, key_valid.stride(0),)
    _attn_bwd_dkdv_kernel[(triton.cdiv(S, cfg["BLOCK_N1"]), B * H)](
        q, k, v, do, lse, delta, dk, dv, *common,
        BLOCK_M=cfg["BLOCK_M1"], BLOCK_N=cfg["BLOCK_N1"], BLOCK_D=bd,
        num_warps=cfg["num_warps"], num_stages=cfg["num_stages"])
    _attn_bwd_dq_kernel[(triton.cdiv(S, cfg["BLOCK_M2"]), B * H)](
        q, k, v, do, lse, delta, dq, *common,
        BLOCK_M=cfg["BLOCK_M2"], BLOCK_N=cfg["BLOCK_N2"], BLOCK_D=bd,
        num_warps=cfg["num_warps"], num_stages=cfg["num_stages"])
    return dq, dk, dv


class _Attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, mask, recompute):
        o, lse = attention_fwd(q, k, v, mask, not recompute)
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.mask = mask
        ctx.recompute = recompute
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        if ctx.recompute:
            o, lse = attention_fwd(q, k, v, ctx.mask, True)
        return (*attention_bwd(q, k, v, o, do, lse, ctx.mask), None, None)


def padded_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, key_valid: torch.Tensor, *, recompute: bool = False):
    """Training attention with optional recomputation; inference omits normalizer stores."""
    if not torch.is_grad_enabled() or not any(t.requires_grad for t in (q, k, v)):
        return attention_fwd(q, k, v, key_valid, False)[0]
    return _Attention.apply(q, k, v, key_valid, recompute)
