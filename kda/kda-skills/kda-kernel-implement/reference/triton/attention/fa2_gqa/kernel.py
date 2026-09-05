"""FlashAttention-2, full (non-causal) grouped-query attention in Triton: ``O = softmax(Q K^T * scale) V``.

Layout ``q, o: (B, H, S_q, D)``, ``k, v: (B, H_kv, S_kv, D)`` with ``H % H_kv == 0`` (query head
``h`` reads K,V head ``h // (H // H_kv)``; ``H_kv == H`` is plain MHA), unit stride in ``D``, any
other strides. ``S_q`` and ``S_kv`` are independent (encoder-style ``S_q >> S_kv``, decode-style
``S_q << S_kv`` both run). ``D <= 256``, padded to a power of two in registers. Storage dtype
bf16/fp16 on the tensor cores, fp32 for scores, softmax, accumulators and the saved ``lse``.

Forward (one program per ``(B*H, BLOCK_M rows of Q)``)::

    for each K,V block n of head h // GROUP:                # all of S_kv: no mask except the tail
        s = Q K_n^T * (scale * log2 e);  online softmax in the log2 domain;  O = O * alpha + p V_n
    O /= l;  lse = m + log2(l)                              # (B, H, S_q) fp32

Backward, deterministic (no atomics)::

    delta = rowsum(O * dO)                                  # (B, H, S_q) fp32
    dK/dV kernel: one program per (B*H_kv, K,V block); loops over the GROUP query heads that
        share this K,V head and, for each, over all Q blocks:
        pT = 2^(K Q^T * scale_log2 - lse);  dV += pT dO;  dpT = V dO^T;  dsT = pT (dpT - delta);  dK += dsT Q
    dQ kernel: one program per (B*H, Q block); loops over all K,V blocks of head h // GROUP:
        p = 2^(Q K^T * scale_log2 - lse);  dp = dO V^T;  ds = p (dp - delta);  dQ += ds K

The GQA reduction of ``dK``, ``dV`` over the ``GROUP`` query heads happens inside one program
(a loop, fp32 accumulator), not with atomics across programs: bitwise reproducible, and the
K,V tile is loaded once for the whole group. ``GROUP`` is a runtime int (the head loop is a
plain ``range``), so one compiled kernel serves every ``H / H_kv``.

Saved for the backward: ``q, k, v, o`` and ``lse``. Offsets derived from program ids are
int64 (``B*H*S_q*S_kv`` is a logical index past 2^31 at ordinary sizes).
"""

from typing import Dict, Optional, Tuple

import torch
import triton
import triton.language as tl

_MMA_DTYPES = (torch.bfloat16, torch.float16)
LOG2E = 1.4426950408889634


@triton.jit
def _attn_fwd_inner(
    acc, l_i, m_i, q, k_base, v_base, stride_kn, stride_vn, offs_d, mask_d,
    lo, hi, S_kv, scale_log2,
    BLOCK_N: tl.constexpr, MASK: tl.constexpr,
):
    """K,V blocks ``[lo, hi)``; ``MASK`` only for the block holding the ``S_kv`` tail."""
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + tl.arange(0, BLOCK_N)
        if MASK:
            mask_n = offs_n < S_kv
            k = tl.load(k_base + offs_d[:, None] + offs_n[None, :] * stride_kn, mask=mask_d[:, None] & mask_n[None, :], other=0.0)
        else:
            k = tl.load(k_base + offs_d[:, None] + offs_n[None, :] * stride_kn, mask=mask_d[:, None], other=0.0)
        s = tl.dot(q, k) * scale_log2  # K loaded transposed: (BLOCK_D, BLOCK_N), no tl.trans
        if MASK:
            s = tl.where(mask_n[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(s - m_new[:, None])
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
    q_ptr, k_ptr, v_ptr, o_ptr, lse_ptr,
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_ob, stride_oh, stride_om,
    H, GROUP, S_q, S_kv, D, scale_log2,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    start_m = tl.program_id(0)
    bh = tl.program_id(1).to(tl.int64)
    b = bh // H
    h = bh % H
    hkv = h // GROUP
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)  # int32: compared against keys in the loop
    offs_m64 = start_m.to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)  # int64: this tile's own addresses
    offs_d = tl.arange(0, BLOCK_D)
    mask_m = offs_m < S_q
    mask_d = offs_d < D

    q = tl.load(q_ptr + b * stride_qb + h * stride_qh + offs_m64[:, None] * stride_qm + offs_d[None, :], mask=mask_m[:, None] & mask_d[None, :], other=0.0)
    k_base = k_ptr + b * stride_kb + hkv * stride_kh
    v_base = v_ptr + b * stride_vb + hkv * stride_vh

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)

    full = (S_kv // BLOCK_N) * BLOCK_N  # whole blocks need no mask; the tail block does
    acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, k_base, v_base, stride_kn, stride_vn, offs_d, mask_d, 0, full, S_kv, scale_log2, BLOCK_N, False)
    acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, k_base, v_base, stride_kn, stride_vn, offs_d, mask_d, full, S_kv, S_kv, scale_log2, BLOCK_N, True)

    acc = acc / l_i[:, None]
    tl.store(o_ptr + b * stride_ob + h * stride_oh + offs_m64[:, None] * stride_om + offs_d[None, :], acc.to(o_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_d[None, :])
    tl.store(lse_ptr + bh * S_q + offs_m, m_i + tl.math.log2(l_i), mask=mask_m)


@triton.jit
def _attn_bwd_preprocess_kernel(
    o_ptr, do_ptr, delta_ptr,
    stride_ob, stride_oh, stride_om,
    stride_dob, stride_doh, stride_dom,
    H, S_q, D,
    BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr,
):
    start_m = tl.program_id(0)
    bh = tl.program_id(1).to(tl.int64)
    b = bh // H
    h = bh % H
    offs_m = start_m.to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    mask = (offs_m < S_q)[:, None] & (offs_d < D)[None, :]
    o = tl.load(o_ptr + b * stride_ob + h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :], mask=mask, other=0.0).to(tl.float32)
    do = tl.load(do_ptr + b * stride_dob + h * stride_doh + offs_m[:, None] * stride_dom + offs_d[None, :], mask=mask, other=0.0).to(tl.float32)
    tl.store(delta_ptr + bh * S_q + offs_m, tl.sum(o * do, axis=1), mask=offs_m < S_q)


@triton.jit
def _attn_bwd_dkdv_kernel(
    q_ptr, k_ptr, v_ptr, do_ptr, lse_ptr, delta_ptr, dk_ptr, dv_ptr,
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_dob, stride_doh, stride_dom,
    H, H_kv, GROUP, S_q, S_kv, D, scale, scale_log2,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    start_n = tl.program_id(0)
    bhkv = tl.program_id(1).to(tl.int64)
    b = bhkv // H_kv
    hkv = bhkv % H_kv
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)  # int32: compared against queries in the loop
    offs_n64 = start_n.to(tl.int64) * BLOCK_N + tl.arange(0, BLOCK_N)  # int64: this tile's own addresses
    offs_d = tl.arange(0, BLOCK_D)
    mask_n = offs_n < S_kv
    mask_d = offs_d < D

    k = tl.load(k_ptr + b * stride_kb + hkv * stride_kh + offs_n64[:, None] * stride_kn + offs_d[None, :], mask=mask_n[:, None] & mask_d[None, :], other=0.0)
    v = tl.load(v_ptr + b * stride_vb + hkv * stride_vh + offs_n64[:, None] * stride_vn + offs_d[None, :], mask=mask_n[:, None] & mask_d[None, :], other=0.0)
    dk = tl.zeros([BLOCK_N, BLOCK_D], tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_D], tl.float32)

    # The GROUP query heads sharing this K,V head, reduced in this program: deterministic, and
    # the K,V tile stays in registers across the whole group.
    for g in range(0, GROUP):
        h = hkv * GROUP + g
        q_base = q_ptr + b * stride_qb + h * stride_qh
        do_base = do_ptr + b * stride_dob + h * stride_doh
        lse_base = lse_ptr + (b * H + h) * S_q
        delta_base = delta_ptr + (b * H + h) * S_q
        for start_m in range(0, S_q, BLOCK_M):
            start_m = tl.multiple_of(start_m, BLOCK_M)
            offs_m = start_m + tl.arange(0, BLOCK_M)
            mask_m = offs_m < S_q
            q = tl.load(q_base + offs_m[:, None] * stride_qm + offs_d[None, :], mask=mask_m[:, None] & mask_d[None, :], other=0.0)
            do = tl.load(do_base + offs_m[:, None] * stride_dom + offs_d[None, :], mask=mask_m[:, None] & mask_d[None, :], other=0.0)
            lse = tl.load(lse_base + offs_m, mask=mask_m, other=float("inf"))  # tail rows: p = 0
            delta = tl.load(delta_base + offs_m, mask=mask_m, other=0.0)
            sT = tl.dot(k, tl.trans(q)) * scale_log2  # (BLOCK_N, BLOCK_M)
            pT = tl.math.exp2(sT - lse[None, :])
            dv += tl.dot(pT.to(do.dtype), do)
            dpT = tl.dot(v, tl.trans(do))
            dsT = pT * (dpT - delta[None, :])
            dk += tl.dot(dsT.to(q.dtype), q)

    tl.store(dk_ptr + b * stride_kb + hkv * stride_kh + offs_n64[:, None] * stride_kn + offs_d[None, :], (dk * scale).to(dk_ptr.dtype.element_ty), mask=mask_n[:, None] & mask_d[None, :])
    tl.store(dv_ptr + b * stride_vb + hkv * stride_vh + offs_n64[:, None] * stride_vn + offs_d[None, :], dv.to(dv_ptr.dtype.element_ty), mask=mask_n[:, None] & mask_d[None, :])


@triton.jit
def _attn_bwd_dq_inner(
    dq, q, do, lse, delta, k_base, v_base, stride_kn, stride_vn, offs_d, mask_d,
    lo, hi, S_kv, scale_log2,
    BLOCK_N: tl.constexpr, MASK: tl.constexpr,
):
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + tl.arange(0, BLOCK_N)
        if MASK:
            mask_n = offs_n < S_kv
            k = tl.load(k_base + offs_n[:, None] * stride_kn + offs_d[None, :], mask=mask_n[:, None] & mask_d[None, :], other=0.0)
            v = tl.load(v_base + offs_n[:, None] * stride_vn + offs_d[None, :], mask=mask_n[:, None] & mask_d[None, :], other=0.0)
        else:
            k = tl.load(k_base + offs_n[:, None] * stride_kn + offs_d[None, :], mask=mask_d[None, :], other=0.0)
            v = tl.load(v_base + offs_n[:, None] * stride_vn + offs_d[None, :], mask=mask_d[None, :], other=0.0)
        s = tl.dot(q, tl.trans(k)) * scale_log2
        p = tl.math.exp2(s - lse[:, None])
        if MASK:
            p = tl.where(mask_n[None, :], p, 0.0)
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None])
        dq += tl.dot(ds.to(k.dtype), k)
    return dq


@triton.jit
def _attn_bwd_dq_kernel(
    q_ptr, k_ptr, v_ptr, do_ptr, lse_ptr, delta_ptr, dq_ptr,
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_dob, stride_doh, stride_dom,
    H, GROUP, S_q, S_kv, D, scale, scale_log2,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    start_m = tl.program_id(0)
    bh = tl.program_id(1).to(tl.int64)
    b = bh // H
    h = bh % H
    hkv = h // GROUP
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)  # int32: compared against keys in the loop
    offs_m64 = start_m.to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)  # int64: this tile's own addresses
    offs_d = tl.arange(0, BLOCK_D)
    mask_m = offs_m < S_q
    mask_d = offs_d < D

    q = tl.load(q_ptr + b * stride_qb + h * stride_qh + offs_m64[:, None] * stride_qm + offs_d[None, :], mask=mask_m[:, None] & mask_d[None, :], other=0.0)
    do = tl.load(do_ptr + b * stride_dob + h * stride_doh + offs_m64[:, None] * stride_dom + offs_d[None, :], mask=mask_m[:, None] & mask_d[None, :], other=0.0)
    lse = tl.load(lse_ptr + bh * S_q + offs_m, mask=mask_m, other=float("inf"))
    delta = tl.load(delta_ptr + bh * S_q + offs_m, mask=mask_m, other=0.0)
    dq = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)
    k_base = k_ptr + b * stride_kb + hkv * stride_kh
    v_base = v_ptr + b * stride_vb + hkv * stride_vh

    full = (S_kv // BLOCK_N) * BLOCK_N
    dq = _attn_bwd_dq_inner(dq, q, do, lse, delta, k_base, v_base, stride_kn, stride_vn, offs_d, mask_d, 0, full, S_kv, scale_log2, BLOCK_N, False)
    dq = _attn_bwd_dq_inner(dq, q, do, lse, delta, k_base, v_base, stride_kn, stride_vn, offs_d, mask_d, full, S_kv, S_kv, scale_log2, BLOCK_N, True)

    tl.store(dq_ptr + b * stride_qb + h * stride_qh + offs_m64[:, None] * stride_qm + offs_d[None, :], (dq * scale).to(dq_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_d[None, :])


# ----------------------------------------------------------------------------------------------
# Host side
# ----------------------------------------------------------------------------------------------


def select_config(D: int, phase: str) -> Dict[str, int]:
    """Launch geometry per head dim and phase, measured on A800 (SNIPPET.md); same table as `fa2_causal`."""
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
    if q.dim() != 4 or k.dim() != 4 or k.shape != v.shape:
        raise ValueError(f"attention: q (B, H, S_q, D), k and v (B, H_kv, S_kv, D) expected, got {tuple(q.shape)}, {tuple(k.shape)}, {tuple(v.shape)}")
    B, H, _, D = q.shape
    if k.shape[0] != B or k.shape[3] != D or H % k.shape[1] != 0:
        raise ValueError(f"attention: k/v batch or head dim mismatch, or H = {H} not a multiple of H_kv = {k.shape[1]}")
    if D > 256:
        raise ValueError(f"attention: head dim {D} > 256")
    for name, t in (("q", q), ("k", k), ("v", v)):
        if t.stride(-1) != 1:
            raise ValueError(f"attention: {name} must be contiguous in the head dim")
        # Offsets *inside* the K,V / Q loops are int32 (int64 there costs 11% on the forward);
        # the (b, h) base is int64. So one (b, h) plane must be addressable in int32.
        if t.shape[2] * t.stride(2) >= 2**31:
            raise ValueError(f"attention: one (batch, head) plane of {name} spans {t.shape[2] * t.stride(2)} elements >= 2^31")


def attention_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Full GQA attention forward. Returns ``o`` (q's shape/dtype/strides) and ``lse`` (``(B, H, S_q)`` fp32, log2 domain)."""
    _check(q, k, v)
    B, H, S_q, D = q.shape
    H_kv, S_kv = k.shape[1], k.shape[2]
    scale = D**-0.5 if scale is None else scale
    o = torch.empty_like(q)
    lse = torch.empty(B, H, S_q, dtype=torch.float32, device=q.device)
    cfg = select_config(D, "fwd")
    _attn_fwd_kernel[(triton.cdiv(S_q, cfg["BLOCK_M"]), B * H)](
        q, k, v, o, lse,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        H, H // H_kv, S_q, S_kv, D, scale * LOG2E,
        BLOCK_M=cfg["BLOCK_M"], BLOCK_N=cfg["BLOCK_N"], BLOCK_D=triton.next_power_of_2(D),
        num_warps=cfg["num_warps"], num_stages=cfg["num_stages"],
    )
    return o, lse


def attention_bwd(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, o: torch.Tensor, do: torch.Tensor, lse: torch.Tensor, scale: Optional[float] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Full GQA attention backward: ``dq (B, H, S_q, D)``, ``dk, dv (B, H_kv, S_kv, D)``, deterministic."""
    B, H, S_q, D = q.shape
    H_kv, S_kv = k.shape[1], k.shape[2]
    GROUP = H // H_kv
    scale = D**-0.5 if scale is None else scale
    if do.stride(-1) != 1:  # autograd hands dO in o's layout (unit stride in D) unless the consumer permuted it
        raise ValueError("attention backward: dO must be contiguous in the head dim (call .contiguous() in the caller)")
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    delta = torch.empty_like(lse)
    BLOCK_D = triton.next_power_of_2(D)
    cfg = select_config(D, "bwd")
    pre_block = 128
    _attn_bwd_preprocess_kernel[(triton.cdiv(S_q, pre_block), B * H)](
        o, do, delta,
        o.stride(0), o.stride(1), o.stride(2),
        do.stride(0), do.stride(1), do.stride(2),
        H, S_q, D, BLOCK_M=pre_block, BLOCK_D=BLOCK_D, num_warps=4,
    )
    strides = (
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        do.stride(0), do.stride(1), do.stride(2),
    )
    _attn_bwd_dkdv_kernel[(triton.cdiv(S_kv, cfg["BLOCK_N1"]), B * H_kv)](
        q, k, v, do, lse, delta, dk, dv, *strides,
        H, H_kv, GROUP, S_q, S_kv, D, scale, scale * LOG2E,
        BLOCK_M=cfg["BLOCK_M1"], BLOCK_N=cfg["BLOCK_N1"], BLOCK_D=BLOCK_D,
        num_warps=cfg["num_warps"], num_stages=cfg["num_stages"],
    )
    _attn_bwd_dq_kernel[(triton.cdiv(S_q, cfg["BLOCK_M2"]), B * H)](
        q, k, v, do, lse, delta, dq, *strides,
        H, GROUP, S_q, S_kv, D, scale, scale * LOG2E,
        BLOCK_M=cfg["BLOCK_M2"], BLOCK_N=cfg["BLOCK_N2"], BLOCK_D=BLOCK_D,
        num_warps=cfg["num_warps"], num_stages=cfg["num_stages"],
    )
    return dq, dk, dv


class _Attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, scale):
        o, lse = attention_fwd(q, k, v, scale)
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.scale = scale
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        dq, dk, dv = attention_bwd(q, k, v, o, do, lse, ctx.scale)
        return dq, dk, dv, None


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: Optional[float] = None) -> torch.Tensor:
    """``softmax(q k^T * scale) v`` over all keys; ``q: (B, H, S_q, D)``, ``k, v: (B, H_kv, S_kv, D)``."""
    return _Attention.apply(q, k, v, scale)


def attention_flops(B: int, H: int, S_q: int, S_kv: int, D: int, phase: str) -> float:
    """Tensor-core FLOPs: every (query, key) pair is attended; 2 GEMMs forward, 5 backward, 2 FLOP each."""
    per_pair = {"fwd": 4 * D, "infer": 4 * D, "bwd": 10 * D}[phase]
    return B * H * S_q * S_kv * per_pair


__all__ = ["attention", "attention_fwd", "attention_bwd", "attention_flops", "select_config", "LOG2E"]
