"""FlashAttention-2, full (non-causal) grouped-query attention in TileLang: ``O = softmax(Q K^T * scale) V``.

The TileLang twin of ``reference/triton/attention/fa2_gqa/kernel.py``: same math, host
signatures and test. Layout ``q, o: (B, H, S_q, D)``, ``k, v: (B, H_kv, S_kv, D)``,
``H % H_kv == 0`` (query head ``h`` reads K,V head ``h // GROUP``), ``S_q`` and ``S_kv``
independent, ``D`` a multiple of 16. The structure is the causal kernel's
(``fa2_causal/kernel.py``, read its module docstring and SNIPPET first) with:

* no diagonal: the K,V loop covers all of ``S_kv`` and the only masked block is the tail
  (``acc_s`` initialised to ``-inf`` where ``n0 + j >= S_kv``, ``T.clear`` elsewhere);
* ``GROUP = H // H_kv`` is a Python int (one variant per group ratio); the K,V head is
  ``by // GROUP`` in the forward and dQ kernels;
* the dK/dV program owns one ``(b, h_kv, K,V block)`` and loops ``for g in T.serial(GROUP)``
  over the query heads sharing it, then over all ``S_q`` blocks: the group reduction lands in
  its fp32 fragments (deterministic, no ``T.atomic_add``), the K,V tile is read once per group.
"""

import logging
from typing import Dict, Optional, Tuple

import tilelang
import tilelang.language as T
import torch

logging.getLogger("tilelang").setLevel(logging.WARNING)

_MMA_DTYPES = (torch.bfloat16, torch.float16)
_TL_DTYPE = {torch.bfloat16: T.bfloat16, torch.float16: T.float16}
LOG2E = 1.4426950408889634


@tilelang.jit
def _attn_fwd_kernel(
    q, k, v,
    sqb: int, sqh: int, sqm: int, skb: int, skh: int, skn: int, svb: int, svh: int, svn: int,
    GROUP: int, scale_log2: float, block_M: int, block_N: int, num_stages: int, threads: int,
    dtype=T.bfloat16, accum_dtype=T.float32,
):
    B, H = T.dynamic("B"), T.dynamic("H")  # grid-only extents: dynamic is free and saves a compile per shape (fa2_causal measured)
    S_q, D = T.const("S_q, D")  # loop bounds and tiles stay static: a dynamic S costs 13% device time
    H_kv, S_kv = T.const("H_kv, S_kv")
    q: T.StridedTensor((B, H, S_q, D), (sqb, sqh, sqm, 1), dtype)
    k: T.StridedTensor((B, H_kv, S_kv, D), (skb, skh, skn, 1), dtype)
    v: T.StridedTensor((B, H_kv, S_kv, D), (svb, svh, svn, 1), dtype)
    o = T.empty((B, H, S_q, D), dtype)
    lse = T.empty((B, H, S_q), accum_dtype)

    with T.Kernel(T.ceildiv(S_q, block_M), H, B, threads=threads) as (bx, by, bz):
        q_s = T.alloc_shared((block_M, D), dtype)
        k_s = T.alloc_shared((block_N, D), dtype)
        v_s = T.alloc_shared((block_N, D), dtype)
        o_s = T.alloc_shared((block_M, D), dtype)
        acc_s = T.alloc_fragment((block_M, block_N), accum_dtype)
        acc_s_cast = T.alloc_fragment((block_M, block_N), dtype)
        acc_o = T.alloc_fragment((block_M, D), accum_dtype)
        m_i = T.alloc_fragment((block_M,), accum_dtype)
        m_prev = T.alloc_fragment((block_M,), accum_dtype)
        alpha = T.alloc_fragment((block_M,), accum_dtype)
        row_sum = T.alloc_fragment((block_M,), accum_dtype)
        l_i = T.alloc_fragment((block_M,), accum_dtype)

        m0 = bx * block_M
        hkv = by // GROUP
        T.copy(q[bz, by, m0 : m0 + block_M, :], q_s)
        T.fill(acc_o, 0)
        T.fill(l_i, 0)
        T.fill(m_i, -T.infinity(accum_dtype))

        for kb in T.Pipelined(T.ceildiv(S_kv, block_N), num_stages=num_stages):
            n0 = kb * block_N
            T.copy(k[bz, hkv, n0 : n0 + block_N, :], k_s)
            if n0 + block_N <= S_kv:
                T.clear(acc_s)
            else:  # the S_kv tail: zero-filled K rows must not score 0, they must be absent
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.if_then_else(n0 + j < S_kv, 0, -T.infinity(accum_dtype))
            T.gemm(q_s, k_s, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
            T.copy(m_i, m_prev)
            T.reduce_max(acc_s, m_i, dim=1, clear=False)
            for i in T.Parallel(block_M):
                alpha[i] = T.exp2((m_prev[i] - m_i[i]) * scale_log2)
            for i, j in T.Parallel(block_M, block_N):
                acc_s[i, j] = T.exp2((acc_s[i, j] - m_i[i]) * scale_log2)
            T.reduce_sum(acc_s, row_sum, dim=1)
            for i in T.Parallel(block_M):
                l_i[i] = l_i[i] * alpha[i] + row_sum[i]
            for i, j in T.Parallel(block_M, D):
                acc_o[i, j] *= alpha[i]
            T.copy(acc_s, acc_s_cast)
            T.copy(v[bz, hkv, n0 : n0 + block_N, :], v_s)
            T.gemm(acc_s_cast, v_s, acc_o, policy=T.GemmWarpPolicy.FullRow)
        for i, j in T.Parallel(block_M, D):
            acc_o[i, j] /= l_i[i]
        T.copy(acc_o, o_s)
        T.copy(o_s, o[bz, by, m0 : m0 + block_M, :])
        for i in T.Parallel(block_M):
            l_i[i] = m_i[i] * scale_log2 + T.log2(l_i[i])
        T.copy(l_i, lse[bz, by, m0 : m0 + block_M])
    return o, lse


@tilelang.jit
def _attn_bwd_preprocess_kernel(
    o, do, sob: int, soh: int, som: int, sdb: int, sdh: int, sdm: int, block_M: int,
    dtype=T.bfloat16, accum_dtype=T.float32,
):
    # delta = rowsum(o * do). Tiles are staged in shared memory and each thread reduces one
    # row with a serial loop: the fragment form (T.copy into a (block_M, D) fp32 fragment +
    # T.reduce_sum) only lowers for power-of-two D ("no available layout found" at D = 96,
    # 160, 192, 288), and this kernel reads 2 tiles per row, so its speed does not matter.
    B, H = T.dynamic("B"), T.dynamic("H")
    S, D = T.const("S, D")  # S is S_q here
    o: T.StridedTensor((B, H, S, D), (sob, soh, som, 1), dtype)
    do: T.StridedTensor((B, H, S, D), (sdb, sdh, sdm, 1), dtype)
    delta = T.empty((B, H, S), accum_dtype)
    with T.Kernel(T.ceildiv(S, block_M), H, B, threads=block_M) as (bx, by, bz):
        o_s = T.alloc_shared((block_M, D), dtype)
        do_s = T.alloc_shared((block_M, D), dtype)
        m0 = bx * block_M
        T.copy(o[bz, by, m0 : m0 + block_M, :], o_s)
        T.copy(do[bz, by, m0 : m0 + block_M, :], do_s)
        for i in T.Parallel(block_M):
            acc = T.alloc_var(accum_dtype)
            acc = 0.0
            for j in T.serial(D):
                acc += T.cast(o_s[i, j], accum_dtype) * T.cast(do_s[i, j], accum_dtype)
            if m0 + i < S:
                delta[bz, by, m0 + i] = acc
    return delta


@tilelang.jit
def _attn_bwd_dkdv_kernel(
    q, k, v, do, lse, delta, dk, dv,
    sqb: int, sqh: int, sqm: int, skb: int, skh: int, skn: int, svb: int, svh: int, svn: int, sdb: int, sdh: int, sdm: int,
    GROUP: int, scale: float, scale_log2: float, block_M: int, block_N: int, num_stages: int, threads: int,
    dtype=T.bfloat16, accum_dtype=T.float32,
):
    B, H = T.dynamic("B"), T.dynamic("H")  # grid-only extents: dynamic is free and saves a compile per shape (fa2_causal measured)
    S_q, D = T.const("S_q, D")  # loop bounds and tiles stay static: a dynamic S costs 13% device time
    H_kv, S_kv = T.const("H_kv, S_kv")
    q: T.StridedTensor((B, H, S_q, D), (sqb, sqh, sqm, 1), dtype)
    k: T.StridedTensor((B, H_kv, S_kv, D), (skb, skh, skn, 1), dtype)
    v: T.StridedTensor((B, H_kv, S_kv, D), (svb, svh, svn, 1), dtype)
    do: T.StridedTensor((B, H, S_q, D), (sdb, sdh, sdm, 1), dtype)
    lse: T.Tensor((B, H, S_q), accum_dtype)
    delta: T.Tensor((B, H, S_q), accum_dtype)
    dk: T.StridedTensor((B, H_kv, S_kv, D), (skb, skh, skn, 1), dtype)
    dv: T.StridedTensor((B, H_kv, S_kv, D), (svb, svh, svn, 1), dtype)

    with T.Kernel(T.ceildiv(S_kv, block_N), H_kv, B, threads=threads) as (bx, by, bz):
        k_s = T.alloc_shared((block_N, D), dtype)
        v_s = T.alloc_shared((block_N, D), dtype)
        q_s = T.alloc_shared((block_M, D), dtype)
        do_s = T.alloc_shared((block_M, D), dtype)
        lse_s = T.alloc_shared((block_M,), accum_dtype)
        delta_s = T.alloc_shared((block_M,), accum_dtype)
        sT = T.alloc_fragment((block_N, block_M), accum_dtype)
        dpT = T.alloc_fragment((block_N, block_M), accum_dtype)
        pT_cast = T.alloc_fragment((block_N, block_M), dtype)
        dsT_cast = T.alloc_fragment((block_N, block_M), dtype)
        dk_f = T.alloc_fragment((block_N, D), accum_dtype)
        dv_f = T.alloc_fragment((block_N, D), accum_dtype)
        dk_s = T.alloc_shared((block_N, D), dtype)
        dv_s = T.alloc_shared((block_N, D), dtype)

        n0 = bx * block_N
        T.copy(k[bz, by, n0 : n0 + block_N, :], k_s)
        T.copy(v[bz, by, n0 : n0 + block_N, :], v_s)
        T.clear(dk_f)
        T.clear(dv_f)

        # The GROUP query heads sharing this K,V head, reduced in this program's fragments.
        for g in T.serial(GROUP):
            h = by * GROUP + g
            for mb in T.Pipelined(T.ceildiv(S_q, block_M), num_stages=num_stages):
                m0 = mb * block_M
                T.copy(q[bz, h, m0 : m0 + block_M, :], q_s)
                T.copy(do[bz, h, m0 : m0 + block_M, :], do_s)
                T.copy(lse[bz, h, m0 : m0 + block_M], lse_s)
                T.copy(delta[bz, h, m0 : m0 + block_M], delta_s)
                T.clear(sT)
                T.gemm(k_s, q_s, sT, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.clear(dpT)
                T.gemm(v_s, do_s, dpT, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                # `m0 + j < S_q`: the 1-D copies of lse/delta past S_q leave the shared tail
                # uninitialised; select the padded query columns out, never multiply by zero.
                for i, j in T.Parallel(block_N, block_M):
                    sT[i, j] = T.if_then_else(m0 + j < S_q, T.exp2(sT[i, j] * scale_log2 - lse_s[j]), 0)
                T.copy(sT, pT_cast)
                T.gemm(pT_cast, do_s, dv_f, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(block_N, block_M):
                    dsT_cast[i, j] = T.if_then_else(m0 + j < S_q, sT[i, j] * (dpT[i, j] - delta_s[j]), 0)
                T.gemm(dsT_cast, q_s, dk_f, policy=T.GemmWarpPolicy.FullRow)

        for i, j in T.Parallel(block_N, D):
            dk_f[i, j] *= scale
        T.copy(dk_f, dk_s)
        T.copy(dk_s, dk[bz, by, n0 : n0 + block_N, :])
        T.copy(dv_f, dv_s)
        T.copy(dv_s, dv[bz, by, n0 : n0 + block_N, :])


@tilelang.jit
def _attn_bwd_dq_kernel(
    q, k, v, do, lse, delta, dq,
    sqb: int, sqh: int, sqm: int, skb: int, skh: int, skn: int, svb: int, svh: int, svn: int, sdb: int, sdh: int, sdm: int,
    GROUP: int, scale: float, scale_log2: float, block_M: int, block_N: int, num_stages: int, threads: int,
    dtype=T.bfloat16, accum_dtype=T.float32,
):
    B, H = T.dynamic("B"), T.dynamic("H")  # grid-only extents: dynamic is free and saves a compile per shape (fa2_causal measured)
    S_q, D = T.const("S_q, D")  # loop bounds and tiles stay static: a dynamic S costs 13% device time
    H_kv, S_kv = T.const("H_kv, S_kv")
    q: T.StridedTensor((B, H, S_q, D), (sqb, sqh, sqm, 1), dtype)
    k: T.StridedTensor((B, H_kv, S_kv, D), (skb, skh, skn, 1), dtype)
    v: T.StridedTensor((B, H_kv, S_kv, D), (svb, svh, svn, 1), dtype)
    do: T.StridedTensor((B, H, S_q, D), (sdb, sdh, sdm, 1), dtype)
    lse: T.Tensor((B, H, S_q), accum_dtype)
    delta: T.Tensor((B, H, S_q), accum_dtype)
    dq: T.StridedTensor((B, H, S_q, D), (sqb, sqh, sqm, 1), dtype)

    with T.Kernel(T.ceildiv(S_q, block_M), H, B, threads=threads) as (bx, by, bz):
        q_s = T.alloc_shared((block_M, D), dtype)
        do_s = T.alloc_shared((block_M, D), dtype)
        k_s = T.alloc_shared((block_N, D), dtype)
        v_s = T.alloc_shared((block_N, D), dtype)
        lse_s = T.alloc_shared((block_M,), accum_dtype)
        delta_s = T.alloc_shared((block_M,), accum_dtype)
        s = T.alloc_fragment((block_M, block_N), accum_dtype)
        dp = T.alloc_fragment((block_M, block_N), accum_dtype)
        ds_cast = T.alloc_fragment((block_M, block_N), dtype)
        dq_f = T.alloc_fragment((block_M, D), accum_dtype)
        out_s = T.alloc_shared((block_M, D), dtype)

        m0 = bx * block_M
        hkv = by // GROUP
        T.copy(q[bz, by, m0 : m0 + block_M, :], q_s)
        T.copy(do[bz, by, m0 : m0 + block_M, :], do_s)
        T.copy(lse[bz, by, m0 : m0 + block_M], lse_s)
        T.copy(delta[bz, by, m0 : m0 + block_M], delta_s)
        T.clear(dq_f)

        for kb in T.Pipelined(T.ceildiv(S_kv, block_N), num_stages=num_stages):
            n0 = kb * block_N
            T.copy(k[bz, hkv, n0 : n0 + block_N, :], k_s)
            T.copy(v[bz, hkv, n0 : n0 + block_N, :], v_s)
            T.clear(s)
            T.gemm(q_s, k_s, s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
            T.clear(dp)
            T.gemm(do_s, v_s, dp, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
            for i, j in T.Parallel(block_M, block_N):
                s[i, j] = T.if_then_else(n0 + j < S_kv, T.exp2(s[i, j] * scale_log2 - lse_s[i]), 0)
            for i, j in T.Parallel(block_M, block_N):
                ds_cast[i, j] = s[i, j] * (dp[i, j] - delta_s[i])
            T.gemm(ds_cast, k_s, dq_f, policy=T.GemmWarpPolicy.FullRow)

        for i, j in T.Parallel(block_M, D):
            dq_f[i, j] *= scale
        T.copy(dq_f, out_s)
        T.copy(out_s, dq[bz, by, m0 : m0 + block_M, :])


# ----------------------------------------------------------------------------------------------
# Host side
# ----------------------------------------------------------------------------------------------


def select_config(D: int, phase: str) -> Dict[str, int]:
    """Launch geometry per head dim and phase, measured on A800 (SNIPPET.md has the sweep)."""
    if phase == "fwd":
        if D <= 128:
            return dict(block_M=128, block_N=128, num_stages=1, threads=256)
        return dict(block_M=64, block_N=64, num_stages=1, threads=128)
    # Backward: num_stages=1 is *required* (2 stages race on the K block whose Q loop has a
    # single, S-tail iteration: nondeterministic dk/dv) and is also faster (11.4 vs 15.3 ms at
    # 8192 tokens): five GEMM operands per iteration leave no shared memory for a second stage.
    if D <= 128:
        return dict(block_M1=64, block_N1=64, block_M2=64, block_N2=64, num_stages=1, threads=128)
    return dict(block_M1=32, block_N1=64, block_M2=64, block_N2=32, num_stages=1, threads=128)


def _check(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    if q.dtype not in _MMA_DTYPES or k.dtype != q.dtype or v.dtype != q.dtype:
        raise TypeError(f"attention: q, k, v must share a bf16/fp16 dtype, got {q.dtype}, {k.dtype}, {v.dtype}")
    if q.dim() != 4 or k.dim() != 4 or k.shape != v.shape:
        raise ValueError(f"attention: q (B, H, S_q, D), k and v (B, H_kv, S_kv, D) expected, got {tuple(q.shape)}, {tuple(k.shape)}, {tuple(v.shape)}")
    B, H, _, D = q.shape
    if k.shape[0] != B or k.shape[3] != D or H % k.shape[1] != 0:
        raise ValueError(f"attention: k/v batch or head dim mismatch, or H = {H} not a multiple of H_kv = {k.shape[1]}")
    if D % 16 or D > 256:
        raise ValueError(f"attention: head dim {D} must be a multiple of 16 and <= 256")
    for name, t in (("q", q), ("k", k), ("v", v)):
        if t.stride(-1) != 1:
            raise ValueError(f"attention: {name} must be contiguous in the head dim")


def _strides(t: torch.Tensor) -> Tuple[int, int, int]:
    return int(t.stride(0)), int(t.stride(1)), int(t.stride(2))


def attention_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Full GQA attention forward. Returns ``o`` (contiguous ``(B, H, S_q, D)``) and ``lse`` (``(B, H, S_q)`` fp32, log2 domain)."""
    _check(q, k, v)
    D = q.shape[-1]
    scale = D**-0.5 if scale is None else scale
    cfg = select_config(D, "fwd")
    return _attn_fwd_kernel(q, k, v, *_strides(q), *_strides(k), *_strides(v), q.shape[1] // k.shape[1], float(scale * LOG2E), **cfg, dtype=_TL_DTYPE[q.dtype])


def attention_bwd(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, o: torch.Tensor, do: torch.Tensor, lse: torch.Tensor, scale: Optional[float] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Full GQA attention backward: ``dq (B, H, S_q, D)``, ``dk, dv (B, H_kv, S_kv, D)``, deterministic."""
    D = q.shape[-1]
    GROUP = q.shape[1] // k.shape[1]
    scale = D**-0.5 if scale is None else scale
    if do.stride(-1) != 1:
        raise ValueError("attention backward: dO must be contiguous in the head dim (call .contiguous() in the caller)")
    dt = _TL_DTYPE[q.dtype]
    delta = _attn_bwd_preprocess_kernel(o, do, *_strides(o), *_strides(do), 128, dtype=dt)
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    cfg = select_config(D, "bwd")
    strides = (*_strides(q), *_strides(k), *_strides(v), *_strides(do))
    _attn_bwd_dkdv_kernel(
        q, k, v, do, lse, delta, dk, dv, *strides, GROUP, float(scale), float(scale * LOG2E),
        cfg["block_M1"], cfg["block_N1"], cfg["num_stages"], cfg["threads"], dtype=dt,
    )
    _attn_bwd_dq_kernel(
        q, k, v, do, lse, delta, dq, *strides, GROUP, float(scale), float(scale * LOG2E),
        cfg["block_M2"], cfg["block_N2"], cfg["num_stages"], cfg["threads"], dtype=dt,
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
