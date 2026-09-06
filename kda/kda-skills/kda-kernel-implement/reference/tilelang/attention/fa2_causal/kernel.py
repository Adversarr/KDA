"""FlashAttention-2, causal, multi-head, in TileLang: ``O = softmax(Q K^T * scale) V``.

The TileLang twin of ``reference/triton/attention/fa2_causal/kernel.py``: same math, host
signatures and test. Read the Triton SNIPPET for the algorithm (log2-domain online softmax,
the causal block skip, the deterministic two-kernel backward); this file documents what is
TileLang-specific. ``reference/tilelang/README.md`` has the general conventions.

Layout ``q, k, v, o: (B, H, S, D)``, unit stride in ``D``, the other three strides are
compile-time ints (``T.StridedTensor``; a ``(B, S, H, D)`` view passes without a copy).
``S_q == S_kv == S`` for the causal mask; ``D`` is the tile's last extent (no padding: ``D`` must
be a multiple of 16; 64/96/128/256 all run).

TileLang specifics:

* **`policy=T.GemmWarpPolicy.FullRow` on every GEMM.** The ``Q K^T`` accumulator is consumed as
  the *A* operand of ``P V``; with the default warp policy their fragment layouts differ and
  lowering fails with ``Layout infer conflict between acc_s and acc_s_cast``. FullRow gives
  each warp whole rows of both tiles, so the layouts agree (TileOPs' attention kernels do
  the same). The backward's ``pT -> dv``, ``dsT -> dk``, ``ds -> dq`` chains need it too.
* **Scores live in a fragment.** ``acc_s = T.alloc_fragment((block_M, block_N), fp32)`` is the
  ``Q K^T`` accumulator; the mask, the online softmax (``T.reduce_max``/``T.reduce_sum`` along
  ``dim=1``, ``T.Parallel`` element loops) and the cast to the storage dtype all happen on it,
  and the cast fragment is the *A* operand of the second ``T.gemm`` (``P V``). No shared round
  trip for ``P``.
* **The mask is an initial value, not a select after the GEMM.** ``acc_s`` is filled with
  ``0`` / ``-inf`` per element *before* ``T.gemm`` accumulates into it (the TileLang idiom);
  the branch on ``kb`` being a diagonal block is a runtime ``if`` around two fills.
* **Tails** are zero-filled by the predicated ``T.copy`` of the out-of-range rows; the
  causal condition covers keys ``>= S`` for every valid query (``j > i``), and the stores are
  predicated so garbage rows never land.
* **Backward tiles.** dK/dV kernel: ``sT (block_N, block_M) = K Q^T`` via
  ``T.gemm(k_s, q_s, sT, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)``; ``pT``, ``dsT`` are fragments cast to the storage
  dtype and used as the *A* operand of ``T.gemm(pT_cast, do_s, dv, policy=T.GemmWarpPolicy.FullRow)`` / ``T.gemm(dsT_cast, q_s,
  dk, policy=T.GemmWarpPolicy.FullRow)``. dQ kernel mirrors it with ``(block_M, block_N)`` tiles. ``lse``/``delta`` rows are
  staged in a shared ``(block,)`` fp32 buffer per iteration.
* **Outputs** via ``T.empty``; the gradient kernels take ``dk``/``dv``/``dq`` as inputs
  allocated by the host with the strides of ``k``/``v``/``q``, stored through a shared tile
  (a fragment stored straight to global is 6x slower, see the GEMM page).
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
    scale_log2: float, block_M: int, block_N: int, num_stages: int, threads: int,
    dtype=T.bfloat16, accum_dtype=T.float32,
):
    B, H = T.dynamic("B"), T.dynamic("H")  # grid-only extents: dynamic is free (2.09 vs 2.13 ms) and saves a compile per shape
    S, D = T.const("S, D")  # loop bounds and tiles: dynamic S costs 13% device time and 50% compile time
    q: T.StridedTensor((B, H, S, D), (sqb, sqh, sqm, 1), dtype)
    k: T.StridedTensor((B, H, S, D), (skb, skh, skn, 1), dtype)
    v: T.StridedTensor((B, H, S, D), (svb, svh, svn, 1), dtype)
    o = T.empty((B, H, S, D), dtype)
    lse = T.empty((B, H, S), accum_dtype)

    with T.Kernel(T.ceildiv(S, block_M), H, B, threads=threads) as (bx, by, bz):
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
        T.copy(q[bz, by, m0 : m0 + block_M, :], q_s)
        T.fill(acc_o, 0)
        T.fill(l_i, 0)
        T.fill(m_i, -T.infinity(accum_dtype))

        # Causal: K,V blocks up to the one holding this tile's last query. Blocks whose last
        # key precedes the tile's first query need no mask (the common case).
        n_blocks = T.ceildiv(T.min(m0 + block_M, S), block_N)
        for kb in T.Pipelined(n_blocks, num_stages=num_stages):
            n0 = kb * block_N
            T.copy(k[bz, by, n0 : n0 + block_N, :], k_s)
            if n0 + block_N <= m0:
                T.clear(acc_s)
            else:
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.if_then_else(m0 + i >= n0 + j, 0, -T.infinity(accum_dtype))
            T.gemm(q_s, k_s, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
            # Online softmax in the log2 domain: scale folds into the exp2 argument.
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
            T.copy(v[bz, by, n0 : n0 + block_N, :], v_s)
            T.gemm(acc_s_cast, v_s, acc_o, policy=T.GemmWarpPolicy.FullRow)
        for i, j in T.Parallel(block_M, D):
            acc_o[i, j] /= l_i[i]
        T.copy(acc_o, o_s)
        T.copy(o_s, o[bz, by, m0 : m0 + block_M, :])
        for i in T.Parallel(block_M):
            l_i[i] = m_i[i] * scale_log2 + T.log2(l_i[i])  # lse in the log2 domain
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
    B, H = T.dynamic("B"), T.dynamic("H")  # grid-only extents: dynamic is free (2.09 vs 2.13 ms) and saves a compile per shape
    S, D = T.const("S, D")  # loop bounds and tiles: dynamic S costs 13% device time and 50% compile time
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
    scale: float, scale_log2: float, block_M: int, block_N: int, num_stages: int, threads: int,
    dtype=T.bfloat16, accum_dtype=T.float32,
):
    B, H = T.dynamic("B"), T.dynamic("H")  # grid-only extents: dynamic is free (2.09 vs 2.13 ms) and saves a compile per shape
    S, D = T.const("S, D")  # loop bounds and tiles: dynamic S costs 13% device time and 50% compile time
    q: T.StridedTensor((B, H, S, D), (sqb, sqh, sqm, 1), dtype)
    k: T.StridedTensor((B, H, S, D), (skb, skh, skn, 1), dtype)
    v: T.StridedTensor((B, H, S, D), (svb, svh, svn, 1), dtype)
    do: T.StridedTensor((B, H, S, D), (sdb, sdh, sdm, 1), dtype)
    lse: T.Tensor((B, H, S), accum_dtype)
    delta: T.Tensor((B, H, S), accum_dtype)
    dk: T.StridedTensor((B, H, S, D), (skb, skh, skn, 1), dtype)
    dv: T.StridedTensor((B, H, S, D), (svb, svh, svn, 1), dtype)

    with T.Kernel(T.ceildiv(S, block_N), H, B, threads=threads) as (bx, by, bz):
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

        # Causal: Q blocks from the one holding this tile's first key to the end.
        lo = T.floordiv(n0, block_M)
        hi = T.ceildiv(S, block_M)
        for mb in T.Pipelined(lo, hi, num_stages=num_stages):
            m0 = mb * block_M
            T.copy(q[bz, by, m0 : m0 + block_M, :], q_s)
            T.copy(do[bz, by, m0 : m0 + block_M, :], do_s)
            T.copy(lse[bz, by, m0 : m0 + block_M], lse_s)
            T.copy(delta[bz, by, m0 : m0 + block_M], delta_s)
            T.clear(sT)
            T.gemm(k_s, q_s, sT, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)  # (block_N, block_M): keys down, queries across
            T.clear(dpT)
            T.gemm(v_s, do_s, dpT, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
            # Masked on every block (not only the diagonal ones): an `if` around the two
            # variants here raced with the in-loop lse_s/delta_s copies under T.Pipelined
            # (nondeterministic dk/dv in the last K block). The compare is cheap next to the GEMMs.
            # `m0 + j < S` is not redundant: the 1-D copies of lse/delta past S leave the shared
            # tail *uninitialised* (2-D T.copy zero-fills, 1-D does not), and `0 * NaN` would
            # poison dk/dv through the padded query columns. Select, do not multiply.
            for i, j in T.Parallel(block_N, block_M):
                sT[i, j] = T.if_then_else((m0 + j < S) and (m0 + j >= n0 + i), T.exp2(sT[i, j] * scale_log2 - lse_s[j]), 0)
            T.copy(sT, pT_cast)
            T.gemm(pT_cast, do_s, dv_f, policy=T.GemmWarpPolicy.FullRow)
            for i, j in T.Parallel(block_N, block_M):
                dsT_cast[i, j] = T.if_then_else(m0 + j < S, sT[i, j] * (dpT[i, j] - delta_s[j]), 0)
            T.gemm(dsT_cast, q_s, dk_f, policy=T.GemmWarpPolicy.FullRow)

        for i, j in T.Parallel(block_N, D):
            dk_f[i, j] *= scale
        # Two staging tiles: reusing one for both stores raced (a few inf rows, nondeterministic)
        # -- TileLang does not insert a barrier between a shared->global copy and the next
        # fragment->shared copy into the same buffer.
        T.copy(dk_f, dk_s)
        T.copy(dk_s, dk[bz, by, n0 : n0 + block_N, :])
        T.copy(dv_f, dv_s)
        T.copy(dv_s, dv[bz, by, n0 : n0 + block_N, :])


@tilelang.jit
def _attn_bwd_dq_kernel(
    q, k, v, do, lse, delta, dq,
    sqb: int, sqh: int, sqm: int, skb: int, skh: int, skn: int, svb: int, svh: int, svn: int, sdb: int, sdh: int, sdm: int,
    scale: float, scale_log2: float, block_M: int, block_N: int, num_stages: int, threads: int,
    dtype=T.bfloat16, accum_dtype=T.float32,
):
    B, H = T.dynamic("B"), T.dynamic("H")  # grid-only extents: dynamic is free (2.09 vs 2.13 ms) and saves a compile per shape
    S, D = T.const("S, D")  # loop bounds and tiles: dynamic S costs 13% device time and 50% compile time
    q: T.StridedTensor((B, H, S, D), (sqb, sqh, sqm, 1), dtype)
    k: T.StridedTensor((B, H, S, D), (skb, skh, skn, 1), dtype)
    v: T.StridedTensor((B, H, S, D), (svb, svh, svn, 1), dtype)
    do: T.StridedTensor((B, H, S, D), (sdb, sdh, sdm, 1), dtype)
    lse: T.Tensor((B, H, S), accum_dtype)
    delta: T.Tensor((B, H, S), accum_dtype)
    dq: T.StridedTensor((B, H, S, D), (sqb, sqh, sqm, 1), dtype)

    with T.Kernel(T.ceildiv(S, block_M), H, B, threads=threads) as (bx, by, bz):
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
        T.copy(q[bz, by, m0 : m0 + block_M, :], q_s)
        T.copy(do[bz, by, m0 : m0 + block_M, :], do_s)
        T.copy(lse[bz, by, m0 : m0 + block_M], lse_s)
        T.copy(delta[bz, by, m0 : m0 + block_M], delta_s)
        T.clear(dq_f)

        n_blocks = T.ceildiv(T.min(m0 + block_M, S), block_N)
        for kb in T.Pipelined(n_blocks, num_stages=num_stages):
            n0 = kb * block_N
            T.copy(k[bz, by, n0 : n0 + block_N, :], k_s)
            T.copy(v[bz, by, n0 : n0 + block_N, :], v_s)
            T.clear(s)
            T.gemm(q_s, k_s, s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
            T.clear(dp)
            T.gemm(do_s, v_s, dp, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
            if n0 + block_N <= m0:
                for i, j in T.Parallel(block_M, block_N):
                    s[i, j] = T.exp2(s[i, j] * scale_log2 - lse_s[i])
            else:
                for i, j in T.Parallel(block_M, block_N):
                    s[i, j] = T.if_then_else(m0 + i >= n0 + j, T.exp2(s[i, j] * scale_log2 - lse_s[i]), 0)
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
    if q.dim() != 4 or k.shape != q.shape or v.shape != q.shape:
        raise ValueError(f"attention (causal, MHA): q, k, v must be (B, H, S, D) of one shape, got {tuple(q.shape)}, {tuple(k.shape)}, {tuple(v.shape)}")
    if q.shape[-1] % 16 or q.shape[-1] > 256:
        raise ValueError(f"attention: head dim {q.shape[-1]} must be a multiple of 16 and <= 256")
    for name, t in (("q", q), ("k", k), ("v", v)):
        if t.stride(-1) != 1:
            raise ValueError(f"attention: {name} must be contiguous in the head dim")


def _strides(t: torch.Tensor) -> Tuple[int, int, int]:
    return int(t.stride(0)), int(t.stride(1)), int(t.stride(2))


def attention_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Causal attention forward. Returns ``o`` (contiguous ``(B, H, S, D)``) and ``lse`` (``(B, H, S)`` fp32, log2 domain)."""
    _check(q, k, v)
    D = q.shape[-1]
    scale = D**-0.5 if scale is None else scale
    cfg = select_config(D, "fwd")
    return _attn_fwd_kernel(q, k, v, *_strides(q), *_strides(k), *_strides(v), float(scale * LOG2E), **cfg, dtype=_TL_DTYPE[q.dtype])


def attention_bwd(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, o: torch.Tensor, do: torch.Tensor, lse: torch.Tensor, scale: Optional[float] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Causal attention backward: ``dq, dk, dv`` in the storage dtype, deterministic."""
    D = q.shape[-1]
    scale = D**-0.5 if scale is None else scale
    if do.stride(-1) != 1:
        raise ValueError("attention backward: dO must be contiguous in the head dim (call .contiguous() in the caller)")
    dt = _TL_DTYPE[q.dtype]
    delta = _attn_bwd_preprocess_kernel(o, do, *_strides(o), *_strides(do), 128, dtype=dt)
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    cfg = select_config(D, "bwd")
    strides = (*_strides(q), *_strides(k), *_strides(v), *_strides(do))
    _attn_bwd_dkdv_kernel(
        q, k, v, do, lse, delta, dk, dv, *strides, float(scale), float(scale * LOG2E),
        cfg["block_M1"], cfg["block_N1"], cfg["num_stages"], cfg["threads"], dtype=dt,
    )
    _attn_bwd_dq_kernel(
        q, k, v, do, lse, delta, dq, *strides, float(scale), float(scale * LOG2E),
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
    """``softmax(q k^T * scale, causal) v`` with ``q, k, v: (B, H, S, D)``; ``scale`` defaults to ``D**-0.5``."""
    return _Attention.apply(q, k, v, scale)


def attention_flops(B: int, H: int, S: int, D: int, phase: str) -> float:
    """Useful tensor-core FLOPs with the causal block skip: the attended area is ``S(S+1)/2`` pairs."""
    pairs = B * H * S * (S + 1) / 2
    per_pair = {"fwd": 4 * D, "infer": 4 * D, "bwd": 10 * D}[phase]
    return pairs * per_pair


__all__ = ["attention", "attention_fwd", "attention_bwd", "attention_flops", "select_config", "LOG2E"]
