"""GEMM with a fused epilogue in TileLang: ``Y = act(X W^T + b)`` on tensor cores (experimental).

**Read `reference/nvmath/epilogue/gemm/` first**: bias, ReLU, tanh-GELU and the aux store are
cuBLASLt epilogues that nvmath-python fuses at cuBLAS speed (A800, 8192 x 8192 x 2048: cuBLAS
1017 us, nvmath 1053, the Triton twin 1273, this kernel ~1600). This twin exists to show the
TileLang idioms of a tensor-core GEMM with an epilogue, for the activations cuBLASLt lacks.
A residual is not an epilogue: in a transformer it is added by the next norm.

The TileLang twin of ``reference/triton/epilogue/gemm/kernel.py``: same math, same host
signatures, same test. Read the Triton SNIPPET for the algorithm; this file documents what is
TileLang-specific (``reference/tilelang/README.md`` has the general conventions).

Forward::

    Z = X @ W^T + b                      # T.gemm on bf16/fp16 shared tiles, fp32 fragment
    Y = act(Z)                           # epilogue on the fragment, one cast at the store

``X: (M, K)`` unit stride along ``K``, any row stride (``T.StridedTensor``, never copied);
``W: (N, K)`` contiguous (``nn.Linear``), consumed with ``transpose_B=True``; ``b: (N,)``;
``Y: (M, N)`` contiguous in ``X.dtype``. ``act`` in {none, relu, gelu (erf), gelu_tanh, silu}.

Saved for the backward: the pre-activation ``Z`` (storage dtype) under ``save_aux``; with
``recompute=True`` nothing is saved and the backward rebuilds ``Z`` with one more GEMM.

Backward::

    dZ = dY * act'(Z)                    # fused adjoint kernel, also sums db partials
    dX = dZ @ W,  dW = dZ^T @ X          # cuBLAS (torch.matmul)

TileLang specifics:

* **Eager JIT**: ``@tilelang.jit`` on a function that takes the tensors and the launch
  parameters. ``N, K`` are ``T.const`` (inferred from the tensors, one compiled variant per
  value: the tile loop depends on them); ``M`` is ``T.dynamic`` (no recompile per row count,
  and measured free: 0.323 ms dynamic vs 0.331 static on 4096x4096x1024, the M tail is a
  predicate either way). Every ``@tilelang.jit`` compile costs 4-6 s (TVM passes ~2.2 s + nvcc
  ~2.7 s + host cc 0.8 s with ``tvm_ffi``; ``execution_backend="nvrtc"`` drops the last two to
  ~2.0 s, same device time, but doubles the launch overhead, 21 vs 10 us per call), so what is
  ``T.const`` decides how long a multi-shape verify takes. Python ``bool``/``int``
  arguments (``act``, ``has_bias``, ``save_aux``, ``split_k``, the tile sizes) specialise the
  kernel: TileLang caches one variant per distinct value, so ``if save_aux:`` is folded at
  compile time exactly like a Triton ``constexpr``.
* **Optional tensors** are still positional tensor arguments: when ``has_bias`` is false the
  host passes a 1-element dummy and the kernel never reads it. Their declarations use
  ``T.dynamic`` extents so the dummy passes the ABI shape check. Never name a tensor argument
  ``res``: the ``nvrtc`` adapter's generated launcher keeps its ``CUresult`` in a local called
  ``res`` and an argument of that name breaks it (``'CUresult' object has no attribute
  'data_ptr'``, TileLang 0.1.13 and 0.1.14; ``config``, ``kernels`` and ``stream`` are the
  other reserved names). The ``cython`` backend rejects a
  ``T.StridedTensor`` with ``T.dynamic`` extents altogether.
* **Outputs** are allocated inside the kernel with ``T.empty`` and returned (``out_idx`` is
  inferred); a 0-element ``T.empty((0,), ...)`` is the placeholder for an aux tensor that is
  compiled out.
* **Tails**: ``T.copy`` predicates out-of-range rows/cols/K (zero fill), so ``M, N, K`` need
  not divide the tile; the epilogue and partial stores guard with ``if m < M and n < N``.
* **Split-K**: the ``K`` blocks are cut across the third grid axis; each split writes an fp32
  partial ``(split_k, M, N)`` and ``_splitk_epilogue_kernel`` sums them and runs the same
  epilogue (no ``T.atomic_add``: the epilogue is not linear, so it must run once on the sum).
"""

import logging
from typing import Dict, Optional, Tuple

import tilelang
import tilelang.language as T
import torch

logging.getLogger("tilelang").setLevel(logging.WARNING)  # the per-compile INFO lines are noise here

ACTS: Dict[str, int] = {"none": 0, "relu": 1, "gelu": 2, "silu": 3, "gelu_tanh": 4}  # gelu = erf; gelu_tanh = what models use and cuBLASLt fuses
_GELU_C = 0.7978845608028654  # sqrt(2/pi)
_MMA_DTYPES = (torch.bfloat16, torch.float16)
_TL_DTYPE = {torch.bfloat16: T.bfloat16, torch.float16: T.float16}


def _act(z, act: int):
    """Activation on an fp32 scalar expression (trace-time helper: ``act`` is a Python int)."""
    if act == 1:
        return T.max(z, 0.0)
    if act == 2:
        return 0.5 * z * (1.0 + T.erf(z * 0.7071067811865476))
    if act == 3:
        return z / (1.0 + T.exp(-z))
    if act == 4:  # z * sigmoid(2u) == 0.5 z (1 + tanh u), u = c (z + 0.044715 z^3)
        return z / (1.0 + T.exp(-2.0 * _GELU_C * (z + 0.044715 * z * z * z)))
    return z


def _act_grad(z, act: int):
    """``d act(z) / dz`` on an fp32 scalar expression."""
    if act == 1:
        return T.if_then_else(z > 0, 1.0, 0.0)
    if act == 2:
        return 0.5 * (1.0 + T.erf(z * 0.7071067811865476)) + z * T.exp(-0.5 * z * z) * 0.3989422804014327
    if act == 3:
        s = 1.0 / (1.0 + T.exp(-z))
        return s * (1.0 + z * (1.0 - s))
    if act == 4:
        s = 1.0 / (1.0 + T.exp(-2.0 * _GELU_C * (z + 0.044715 * z * z * z)))
        return s + z * s * (1.0 - s) * 2.0 * _GELU_C * (1.0 + 3.0 * 0.044715 * z * z)
    return 1.0


@T.macro
def _epilogue_tile(acc, bias, y, z, m0, n0, block_M, block_N, act, has_bias, save_aux, dtype, accum_dtype):
    """Bias, aux store of Z, activation, store of Y on one accumulator tile.

    A ``T.macro`` is expanded inline into the calling kernel (a plain Python helper cannot hold
    ``T.Parallel`` loops: only the JIT function's own AST is lowered); every flag is a Python
    bool folded at compile time. Tiles move with ``T.copy`` (predicated at the M/N tails,
    fp32 -> storage cast in the copy). The two output stores are staged through one shared
    tile: written straight from the MMA fragment each thread stores 4-byte pairs across
    scattered rows, which costs ~100 us on the up-projection's Z; through shared memory the
    global store is 16-byte and coalesced (~16 us). A scalar ``y[m, n] = ...`` loop with bounds
    guards is slower still (SNIPPET.md has the breakdown).
    """
    out_s = T.alloc_shared((block_M, block_N), dtype)
    if has_bias:
        bias_s = T.alloc_shared((block_N,), dtype)
        T.copy(bias[n0], bias_s)
        for i, j in T.Parallel(block_M, block_N):
            acc[i, j] += T.cast(bias_s[j], accum_dtype)
    if save_aux:  # Z is written only when a backward will read it
        T.copy(acc, out_s)
        T.copy(out_s, z[m0, n0])
    if act != 0:
        for i, j in T.Parallel(block_M, block_N):
            acc[i, j] = _act(acc[i, j], act)
    T.copy(acc, out_s)
    T.copy(out_s, y[m0, n0])


@tilelang.jit
def _gemm_epilogue_kernel(
    x, w, bias,
    stride_xm: int,
    block_M: int, block_N: int, block_K: int, num_stages: int, threads: int, split_k: int,
    act: int, has_bias: bool, save_aux: bool,
    dtype=T.bfloat16, accum_dtype=T.float32,
):
    # Strides are Python ints, i.e. compile-time constants like M, N, K (T.const already gives
    # one variant per shape, so a stride adds no variants in practice). A T.dynamic stride
    # halves the mainloop speed: the copies can no longer prove alignment and stop vectorising.
    M = T.dynamic("M")  # rows: no recompile per M, measured free (0.323 vs 0.331 ms on 4096x4096x1024)
    N, K = T.const("N, K")  # the tile loop and the swizzle depend on these: one variant per value
    nb = T.dynamic("nb")  # extent the bias dummy satisfies
    x: T.StridedTensor((M, K), (stride_xm, 1), dtype)
    w: T.Tensor((N, K), dtype)
    bias: T.Tensor((nb,), dtype)
    y = T.empty((M, N), dtype)
    z = T.empty((M, N) if save_aux else (0, N), dtype)
    # split-K partials: one fp32 slab per split, stacked along rows. Slabs are padded to whole
    # tiles (M_pad) so a tile at the M tail cannot spill into the next split's slab.
    M_pad = T.ceildiv(M, block_M) * block_M
    partial = T.empty((split_k * M_pad, N) if split_k > 1 else (0, N), accum_dtype)
    k_blocks_per_split = T.ceildiv(T.ceildiv(K, block_K), split_k)

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), split_k, threads=threads) as (bx, by, bz):
        xs = T.alloc_shared((block_M, block_K), dtype)
        ws = T.alloc_shared((block_N, block_K), dtype)
        acc = T.alloc_fragment((block_M, block_N), accum_dtype)
        # Rasterise the tile order so neighbouring CTAs share W tiles in L2 (Triton's GROUP_M);
        # the swizzle assumes a 2-D grid, so it is off on the split-K path.
        T.use_swizzle(panel_size=8, enable=(split_k == 1))
        T.clear(acc)
        k0 = 0 if split_k == 1 else bz * k_blocks_per_split  # first K block of this split
        for ko in T.Pipelined(k_blocks_per_split, num_stages=num_stages):
            # Operands stay in storage dtype in shared memory; only the accumulator is fp32.
            # T.copy predicates the K tail (and a last split that runs past K) with zeros.
            T.copy(x[by * block_M, (k0 + ko) * block_K], xs)
            T.copy(w[bx * block_N, (k0 + ko) * block_K], ws)
            T.gemm(xs, ws, acc, transpose_B=True)
        if split_k == 1:
            _epilogue_tile(acc, bias, y, z, by * block_M, bx * block_N, block_M, block_N,
                           act, has_bias, save_aux, dtype, accum_dtype)
        else:
            T.copy(acc, partial[bz * M_pad + by * block_M, bx * block_N])
    return y, z, partial


@tilelang.jit
def _splitk_epilogue_kernel(
    partial, bias, y, z, split_k: int,
    block_M: int, block_N: int, act: int, has_bias: bool, save_aux: bool,
    dtype=T.bfloat16, accum_dtype=T.float32,
):
    # Sums the split_k fp32 partial slabs of one (M, N) tile and applies the epilogue into the
    # forward's y (and z), which arrive as inputs here (no T.empty: nothing is allocated).
    # block_M must be the forward's (the slab padding M_pad is per forward tile).
    M = T.dynamic("M")
    N = T.const("N")
    M_pad = T.ceildiv(M, block_M) * block_M
    nb, mz, mp = T.dynamic("nb"), T.dynamic("mz"), T.dynamic("mp")
    partial: T.Tensor((mp, N), accum_dtype)
    bias: T.Tensor((nb,), dtype)
    y: T.Tensor((M, N), dtype)
    z: T.Tensor((mz, N), dtype)  # (M, N) when save_aux, else the (0, N) placeholder
    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
        acc = T.alloc_fragment((block_M, block_N), accum_dtype)
        p_f = T.alloc_fragment((block_M, block_N), accum_dtype)
        T.clear(acc)
        for s in T.serial(split_k):
            T.copy(partial[s * M_pad + by * block_M, bx * block_N], p_f)
            for i, j in T.Parallel(block_M, block_N):
                acc[i, j] += p_f[i, j]
        _epilogue_tile(acc, bias, y, z, by * block_M, bx * block_N, block_M, block_N,
                       act, has_bias, save_aux, dtype, accum_dtype)


@tilelang.jit
def _epilogue_bwd_kernel(
    dy, z, stride_dym: int, n_prog_m: int, block_M: int, block_N: int, act: int, has_bias: bool,
    dtype=T.bfloat16, accum_dtype=T.float32,
):
    # Adjoint of the epilogue: dZ = dY * act'(Z) streamed in [block_M, block_N] tiles. A fixed
    # number of programs along M strides over the row tiles and keeps a fragment accumulator
    # for db, writing one fp32 partial row per program (the host reduces (n_prog_m, N)).
    M = T.dynamic("M")
    N = T.const("N")
    mz = T.dynamic("mz")
    dy: T.StridedTensor((M, N), (stride_dym, 1), dtype)
    z: T.Tensor((mz, N), dtype)  # the (M, N) pre-activation, or a (0, N) placeholder when act = none
    dz = T.empty((M, N), dtype)
    db_partial = T.empty((n_prog_m if has_bias else 0, N), accum_dtype)
    row_tiles = T.ceildiv(M, block_M)
    with T.Kernel(T.ceildiv(N, block_N), n_prog_m, threads=128) as (bx, pm):
        # Tiles move with T.copy (predicated at the M/N tails, and it is what gives a fragment
        # its layout: a fragment filled only by scalar stores cannot be laid out). db accumulates
        # tile-wise in a 2-D fragment and is reduced once per program.
        dy_frag = T.alloc_fragment((block_M, block_N), dtype)
        z_frag = T.alloc_fragment((block_M, block_N), dtype)
        dz_frag = T.alloc_fragment((block_M, block_N), accum_dtype)  # fp32 until the store; db sums the unrounded dz
        if has_bias:  # a fragment that is only cleared and never read has no layout to infer
            db_acc = T.alloc_fragment((block_M, block_N), accum_dtype)
            db_row = T.alloc_fragment((block_N,), accum_dtype)
            T.clear(db_acc)
        for it in T.serial(T.ceildiv(row_tiles - pm, n_prog_m)):
            m0 = (pm + it * n_prog_m) * block_M
            T.copy(dy[m0, bx * block_N], dy_frag)
            if act == 0:
                T.copy(dy_frag, dz_frag)
            else:
                T.copy(z[m0, bx * block_N], z_frag)
                for i, j in T.Parallel(block_M, block_N):
                    dz_frag[i, j] = T.cast(dy_frag[i, j], accum_dtype) * _act_grad(T.cast(z_frag[i, j], accum_dtype), act)
            T.copy(dz_frag, dz[m0, bx * block_N])  # the one cast to storage dtype
            if has_bias:
                # The tail tile's out-of-range rows arrived as zeros from the predicated copy, so
                # they add nothing (a guard here would also break the layout propagation).
                for i, j in T.Parallel(block_M, block_N):
                    db_acc[i, j] += dz_frag[i, j]
        if has_bias:
            T.reduce_sum(db_acc, db_row, dim=0)
            for j in T.Parallel(block_N):
                if bx * block_N + j < N:
                    db_partial[pm, bx * block_N + j] = db_row[j]
    return dz, db_partial


def _num_sms(device: torch.device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def select_config(M: int, N: int, K: int, num_sms: int, split_k: Optional[int] = None) -> Dict[str, int]:
    """Tile shape, pipeline depth, thread count and split-K for one problem size (A800, SNIPPET.md).

    Same policy as the Triton twin: 128x128 tiles once that grid has >= 4 waves, 64x128 below,
    64x64 with split-K for skinny ``M <= 128``; ``threads`` is the CTA size TileLang maps the
    fragment onto (128 = 4 warps). The sweep (SNIPPET.md) found ``block_K = 32`` with 128
    threads best for the 128x128 tile on sm80 (256 threads / ``block_K = 64`` is ~6% slower),
    and ``block_K = 64`` best for the 64-row tiles. ``split_k`` forces a value (tests and the
    bench sweep it).
    """
    if M <= 128:
        cfg = dict(block_M=64, block_N=64, block_K=64, num_stages=4, threads=128)
    elif _cdiv(M, 128) * _cdiv(N, 128) >= 4 * num_sms:
        cfg = dict(block_M=128, block_N=128, block_K=32, num_stages=3, threads=128)
    else:
        cfg = dict(block_M=64, block_N=128, block_K=64, num_stages=3, threads=128)
    if split_k is None:
        split_k = 1
        tiles = _cdiv(M, cfg["block_M"]) * _cdiv(N, cfg["block_N"])
        if 2 * tiles < num_sms:
            k_blocks = K // cfg["block_K"]
            want = _cdiv(num_sms, 2 * tiles)
            split_k = min(1 << (want - 1).bit_length(), max(1, k_blocks // 8), 16)
    cfg["split_k"] = max(1, split_k)
    return cfg


def _dummy(like: torch.Tensor, dims: int) -> torch.Tensor:
    return torch.empty((1,) * dims, dtype=like.dtype, device=like.device)


def gemm_epilogue_fwd(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: Optional[torch.Tensor],
    act: str,
    save_aux: bool = True,
    *,
    split_k: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``x`` (M, K) unit last stride, ``w`` (N, K) contiguous -> ``(y, z)``; ``y`` is contiguous.

    ``z`` (pre-activation, storage dtype) is written only when ``save_aux``; otherwise it is a
    0-element placeholder and the store is compiled out.
    """
    M, K = x.shape
    N = w.shape[0]
    assert x.dtype in _MMA_DTYPES and w.dtype == x.dtype, "tensor-core GEMM: bf16/fp16 operands"
    assert x.stride(1) == 1 and w.is_contiguous() and w.shape[1] == K
    cfg = select_config(M, N, K, _num_sms(x.device), split_k)
    s = cfg["split_k"]
    flags = dict(act=ACTS[act], has_bias=bias is not None, save_aux=save_aux)
    b = bias if bias is not None else _dummy(x, 1)
    y, z, partial = _gemm_epilogue_kernel(
        x, w, b, x.stride(0),
        cfg["block_M"], cfg["block_N"], cfg["block_K"], cfg["num_stages"], cfg["threads"], s,
        **flags, dtype=_TL_DTYPE[x.dtype],
    )
    if s > 1:
        _splitk_epilogue_kernel(partial, b, y, z, s, cfg["block_M"], cfg["block_N"], **flags, dtype=_TL_DTYPE[x.dtype])
    return y, z


def epilogue_bwd(dy: torch.Tensor, z: torch.Tensor, act: str, has_bias: bool) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """``dZ = dY * act'(Z)`` and ``db = sum_rows(dZ)`` (fp32, from per-program partials); ``dZ`` contiguous."""
    M, N = dy.shape
    assert dy.stride(1) == 1 and (z.shape == dy.shape or act == "none")
    block_m, block_n = 32, 128
    n_blocks_n = _cdiv(N, block_n)
    n_prog_m = max(1, min(_cdiv(M, block_m), 2 * _num_sms(dy.device) // n_blocks_n))
    z2d = z if z.numel() else dy.new_empty((0, N))  # (0, N) placeholder, contiguous as declared
    dz, db_partial = _epilogue_bwd_kernel(dy, z2d, dy.stride(0), n_prog_m, block_m, block_n, ACTS[act], has_bias, dtype=_TL_DTYPE[dy.dtype])
    return dz, (db_partial.sum(0) if has_bias else None)


def gemm_epilogue_bwd(
    dy: torch.Tensor, x: torch.Tensor, w: torch.Tensor, bias: Optional[torch.Tensor], z: torch.Tensor, act: str,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Returns ``(dx, dw, db)``.

    ``z`` is the saved pre-activation or, under recompute, a 0-element placeholder: then it is
    rebuilt here with the forward kernel (``act="none"``, nothing saved).
    """
    if z.numel() == 0 and act != "none":
        z, _ = gemm_epilogue_fwd(x, w, bias, "none", save_aux=False)
    if act == "none" and bias is None:
        dz = dy  # the epilogue is the identity: nothing to compute
        db = None
    else:
        dz, db = epilogue_bwd(dy, z, act, bias is not None)
    dx = torch.matmul(dz, w)  # (M, N) @ (N, K)
    dw = torch.matmul(dz.t(), x)  # (N, M) @ (M, K); x may be row-strided
    return dx, dw, (db.to(bias.dtype) if db is not None else None)


def _rows(t: torch.Tensor) -> torch.Tensor:
    """``(..., D)`` -> ``(N, D)`` without a copy."""
    return t if t.dim() == 2 else t.view(-1, t.shape[-1])


class _GemmEpilogue(torch.autograd.Function):
    # Standalone snippet: plain autograd.Function. Inside a kernel package use
    # `_common.compat.register_kernel` + `make_differentiable`.
    @staticmethod
    def forward(ctx, x, w, bias, act, save_aux):
        x2d = _rows(x)
        y, z = gemm_epilogue_fwd(x2d, w, bias, act, save_aux)
        ctx.act = act
        ctx.save_for_backward(x2d, w, bias, z)
        return y.view(*x.shape[:-1], w.shape[0])

    @staticmethod
    def backward(ctx, dy):
        x2d, w, bias, z = ctx.saved_tensors
        dx, dw, db = gemm_epilogue_bwd(_rows(dy), x2d, w, bias, z, ctx.act)
        return dx.view(*dy.shape[:-1], w.shape[1]), dw, db, None, None


def gemm_epilogue(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    act: str = "none",
    *,
    recompute: bool = False,
) -> torch.Tensor:
    """``act(x @ w.T + bias)`` over the last dim of ``x``; output in ``x.dtype``.

    ``recompute=True`` saves no pre-activation and rebuilds it in the backward.
    """
    needs_grad = torch.is_grad_enabled() and any(t is not None and t.requires_grad for t in (x, w, bias))
    save_aux = needs_grad and not recompute and act != "none"
    return _GemmEpilogue.apply(x, w, bias, act, save_aux)


__all__ = ["ACTS", "gemm_epilogue", "gemm_epilogue_fwd", "gemm_epilogue_bwd", "epilogue_bwd", "select_config"]
