"""GEMM with a fused epilogue in Triton: ``Y = act(X W^T + b)`` on tensor cores.

**Read `reference/nvmath/epilogue/gemm/` first.** Bias, ReLU, tanh-GELU and the aux store are
cuBLASLt epilogues that nvmath-python fuses into the cuBLAS GEMM at no cost (A800, 8192 x 8192
x 2048: cuBLAS 1017 us, nvmath ``GELU_AUX_BIAS`` 1053, this kernel 1273). This file is for what
cuBLASLt cannot express (erf-GELU, SiLU, a SwiGLU gate, quantised epilogues) and for GEMMs too
short for nvmath's 130 us of host time per call; and it is the source of the streaming pieces
the nvmath twin reuses (`_pointwise_epilogue_kernel`, `epilogue_bwd`). A residual is **not** an
epilogue: in a transformer it is added by the next norm (`residual_then_norm`), never here.

Forward::

    Z = X @ W^T                          # tl.dot on bf16/fp16 operands, fp32 accumulator
    Y = act(Z + b)                       # epilogue on the accumulator, one cast at the store

Two mainloops behind one host signature (``mainloop=``): ``"triton"`` is the fused kernel
described below; ``"cublas"`` runs the GEMM in ``torch.matmul`` and the same epilogue as one
pointwise kernel over the stored ``Z`` (one extra read of ``Z``, nothing else). Which is faster
is a shape question, decided by ``select_mainloop`` from the SNIPPET.md table: on A800 the
Triton mainloop trails cuBLAS by 7-14% on the ``D x D``, down-projection and skinny shapes,
where the fused epilogue still nets a win or a tie, but by 23-53% on the wide-output shapes
(``N >= 4096``: the up-projection, the 8192-wide MLP), where cuBLAS + epilogue kernel wins the
training forward by 18% / 2-4% and is level with or ahead of ``torch.compile``. Measure again on
another GPU or Triton version.

``X: (M, K)`` with a unit stride along ``K`` and any row stride (a padded row or a slice of a
wider activation is passed as ``stride_xm``, never copied); ``W: (N, K)`` contiguous, the
``nn.Linear`` layout; ``b: (N,)``; ``Y: (M, N)`` contiguous in ``X.dtype``. ``act`` in {none,
relu, gelu (erf), gelu_tanh, silu}.

Saved for the backward: ``Z = X W^T`` *without* the bias, in storage dtype (``M*N*es`` bytes;
the adjoint re-adds ``b`` before ``act'``, so both mainloops save the same tensor and the
cuBLAS output needs no rewrite), written under ``SAVE_AUX`` only when a backward can follow.
With ``recompute=True`` nothing is saved and the backward rebuilds ``Z`` with one more GEMM
(``torch.matmul``), trading ``2*M*N*K`` FLOPs for ``M*N*es`` bytes of activation memory: worth
it for the ``D -> 4D`` up-projection where ``Z`` is four times the size of ``X``.

Backward (``dY`` may carry a row stride; ``dZ`` is materialised in storage dtype)::

    dZ = dY * act'(Z + b)                # fused adjoint kernel, also sums db partials
    dX = dZ @ W,  dW = dZ^T @ X          # cuBLAS (torch.matmul); the op owns these launches

Geometry: a 2-D tile grid over ``(M, N)`` in grouped order (``GROUP_M`` row tiles share a
column of ``W`` tiles in L2), ``K`` loop of ``BLOCK_K`` steps, ``num_stages``-deep software
pipeline. **Split-K** (``SPLIT_K > 1``): the ``K`` loop is cut across a second grid axis, every
split writes an fp32 partial ``(M, N)`` and ``_splitk_epilogue_kernel`` sums the partials and
applies the same epilogue. ``select_config`` turns it on when the tile grid is far below the SM
count and ``K`` is long (the ``4D -> D`` down-projection with few tokens), see SNIPPET.md for
the measured table.
"""

from typing import Dict, Optional, Tuple

import torch
import triton
import triton.language as tl

ACTS: Dict[str, int] = {"none": 0, "relu": 1, "gelu": 2, "silu": 3, "gelu_tanh": 4}  # gelu = erf form; gelu_tanh = the approximation cuBLASLt/nvmath fuse
_MMA_DTYPES = (torch.bfloat16, torch.float16)


@triton.jit
def _act(z, ACT: tl.constexpr):
    """Activation on an fp32 tile. ACT codes follow ``ACTS``; gelu is the exact erf form."""
    if ACT == 1:
        return tl.maximum(z, 0.0)
    if ACT == 2:
        return 0.5 * z * (1.0 + tl.erf(z * 0.7071067811865476))
    if ACT == 3:
        return z * tl.sigmoid(z)
    if ACT == 4:  # tanh(u) = 2 sigmoid(2u) - 1
        u = 0.7978845608028654 * (z + 0.044715 * z * z * z)
        return z * tl.sigmoid(2.0 * u)
    return z


@triton.jit
def _act_grad(z, ACT: tl.constexpr):
    """``d act(z) / dz`` on an fp32 tile."""
    if ACT == 1:
        return tl.where(z > 0, 1.0, 0.0)
    if ACT == 2:
        cdf = 0.5 * (1.0 + tl.erf(z * 0.7071067811865476))
        pdf = tl.exp(-0.5 * z * z) * 0.3989422804014327
        return cdf + z * pdf
    if ACT == 3:
        s = tl.sigmoid(z)
        return s * (1.0 + z * (1.0 - s))
    if ACT == 4:
        u = 0.7978845608028654 * (z + 0.044715 * z * z * z)
        t = 2.0 * tl.sigmoid(2.0 * u) - 1.0
        return 0.5 * (1.0 + t) + 0.5 * z * (1.0 - t * t) * 0.7978845608028654 * (1.0 + 3.0 * 0.044715 * z * z)
    return tl.zeros_like(z) + 1.0


@triton.jit
def _epilogue(
    acc, bias_ptr, z_ptr, y_ptr, rm, rn, mask, N,
    ACT: tl.constexpr, HAS_BIAS: tl.constexpr, SAVE_AUX: tl.constexpr,
):
    # Shared by the fused kernel (SPLIT_K == 1) and the split-K reduce: aux store of the raw
    # product, bias, activation, one cast at the store. `rm` is [BM, 1] int64, `rn` [1, BN].
    if SAVE_AUX:  # Z (without bias, like the cuBLAS output) is written only when a backward will read it
        tl.store(z_ptr + rm * N + rn, acc.to(z_ptr.dtype.element_ty), mask=mask)
    if HAS_BIAS:
        acc += tl.load(bias_ptr + rn, mask=rn < N, other=0.0).to(tl.float32)
    y = _act(acc, ACT)
    tl.store(y_ptr + rm * N + rn, y.to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _gemm_epilogue_kernel(
    x_ptr, w_ptr, bias_ptr, y_ptr, z_ptr, partial_ptr,
    M, N, K, stride_xm,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr,
    SPLIT_K: tl.constexpr, EVEN_M: tl.constexpr, EVEN_N: tl.constexpr, EVEN_K: tl.constexpr,
    ACT: tl.constexpr, HAS_BIAS: tl.constexpr, SAVE_AUX: tl.constexpr,
):
    # Grouped tile order: GROUP_M consecutive programs walk down one column of N tiles, so the
    # W tile they share stays in L2 (the classic Triton matmul swizzle). int64 from the start:
    # M * N and M * K exceed 2^31 elements at LLM sizes.
    pid = tl.program_id(0).to(tl.int64)
    pid_k = tl.program_id(1).to(tl.int64)
    num_m = tl.cdiv(M, BLOCK_M)
    num_n = tl.cdiv(N, BLOCK_N)
    group_size = GROUP_M * num_n
    first_m = (pid // group_size) * GROUP_M
    rows_in_group = tl.minimum(num_m - first_m, GROUP_M)
    pid_m = first_m + (pid % group_size) % rows_in_group
    pid_n = (pid % group_size) // rows_in_group

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    # Row/col indices for the K-loop loads. When the tile divides M (N) the indices are told to
    # be BLOCK-aligned and contiguous, which is what lets Triton emit 16-byte vector loads (the
    # `% M` below hides that and costs ~20% of GEMM throughput). Otherwise out-of-range rows
    # re-read a valid row (modulo) instead of masking inside the K loop; the store masks them.
    if EVEN_M:
        rm_load = tl.max_contiguous(tl.multiple_of(rm, BLOCK_M), BLOCK_M)
    else:
        rm_load = rm % M
    if EVEN_N:
        rn_load = tl.max_contiguous(tl.multiple_of(rn, BLOCK_N), BLOCK_N)
    else:
        rn_load = rn % N

    # This split's share of the K blocks (all of them when SPLIT_K == 1).
    k_blocks = tl.cdiv(K, BLOCK_K)
    per_split = tl.cdiv(k_blocks, SPLIT_K)
    k_start = pid_k * per_split
    k_end = tl.minimum(k_start + per_split, k_blocks)

    x_ptrs = x_ptr + rm_load[:, None] * stride_xm + (k_start * BLOCK_K + rk)[None, :]
    w_ptrs = w_ptr + rn_load[:, None] * K + (k_start * BLOCK_K + rk)[None, :]  # W is (N, K): a [BN, BK] tile
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for kb in range(k_start, k_end):
        if EVEN_K:
            a = tl.load(x_ptrs)
            b = tl.load(w_ptrs)
        else:
            kmask = (kb * BLOCK_K + rk) < K
            a = tl.load(x_ptrs, mask=kmask[None, :], other=0.0)
            b = tl.load(w_ptrs, mask=kmask[None, :], other=0.0)
        # Operands stay in storage dtype (bf16/fp16 tensor cores); only the accumulator is fp32.
        acc = tl.dot(a, tl.trans(b), acc)
        x_ptrs += BLOCK_K
        w_ptrs += BLOCK_K

    rm2 = rm[:, None]
    rn2 = rn[None, :]
    mask = (rm2 < M) & (rn2 < N)
    if SPLIT_K == 1:
        _epilogue(acc, bias_ptr, z_ptr, y_ptr, rm2, rn2, mask, N, ACT, HAS_BIAS, SAVE_AUX)
    else:
        # fp32 partial per split; the epilogue runs once in _splitk_epilogue_kernel.
        tl.store(partial_ptr + pid_k * M * N + rm2 * N + rn2, acc, mask=mask)


@triton.jit
def _splitk_epilogue_kernel(
    partial_ptr, bias_ptr, y_ptr, z_ptr, M, N,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, SPLIT_K: tl.constexpr,
    ACT: tl.constexpr, HAS_BIAS: tl.constexpr, SAVE_AUX: tl.constexpr,
):
    # Sums the SPLIT_K fp32 partials of one (M, N) tile and applies the epilogue.
    pid_m = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1).to(tl.int64)
    rm2 = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M))[:, None]
    rn2 = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N))[None, :]
    mask = (rm2 < M) & (rn2 < N)
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for s in tl.static_range(SPLIT_K):
        acc += tl.load(partial_ptr + s * M * N + rm2 * N + rn2, mask=mask, other=0.0)
    _epilogue(acc, bias_ptr, z_ptr, y_ptr, rm2, rn2, mask, N, ACT, HAS_BIAS, SAVE_AUX)


@triton.jit
def _pointwise_epilogue_kernel(
    z_ptr, bias_ptr, y_ptr, M, N,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, ACT: tl.constexpr, HAS_BIAS: tl.constexpr,
):
    # The epilogue as its own pass over a Z that cuBLAS wrote (`mainloop="cublas"`): read Z
    # once, write Y once. Z itself is the saved aux, so nothing else is stored.
    pid_m = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1).to(tl.int64)
    rm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M))[:, None]
    rn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N))[None, :]
    mask = (rm < M) & (rn < N)
    z = tl.load(z_ptr + rm * N + rn, mask=mask, other=0.0).to(tl.float32)
    if HAS_BIAS:
        z += tl.load(bias_ptr + rn, mask=rn < N, other=0.0).to(tl.float32)
    y = _act(z, ACT)
    tl.store(y_ptr + rm * N + rn, y.to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _epilogue_bwd_kernel(
    dy_ptr, z_ptr, bias_ptr, dz_ptr, db_partial_ptr, M, N, stride_dym,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, ACT: tl.constexpr, HAS_BIAS: tl.constexpr, Z_HAS_BIAS: tl.constexpr,
):
    # Adjoint of the epilogue: dZ = dY * act'(Z + b), streamed in [BLOCK_M, BLOCK_N] tiles (Z_HAS_BIAS:
    # the saved Z already contains b, as cuBLASLt's `*_AUX_BIAS` aux does; db is still reduced). A fixed
    # number of programs along M strides over the row tiles and keeps a register accumulator for
    # db, writing one fp32 partial row per program (host reduces (n_prog_m, N)), the same scheme
    # as the rmsnorm dw reduction.
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1).to(tl.int64)
    n_prog_m = tl.num_programs(0)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    nmask = rn < N
    if HAS_BIAS and not Z_HAS_BIAS:
        bias = tl.load(bias_ptr + rn, mask=nmask, other=0.0).to(tl.float32)
    db = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for m0 in range(pid_m * BLOCK_M, M, n_prog_m * BLOCK_M):
        rm = (m0 + tl.arange(0, BLOCK_M)).to(tl.int64)
        mask = (rm[:, None] < M) & nmask[None, :]
        dy = tl.load(dy_ptr + rm[:, None] * stride_dym + rn[None, :], mask=mask, other=0.0).to(tl.float32)
        if ACT == 0:
            dz = dy
        else:
            z = tl.load(z_ptr + rm[:, None] * N + rn[None, :], mask=mask, other=0.0).to(tl.float32)
            if HAS_BIAS and not Z_HAS_BIAS:
                z += bias[None, :]
            dz = dy * _act_grad(z, ACT)
        tl.store(dz_ptr + rm[:, None] * N + rn[None, :], dz.to(dz_ptr.dtype.element_ty), mask=mask)
        if HAS_BIAS:
            db += dz
    if HAS_BIAS:
        tl.store(db_partial_ptr + pid_m.to(tl.int64) * N + rn, tl.sum(db, axis=0), mask=nmask)


def _num_sms(device: torch.device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


def select_config(M: int, N: int, K: int, num_sms: int, split_k: Optional[int] = None) -> Dict[str, int]:
    """Tile shape, pipeline depth and split-K for one problem size (A800 measurements, SNIPPET.md).

    - 128x128x64, 3 stages, 4 warps (the tile inductor's max-autotune also picks on sm80) once
      the 128x128 grid has >= 4 waves of tiles; 64x128x64 below that (more, smaller tiles keep
      every SM busy to the end and overlap one tile's epilogue with another's K loop).
    - Skinny M (<= 128 rows, the down-projection at a small token count): 64x64x64, 4 stages,
      and split-K so that ``tiles * SPLIT_K`` reaches about half the SM count (more splits pay
      more in partials than they gain), each split keeping at least 8 K blocks.
    ``split_k`` forces a value for a benchmark sweep. Measure with interleaved
    repeats: on an unlocked GPU two configs 10% apart swap places
    between clock states when timed one after the other.
    """
    if M <= 128:
        cfg = dict(BLOCK_M=64, BLOCK_N=64, BLOCK_K=64, GROUP_M=1, num_warps=4, num_stages=4)
    elif triton.cdiv(M, 128) * triton.cdiv(N, 128) >= 4 * num_sms:
        cfg = dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, GROUP_M=8, num_warps=4, num_stages=3)
    else:
        cfg = dict(BLOCK_M=64, BLOCK_N=128, BLOCK_K=64, GROUP_M=8, num_warps=4, num_stages=3)
    if split_k is None:
        split_k = 1
        tiles = triton.cdiv(M, cfg["BLOCK_M"]) * triton.cdiv(N, cfg["BLOCK_N"])
        if 2 * tiles < num_sms:
            k_blocks = K // cfg["BLOCK_K"]
            split_k = min(triton.next_power_of_2(triton.cdiv(num_sms, 2 * tiles)), max(1, k_blocks // 8), 16)
    cfg["SPLIT_K"] = max(1, split_k)
    return cfg


def select_mainloop(M: int, N: int, K: int) -> str:
    """``"cublas"`` or ``"triton"`` for one problem size (A800 measurements, SNIPPET.md).

    The Triton mainloop loses most to cuBLAS on wide outputs (``N >= 4096``: 499 vs 327 us on
    the ``8192 x 4096 x 1024`` up-projection, 1250 vs 1019 on ``8192 x 8192 x 2048``), more than
    the epilogue pass costs there, so those go to cuBLAS (443 vs 541 fused, 1250-1269 vs 1299);
    on ``N <= 1024`` shapes the fused kernel ties or wins (down-projection 359 vs 363, ``D x D``
    89 vs 95, skinny 24.2 vs 24.4). The threshold is the boundary the measured shapes give, not
    a model of the cause.
    """
    return "cublas" if N >= 4096 else "triton"


def gemm_epilogue_fwd(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: Optional[torch.Tensor],
    act: str,
    save_aux: bool = True,
    *,
    mainloop: Optional[str] = None,
    split_k: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``x`` (M, K) unit last stride, ``w`` (N, K) contiguous -> ``(y, z)``; ``y`` is contiguous.

    ``z`` (``X W^T`` without bias, storage dtype) is returned for the backward when ``save_aux``;
    otherwise it is a 0-element placeholder (the fused kernel skips the store; the cuBLAS path
    frees its temporary). ``mainloop`` picks the GEMM: ``"cublas"`` (``torch.matmul`` +
    ``_pointwise_epilogue_kernel``), ``"triton"`` (the fused kernel) or ``None`` for
    ``select_mainloop``. ``split_k`` forces a split of the Triton mainloop (tests and the
    bench); ``None`` lets ``select_config`` decide.
    """
    M, K = x.shape
    N = w.shape[0]
    assert x.dtype in _MMA_DTYPES and w.dtype == x.dtype, "tensor-core GEMM: bf16/fp16 operands"
    assert x.stride(1) == 1 and w.is_contiguous() and w.shape[1] == K
    if mainloop is None:
        mainloop = select_mainloop(M, N, K)
    if mainloop == "cublas":
        z = torch.matmul(x, w.t())  # cuBLAS accumulates in fp32 and rounds once to the storage dtype
        y = torch.empty_like(z)
        _pointwise_epilogue_kernel[(triton.cdiv(M, 32), triton.cdiv(N, 128))](
            z, bias, y, M, N, BLOCK_M=32, BLOCK_N=128, ACT=ACTS[act], HAS_BIAS=bias is not None, num_warps=4,
        )
        return y, (z if save_aux else z.new_empty(0))
    assert mainloop == "triton", mainloop
    y = torch.empty((M, N), dtype=x.dtype, device=x.device)
    z = torch.empty((M, N) if save_aux else (0,), dtype=x.dtype, device=x.device)
    cfg = select_config(M, N, K, _num_sms(x.device), split_k)
    s = cfg["SPLIT_K"]
    partial = torch.empty((s, M, N) if s > 1 else (0,), dtype=torch.float32, device=x.device)
    common = dict(ACT=ACTS[act], HAS_BIAS=bias is not None, SAVE_AUX=save_aux)
    grid = (triton.cdiv(M, cfg["BLOCK_M"]) * triton.cdiv(N, cfg["BLOCK_N"]), s)
    _gemm_epilogue_kernel[grid](
        x, w, bias, y, z, partial, M, N, K, x.stride(0),
        BLOCK_M=cfg["BLOCK_M"], BLOCK_N=cfg["BLOCK_N"], BLOCK_K=cfg["BLOCK_K"], GROUP_M=cfg["GROUP_M"],
        SPLIT_K=s, EVEN_M=(M % cfg["BLOCK_M"] == 0), EVEN_N=(N % cfg["BLOCK_N"] == 0), EVEN_K=(K % (cfg["BLOCK_K"] * s) == 0),
        num_warps=cfg["num_warps"], num_stages=cfg["num_stages"], **common,
    )
    if s > 1:
        _splitk_epilogue_kernel[(triton.cdiv(M, 32), triton.cdiv(N, 128))](
            partial, bias, y, z, M, N,
            BLOCK_M=32, BLOCK_N=128, SPLIT_K=s, num_warps=4, **common,
        )
    return y, z


def epilogue_bwd(
    dy: torch.Tensor, z: torch.Tensor, bias: Optional[torch.Tensor], act: str, *, z_has_bias: bool = False
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """``dZ = dY * act'(Z + b)`` and ``db = sum_rows(dZ)`` (fp32, from per-program partials); ``dZ`` contiguous.

    ``z_has_bias``: ``z`` is the pre-activation *including* the bias (the ``gelu_aux`` that
    cuBLASLt's ``GELU_AUX_BIAS`` epilogue writes, see ``reference/nvmath``); the add is skipped,
    ``db`` is still reduced.
    """
    M, N = dy.shape
    assert dy.stride(1) == 1 and (z.shape == dy.shape or act == "none")
    assert act == "none" or z.is_contiguous(), "the adjoint reads z with row stride N"
    has_bias = bias is not None
    dz = torch.empty((M, N), dtype=dy.dtype, device=dy.device)
    block_m, block_n = 32, 128
    n_blocks_n = triton.cdiv(N, block_n)
    n_prog_m = max(1, min(triton.cdiv(M, block_m), 2 * _num_sms(dy.device) // n_blocks_n))
    db_partial = torch.empty((n_prog_m if has_bias else 0, N), dtype=torch.float32, device=dy.device)
    _epilogue_bwd_kernel[(n_prog_m, n_blocks_n)](
        dy, z, bias, dz, db_partial, M, N, dy.stride(0),
        BLOCK_M=block_m, BLOCK_N=block_n, ACT=ACTS[act], HAS_BIAS=has_bias, Z_HAS_BIAS=z_has_bias, num_warps=4,
    )
    return dz, (db_partial.sum(0) if has_bias else None)


def gemm_epilogue_bwd(
    dy: torch.Tensor, x: torch.Tensor, w: torch.Tensor, bias: Optional[torch.Tensor], z: torch.Tensor, act: str,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Returns ``(dx, dw, db)``.

    ``z`` is the saved ``X W^T`` or, under recompute, a 0-element placeholder: then it is rebuilt
    here with one GEMM (cuBLAS; the fused kernel with ``act="none"`` and no bias is the same
    tensor).
    """
    if z.numel() == 0 and act != "none":
        z = torch.matmul(x, w.t())
    if act == "none" and bias is None:
        dz = dy  # the epilogue is the identity: nothing to compute
        db = None
    else:
        dz, db = epilogue_bwd(dy, z, bias, act)
    dx = torch.matmul(dz, w)  # (M, N) @ (N, K); cuBLAS handles a row-strided dz
    dw = torch.matmul(dz.t(), x)  # (N, M) @ (M, K); x may be row-strided
    return dx, dw, (db.to(bias.dtype) if db is not None else None)


def _rows(t: torch.Tensor) -> torch.Tensor:
    """``(..., D)`` -> ``(N, D)`` without a copy (see the rmsnorm snippet for the layout caveat)."""
    return t if t.dim() == 2 else t.view(-1, t.shape[-1])


class _GemmEpilogue(torch.autograd.Function):
    # Standalone snippet: plain autograd.Function. Inside a kernel package use
    # `_common.compat.register_kernel` + `make_differentiable`, which also sets `save_aux` and
    # `recompute` per call from grad mode and `KDA_RECOMPUTE`.
    @staticmethod
    def forward(ctx, x, w, bias, act, save_aux, mainloop):
        x2d = _rows(x)
        y, z = gemm_epilogue_fwd(x2d, w, bias, act, save_aux, mainloop=mainloop)
        ctx.act = act
        ctx.save_for_backward(x2d, w, bias, z)  # z is empty under recompute (bias rebuilds it) or act=none
        return y.view(*x.shape[:-1], w.shape[0])

    @staticmethod
    def backward(ctx, dy):
        x2d, w, bias, z = ctx.saved_tensors
        dx, dw, db = gemm_epilogue_bwd(_rows(dy), x2d, w, bias, z, ctx.act)
        return dx.view(*dy.shape[:-1], w.shape[1]), dw, db, None, None, None


def gemm_epilogue(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    act: str = "none",
    *,
    recompute: bool = False,
    mainloop: Optional[str] = None,
) -> torch.Tensor:
    """``act(x @ w.T + bias)`` over the last dim of ``x``; output in ``x.dtype``.

    ``recompute=True`` saves no pre-activation and rebuilds it in the backward. ``mainloop`` is
    ``"cublas"``, ``"triton"`` or ``None`` (``select_mainloop``), see ``gemm_epilogue_fwd``.
    """
    needs_grad = torch.is_grad_enabled() and any(t is not None and t.requires_grad for t in (x, w, bias))
    save_aux = needs_grad and not recompute and act != "none"  # with act=none, dZ = dY needs no Z
    return _GemmEpilogue.apply(x, w, bias, act, save_aux, mainloop)


__all__ = ["ACTS", "gemm_epilogue", "gemm_epilogue_fwd", "gemm_epilogue_bwd", "epilogue_bwd", "select_config", "select_mainloop"]
