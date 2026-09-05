"""Row-wise RMSNorm in Triton: fp32 math, low-precision I/O, fused forward and backward.

Forward, per row of ``D`` elements::

    rstd = rsqrt(mean(x^2) + eps)          # fp32, saved for the backward
    y    = (x * rstd) * w                  # cast to x.dtype on store

Backward (``xhat = x * rstd``)::

    dxhat = dy * w
    dx    = rstd * (dxhat - xhat * mean(dxhat * xhat))
    dw    = sum_rows(dy * xhat)

Layout: ``x`` is ``(N, D)`` with a unit stride along ``D`` and any row stride (a padded or
sliced row is passed as ``stride_row``, never copied); ``y`` has the same shape and is
contiguous; ``dy`` may carry its own row stride. Any leading shape is flattened by the wrapper
with ``view`` (no copy). ``w`` is ``(D,)`` in any float dtype. Every intermediate is fp32.

``rstd`` exists only for the backward: the forward writes it under ``SAVE_RSTD`` when a
backward can follow (grad mode on and ``x`` or ``w`` requires grad) and skips the store and
the allocation otherwise, so inference pays for the outputs alone.

Two forward geometries, chosen by ``_fwd_geometry``:

* **one program per row** (``_rmsnorm_fwd_kernel``): the whole row in registers. Measured at
  the copy roof for every ``D <= 32768`` and every row count, including rows below the SM
  count (a 32 x 8192 bf16 forward is latency-bound at ~5 us, as is the copy it is compared
  with, so filling the SMs buys nothing).
* **split-D, two stages** (``_rmsnorm_fwd_sumsq_kernel`` + ``_rmsnorm_fwd_normalize_kernel``):
  ``N_CHUNKS`` programs per row write a partial sum of squares each, then normalise their chunk
  with the row's ``rstd``. The second read of ``x`` hits L2. This is the *wide-row* path: at
  ``D = 65536`` the one-row kernel spills and runs 8x slower than copy, split-D runs at 1.3x
  copy. It is never faster below that (a second launch and a second read), see SNIPPET.md.
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

# Widest row the one-program-per-row kernels keep in registers without spilling (A800, 8 warps).
ONE_ROW_MAX_D = 32768
# The fused backward keeps a [R, BLOCK_D] tile of x, dy and the dw accumulator in registers.
BWD_MAX_D = 8192


@triton.jit
def _rmsnorm_fwd_kernel(
    x_ptr, w_ptr, y_ptr, rstd_ptr,
    D, stride_row, eps,
    BLOCK_D: tl.constexpr, COMPUTE: tl.constexpr, SAVE_RSTD: tl.constexpr,
):
    # One program per row. int64 row offset: N * D can exceed 2^31 elements.
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D
    x = tl.load(x_ptr + row * stride_row + cols, mask=mask, other=0.0).to(COMPUTE)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(COMPUTE)
    rstd = tl.rsqrt(tl.sum(x * x, axis=0) / D + eps)
    y = x * rstd * w
    tl.store(y_ptr + row * D + cols, y.to(y_ptr.dtype.element_ty), mask=mask)
    if SAVE_RSTD:  # saved-for-backward tensors are written only when a backward can follow
        tl.store(rstd_ptr + row, rstd)


@triton.jit
def _rmsnorm_fwd_sumsq_kernel(
    x_ptr, partial_ptr, D, stride_row,
    N_CHUNKS: tl.constexpr, BLOCK_C: tl.constexpr, COMPUTE: tl.constexpr,
):
    # Split-D stage 1: program (row, chunk) reduces BLOCK_C elements to one fp32 partial.
    row = tl.program_id(0).to(tl.int64)
    chunk = tl.program_id(1).to(tl.int64)
    cols = chunk * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = cols < D
    x = tl.load(x_ptr + row * stride_row + cols, mask=mask, other=0.0).to(COMPUTE)
    tl.store(partial_ptr + row * N_CHUNKS + chunk, tl.sum(x * x, axis=0))


@triton.jit
def _rmsnorm_fwd_normalize_kernel(
    x_ptr, w_ptr, y_ptr, rstd_ptr, partial_ptr, D, stride_row, eps,
    N_CHUNKS: tl.constexpr, BLOCK_C: tl.constexpr, COMPUTE: tl.constexpr, SAVE_RSTD: tl.constexpr,
):
    # Split-D stage 2: every program of a row sums the N_CHUNKS partials (a few floats) to the
    # same rstd, normalises its chunk, and chunk 0 stores rstd. N_CHUNKS is a power of two.
    row = tl.program_id(0).to(tl.int64)
    chunk = tl.program_id(1).to(tl.int64)
    partial = tl.load(partial_ptr + row * N_CHUNKS + tl.arange(0, N_CHUNKS))
    rstd = tl.rsqrt(tl.sum(partial, axis=0) / D + eps)
    cols = chunk * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = cols < D
    x = tl.load(x_ptr + row * stride_row + cols, mask=mask, other=0.0).to(COMPUTE)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(COMPUTE)
    tl.store(y_ptr + row * D + cols, (x * rstd * w).to(y_ptr.dtype.element_ty), mask=mask)
    if SAVE_RSTD:
        if chunk == 0:
            tl.store(rstd_ptr + row, rstd)


@triton.jit
def _rmsnorm_bwd_kernel(
    dy_ptr, x_ptr, w_ptr, rstd_ptr, dx_ptr, dw_partial_ptr,
    N, D, stride_x, stride_dy,
    BLOCK_D: tl.constexpr, R: tl.constexpr, COMPUTE: tl.constexpr,
):
    # A fixed number of programs stride over the rows, R rows per step as a [R, BLOCK_D] tile
    # so several rows' loads are in flight together. Each program keeps a register accumulator
    # for dw and writes one partial row at the end: a per-row dw partial would add N*D*4 bytes
    # of traffic, more than the whole backward moves otherwise. x and dy carry their own row
    # strides (dy arrives from autograd with whatever layout the graph gave it); dx is contiguous.
    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)
    cols = tl.arange(0, BLOCK_D)[None, :]
    cmask = cols < D
    w = tl.load(w_ptr + cols, mask=cmask, other=0.0).to(COMPUTE)
    dw_acc = tl.zeros([R, BLOCK_D], dtype=COMPUTE)
    for row0 in range(pid, N, n_programs * R):
        rows = (row0 + tl.arange(0, R) * n_programs).to(tl.int64)[:, None]
        mask = (rows < N) & cmask
        x = tl.load(x_ptr + rows * stride_x + cols, mask=mask, other=0.0).to(COMPUTE)
        dy = tl.load(dy_ptr + rows * stride_dy + cols, mask=mask, other=0.0).to(COMPUTE)
        rstd = tl.load(rstd_ptr + rows, mask=rows < N, other=0.0)
        xhat = x * rstd
        dxhat = dy * w
        c = tl.sum(dxhat * xhat, axis=1)[:, None] / D
        dx = rstd * (dxhat - xhat * c)
        tl.store(dx_ptr + rows * D + cols, dx.to(dx_ptr.dtype.element_ty), mask=mask)
        dw_acc += dy * xhat
    dw = tl.sum(dw_acc, axis=0)
    dcols = tl.arange(0, BLOCK_D)
    tl.store(dw_partial_ptr + pid.to(tl.int64) * D + dcols, dw, mask=dcols < D)


def _num_warps(block: int) -> int:
    return 4 if block <= 4096 else (8 if block <= 16384 else 16)


def _fwd_geometry(D: int, split_d: Optional[bool] = None) -> Tuple[int, int, int]:
    """``(N_CHUNKS, BLOCK, num_warps)`` for the forward; ``N_CHUNKS == 1`` is one program per row.

    ``split_d=None`` applies the measured rule: one program per row up to ``ONE_ROW_MAX_D``,
    split-D beyond it. Chunks are a quarter of the row, capped at 8192 elements (``D = 65536``
    gives 8 chunks of 8192). Pass ``True``/``False`` to force a path when comparing both
    geometries, as in the table in SNIPPET.md.
    """
    block = triton.next_power_of_2(D)
    if split_d is None:
        split_d = block > ONE_ROW_MAX_D
    if not split_d:
        return 1, block, _num_warps(block)
    block_c = max(128, min(block // 4, 8192))
    n_chunks = triton.next_power_of_2(triton.cdiv(D, block_c))
    if n_chunks == 1:
        return 1, block, _num_warps(block)
    return n_chunks, block_c, _num_warps(block_c)


def _bwd_geometry(D: int) -> Tuple[int, int, int]:
    """``(BLOCK_D, R, num_warps)``. R rows per backward tile keeps the tile near 4K elements.

    Measured on A800 (16K x 1K bf16): backward reaches ~86% of copy with R=4 and two programs
    per SM (fewer partials to reduce). The tile holds x, dy and the dw accumulator, which caps
    D at ``BWD_MAX_D`` (a split-D backward would need a two-stage `mean(dxhat * xhat)`).
    """
    if D > BWD_MAX_D:
        raise ValueError(f"rmsnorm backward: D={D} > {BWD_MAX_D} needs a split-D backward (not in this snippet)")
    block = triton.next_power_of_2(D)
    return block, max(1, 4096 // block), _num_warps(block)


def _num_sms(device: torch.device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


def rmsnorm_fwd(x: torch.Tensor, w: torch.Tensor, eps: float, save_rstd: bool = True, *, split_d: Optional[bool] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """``x`` (N, D) with unit last stride (any row stride) -> ``(y, rstd)``; ``y`` is contiguous.

    ``rstd`` is fp32 (N,) when ``save_rstd``, else a 0-element placeholder: nothing is
    allocated or written for a forward that no backward will follow. ``split_d`` forces a
    forward geometry (see ``_fwd_geometry``); leave it ``None`` outside tests.
    """
    N, D = x.shape
    assert x.stride(1) == 1 and w.is_contiguous()
    y = torch.empty((N, D), dtype=x.dtype, device=x.device)
    rstd = torch.empty(N if save_rstd else 0, dtype=torch.float32, device=x.device)
    n_chunks, block, warps = _fwd_geometry(D, split_d)
    if n_chunks == 1:
        _rmsnorm_fwd_kernel[(N,)](
            x, w, y, rstd, D, x.stride(0), eps,
            BLOCK_D=block, COMPUTE=tl.float32, SAVE_RSTD=save_rstd, num_warps=warps,
        )
    else:
        partial = torch.empty((N, n_chunks), dtype=torch.float32, device=x.device)
        _rmsnorm_fwd_sumsq_kernel[(N, n_chunks)](
            x, partial, D, x.stride(0), N_CHUNKS=n_chunks, BLOCK_C=block, COMPUTE=tl.float32, num_warps=warps,
        )
        _rmsnorm_fwd_normalize_kernel[(N, n_chunks)](
            x, w, y, rstd, partial, D, x.stride(0), eps,
            N_CHUNKS=n_chunks, BLOCK_C=block, COMPUTE=tl.float32, SAVE_RSTD=save_rstd, num_warps=warps,
        )
    return y, rstd


def rmsnorm_bwd(dy: torch.Tensor, x: torch.Tensor, w: torch.Tensor, rstd: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns ``(dx, dw)``; ``dw`` is reduced from per-program fp32 partials on the host.

    ``dy`` and ``x`` are ``(N, D)`` with unit last stride and any row stride; ``dx`` is contiguous.
    """
    N, D = x.shape
    assert dy.stride(1) == 1 and dy.shape == x.shape
    dx = torch.empty((N, D), dtype=x.dtype, device=x.device)
    block, rows, warps = _bwd_geometry(D)
    n_programs = max(1, min(triton.cdiv(N, rows), 2 * _num_sms(x.device)))
    dw_partial = torch.empty(n_programs, D, dtype=torch.float32, device=x.device)
    _rmsnorm_bwd_kernel[(n_programs,)](
        dy, x, w, rstd, dx, dw_partial, N, D, x.stride(0), dy.stride(0),
        BLOCK_D=block, R=rows, COMPUTE=tl.float32, num_warps=warps,
    )
    return dx, dw_partial.sum(0).to(w.dtype)


def _rows(t: torch.Tensor) -> torch.Tensor:
    """``(..., D)`` -> ``(N, D)`` without a copy. Works for contiguous tensors and for row-strided
    2-D views; a layout `view` cannot express (e.g. a padded 3-D row) is the caller's to fix
    (a kernel package passes the row stride from `_helpers.rows_and_stride` instead)."""
    return t if t.dim() == 2 else t.view(-1, t.shape[-1])


class _RMSNorm(torch.autograd.Function):
    # Standalone snippet: plain autograd.Function. Inside a kernel package use
    # `_common.compat.register_kernel` + `make_differentiable` instead (torch.compile-safe;
    # it also sets `save_aux` per call the way `rmsnorm()` sets `save_rstd` here).
    @staticmethod
    def forward(ctx, x, w, eps, save_rstd):
        x2d = _rows(x)
        y, rstd = rmsnorm_fwd(x2d, w, eps, save_rstd)
        if save_rstd:
            ctx.save_for_backward(x2d, w, rstd)
        return y.view(x.shape)

    @staticmethod
    def backward(ctx, dy):
        x2d, w, rstd = ctx.saved_tensors
        dx, dw = rmsnorm_bwd(_rows(dy), x2d, w, rstd)
        return dx.view(dy.shape), dw, None, None


def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm over the last dim; fp32 math, output in ``x.dtype``. Differentiable for
    ``D <= BWD_MAX_D``; wider rows run forward-only (inference) through the split-D path."""
    # Grad mode is off inside Function.forward, so the decision is made here.
    save_rstd = torch.is_grad_enabled() and (x.requires_grad or w.requires_grad)
    if save_rstd and x.shape[-1] > BWD_MAX_D:
        raise ValueError(f"rmsnorm: D={x.shape[-1]} > {BWD_MAX_D} is forward-only in this snippet (run under no_grad)")
    return _RMSNorm.apply(x, w, eps, save_rstd)


__all__ = ["rmsnorm", "rmsnorm_fwd", "rmsnorm_bwd"]
