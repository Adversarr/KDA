"""Golden fused residual-add + RMSNorm (fp32 residual stream), forward and backward.

The region being fused (``minilm/model.py::add_rmsnorm``)::

    h    = residual + x.float()                       # fp32 residual stream, returned
    rstd = rsqrt(mean(h^2) + eps)                     # saved
    y    = (h * rstd * w).to(x.dtype)                 # low-precision sublayer input, returned

Backward, with ``dh`` the gradient flowing into ``h`` from downstream and ``dy`` into ``y``::

    xhat  = h * rstd
    dxhat = dy * w
    g     = dh + rstd * (dxhat - xhat * mean(dxhat * xhat))    # fp32
    dresidual = g;  dx = g.to(x.dtype);  dw = sum_rows(dy * xhat)

Traffic: forward reads x (es) and residual (4), writes h (4), y (es), rstd; backward reads
dy (es), dh (4), h (4), rstd, writes dx (es), dresidual (4) plus dw partials. ``h`` is an
output, so saving it for the backward costs nothing extra.

Layout: every input is ``(..., D)`` with a unit stride along ``D`` and any row stride (a
padded row or a slice of a wider buffer is passed as ``stride_*``, never copied); ``dy`` and
``dh`` arrive from autograd with whatever strides the graph gave them and go to the kernel as
they are. Outputs are contiguous. Leading dims are flattened with ``view`` (no copy).

Two forward geometries (the rmsnorm snippet's row width rule): one program per row keeps the
whole row in registers and is at the copy roof for every ``D <= 32768`` and every row count;
beyond that the row spills, and a two-stage split-D path takes over (stage 1 writes ``h`` and a
partial sum of squares per chunk, stage 2 reads the fp32 ``h`` chunk back, so ``x`` and
``residual`` are still read once). The backward keeps an ``[R, BLOCK_D]`` tile of ``h``, ``dy``,
``dh`` and the ``dw`` accumulator in registers and is limited to ``D <= 8192``; wider rows are
forward-only here.

This is a standalone reference kernel (GOLDEN.md has the A800 numbers);
a kernel package would wire the same launchers through ``_common.compat``.
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

ONE_ROW_MAX_D = 32768  # widest row one program keeps in registers without spilling (A800, 8 warps)
BWD_MAX_D = 8192  # the fused backward tile holds h, dy, dh and the dw accumulator


@triton.jit
def _fwd_kernel(
    x_ptr, r_ptr, w_ptr, h_ptr, y_ptr, rstd_ptr,
    D, stride_x, stride_r, eps,
    BLOCK_D: tl.constexpr, COMPUTE: tl.constexpr, SAVE_RSTD: tl.constexpr,
):
    # One program per row; int64 row offset because N * D can exceed 2^31 elements.
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D
    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(COMPUTE)
    r = tl.load(r_ptr + row * stride_r + cols, mask=mask, other=0.0).to(COMPUTE)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(COMPUTE)
    h = r + x
    rstd = tl.rsqrt(tl.sum(h * h, axis=0) / D + eps)
    tl.store(h_ptr + row * D + cols, h, mask=mask)
    tl.store(y_ptr + row * D + cols, (h * rstd * w).to(y_ptr.dtype.element_ty), mask=mask)
    if SAVE_RSTD:  # saved for the backward only when one can follow
        tl.store(rstd_ptr + row, rstd)


@triton.jit
def _fwd_sumsq_kernel(
    x_ptr, r_ptr, h_ptr, partial_ptr, D, stride_x, stride_r,
    N_CHUNKS: tl.constexpr, BLOCK_C: tl.constexpr, COMPUTE: tl.constexpr,
):
    # Split-D stage 1: program (row, chunk) forms its chunk of h, stores it (h is an output
    # anyway) and reduces it to one fp32 partial sum of squares.
    row = tl.program_id(0).to(tl.int64)
    chunk = tl.program_id(1).to(tl.int64)
    cols = chunk * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = cols < D
    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(COMPUTE)
    r = tl.load(r_ptr + row * stride_r + cols, mask=mask, other=0.0).to(COMPUTE)
    h = r + x
    tl.store(h_ptr + row * D + cols, h, mask=mask)
    tl.store(partial_ptr + row * N_CHUNKS + chunk, tl.sum(h * h, axis=0))


@triton.jit
def _fwd_normalize_kernel(
    h_ptr, w_ptr, y_ptr, rstd_ptr, partial_ptr, D, eps,
    N_CHUNKS: tl.constexpr, BLOCK_C: tl.constexpr, COMPUTE: tl.constexpr, SAVE_RSTD: tl.constexpr,
):
    # Split-D stage 2: every program of a row sums the N_CHUNKS partials to the same rstd,
    # normalises its chunk of the fp32 h (a second read of h, not of x and residual), and
    # chunk 0 stores rstd. N_CHUNKS is a power of two.
    row = tl.program_id(0).to(tl.int64)
    chunk = tl.program_id(1).to(tl.int64)
    partial = tl.load(partial_ptr + row * N_CHUNKS + tl.arange(0, N_CHUNKS))
    rstd = tl.rsqrt(tl.sum(partial, axis=0) / D + eps)
    cols = chunk * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = cols < D
    h = tl.load(h_ptr + row * D + cols, mask=mask, other=0.0)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(COMPUTE)
    tl.store(y_ptr + row * D + cols, (h * rstd * w).to(y_ptr.dtype.element_ty), mask=mask)
    if SAVE_RSTD:
        if chunk == 0:
            tl.store(rstd_ptr + row, rstd)


@triton.jit
def _small_bwd_kernel(
    DY, DH, H, W, RS, DX, DR, DW, N: tl.constexpr, D: tl.constexpr,
    SY, SYD, SY2, SY3, SH, SHD, SH2, SH3, N1, N2,
    BD: tl.constexpr, BN: tl.constexpr, BC: tl.constexpr,
):
    """Each block owns one activation row and a disjoint weight-gradient slice."""
    row = tl.program_id(0).to(tl.int64)
    d = tl.arange(0, BD)
    h = tl.load(H + row * D + d, d < D, other=0)
    yo = row % N1 * SY + (row // N1) % N2 * SY2 + row // (N1*N2) * SY3
    ho = row % N1 * SH + (row // N1) % N2 * SH2 + row // (N1*N2) * SH3
    dy = tl.load(DY + yo + d * SYD, d < D, other=0).to(tl.float32)
    dh = tl.load(DH + ho + d * SHD, d < D, other=0).to(tl.float32)
    w = tl.load(W + d, d < D, other=0).to(tl.float32)
    rs = tl.load(RS + row)
    xhat = h * rs
    g = dh + rs * (dy * w - xhat * tl.sum(dy * w * xhat, 0) / D)
    tl.store(DX + row * D + d, g, d < D)
    tl.store(DR + row * D + d, g, d < D)
    # Columns have a single owner, so the small reduction needs neither a partial
    # allocation nor a second launch. The extra reads are confined to tiny inputs.
    cols = row * BC + tl.arange(0, BC)
    rows = tl.arange(0, BN).to(tl.int64)
    mask = (rows[:, None] < N) & (cols[None, :] < D)
    hh = tl.load(H + rows[:, None] * D + cols[None, :], mask, other=0)
    offsets = rows % N1 * SY + (rows // N1) % N2 * SY2 + rows // (N1*N2) * SY3
    yy = tl.load(DY + offsets[:, None] + cols[None, :] * SYD, mask, other=0).to(tl.float32)
    rr = tl.load(RS + rows, rows < N, other=0)
    tl.store(DW + cols, tl.sum(yy * hh * rr[:, None], 0), cols < D)


@triton.jit
def _bwd_kernel(
    dy_ptr, dh_ptr, h_ptr, w_ptr, rstd_ptr, dx_ptr, dr_ptr, dw_partial_ptr,
    N, D, stride_dy, stride_dyd, sy2, sy3, stride_dh, stride_dhd, sh2, sh3, N1, N2,
    BLOCK_D: tl.constexpr, R: tl.constexpr, COMPUTE: tl.constexpr,
):
    # Fixed grid striding over rows, R rows per tile; one fp32 dw partial per program. dy and
    # dh carry their own row strides; h, dx and dresidual are contiguous.
    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)
    cols = tl.arange(0, BLOCK_D)[None, :]
    cmask = cols < D
    w = tl.load(w_ptr + cols, mask=cmask, other=0.0).to(COMPUTE)
    dw_acc = tl.zeros([R, BLOCK_D], dtype=COMPUTE)
    for row0 in range(pid, N, n_programs * R):
        rows = (row0 + tl.arange(0, R) * n_programs).to(tl.int64)[:, None]
        mask = (rows < N) & cmask
        h = tl.load(h_ptr + rows * D + cols, mask=mask, other=0.0)
        yo = rows % N1 * stride_dy + (rows // N1) % N2 * sy2 + rows // (N1*N2) * sy3
        ho = rows % N1 * stride_dh + (rows // N1) % N2 * sh2 + rows // (N1*N2) * sh3
        dy = tl.load(dy_ptr + yo + cols * stride_dyd, mask=mask, other=0.0).to(COMPUTE)
        dh = tl.load(dh_ptr + ho + cols * stride_dhd, mask=mask, other=0.0).to(COMPUTE)
        rstd = tl.load(rstd_ptr + rows, mask=rows < N, other=0.0)
        xhat = h * rstd
        dxhat = dy * w
        c = tl.sum(dxhat * xhat, axis=1)[:, None] / D
        g = dh + rstd * (dxhat - xhat * c)
        tl.store(dr_ptr + rows * D + cols, g, mask=mask)
        tl.store(dx_ptr + rows * D + cols, g.to(dx_ptr.dtype.element_ty), mask=mask)
        dw_acc += dy * xhat
    dw = tl.sum(dw_acc, axis=0)
    dcols = tl.arange(0, BLOCK_D)
    tl.store(dw_partial_ptr + pid.to(tl.int64) * D + dcols, dw, mask=dcols < D)


def _num_warps(block: int) -> int:
    return 4 if block <= 4096 else (8 if block <= 16384 else 16)


def _fwd_geometry(D: int, split_d: Optional[bool] = None) -> Tuple[int, int, int]:
    """``(N_CHUNKS, BLOCK, num_warps)``; ``N_CHUNKS == 1`` is one program per row.

    ``split_d=None`` applies the row width rule (one row per program up to ``ONE_ROW_MAX_D``);
    ``True``/``False`` forces a path. Chunks are a quarter of the
    row, capped at 8192 elements.
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
    """``(BLOCK_D, rows per tile, num_warps)``; the tile stays near 4K elements."""
    if D > BWD_MAX_D:
        raise ValueError(f"add_rmsnorm backward: D={D} > {BWD_MAX_D} needs a split-D backward (forward-only here)")
    block = triton.next_power_of_2(D)
    return block, max(1, 4096 // block), _num_warps(block)


def add_rmsnorm_fwd(
    x: torch.Tensor, residual: torch.Tensor, w: torch.Tensor, eps: float, save_rstd: bool = True, *, split_d: Optional[bool] = None
):
    """``x`` (N, D) low precision, ``residual`` (N, D) fp32, both with unit last stride and any
    row stride -> ``(h, y, rstd)``, all contiguous.

    ``rstd`` is a 0-element placeholder when ``save_rstd`` is false (inference). ``split_d``
    forces a forward geometry; leave it ``None`` outside tests.
    """
    N, D = x.shape
    assert x.stride(1) == 1 and residual.stride(1) == 1 and w.is_contiguous()
    h = torch.empty(N, D, dtype=torch.float32, device=x.device)
    y = torch.empty(N, D, dtype=x.dtype, device=x.device)
    rstd = torch.empty(N if save_rstd else 0, dtype=torch.float32, device=x.device)
    n_chunks, block, warps = _fwd_geometry(D, split_d)
    if n_chunks == 1:
        _fwd_kernel[(N,)](
            x, residual, w, h, y, rstd, D, x.stride(0), residual.stride(0), eps,
            BLOCK_D=block, COMPUTE=tl.float32, SAVE_RSTD=save_rstd, num_warps=warps,
        )
    else:
        partial = torch.empty((N, n_chunks), dtype=torch.float32, device=x.device)
        _fwd_sumsq_kernel[(N, n_chunks)](
            x, residual, h, partial, D, x.stride(0), residual.stride(0),
            N_CHUNKS=n_chunks, BLOCK_C=block, COMPUTE=tl.float32, num_warps=warps,
        )
        _fwd_normalize_kernel[(N, n_chunks)](
            h, w, y, rstd, partial, D, eps,
            N_CHUNKS=n_chunks, BLOCK_C=block, COMPUTE=tl.float32, SAVE_RSTD=save_rstd, num_warps=warps,
        )
    return h, y, rstd


def add_rmsnorm_bwd(dy: torch.Tensor, dh: torch.Tensor, h: torch.Tensor, w: torch.Tensor, rstd: torch.Tensor, x_dtype: torch.dtype):
    """Returns ``(dx, dresidual, dw)``. ``dy`` and ``dh`` keep their row strides (no copy)."""
    N, D = h.shape
    if dy.shape != dh.shape or dy.numel() != h.numel() or not 2 <= dy.ndim <= 4:
        raise ValueError("backward gradients must match and have two to four dimensions")
    def layout(t):
        return (t.stride(-2), t.stride(-1), t.stride(-3) if t.ndim >= 3 else 0,
                t.stride(-4) if t.ndim >= 4 else 0)
    n1, n2 = dy.shape[-2], dy.shape[-3] if dy.ndim >= 3 else 1
    if 0 < N <= 128 and D <= 8192:
        dx = torch.empty_like(h, dtype=x_dtype)
        dr = torch.empty_like(h)
        dw = torch.empty_like(w)
        _small_bwd_kernel[(N,)](
            dy, dh, h, w, rstd, dx, dr, dw, N, D,
            *layout(dy), *layout(dh), n1, n2,
            BD=triton.next_power_of_2(D), BN=triton.next_power_of_2(N),
            BC=triton.next_power_of_2(triton.cdiv(D, N)), num_warps=4 if D <= 2048 else 8,
        )
        return dx, dr, dw
    dx = torch.empty(N, D, dtype=x_dtype, device=h.device)
    dr = torch.empty(N, D, dtype=torch.float32, device=h.device)
    block, rows, warps = _bwd_geometry(D)
    sms = torch.cuda.get_device_properties(h.device).multi_processor_count
    n_programs = max(1, min(triton.cdiv(N, rows), 2 * sms))
    dw_partial = torch.empty(n_programs, D, dtype=torch.float32, device=h.device)
    _bwd_kernel[(n_programs,)](
        dy, dh, h, w, rstd, dx, dr, dw_partial, N, D, *layout(dy), *layout(dh), n1, n2,
        BLOCK_D=block, R=rows, COMPUTE=tl.float32, num_warps=warps,
    )
    return dx, dr, dw_partial.sum(0).to(w.dtype)


def _rows(t: torch.Tensor) -> torch.Tensor:
    """``(..., D)`` -> ``(N, D)`` without a copy: ``view`` for dense leading dims, ``as_strided``
    for a row-strided block (padded rows, a slice of a wider buffer). Raises on a layout that
    is not one block of rows instead of copying silently."""
    if t.dim() == 2:
        return t
    D = t.shape[-1]
    n_rows = t.numel() // D if D else 0
    for i in range(t.dim() - 2):
        if t.shape[i] > 1 and t.stride(i) != t.shape[i + 1] * t.stride(i + 1):
            raise ValueError(f"add_rmsnorm: leading dims of shape {tuple(t.shape)} / strides {t.stride()} are not one block of rows")
    return t.as_strided((n_rows, D), (t.stride(-2), t.stride(-1)))


class _AddRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, residual, w, eps, save_rstd):
        shape = x.shape
        h, y, rstd = add_rmsnorm_fwd(_rows(x), _rows(residual), w, eps, save_rstd)
        if save_rstd:
            ctx.save_for_backward(h, w, rstd)
        ctx.x_dtype = x.dtype
        ctx.shape = shape
        return h.view(shape), y.view(shape)

    @staticmethod
    def backward(ctx, dh, dy):
        h, w, rstd = ctx.saved_tensors
        # Either output may be unused downstream; autograd then passes None.
        dh2 = torch.zeros_like(h).view(ctx.shape) if dh is None else dh
        dy2 = torch.zeros(ctx.shape, dtype=ctx.x_dtype, device=h.device) if dy is None else dy
        dx, dr, dw = add_rmsnorm_bwd(dy2, dh2, h, w, rstd, ctx.x_dtype)
        shape = ctx.shape
        return dx.view(shape), dr.view(shape), dw, None, None


def add_rmsnorm(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
    """Fused ``(h, y) = (residual + x, rmsnorm(h) * weight)``; ``h`` fp32, ``y`` in ``x.dtype``.

    Differentiable for ``D <= BWD_MAX_D``; wider rows run forward-only through split-D.
    """
    # rstd is written only when a backward can follow (grad mode is off inside Function.forward).
    save_rstd = torch.is_grad_enabled() and any(t.requires_grad for t in (x, residual, weight))
    if save_rstd and x.shape[-1] > BWD_MAX_D:
        raise ValueError(f"add_rmsnorm: D={x.shape[-1]} > {BWD_MAX_D} is forward-only (run under no_grad)")
    return _AddRMSNorm.apply(x, residual, weight, eps, save_rstd)


__all__ = ["add_rmsnorm", "add_rmsnorm_fwd", "add_rmsnorm_bwd"]
