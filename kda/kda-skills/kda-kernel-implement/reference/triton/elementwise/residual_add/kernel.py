"""Residual add in Triton: ``h = x + residual`` with fp32 accumulation, low-precision I/O.

The minimal streaming kernel: flat 1-D tiles, int64 offsets, masked tail, vectorised loads.
Its backward is the identity on both inputs (``dx = dres = dh``), so no backward kernel exists;
the value of this snippet is the load/store skeleton every fused kernel starts from and the
fp32 residual-stream variant (``out_dtype=torch.float32``) that pre-norm blocks need.
"""

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _residual_add_kernel(
    x_ptr, r_ptr, h_ptr, n_elements,
    BLOCK: tl.constexpr, COMPUTE: tl.constexpr,
):
    # int64 block offset: n_elements can exceed 2^31 for activations of large models.
    start = tl.program_id(0).to(tl.int64) * BLOCK
    offs = start + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(COMPUTE)
    r = tl.load(r_ptr + offs, mask=mask, other=0.0).to(COMPUTE)
    tl.store(h_ptr + offs, (x + r).to(h_ptr.dtype.element_ty), mask=mask)


def residual_add(x: torch.Tensor, residual: torch.Tensor, out_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """``x + residual`` computed in fp32, stored as ``out_dtype`` (default ``x.dtype``). Differentiable."""
    return _ResidualAdd.apply(x, residual, out_dtype)


def residual_add_fwd(x: torch.Tensor, residual: torch.Tensor, out_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    assert x.shape == residual.shape and x.is_contiguous() and residual.is_contiguous()
    h = torch.empty(x.shape, dtype=out_dtype or x.dtype, device=x.device)
    n = x.numel()
    # 2K elements per program: 4 warps x 32 lanes x 16 elements = 16-byte vector loads of bf16.
    BLOCK = 2048
    _residual_add_kernel[(triton.cdiv(n, BLOCK),)](x, residual, h, n, BLOCK=BLOCK, COMPUTE=tl.float32, num_warps=4)
    return h


class _ResidualAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, residual, out_dtype):
        ctx.dtypes = (x.dtype, residual.dtype)
        return residual_add_fwd(x, residual, out_dtype)

    @staticmethod
    def backward(ctx, dh):
        xd, rd = ctx.dtypes
        return dh.to(xd), dh.to(rd), None


__all__ = ["residual_add", "residual_add_fwd"]
