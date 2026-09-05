"""Backward kernel(s) and host launcher for `{{op}}`.

Same conventions as `_impl_fwd.py`: strides are arguments (``dy`` arrives with whatever strides
autograd gave it; never ``.contiguous()`` it), int64 offsets, masks, ``COMPUTE`` dtype.

``RECOMPUTE: tl.constexpr`` selects the second training path (SPEC ``recompute``): with
``RECOMPUTE`` the kernel ignores ``aux_ptr`` and recomputes the statistic from the saved inputs
(the forward ran with ``save_aux=False``); without it the aux is loaded. The launcher takes a
``recompute: bool`` so both variants share one schema; with ``recompute=True`` the ``aux``
argument is the 0-element placeholder.

Per-program weight-gradient partials are written as ``(n_programs, D)`` fp32 and reduced with
``torch.sum`` on the GPU; never accumulate with atomics across rows.
"""

from typing import Tuple

import torch
import triton
import triton.language as tl

from .._helpers import rows_and_stride
from ._configs import select_config


@triton.jit
def _{{op}}_bwd_kernel(
    dy_ptr,
    x_ptr,
    aux_ptr,
    dx_ptr,
    n_rows,
    n_cols,
    stride_dy_row,
    stride_x_row,
    stride_dx_row,
    BLOCK_D: tl.constexpr,
    COMPUTE: tl.constexpr,
    RECOMPUTE: tl.constexpr,
):
    # TODO(implementer)
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_D).to(tl.int64)
    mask = (row < n_rows) & (cols < n_cols)
    dy = tl.load(dy_ptr + row * stride_dy_row + cols, mask=mask, other=0).to(COMPUTE)
    if RECOMPUTE:
        # TODO(implementer): recompute the statistic from x (same code as the forward).
        x = tl.load(x_ptr + row * stride_x_row + cols, mask=mask, other=0).to(COMPUTE)
        stat = tl.sum(x, axis=0)
    else:
        stat = tl.load(aux_ptr + row)
    tl.store(dx_ptr + row * stride_dx_row + cols, (dy + stat * 0).to(dx_ptr.dtype.element_ty), mask=mask)


def {{op}}_bwd(dy: torch.Tensor, x: torch.Tensor, aux: torch.Tensor, recompute: bool) -> Tuple[torch.Tensor]:
    """Launch the backward. Returns one gradient per differentiable forward input."""
    # TODO(implementer)
    raise NotImplementedError("{{op}}_bwd: implement the Triton backward launcher")


def {{op}}_bwd_fake(dy: torch.Tensor, x: torch.Tensor, aux: torch.Tensor, recompute: bool) -> Tuple[torch.Tensor]:
    # TODO(implementer)
    raise NotImplementedError("{{op}}_bwd_fake: describe the backward outputs")


__all__ = ["{{op}}_bwd", "{{op}}_bwd_fake"]
