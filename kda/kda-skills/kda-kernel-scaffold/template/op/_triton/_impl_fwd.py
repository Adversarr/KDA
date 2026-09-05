"""Forward kernel(s) and host launcher for `{{op}}`.

Conventions (kda-kernel-implement/SKILL.md, checked by ``lint_kernel.py``):

* Arithmetic in SPEC ``compute_dtype`` through ``COMPUTE: tl.constexpr``; one cast back to the
  storage dtype at the store. Tensor-core operands (``tl.dot``) stay in storage dtype.
* Strides are launcher arguments (``x.stride(0)``), never ``.contiguous()``/``.reshape`` on an
  input: `interface.py` already rejected a non-unit last stride, leading strides follow the explicit SPEC layout contract.
* int64 offsets (``.to(tl.int64)`` on the program id), masked loads and stores on every tail,
  ``triton.next_power_of_2`` for the block over a non-power-of-two dim.
* The launcher returns the user-visible outputs followed by the aux tensors the backward needs
  (SPEC ``saved_for_backward``). Both functions must be fully type-annotated: the op schema is
  inferred from the launcher's signature.

Aux tensors are optional work. ``save_aux`` (set per call by ``_common.compat`` from grad mode,
``requires_grad`` and the recompute flag) gates every aux store behind ``SAVE_AUX: tl.constexpr``
in the kernel, and with ``save_aux=False`` the launcher returns a 0-element placeholder per aux
output instead of allocating and writing it. Inference (and the recompute path) never pays for
the backward.
"""

from typing import Tuple

import torch
import triton
import triton.language as tl

from .._helpers import rows_and_stride
from ._configs import select_config


@triton.jit
def _{{op}}_fwd_kernel(
    x_ptr,
    y_ptr,
    aux_ptr,
    n_rows,
    n_cols,
    stride_x_row,
    stride_y_row,
    BLOCK_D: tl.constexpr,
    COMPUTE: tl.constexpr,
    SAVE_AUX: tl.constexpr,
):
    # TODO(implementer): one program per row (or per ROWS_PER_PROGRAM rows) for every row count;
    # start with the measured width rule; change geometry only with workload evidence (compute-patterns.md).
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_D).to(tl.int64)
    mask = (row < n_rows) & (cols < n_cols)
    x = tl.load(x_ptr + row * stride_x_row + cols, mask=mask, other=0).to(COMPUTE)
    tl.store(y_ptr + row * stride_y_row + cols, x.to(y_ptr.dtype.element_ty), mask=mask)
    if SAVE_AUX:
        # TODO(implementer): the per-row statistic the backward reuses, e.g. rstd.
        tl.store(aux_ptr + row, tl.sum(x, axis=0))


def {{op}}_fwd(x: torch.Tensor, save_aux: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    """Launch the forward. Returns ``(y, aux)``; see SPEC for what aux is.

    ``aux`` is allocated and written only when ``save_aux``; otherwise it is
    ``torch.empty(0, ...)`` so the op schema (two outputs) is unchanged.
    """
    # TODO(implementer): replace this placeholder. Sketch:
    # n_rows, stride_x = rows_and_stride(x)
    # y = torch.empty_like(x)  # same strides as x when x is dense; else torch.empty + strides
    # aux = torch.empty(n_rows, dtype=torch.float32, device=x.device) if save_aux else torch.empty(0, ...)
    # if n_rows == 0:  # a zero-size grid is a CUDA error; the contract probe checks this row
    #     return y, aux
    # cfg = select_config({"n_rows": n_rows, "d": x.shape[-1]})
    # _{{op}}_fwd_kernel[(n_rows,)](x, y, aux, n_rows, x.shape[-1], stride_x, y.stride(-2), BLOCK_D=..., COMPUTE=tl.float32, SAVE_AUX=save_aux, num_warps=...)
    raise NotImplementedError("{{op}}_fwd: implement the Triton forward launcher")


def {{op}}_fwd_fake(x: torch.Tensor, save_aux: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    """Output metadata only (what torch.compile traces); no computation."""
    # TODO(implementer): mirror the shapes/dtypes returned by {{op}}_fwd, including the
    # 0-element aux when not save_aux.
    raise NotImplementedError("{{op}}_fwd_fake: describe the forward outputs")


__all__ = ["{{op}}_fwd", "{{op}}_fwd_fake"]
