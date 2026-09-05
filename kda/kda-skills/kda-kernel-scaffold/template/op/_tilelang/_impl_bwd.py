"""Backward kernel(s) and host launcher for `{{op}}` (TileLang backend).

Same conventions as `_impl_fwd.py`: strides declared with ``T.StridedTensor`` and passed as
Python ints (``dy`` arrives with whatever strides autograd gave it; never ``.contiguous()`` it,
pass ``dy.stride(0)``), ``accum_dtype`` math.

``recompute: bool`` (a Python argument, hence a compile-time specialisation) selects the
second training path (SPEC ``recompute``): with ``recompute`` the kernel ignores ``aux`` and
recomputes the statistic from the saved inputs (the forward ran with ``save_aux=False``);
without it the aux is loaded. Both variants share one launcher schema; with ``recompute=True``
the ``aux`` argument is the 0-element placeholder.

Per-program weight-gradient partials are written as ``(n_programs, D)`` fp32 and reduced with
``torch.sum`` on the GPU; never accumulate with atomics across rows.
"""

from typing import Tuple

import torch
import tilelang
import tilelang.language as T

from .._helpers import rows_and_stride
from ._configs import select_config


@tilelang.jit
def _{{op}}_bwd_kernel(
    dy, x, aux, stride_dy: int, stride_x: int, recompute: bool, block: int, dtype=T.bfloat16, accum_dtype=T.float32
):
    # TODO(implementer)
    D = T.const("D")
    R = T.dynamic("R")
    dy: T.StridedTensor((R, D), (stride_dy, 1), dtype)
    x: T.StridedTensor((R, D), (stride_x, 1), dtype)
    aux: T.Tensor((0,) if recompute else (R,), accum_dtype)
    dx = T.empty((R, D), dtype)
    with T.Kernel(R, threads=128) as (row,):
        dyl = T.alloc_fragment((block,), accum_dtype)
        stat = T.alloc_fragment((1,), accum_dtype)
        for i in T.Parallel(block):
            dyl[i] = T.cast(dy[row, i], accum_dtype) if i < D else T.cast(0, accum_dtype)
        if recompute:
            # TODO(implementer): recompute the statistic from x (same code as the forward).
            xl = T.alloc_fragment((block,), accum_dtype)
            for i in T.Parallel(block):
                xl[i] = T.cast(x[row, i], accum_dtype) if i < D else T.cast(0, accum_dtype)
            T.reduce_sum(xl, stat, dim=0)
        else:
            stat[0] = aux[row]
        for i in T.Parallel(block):
            if i < D:
                dx[row, i] = T.cast(dyl[i] + stat[0] * 0, dtype)
    return dx


def {{op}}_bwd(dy: torch.Tensor, x: torch.Tensor, aux: torch.Tensor, recompute: bool) -> Tuple[torch.Tensor]:
    """Launch the backward. Returns one gradient per differentiable forward input."""
    # TODO(implementer)
    raise NotImplementedError("{{op}}_bwd: implement the TileLang backward launcher")


def {{op}}_bwd_fake(dy: torch.Tensor, x: torch.Tensor, aux: torch.Tensor, recompute: bool) -> Tuple[torch.Tensor]:
    # TODO(implementer)
    raise NotImplementedError("{{op}}_bwd_fake: describe the backward outputs")


__all__ = ["{{op}}_bwd", "{{op}}_bwd_fake"]
