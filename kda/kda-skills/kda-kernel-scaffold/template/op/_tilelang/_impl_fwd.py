"""Forward kernel(s) and host launcher for `{{op}}` (TileLang backend).

Conventions (kda-kernel-implement/SKILL.md and reference/tilelang/README.md, checked by
``lint_kernel.py``):

* Eager JIT: ``@tilelang.jit`` on a Python function that takes the tensors and the launch
  parameters directly. Shapes the tile depends on are ``T.const`` (one compiled variant per
  value); the row count is ``T.dynamic`` (no recompile per call). Python ``bool`` / ``int``
  arguments (``save_aux``, ``recompute``, ``block``, the strides) specialise the kernel:
  TileLang caches one variant per distinct value, so ``if save_aux:`` is resolved at compile
  time exactly like a Triton ``constexpr``.
* Every input is declared ``T.StridedTensor((R, D), (stride_x, 1), dt)`` with the row stride
  passed as a **Python int** (``x.stride(0)``), not ``T.dynamic``: a dynamic stride stops the
  copies from vectorising and halves throughput (reference/tilelang/README.md, measured).
  TileLang's packed-ABI check rejects a tensor whose strides differ from the declaration, so
  never ``.contiguous()`` an input on the host; pass the stride through. `interface.py` already
  rejected a non-unit last stride.
* Fill fragments with ``T.copy`` (predicated at the tails) where a tile has two dimensions, and
  stage 2-D stores through a shared tile; a fragment that is only cleared, or only written by
  guarded scalar stores, fails layout inference. The pitfall list is in
  reference/tilelang/README.md; read it before the first compile.
* Arithmetic in SPEC ``compute_dtype`` (``accum_dtype``); one ``T.cast`` back to the storage
  dtype at the store. Tensor-core operands (``T.gemm``) stay in storage dtype; the accumulator
  fragment is ``accum_dtype``.
* Outputs are allocated inside the kernel with ``T.empty`` and returned; the launcher hands
  them on. The launcher returns the user-visible outputs followed by the aux tensors the
  backward needs (SPEC ``saved_for_backward``) and must be fully type-annotated (the op schema
  is inferred from its signature).

Aux tensors are optional work: with ``save_aux=False`` (inference, or the recompute training
path) the aux stores are compiled out and the launcher returns a 0-element placeholder per
aux output so the schema stays fixed.
"""

from typing import Tuple

import torch
import tilelang
import tilelang.language as T

from .._helpers import rows_and_stride
from ._configs import select_config


@tilelang.jit
def _{{op}}_fwd_kernel(x, stride_x: int, save_aux: bool, block: int, dtype=T.bfloat16, accum_dtype=T.float32):
    # TODO(implementer): one program per row (or per ROWS_PER_PROGRAM rows) for every row count;
    # start with the measured width rule; change geometry only with workload evidence (compute-patterns.md).
    D = T.const("D")
    R = T.dynamic("R")
    x: T.StridedTensor((R, D), (stride_x, 1), dtype)
    y = T.empty((R, D), dtype)
    aux = T.empty((R,) if save_aux else (0,), accum_dtype)
    with T.Kernel(R, threads=128) as (row,):
        xl = T.alloc_fragment((block,), accum_dtype)
        for i in T.Parallel(block):
            xl[i] = T.cast(x[row, i], accum_dtype) if i < D else T.cast(0, accum_dtype)
        for i in T.Parallel(block):
            if i < D:
                y[row, i] = T.cast(xl[i], dtype)
        if save_aux:
            # TODO(implementer): the per-row statistic the backward reuses, e.g. rstd.
            s = T.alloc_fragment((1,), accum_dtype)
            T.reduce_sum(xl, s, dim=0)
            aux[row] = s[0]
    return y, aux


def {{op}}_fwd(x: torch.Tensor, save_aux: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    """Launch the forward. Returns ``(y, aux)``; see SPEC for what aux is.

    ``aux`` is allocated and written only when ``save_aux``; otherwise it is the 0-element
    placeholder so the op schema (two outputs) is unchanged.
    """
    # TODO(implementer): replace this placeholder. Sketch:
    # n_rows, stride_x = rows_and_stride(x)
    # if n_rows == 0:  # a zero-size grid is a CUDA error; the contract probe checks this row
    #     return torch.empty_like(x), torch.empty(0, dtype=torch.float32, device=x.device)
    # cfg = select_config({"n_rows": n_rows, "d": x.shape[-1]})
    # rows = x.view(n_rows, x.shape[-1])  # view, never reshape: the kernel takes the stride
    # y, aux = _{{op}}_fwd_kernel(rows, stride_x, save_aux, cfg["BLOCK_D"], dtype=_TL_DTYPE[x.dtype])
    # return y.view(x.shape), aux
    raise NotImplementedError("{{op}}_fwd: implement the TileLang forward launcher")


def {{op}}_fwd_fake(x: torch.Tensor, save_aux: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    """Output metadata only (what torch.compile traces); no computation."""
    # TODO(implementer): mirror the shapes/dtypes returned by {{op}}_fwd, including the
    # 0-element aux when not save_aux.
    raise NotImplementedError("{{op}}_fwd_fake: describe the forward outputs")


__all__ = ["{{op}}_fwd", "{{op}}_fwd_fake"]
