"""Backward for `{{op}}`: a Triton adjoint of the epilogue, then cuBLAS for the two data GEMMs.

cuBLASLt has ``DGELU`` / ``DRELU`` / ``BGRAD`` epilogs, but they apply to a GEMM *result* (the
next layer's ``dX`` GEMM); inside one op the incoming gradient is a tensor, so the adjoint is a
streaming kernel: ``dZ = dY * act'(Z)`` with per-program fp32 ``db`` partials, `[32, 128]`
tiles, a fixed number of programs along ``M`` (the `epilogue_bwd` of
`kda-kernel-implement/reference/triton/epilogue/gemm/kernel.py`; copy it, it is 40 lines).
Then ``dX = dZ @ W`` and ``dW = dZ^T @ X`` in `torch.matmul` (cuBLAS; nothing to fuse).

Same conventions as every Triton kernel in this tree: strides are arguments (``dy`` arrives with
whatever strides autograd gave it; never ``.contiguous()`` it), int64 program ids, masks on the
tails. ``recompute`` (SPEC ``recompute``): the forward ran without the aux epilog and ``aux`` is
the 0-element placeholder; rebuild ``Z + b`` here with one ``BIAS`` GEMM through `_impl_fwd._plan`.
"""

from typing import Tuple

import torch
import triton
import triton.language as tl

from ._impl_fwd import _plan  # noqa: F401  (recompute path)


@triton.jit
def _{{op}}_epilogue_bwd_kernel(
    dy_ptr,
    z_ptr,
    dz_ptr,
    db_partial_ptr,
    M,
    N,
    stride_dym,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    # TODO(implementer): dZ = dY * act'(Z) streamed in [BLOCK_M, BLOCK_N] tiles; a fixed number
    # of programs along M strides over the row tiles and keeps a register accumulator for db,
    # writing one fp32 partial row per program (reduced with torch.sum on the host). int64 on
    # the program ids: M * N passes 2^31 at LLM sizes.
    pid_n = tl.program_id(0).to(tl.int64)
    pid_m = tl.program_id(1).to(tl.int64)
    raise NotImplementedError


def {{op}}_bwd(
    dy: torch.Tensor, x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor, aux: torch.Tensor, recompute: bool
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns ``(dx, dw, dbias)``. ``aux`` is the saved pre-activation (with the bias) or, under recompute, empty."""
    # TODO(implementer): sketch:
    # M, N = dy.shape
    # z = aux if not recompute else _plan(w, x.t(), MatmulEpilog.BIAS, bias).execute().t().contiguous()
    # n_prog_m = ...; dz = torch.empty_like(z); db_partial = torch.empty(n_prog_m, N, dtype=torch.float32, device=dy.device)
    # _{{op}}_epilogue_bwd_kernel[(triton.cdiv(N, 128), n_prog_m)](dy, z, dz, db_partial, M, N, dy.stride(0), BLOCK_M=32, BLOCK_N=128, HAS_BIAS=True, num_warps=4)
    # dx = torch.matmul(dz, w); dw = torch.matmul(dz.t(), x); db = db_partial.sum(0).to(bias.dtype)
    # return dx, dw, db
    raise NotImplementedError("{{op}}_bwd: implement the adjoint launcher")


def {{op}}_bwd_fake(
    dy: torch.Tensor, x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor, aux: torch.Tensor, recompute: bool
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Output metadata only."""
    raise NotImplementedError("{{op}}_bwd_fake: describe the backward outputs")


__all__ = ["{{op}}_bwd", "{{op}}_bwd_fake"]
