"""GEMM with a fused epilogue through nvmath-python (cuBLASLt): ``Y = act(X W^T + b)``, no kernel written.

**This is the preferred path for every GEMM + epilogue in KDA.** Same host signatures and test
as the Triton twin (`reference/triton/epilogue/gemm/`). The GEMM, the bias add, a ReLU or
tanh-GELU activation and the aux store for the backward run inside one cuBLASLt kernel picked
by `nvmath.linalg.advanced.Matmul`; an activation outside cuBLASLt's set is one streaming pass
borrowed from the Triton twin. The backward is the Triton twin's: its adjoint kernel
`epilogue_bwd` for ``dZ = dY * act'(Z)`` and ``db``, cuBLAS for ``dX`` and ``dW``.

Why (A800, bf16, 8192 x 8192 x 2048, `torch.profiler` device time, min of 3 interleaved
rounds): cuBLAS runs the bare GEMM at 1017 us (270 TFLOP/s, 87% of the datasheet peak); nvmath
with `GELU_BIAS` fused: 1018 us, with `GELU_AUX_BIAS` (aux Z stored): 1053 us; the Triton fused
kernel: 1273 us; the TileLang twin: ~1600 us; `torch.compile`: 1273 us; eager: 1438 us. A
hand-written GEMM in Triton or TileLang starts 20-50% behind cuBLAS's mainloop on this GPU
before its epilogue does anything; the cuBLASLt epilogue is free. So: **when the epilogue is
in cuBLASLt's set, use nvmath and write no GEMM**; write a Triton GEMM only for what cuBLASLt
cannot express, and a TileLang one never (experimental backend).

What cuBLASLt (12.8, sm80) fuses, in nvmath's `MatmulEpilog` names, all measured here:

* ``BIAS``: ``+ b`` with ``b`` of length **M in nvmath's orientation** (rows of ``a @ b``).
  A per-output-feature bias (length N, the `nn.Linear` case) therefore needs the transposed
  product ``Y^T = W X^T`` (``a = w`` (N, K), ``b = x.t()``): nvmath returns ``Y^T`` (N, M)
  **column-major**, so ``.t()`` is the contiguous row-major ``Y`` (M, N) at no cost.
* ``RELU``, ``GELU`` (the **tanh approximation**, `F.gelu(approximate="tanh")`, which is what
  transformer code calls GELU; the erf form ``F.gelu`` defaults to is not in the set and
  differs by up to 2e-3), each with ``_BIAS`` and ``_AUX`` variants. ``GELU_AUX[_BIAS]`` also
  returns ``gelu_aux`` = the pre-activation ``Z (+ b)`` in the storage dtype, laid out like the
  result (rows padded to a multiple of 8: slice ``[:N]`` before ``.t()``); ``RELU_AUX`` returns
  a bitmask instead (useless for our adjoint, so ReLU with a backward takes the ``BIAS``
  epilogue and a streaming ReLU pass here).
* ``beta * C``: a full ``(M, N)`` addend **inside** the activation, ``act(Z + b + C)``. That is
  not a transformer pattern (a residual is added by the next norm, `residual_then_norm`), so it
  is not exposed here.
* ``BGRADA`` / ``BGRADB``: the column sum of an operand fused into a GEMM (``db`` inside the
  ``dW`` GEMM, +3.5%); ``DGELU[_BGRAD]`` / ``DRELU[_BGRAD]``: the activation gradient applied to
  a GEMM *result* (fuses the next layer's ``dX`` GEMM with this layer's ``act'``; inside one op
  the gradient arrives as a tensor, so it is not used here). Both write the reduction in the
  storage dtype; the Triton adjoint's fp32 partials are kept for ``db``.

Not fused, and why: erf-GELU / SiLU / a SwiGLU gate (not in the set: ``BIAS`` epilogue then
the Triton pointwise pass; for SwiGLU run the two projections as one concatenated cuBLAS GEMM
and fuse ``silu(g) * u`` as a pointwise kernel), an fp32 ``db`` (Triton adjoint), quantised
epilogues (Triton).

Mechanics that matter:

* **Plan once, execute many.** The function form `nvmath.linalg.advanced.matmul()` plans on
  every call (1.6 ms wall here). Keep a `Matmul` object per (shape, strides, dtype, epilogue)
  in a dict, `plan()` once (`autotune()` is optional: 0.2 s, same winner as the heuristic on
  this shape), then `reset_operands(...)` + `execute()` per call. `execute()` costs **~130 us
  of host time** (Python-level validation), against 14 us for `torch.matmul`: below ~200 us of
  device time per GEMM the CPU cannot keep the GPU fed, and the Triton twin's
  ``mainloop="cublas"`` path (`torch.matmul` + one pointwise pass) is the better choice there.
  `select_backend` below encodes that boundary; a training step with CUDA graphs or a
  `torch.compile`d wrapper hides it.
* **Operands must match the plan's layout**: a row-strided ``x`` (``x.stride(0) > K``) is a
  different plan (the stride is part of the cache key; nvmath accepts the strided ``x.t()``).
  `reset_operands` re-checks shapes and strides; `reset_operands_unchecked` skips it.
* The `Matmul` object keeps references to its operands until the next `reset_operands`; the
  cache therefore pins one activation per plan. Acceptable for a training loop; call
  `release_operands()` if not.
* nvmath runs on the operands' current torch stream; `torch.compile` sees the launcher as an
  opaque custom op like any other package (`_common.compat.register_kernel`).
"""

import importlib.util
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from nvmath.linalg.advanced import Matmul, MatmulEpilog

# The Triton twin supplies the streaming pieces (pointwise epilogue pass, adjoint kernel).
_TRITON_TWIN = Path(__file__).resolve().parents[3] / "triton" / "epilogue" / "gemm" / "kernel.py"
_spec = importlib.util.spec_from_file_location("kda_ref_triton_gemm_epilogue", _TRITON_TWIN)
_tr = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _tr
_spec.loader.exec_module(_tr)
ACTS = _tr.ACTS
_MMA_DTYPES = (torch.bfloat16, torch.float16)

# Activations cuBLASLt applies inside the GEMM. `gelu` (erf) and `silu` are not among them.
_FUSED_ACT: Dict[str, Tuple[MatmulEpilog, MatmulEpilog]] = {  # act -> (epilog, epilog with bias)
    "none": (MatmulEpilog.DEFAULT, MatmulEpilog.BIAS),
    "relu": (MatmulEpilog.RELU, MatmulEpilog.RELU_BIAS),
    "gelu_tanh": (MatmulEpilog.GELU, MatmulEpilog.GELU_BIAS),
}
_FUSED_ACT_AUX: Dict[str, Tuple[MatmulEpilog, MatmulEpilog]] = {  # same, also storing Z (+ b) for the backward
    "gelu_tanh": (MatmulEpilog.GELU_AUX, MatmulEpilog.GELU_AUX_BIAS),
}

_PLANS: Dict[tuple, Matmul] = {}


def _plan(w: torch.Tensor, xt: torch.Tensor, epilog: MatmulEpilog, bias: Optional[torch.Tensor]) -> Matmul:
    """One planned `Matmul` per (shapes, strides, dtype, epilogue); operands are reset on every call."""
    key = (tuple(w.shape), tuple(xt.shape), xt.stride(), w.dtype, epilog, bias is not None, w.device.index)
    mm = _PLANS.get(key)
    inputs = {"bias": bias} if bias is not None else None
    if mm is None:
        mm = Matmul(w, xt)
        # `plan(epilog=MatmulEpilog.DEFAULT)` is rejected ("Not supported."): no epilogue is `None`.
        mm.plan(epilog=None if epilog is MatmulEpilog.DEFAULT else epilog, epilog_inputs=inputs)
        _PLANS[key] = mm
    else:
        mm.reset_operands(a=w, b=xt, epilog_inputs=inputs)
    return mm


def _product(w: torch.Tensor, xt: torch.Tensor, epilog: MatmulEpilog, bias: Optional[torch.Tensor]):
    """``(W X^T)^T`` (+ epilogue) as a contiguous row-major ``(M, N)`` tensor, plus nvmath's aux dict."""
    out = _plan(w, xt, epilog, bias).execute()
    yT, aux = out if isinstance(out, tuple) else (out, None)
    y = yT.t()
    if not y.is_contiguous():
        y = y.contiguous()  # not seen on any tested shape; nvmath returns Y^T column-major
    return y, aux


def select_backend(M: int, N: int, K: int) -> str:
    """``"nvmath"`` or ``"triton"`` (the twin's own `select_mainloop`) for one problem size.

    nvmath's `execute()` spends ~130 us on the host per call; a GEMM shorter than that on the
    device (about ``2 M N K < 6e10`` FLOP on an A800: ``4096 x 1024 x 1024`` runs in 90 us) is
    CPU-bound through nvmath and goes to the Triton twin (whose cuBLAS path is `torch.matmul`
    at 14 us of host time). Above it, the cuBLASLt epilogue wins by the whole epilogue cost.
    """
    return "nvmath" if 2 * M * N * K >= 6e10 else "triton"


def gemm_epilogue_fwd(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: Optional[torch.Tensor],
    act: str,
    save_aux: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``x`` (M, K) unit last stride, ``w`` (N, K) contiguous -> ``(y, z)``; ``y`` is contiguous.

    ``z`` is the pre-activation **including the bias** (``X W^T + b``, storage dtype) when
    ``save_aux`` and the activation needs it in the backward; otherwise a 0-element placeholder.
    (The Triton twin saves ``X W^T`` without the bias; its adjoint takes ``z_has_bias=True`` here.)
    """
    M, K = x.shape
    N = w.shape[0]
    assert x.dtype in _MMA_DTYPES and w.dtype == x.dtype, "tensor-core GEMM: bf16/fp16 operands"
    assert x.stride(1) == 1 and w.is_contiguous() and w.shape[1] == K
    if act not in ACTS:
        raise ValueError(f"unknown act {act!r}")
    want_aux = save_aux and act != "none"
    xt = x.t()  # (K, M) view, strides (1, stride_xm): nvmath reads the layout, no copy
    if act in _FUSED_ACT and not (want_aux and act not in _FUSED_ACT_AUX):
        # Whole epilogue in cuBLASLt.
        table = _FUSED_ACT_AUX if want_aux else _FUSED_ACT
        y, aux = _product(w, xt, table[act][bias is not None], bias)
        if want_aux:
            z = aux["gelu_aux"][:N].t()  # cuBLASLt pads aux rows to a multiple of 8
            if not z.is_contiguous():
                z = z.contiguous()  # only when N % 8 != 0; the adjoint reads z with row stride N
        else:
            z = x.new_empty(0)
        return y, z
    # Activation outside cuBLASLt's set (erf-GELU, SiLU) or ReLU with a backward (its aux is a
    # bitmask): cuBLASLt does GEMM + bias and writes Z; one Triton pass applies the activation.
    z, _ = _product(w, xt, MatmulEpilog.BIAS if bias is not None else MatmulEpilog.DEFAULT, bias)
    y = torch.empty_like(z)
    _tr._pointwise_epilogue_kernel[(_tr.triton.cdiv(M, 32), _tr.triton.cdiv(N, 128))](
        z, bias, y, M, N, BLOCK_M=32, BLOCK_N=128, ACT=ACTS[act], HAS_BIAS=False, num_warps=4,
    )
    return y, (z if want_aux else x.new_empty(0))


def gemm_epilogue_bwd(
    dy: torch.Tensor, x: torch.Tensor, w: torch.Tensor, bias: Optional[torch.Tensor], z: torch.Tensor, act: str,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Returns ``(dx, dw, db)``. ``z`` holds ``X W^T + b`` (or is empty under recompute)."""
    if z.numel() == 0 and act != "none":  # recompute: cuBLASLt rebuilds Z + b in one GEMM
        z, _ = _product(w, x.t(), MatmulEpilog.BIAS if bias is not None else MatmulEpilog.DEFAULT, bias)
    if act == "none" and bias is None:
        dz, db = dy, None
    else:
        dz, db = _tr.epilogue_bwd(dy, z, bias, act, z_has_bias=True)
    dx = torch.matmul(dz, w)  # cuBLAS; the two data GEMMs have no epilogue to fuse
    dw = torch.matmul(dz.t(), x)
    return dx, dw, (db.to(bias.dtype) if db is not None else None)


def _rows(t: torch.Tensor) -> torch.Tensor:
    return t if t.dim() == 2 else t.view(-1, t.shape[-1])


class _GemmEpilogue(torch.autograd.Function):
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

    ``act`` in ``"none" | "relu" | "gelu_tanh"`` runs entirely in cuBLASLt (``gelu_tanh`` also
    stores its aux there); ``"gelu"`` (erf) and ``"silu"`` add one streaming pass.
    """
    needs_grad = torch.is_grad_enabled() and any(t is not None and t.requires_grad for t in (x, w, bias))
    save_aux = needs_grad and not recompute and act != "none"
    return _GemmEpilogue.apply(x, w, bias, act, save_aux)


__all__ = ["ACTS", "gemm_epilogue", "gemm_epilogue_fwd", "gemm_epilogue_bwd", "select_backend"]
