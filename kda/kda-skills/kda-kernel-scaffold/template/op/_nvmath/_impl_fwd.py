"""Forward launcher for `{{op}}`: the GEMM and its epilogue in one cuBLASLt kernel via nvmath-python.

No GEMM kernel is written here. `nvmath.linalg.advanced.Matmul` plans one cuBLASLt kernel per
(shapes, strides, dtype, epilog) and `execute()` runs it; the epilogue (bias, ReLU, tanh-GELU,
the aux store of the pre-activation) is applied inside that kernel. Read
`kda-kernel-implement/reference/nvmath/epilogue/gemm/` first; the rules that matter:

* **Orientation.** cuBLASLt's ``BIAS`` runs along the rows of ``a @ b``. For the per-output-
  feature bias of ``nn.Linear`` compute ``Y^T = W X^T`` (``a = w`` (N, K), ``b = x.t()``); the
  result is ``Y^T`` column-major, so ``.t()`` is the row-major ``Y`` for free. The aux
  (``gelu_aux``) comes back the same way with rows padded to a multiple of 8: slice ``[:N]``.
* **Plan once, execute many.** Keep the planned `Matmul` per key in `_PLANS`; `reset_operands`
  + `execute` per call (the function form `nvmath.linalg.advanced.matmul` replans every call).
  `plan(epilog=None)` for no epilogue: ``MatmulEpilog.DEFAULT`` is rejected.
* **GELU is the tanh form** (`F.gelu(approximate="tanh")`); the erf form is not fusable and
  differs by up to 2e-3. The SPEC names which form the user's code calls.
* **Strides are part of the plan key**; a row-strided ``x`` (a slice of a wider activation) is
  accepted as is through ``x.t()``, never copied. The lint rules on hidden copies still apply.
* ``save_aux`` (set per call by ``_common.compat``) selects the ``*_AUX*`` epilog; with
  ``save_aux=False`` the launcher returns a 0-element placeholder per aux output.
* Host cost: `execute()` is ~130-175 us of Python per call. Below ~200 us of device time the
  op is CPU-bound in eager mode; `torch.matmul` + a Triton pointwise epilogue is the better
  choice there (`reference/nvmath/epilogue/gemm/kernel.py::select_backend`), and the SPEC
  says which shapes go which way.
"""

from typing import Dict, Optional, Tuple

import torch
from nvmath.linalg.advanced import Matmul, MatmulEpilog

_PLANS: Dict[tuple, Matmul] = {}


def _plan(a: torch.Tensor, b: torch.Tensor, epilog: Optional[MatmulEpilog], bias: Optional[torch.Tensor]) -> Matmul:
    """One planned `Matmul` per (shapes, strides, dtype, epilog, has_bias, device); operands reset per call."""
    key = (tuple(a.shape), tuple(b.shape), a.stride(), b.stride(), a.dtype, epilog, bias is not None, a.device.index)
    inputs = {"bias": bias} if bias is not None else None
    mm = _PLANS.get(key)
    if mm is None:
        mm = Matmul(a, b)
        mm.plan(epilog=None if epilog in (None, MatmulEpilog.DEFAULT) else epilog, epilog_inputs=inputs)
        _PLANS[key] = mm
    else:
        mm.reset_operands(a=a, b=b, epilog_inputs=inputs)
    return mm


def {{op}}_fwd(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor, save_aux: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    """Launch the forward. Returns ``(y, aux)``; see SPEC for what aux is.

    ``aux`` is written by the ``*_AUX*`` epilog only when ``save_aux``; otherwise it is
    ``torch.empty(0, ...)`` so the op schema (two outputs) is unchanged.
    """
    # TODO(implementer): replace this placeholder. Sketch for gelu_tanh(x @ w.T + bias):
    # M, K = x.shape; N = w.shape[0]
    # xt = x.t()                                   # (K, M) view, strides (1, stride_xm): no copy
    # epilog = MatmulEpilog.GELU_AUX_BIAS if save_aux else MatmulEpilog.GELU_BIAS
    # out = _plan(w, xt, epilog, bias).execute()
    # yT, auxd = out if save_aux else (out, None)
    # y = yT.t()                                   # row-major (M, N) for free
    # aux = auxd["gelu_aux"][:N].t().contiguous() if save_aux else x.new_empty(0)  # `Z + b`; contiguous only copies when N % 8 != 0
    # return y, aux
    raise NotImplementedError("{{op}}_fwd: implement the nvmath forward launcher")


def {{op}}_fwd_fake(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor, save_aux: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    """Output metadata only (what torch.compile traces); no computation."""
    # TODO(implementer): mirror the shapes/dtypes returned by {{op}}_fwd, including the
    # 0-element aux when not save_aux.
    raise NotImplementedError("{{op}}_fwd_fake: describe the forward outputs")


__all__ = ["{{op}}_fwd", "{{op}}_fwd_fake"]
