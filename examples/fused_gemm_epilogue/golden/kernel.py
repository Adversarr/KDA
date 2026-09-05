"""Golden for ``fc1_gelu``: ``gelu_tanh(x @ weight.T + bias)`` with the epilogue fused into the GEMM.

The rule is two lines long:

    if nvmath-python is importable -> the cuBLASLt epilogue (``GELU_AUX_BIAS``) through
                                      ``reference/nvmath/epilogue/gemm/``: cuBLAS speed, no kernel
    else                            -> ``reference/triton/epilogue/gemm/`` (``select_mainloop`` sends
                                      this 8192-wide shape to cuBLAS + one epilogue kernel)

Both twins share the host signature ``gemm_epilogue(x, w, bias, act, recompute=)`` and the same
backward (the Triton adjoint ``dZ = dY * gelu'(Z)`` with fp32 ``db`` partials, then cuBLAS for
``dX`` and ``dW``). The user's parameters arrive in fp32 under autocast; the eager code casts
them to ``x.dtype`` by hand and so does this wrapper.
"""

import importlib.util
import sys
from pathlib import Path
from typing import Optional

import torch

_REF = Path(__file__).resolve().parents[3] / "kda" / "kda-skills" / "kda-kernel-implement" / "reference"


_LOADED = {}


def _load(backend: str):
    if backend not in _LOADED:
        path = _REF / backend / "epilogue" / "gemm" / "kernel.py"
        spec = importlib.util.spec_from_file_location(f"kda_ref_{backend}_gemm_epilogue", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        _LOADED[backend] = mod
    return _LOADED[backend]


try:
    import nvmath  # noqa: F401

    BACKEND = "nvmath"
except ImportError:
    BACKEND = "triton"

_impl = _load(BACKEND)
gemm_epilogue = _impl.gemm_epilogue
gemm_epilogue_fwd = _impl.gemm_epilogue_fwd


def fc1_gelu(
    x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor], *, recompute: bool = False, backend: Optional[str] = None
) -> torch.Tensor:
    """Drop-in for ``minilm.model.fc1_gelu``; ``backend`` forces ``"nvmath"`` or ``"triton"`` (tests, bench)."""
    impl = _impl if backend in (None, BACKEND) else _load(backend)
    return impl.gemm_epilogue(x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype), "gelu_tanh", recompute=recompute)


__all__ = ["BACKEND", "fc1_gelu", "gemm_epilogue", "gemm_epilogue_fwd"]
