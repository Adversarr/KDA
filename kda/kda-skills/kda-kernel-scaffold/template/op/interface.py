"""User-facing API for `{{op}}`: contract checks, backend/recompute resolution, dispatch."""

from typing import Optional, Tuple
from importlib.util import find_spec
from importlib.metadata import version, PackageNotFoundError

import torch

from .._common.compat import check_versions, needs_backward
from .._common.env import resolve_backend, resolve_recompute
from .backends import BACKEND_DEFAULT, BACKENDS, RECOMPUTE_AVAILABLE, RECOMPUTE_DEFAULT, CAPABILITY
from ._dispatch import dispatch
from ._helpers import check_cuda, check_dtype_in, check_last_dim_contiguous

check_versions()
_HAS_DEPENDENCY = find_spec("{{kernel_backend}}") is not None
if _HAS_DEPENDENCY and "{{kernel_backend}}" == "nvmath":
    try:
        _HAS_DEPENDENCY = int(version("nvmath-python").split(".")[0]) == 1
    except PackageNotFoundError:
        _HAS_DEPENDENCY = False


# Process-wide defaults for call sites that cannot pass ``backend=`` / ``recompute=`` (a free
# function that never sees the config). Set once where the model is built; ``KDA_BACKEND`` and
# ``KDA_RECOMPUTE`` still win.
_default_backend: Optional[str] = None
_default_recompute: Optional[bool] = None


def set_default_backend(backend: Optional[str]) -> None:
    """Default for calls without ``backend=``; ``None`` restores the package default."""
    global _default_backend
    if backend is not None and backend not in BACKENDS:
        raise ValueError(f"Unknown backend {backend!r}; allowed: {list(BACKENDS)}")
    _default_backend = backend


def set_default_recompute(recompute: Optional[bool]) -> None:
    """Default for calls without ``recompute=``; ``None`` restores SPEC's ``recompute.default``."""
    global _default_recompute
    _default_recompute = recompute


def _check_contract(x: torch.Tensor) -> None:
    """Generic rows every input must satisfy, then the op-specific ones. Raises before any launch."""
    check_cuda("x", x)
    check_dtype_in("x", x)
    check_last_dim_contiguous("x", x)
    # TODO(scaffold): op-specific rules with clear messages (D <= 8192, weight.shape[0] == D,
    # check_same_shape("x", x, "residual", residual), ...). Every input gets the three generic
    # checks above; the tensors of the eager reference are the model of what is allowed.


def {{op}}(
    x: torch.Tensor,
    *,
    backend: Optional[str] = None,
    recompute: Optional[bool] = None,
) -> Tuple[torch.Tensor, ...]:
    """TODO(scaffold): copy the math and the shape/dtype tables from SPEC.md into this docstring.

    Args:
        x: TODO.
        backend: one of ``BACKENDS``; ``None`` means ``KDA_BACKEND`` from the environment,
            else ``set_default_backend(...)``, else the package default (``auto``).
        recompute: recompute the saved-for-backward tensors in the backward instead of
            storing them (SPEC ``recompute``); ``None`` means ``KDA_RECOMPUTE`` from the
            environment, else ``set_default_recompute(...)``, else SPEC's default. Ignored
            by the eager backend and when the op has no recompute path.

    Returns:
        A tuple of tensors, see SPEC.md.
    """
    _check_math(x)
    chosen = resolve_backend(backend or _default_backend, BACKEND_DEFAULT, BACKENDS)
    rc = resolve_recompute(recompute if recompute is not None else _default_recompute, RECOMPUTE_DEFAULT)
    if chosen == "auto":
        chosen = "{{kernel_backend}}" if _supports_kernel(x) else "eager"
    return call_explicit(chosen, x, recompute=rc and RECOMPUTE_AVAILABLE)


# TODO(scaffold): only when the user's source op is an nn.Module, add a drop-in Module here
# that owns the parameters and calls the functional API above; otherwise delete this comment.

__all__ = ["{{op}}", "set_default_backend", "set_default_recompute"]


def call_explicit(backend: str, x: torch.Tensor, *, recompute: bool = False) -> Tuple[torch.Tensor, ...]:
    """Execute the named verification path without environment overrides or fallback.

    Mirror the public operation signature when adapting this template.
    """
    _check_math(x)
    if backend == "eager":
        from ._eager import {{op}}_eager
        return {{op}}_eager(x)
    if not _supports_kernel(x):
        raise ValueError("input or dependency is outside the accelerated contract")
    _check_contract(x)
    return dispatch(backend, x, recompute=recompute)


def _check_math(x: torch.Tensor) -> None:
    """Validate mathematical inputs independently of acceleration support."""
    if not isinstance(x, torch.Tensor) or not x.is_floating_point():
        raise TypeError("x must be a floating-point tensor")
    if CAPABILITY == "inference" and needs_backward(x):
        raise ValueError("inference capability does not support gradient-bearing calls")
    # TODO: Add the original operation's shape and scalar constraints here.


def _supports_kernel(x: torch.Tensor) -> bool:
    """Decide support before importing or executing the accelerated implementation."""
    dependency = "{{kernel_backend}}"
    if not x.is_cuda or x.dim() == 0 or x.stride(-1) != 1:
        return False
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if not _HAS_DEPENDENCY:
        return False
    if dependency == "nvmath" and torch.cuda.is_current_stream_capturing():
        return False
    # TODO: Include every operation-specific acceleration constraint here.
    return True
