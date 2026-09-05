"""User-facing API for `{{op}}`: contract checks, backend/recompute resolution, dispatch."""

from typing import Optional, Tuple

import torch

from .._common.compat import check_versions
from .._common.env import resolve_backend, resolve_recompute
from .backends import BACKEND_DEFAULT, BACKENDS, RECOMPUTE_AVAILABLE, RECOMPUTE_DEFAULT
from ._dispatch import dispatch
from ._helpers import check_cuda, check_dtype_in, check_last_dim_contiguous

check_versions()

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
            else ``set_default_backend(...)``, else the package default (``{{kernel_backend}}``).
        recompute: recompute the saved-for-backward tensors in the backward instead of
            storing them (SPEC ``recompute``); ``None`` means ``KDA_RECOMPUTE`` from the
            environment, else ``set_default_recompute(...)``, else SPEC's default. Ignored
            by the eager backend and when the op has no recompute path.

    Returns:
        A tuple of tensors, see SPEC.md.
    """
    _check_contract(x)
    chosen = resolve_backend(backend or _default_backend, BACKEND_DEFAULT, BACKENDS)
    rc = resolve_recompute(recompute if recompute is not None else _default_recompute, RECOMPUTE_DEFAULT)
    return dispatch(chosen, x, recompute=rc and RECOMPUTE_AVAILABLE)


# TODO(scaffold): only when the user's source op is an nn.Module, add a drop-in Module here
# that owns the parameters and calls the functional API above; otherwise delete this comment.

__all__ = ["{{op}}", "set_default_backend", "set_default_recompute"]
