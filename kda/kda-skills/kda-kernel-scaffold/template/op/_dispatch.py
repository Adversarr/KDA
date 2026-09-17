"""Explicit backend entrypoints without environment selection or fallback."""

from .backends import BACKEND_EAGER, BACKEND_KERNEL, BACKEND_KERNEL_FWD_ONLY
from ._eager import {{op}}_eager


def implementation(backend):
    """Resolve the callable actually used by dispatch."""
    if backend == BACKEND_EAGER:
        return {{op}}_eager
    if backend in (BACKEND_KERNEL, BACKEND_KERNEL_FWD_ONLY):
        from ._{{kernel_backend}} import {{op}}_{{kernel_backend}}, {{op}}_{{kernel_backend}}_fwd_only
        function = {{op}}_{{kernel_backend}}_fwd_only if backend == BACKEND_KERNEL_FWD_ONLY else {{op}}_{{kernel_backend}}
        if function is {{op}}_eager:
            raise ValueError("accelerated backend is bound to the eager oracle")
        return function
    raise ValueError(f"unknown backend {backend!r}")


def dispatch(backend: str, *args, recompute: bool = False, **kwargs):
    """Execute exactly the named implementation."""
    function = implementation(backend)
    if backend == BACKEND_EAGER:
        return function(*args, **kwargs)
    return function(*args, **kwargs, recompute=recompute)
