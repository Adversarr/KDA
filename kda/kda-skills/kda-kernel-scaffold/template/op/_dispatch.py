"""Route a validated call to one backend. `interface.py` has already checked the contract.

``recompute`` is a keyword of the kernel backend's differentiable callable (popped by
``_common.compat`` before the launch); the eager reference has nothing to recompute and
drops it.
"""

from .backends import BACKEND_EAGER, BACKEND_KERNEL, BACKEND_KERNEL_FWD_ONLY
from ._eager import {{op}}_eager


def dispatch(backend: str, *args, recompute: bool = False, **kwargs):
    if backend == BACKEND_EAGER:
        return {{op}}_eager(*args, **kwargs)
    if backend == BACKEND_KERNEL:
        from ._{{kernel_backend}} import {{op}}_{{kernel_backend}}  # lazy: eager must work without the DSL installed

        return {{op}}_{{kernel_backend}}(*args, **kwargs, recompute=recompute)
    if backend == BACKEND_KERNEL_FWD_ONLY:
        from ._{{kernel_backend}} import {{op}}_{{kernel_backend}}_fwd_only

        return {{op}}_{{kernel_backend}}_fwd_only(*args, **kwargs, recompute=recompute)
    raise ValueError(f"{{op}}: unknown backend {backend!r}")


__all__ = ["dispatch"]
