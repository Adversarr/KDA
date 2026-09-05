"""`{{op}}`: TODO one-line description (see SPEC.md).

Usage::

    from kda_kernels.{{op}} import {{op}}
    (y,) = {{op}}(x)                      # backend from KDA_BACKEND, else the default
    (y,) = {{op}}(x, backend="eager")     # force the reference implementation
"""

from .interface import {{op}}

__all__ = ["{{op}}"]
