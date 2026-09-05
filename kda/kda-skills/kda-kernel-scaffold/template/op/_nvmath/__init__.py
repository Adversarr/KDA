"""nvmath-python (cuBLASLt) backend of `{{op}}`: a library GEMM with its epilogue fused, no GEMM kernel."""

from ._register import {{op}}_nvmath, {{op}}_nvmath_fwd_only

__all__ = ["{{op}}_nvmath", "{{op}}_nvmath_fwd_only"]
