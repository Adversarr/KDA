"""Triton backend of `{{op}}`."""

from ._register import {{op}}_triton, {{op}}_triton_fwd_only

__all__ = ["{{op}}_triton", "{{op}}_triton_fwd_only"]
