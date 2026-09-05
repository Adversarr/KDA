#!/usr/bin/env python3
"""Helpers for building Nsight Compute kernel-name regex selectors.

The default DSL-first convention in this skill is:

    kernel token: some_kernel
    NCU selector: regex:.*some_kernel.*

Use the smallest stable token that identifies the generated CUDA kernel family.
"""

from __future__ import annotations

import re


def kernel_name_regex(token: str) -> str:
    """Return the canonical NCU regex selector for a kernel token."""
    token = token.strip()
    if not token:
        raise ValueError("kernel token must be non-empty")
    return f"regex:.*{re.escape(token)}.*"


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Build an NCU regex selector from a kernel token.")
    ap.add_argument("token", help="Stable kernel token, usually the Python/kernel symbol name.")
    args = ap.parse_args()
    print(kernel_name_regex(args.token))
