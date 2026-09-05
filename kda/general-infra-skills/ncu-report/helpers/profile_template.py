#!/usr/bin/env python3
"""Template for a Python DSL profile driver.

This is the preferred profiling target for `ncu-report`.

The intended workflow is:

1. Import the kernel implementation module.
2. Compile or JIT the kernel explicitly.
3. Build representative inputs.
4. Print the kernel token and the recommended NCU selector.
5. Launch the steady-state kernel once for Nsight Compute.

Customize the TODO sections for your runtime.

This template assumes correctness has already been established elsewhere.
"""

from __future__ import annotations

import argparse
import re
from typing import Any

try:
    from kernel_name_regex import kernel_name_regex
except ImportError:
    def kernel_name_regex(token: str) -> str:
        token = token.strip()
        if not token:
            raise ValueError("kernel token must be non-empty")
        return f"regex:.*{re.escape(token)}.*"


KERNEL_TOKEN = "some_kernel"


def build_inputs() -> tuple[Any, ...]:
    """TODO: return the representative inputs for the target kernel."""
    raise NotImplementedError("Replace build_inputs() with workload construction")


def compile_kernel() -> Any:
    """TODO: import the implementation module and materialize the compiled kernel."""
    raise NotImplementedError("Replace compile_kernel() with your runtime-specific compile/JIT step")


def launch_kernel(compiled_kernel: Any, *inputs: Any) -> None:
    """TODO: launch the steady-state kernel exactly once."""
    raise NotImplementedError("Replace launch_kernel() with the actual kernel invocation")


def run() -> None:
    compiled_kernel = compile_kernel()
    inputs = build_inputs()

    regex = kernel_name_regex(KERNEL_TOKEN)
    print(f"[profile] kernel token: {KERNEL_TOKEN}")
    print(f"[profile] ncu selector: {regex}")

    launch_kernel(compiled_kernel, *inputs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args()
    run()


if __name__ == "__main__":
    main()
