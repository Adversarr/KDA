"""Numerical comparison against the eager reference at fixed per-dtype tolerances.

The pass rule is elementwise ``|actual - expected| <= atol + rtol * |expected|`` with
non-finite ``actual`` values always counted as failures. ``SPEC.md`` may override the
defaults per op with a written justification.

Two refinements keep the fixed table honest for gradients:

* Which dtype's tolerance applies is decided by the *lowest precision in the computation*,
  not by the compared tensor's dtype (`lowest_precision`): an fp32 weight gradient of a bf16
  workload inherits bf16 rounding from the eager chain's bf16 intermediates, which a fused
  kernel keeping fp32 in registers legitimately does not reproduce.
* A tensor produced by reducing over ``R`` rows (weight and bias gradients) accumulates
  ``~sqrt(R)`` rounding errors of the term magnitude, independent of how small the result is,
  so ``compare(..., reduced_over=R)`` scales ``atol`` by ``sqrt(R)``. Elementwise outputs use
  ``R = 1``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

DTYPES: Dict[str, torch.dtype] = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}
DTYPE_NAMES: Dict[torch.dtype, str] = {v: k for k, v in DTYPES.items()}

# (atol, rtol) defaults per dtype.
TOLERANCES: Dict[torch.dtype, Tuple[float, float]] = {
    torch.float32: (1e-5, 1e-5),
    torch.float16: (1e-3, 1e-3),
    torch.bfloat16: (1.6e-2, 1.6e-2),
}
_MANTISSA_BITS: Dict[torch.dtype, int] = {torch.bfloat16: 8, torch.float16: 11, torch.float32: 24, torch.float64: 53}
# Elements per chunk of `compare`'s fp64 evaluation (2^25: about 1.3 GiB of transient planes).
COMPARE_CHUNK = 1 << 25


def lowest_precision(*dtypes: torch.dtype) -> torch.dtype:
    """The dtype with the fewest mantissa bits among ``dtypes`` (non-float dtypes are ignored)."""
    floats = [d for d in dtypes if d in _MANTISSA_BITS]
    return min(floats, key=_MANTISSA_BITS.__getitem__) if floats else torch.float32


@dataclass
class CompareResult:
    max_abs: float
    max_rel: float
    atol: float  # effective, after the sqrt(reduced_over) scaling
    rtol: float
    reduced_over: int
    n_bad: int
    n: int
    passed: bool

    def as_dict(self) -> dict:
        return asdict(self)


def compare(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: Optional[float] = None,
    rtol: Optional[float] = None,
    reduced_over: int = 1,
) -> CompareResult:
    """Compare ``actual`` to ``expected``; tolerances default to ``expected.dtype``'s entry.

    ``reduced_over`` is the number of rows summed to produce each element (see module docstring).
    """
    default_atol, default_rtol = TOLERANCES.get(expected.dtype, TOLERANCES[torch.float32])
    atol = (default_atol if atol is None else atol) * math.sqrt(max(1, reduced_over))
    rtol = default_rtol if rtol is None else rtol

    if actual.shape != expected.shape:
        return CompareResult(float("inf"), float("inf"), atol, rtol, reduced_over, expected.numel(), expected.numel(), False)

    # Chunked over the flattened tensors: the fp64 upcast of both sides plus the diff and mask
    # planes is ~40 bytes per element, which on a 2^31-element GEMM output (the `edge_huge`
    # workload) is 80 GiB on top of the tensors themselves. `reshape(-1)` copies a strided
    # view once at its own dtype; the chunk holds the fp64 planes to about 1.3 GiB.
    a_flat = actual.detach().reshape(-1)
    e_flat = expected.detach().reshape(-1)
    n = e_flat.numel()
    n_bad, max_abs, max_rel = 0, 0.0, 0.0
    for start in range(0, n, COMPARE_CHUNK):
        a = a_flat[start : start + COMPARE_CHUNK].double()
        e = e_flat[start : start + COMPARE_CHUNK].double()
        diff = (a - e).abs()
        bad = ~(diff <= atol + rtol * e.abs()) | ~torch.isfinite(a)
        n_bad += int(bad.sum())
        finite_diff = torch.where(torch.isfinite(diff), diff, torch.full_like(diff, float("inf")))
        max_abs = max(max_abs, float(finite_diff.max()))
        max_rel = max(max_rel, float((finite_diff / e.abs().clamp_min(atol)).max()))
    return CompareResult(max_abs, max_rel, atol, rtol, reduced_over, n_bad, n, n_bad == 0)


def grads(
    outputs: Sequence[torch.Tensor],
    inputs: Sequence[torch.Tensor],
    grad_outputs: Sequence[torch.Tensor],
) -> List[Optional[torch.Tensor]]:
    """Gradients of ``outputs`` w.r.t. each of ``inputs`` (``None`` where not required)."""
    wrt = [t for t in inputs if t.requires_grad]
    if not wrt:
        return [None] * len(inputs)
    got = iter(torch.autograd.grad(outputs, wrt, grad_outputs, allow_unused=True, retain_graph=True))
    return [next(got) if t.requires_grad else None for t in inputs]


__all__ = ["DTYPES", "DTYPE_NAMES", "TOLERANCES", "CompareResult", "compare", "grads", "lowest_precision"]
