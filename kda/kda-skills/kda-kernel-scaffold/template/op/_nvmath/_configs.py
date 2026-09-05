"""Frozen dispatch for `{{op}}` on the nvmath backend (the M5 output).

There are no tile sizes to freeze: cuBLASLt picks the kernel. What `select_config` decides is
the **path** per problem size and GPU, and `TUNED.json` records where each branch came from:

* ``"nvmath"``: the planned `Matmul` with the fused epilog. `execute()` costs ~130-175 us of
  host time per call (A800, nvmath-python 1.0), so a GEMM shorter than that on the device is
  CPU-bound in eager mode.
* ``"matmul"``: `torch.matmul` (18 us of host time) plus one Triton pointwise epilogue pass,
  for the short GEMMs; the SPEC lists which workloads land here.

``dims`` is ``{"m", "n", "k"}``; the launcher builds it, the branches read it. The boundary
measured in `reference/nvmath/epilogue/gemm/` is ``2 m n k >= 6e10`` FLOP (~200 us on A800);
under CUDA graphs or a `torch.compile`d step the host cost is hidden and everything can go to
``"nvmath"``.
"""

import warnings
from typing import Dict, Optional, Set

import torch

# Device-name substrings for which the branch below was measured (M5 adds entries).
_VALIDATED: Set[str] = set()
_DEFAULT: Dict[str, object] = {"path": "nvmath", "autotune": False}
_warned: Set[str] = set()

NVMATH_MIN_FLOP = 6e10  # below this a GEMM is faster through torch.matmul + a pointwise pass in eager mode (A800)


def select_config(dims: Dict[str, int], gpu: Optional[str] = None) -> Dict[str, object]:
    """Path and plan options for the problem described by ``dims`` on ``gpu`` (default: current)."""
    gpu = gpu or torch.cuda.get_device_name()
    if not any(key in gpu for key in _VALIDATED):
        if gpu not in _warned:
            _warned.add(gpu)
            warnings.warn(
                f"{{op}}: no tuned dispatch for {gpu!r}; using the conservative default (nvmath everywhere). "
                "Re-run the M4 tuning loop on this GPU.",
                stacklevel=2,
            )
        return dict(_DEFAULT)
    # TODO(implementer): <= 4 branches, e.g.
    # if 2 * dims["m"] * dims["n"] * dims["k"] < NVMATH_MIN_FLOP:
    #     return {"path": "matmul", "autotune": False}
    return dict(_DEFAULT)


__all__ = ["NVMATH_MIN_FLOP", "select_config"]
