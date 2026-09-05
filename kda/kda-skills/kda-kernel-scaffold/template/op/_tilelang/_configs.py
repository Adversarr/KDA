"""Frozen launch configurations for `{{op}}` (the M5 output; TileLang backend).

`select_config` replaces `tilelang.autotune` at deploy time: at most four branches keyed on the
workload dims and the GPU, plus a conservative default for GPUs that were never validated.
`TUNED.json` next to this file records where each branch came from and on which GPU.

``dims`` is op-specific (``{"n_rows": .., "d": ..}`` for token-wise ops, ``{"m", "n", "k"}``
for a GEMM); the launcher builds it, the branches read it.

Row width rule (compute-patterns.md): one program per row is at the copy roof for every
row count, including grids below the SM count, so ``n_rows`` alone never selects a second
geometry. A row wider than the register budget does: forward past ``ONE_ROW_MAX_D`` (32768 in
the rmsnorm reference), fused backward past 8192; the branch then selects the split-D path
(several programs per row, partial reductions). ``num_sms`` is for persistent grids (a GEMM
tile scheduler, the backward's ``2 * SMs`` programs), not for splitting rows.
"""

import warnings
from typing import Dict, Optional, Set

import torch

# Device-name substrings for which the branches below were measured (M5 adds entries).
_VALIDATED: Set[str] = set()
_DEFAULT: Dict[str, int] = {"BLOCK_D": 1024, "ROWS_PER_PROGRAM": 1, "SPLIT_D": 1, "num_warps": 4, "num_stages": 1}
_warned: Set[str] = set()
_sms: Dict[int, int] = {}

ONE_ROW_MAX_D = 32768  # widest row one program keeps in registers (A800, 16 warps); measured in the rmsnorm reference


def num_sms(device: Optional[torch.device] = None) -> int:
    """SM count of ``device`` (default: current), cached per device index."""
    index = torch.cuda.current_device() if device is None or device.index is None else device.index
    if index not in _sms:
        _sms[index] = torch.cuda.get_device_properties(index).multi_processor_count
    return _sms[index]


def select_config(dims: Dict[str, int], gpu: Optional[str] = None) -> Dict[str, int]:
    """Launch parameters for the problem described by ``dims`` on ``gpu`` (default: current)."""
    gpu = gpu or torch.cuda.get_device_name()
    if not any(key in gpu for key in _VALIDATED):
        if gpu not in _warned:
            _warned.add(gpu)
            warnings.warn(
                f"{{op}}: no tuned configuration for {gpu!r}; using the conservative default. "
                "Re-run the M4 tuning loop on this GPU.",
                stacklevel=2,
            )
        return dict(_DEFAULT)
    # TODO(implementer): <= 4 branches, e.g.
    # n_rows, d = dims["n_rows"], dims["d"]
    # if d > ONE_ROW_MAX_D:
    #     return {"BLOCK_D": 8192, "ROWS_PER_PROGRAM": 1, "SPLIT_D": 8, "num_warps": 8, "num_stages": 1}
    # if d <= 256 and n_rows >= 16384:
    #     return {"BLOCK_D": 256, "ROWS_PER_PROGRAM": 8, "SPLIT_D": 1, "num_warps": 1, "num_stages": 1}
    return dict(_DEFAULT)


__all__ = ["ONE_ROW_MAX_D", "num_sms", "select_config"]
