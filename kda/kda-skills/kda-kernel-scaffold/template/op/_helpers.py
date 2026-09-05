"""Helpers shared by every backend of `{{op}}`: contract checks and shape helpers.

The contract checks raise ``ValueError``/``TypeError`` *before any launch*; `interface.py`
calls the generic ones on every input and `_run_dev.py --contract` probes them (a CPU tensor,
a non-unit last stride, a shape mismatch, an integer dtype must all be rejected here, never
by a CUDA-side assertion).
"""

from typing import Callable, Sequence, Tuple

import torch

FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def check_cuda(name: str, x: torch.Tensor) -> None:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{{op}}: `{name}` must be a torch.Tensor, got {type(x).__name__}")
    if not x.is_cuda:
        raise ValueError(f"{{op}}: `{name}` must be a CUDA tensor, got device {x.device}")


def check_dtype_in(name: str, x: torch.Tensor, allowed: Sequence[torch.dtype] = FLOAT_DTYPES) -> None:
    if x.dtype not in allowed:
        raise TypeError(f"{{op}}: `{name}` dtype must be one of {[str(d) for d in allowed]}, got {x.dtype}")


def check_last_dim_contiguous(name: str, x: torch.Tensor) -> None:
    """Check last-stride contiguity; the operation also validates its leading-layout contract."""
    if x.dim() == 0 or (x.shape[-1] > 1 and x.stride(-1) != 1):
        raise ValueError(f"{{op}}: `{name}` must be contiguous along its last dimension (stride {x.stride()})")


def check_same_shape(name_a: str, a: torch.Tensor, name_b: str, b: torch.Tensor) -> None:
    if a.shape != b.shape:
        raise ValueError(f"{{op}}: `{name_a}` {tuple(a.shape)} and `{name_b}` {tuple(b.shape)} must have the same shape")


def check_last_dim(name: str, x: torch.Tensor, d: int, what: str) -> None:
    if x.shape[-1] != d:
        raise ValueError(f"{{op}}: `{name}` last dim {x.shape[-1]} must equal {what} ({d})")


def rows_and_stride(x: torch.Tensor) -> Tuple[int, int]:
    """``(n_rows, row_stride)`` of a token-wise input ``(..., D)`` without copying it.

    Leading dims must be collapsible to one row index (contiguous among themselves or a
    single leading dim); a padded row (``stride(-2) > D``) is fine and is what the kernel
    receives as its row stride. Never ``.reshape``/``.contiguous()`` here: both copy.
    """
    if x.dim() == 1:
        return 1, x.shape[-1]
    lead = x.shape[:-1]
    n_rows = 1
    for s in lead:
        n_rows *= s
    if x.dim() > 2:
        # Leading dims collapse only when they are laid out as one dense block of rows.
        for i in range(x.dim() - 2):
            if x.shape[i] > 1 and x.stride(i) != x.shape[i + 1] * x.stride(i + 1):
                raise ValueError(f"{{op}}: leading dims of a {tuple(x.shape)} tensor with strides {x.stride()} do not form a row block")
    return n_rows, x.stride(-2)


def as_rows(x: torch.Tensor) -> Tuple[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]:
    """View ``x`` as ``(rows, D)`` for row-wise kernels; returns the view and an un-flattener.

    Uses ``view`` (never ``reshape``) so an input the kernel can address directly is not
    copied; a layout that cannot be viewed raises instead of silently copying.
    """
    shape = x.shape
    rows = x.view(-1, shape[-1])
    return rows, (lambda y: y.view(shape))


__all__ = [
    "FLOAT_DTYPES",
    "check_cuda",
    "check_dtype_in",
    "check_last_dim_contiguous",
    "check_same_shape",
    "check_last_dim",
    "rows_and_stride",
    "as_rows",
]
