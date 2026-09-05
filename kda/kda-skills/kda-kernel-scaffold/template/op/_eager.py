"""Golden reference for `{{op}}`.

A verbatim extraction of the user's PyTorch code (see SPEC.md `source`), wrapped in the
functional signature of `interface.py`. Autograd through this function is the reference
backward. Do not "improve" it: every numerical difference from the user's code is a bug.
"""

from typing import Tuple

import torch


def {{op}}_eager(x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    # TODO(scaffold): paste the user's implementation here, keeping their dtype casts.
    raise NotImplementedError("{{op}}_eager: fill in the user's reference implementation")


__all__ = ["{{op}}_eager"]
