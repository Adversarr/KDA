"""Wire the Triton launchers into autograd through `_common.compat` (torch-version aware).

Two callables come out of this module:

* ``{{op}}_triton``: fused forward and backward kernels. Accepts ``recompute=`` (popped by
  compat): the forward then skips the aux stores and the backward recomputes them.
* ``{{op}}_triton_fwd_only``: Triton forward, backward through autograd on the eager
  reference. Used to bisect a divergence between the forward and backward kernels.
"""

from typing import Any, Sequence, Tuple

import torch

from ..._common.compat import make_differentiable, register_kernel, with_eager_backward
from .._eager import {{op}}_eager
from ._impl_bwd import {{op}}_bwd, {{op}}_bwd_fake
from ._impl_fwd import {{op}}_fwd, {{op}}_fwd_fake

# Number of user-visible outputs (the SPEC `outputs` list), not counting aux: the forward
# launcher returns them first and the aux tensors after them; make_differentiable splits the
# tuple there. `1` is the template placeholder; set it at M2 (an op returning (h, y) has 2).
N_OUTPUTS = 1

_fwd = register_kernel("kda::{{op}}_fwd", {{op}}_fwd, {{op}}_fwd_fake)
_bwd = register_kernel("kda::{{op}}_bwd", {{op}}_bwd, {{op}}_bwd_fake)


def _setup_context(ctx: Any, inputs: Sequence[Any], output: Tuple[torch.Tensor, ...]) -> None:
    # Only reached when a backward can follow. ``ctx.recompute`` (set by compat) says which
    # path: False -> the launcher ran with save_aux=True and the aux tensors are real;
    # True -> aux is the 0-element placeholder and the backward recomputes it from the inputs.
    # TODO(implementer): save SPEC auxiliaries plus the separately documented backward inputs/outputs, e.g.
    # (x, _save_aux) = inputs
    # (_y, aux) = output
    # ctx.save_for_backward(x, aux)   # aux is empty under recompute; x is needed on both paths
    raise NotImplementedError("{{op}}: _setup_context")


def _backward(ctx: Any, *grad_outputs: torch.Tensor) -> Tuple[Any, ...]:
    # TODO(implementer): one gradient (or None) per forward input, save_aux included (None);
    # aux gradients are ignored.
    # x, aux = ctx.saved_tensors
    # (dy, _d_aux) = grad_outputs
    # (dx,) = _bwd(dy, x, aux, ctx.recompute)
    # return (dx, None)
    raise NotImplementedError("{{op}}: _backward")


# save_aux_arg: compat sets the launcher's `save_aux` per call (grad mode, requires_grad and
# not recompute), so aux tensors are written only when a backward will read them.
{{op}}_triton = make_differentiable(
    _fwd,
    n_outputs=N_OUTPUTS,
    setup_context=_setup_context,
    backward=_backward,
    save_aux_arg="save_aux",
    recompute_arg="recompute",
)
{{op}}_triton_fwd_only = with_eager_backward(
    _fwd, {{op}}_eager, n_outputs=N_OUTPUTS, save_aux_arg="save_aux", recompute_arg="recompute"
)

__all__ = ["{{op}}_triton", "{{op}}_triton_fwd_only"]
