"""Torch-version-aware registration of GPU kernels as differentiable ops.

Two paths, selected once at import time from ``torch.__version__``:

* torch >= 2.6: ``torch.library.custom_op`` + ``register_fake`` + ``register_autograd``.
  ``torch.compile`` treats the op as opaque and never traces into the kernel launch.
* 2.1 <= torch < 2.6: plain ``torch.autograd.Function``. Works everywhere, but
  ``torch.compile`` graph-breaks at the op.

Both paths expose the same two helpers, so kernel packages contain no version checks.

Saved-for-backward tensors are optional work. A forward launcher that takes ``save_aux: bool``
writes its aux tensors (``rstd`` and the like) only when asked; ``make_differentiable`` asks
exactly when a backward can follow (grad mode on and some tensor input requires grad), so
inference and ``torch.no_grad()`` evaluation never pay for them. With ``save_aux=False`` the
launcher returns a 0-element placeholder for each aux output so the op schema stays fixed.

Recompute is the second training path: when the caller passes ``recompute=True`` (see
``env.resolve_recompute``) the forward runs with ``save_aux=False`` even though a backward
follows, and the backward kernel recomputes the aux from the saved inputs instead
(``RECOMPUTE: tl.constexpr``). ``ctx.recompute`` tells ``setup_context`` and ``backward``
which path they are on.
"""

from __future__ import annotations

import inspect
import warnings
from typing import Any, Callable, Optional, Sequence, Tuple

import torch

MIN_TORCH = (2, 1)
CUSTOM_OP_TORCH = (2, 6)
MIN_TRITON = (3, 0)


def _parse_version(version: str) -> Tuple[int, int]:
    major, minor = version.split("+")[0].split(".")[:2]
    return int(major), int(minor)


TORCH_VERSION = _parse_version(torch.__version__)
USE_CUSTOM_OP = TORCH_VERSION >= CUSTOM_OP_TORCH


def check_versions(*, min_torch: Tuple[int, int] = MIN_TORCH, min_triton: Tuple[int, int] = MIN_TRITON) -> None:
    """Warn (never raise) when the toolchain is older than what the package was validated with."""
    if TORCH_VERSION < min_torch:
        warnings.warn(
            f"torch {torch.__version__} is older than the validated minimum {min_torch}; "
            "kernels may not load.",
            stacklevel=2,
        )
    try:
        import triton
    except ImportError:
        warnings.warn("triton is not installed; only the eager backend is available.", stacklevel=2)
        return
    if _parse_version(triton.__version__) < min_triton:
        warnings.warn(
            f"triton {triton.__version__} is older than the validated minimum {min_triton}.",
            stacklevel=2,
        )


def register_kernel(name: str, impl: Callable[..., Any], fake: Callable[..., Any]) -> Callable[..., Any]:
    """Expose a host-side kernel launcher as an opaque op.

    ``impl(*args)`` must be fully type-annotated (the op schema is inferred from the
    annotations) and must return fresh tensors, either one ``Tensor`` or a tuple of them.
    ``fake`` has the same signature and returns empty tensors carrying the output metadata;
    it is what ``torch.compile`` runs at trace time.

    On torch < 2.6 ``impl`` is returned unchanged and ``fake`` is unused.
    """
    if not USE_CUSTOM_OP:
        return impl
    op = torch.library.custom_op(name, impl, mutates_args=())
    op.register_fake(fake)
    return op


def needs_backward(*values: Any) -> bool:
    """True when autograd will record this call: grad mode on and a tensor input requires grad."""
    return torch.is_grad_enabled() and any(isinstance(v, torch.Tensor) and v.requires_grad for v in values)


def make_differentiable(
    fwd: Callable[..., Tuple[torch.Tensor, ...]],
    *,
    n_outputs: int,
    setup_context: Callable[[Any, Sequence[Any], Tuple[torch.Tensor, ...]], None],
    backward: Callable[..., Tuple[Any, ...]],
    save_aux_arg: Optional[str] = None,
    recompute_arg: Optional[str] = None,
) -> Callable[..., Tuple[torch.Tensor, ...]]:
    """Attach autograd to a registered forward op and return the user-callable.

    ``fwd(*inputs)`` returns a tuple: ``n_outputs`` user-visible tensors followed by
    auxiliary tensors (e.g. ``rstd``) kept only for the backward pass; the callable
    returned here yields only the first ``n_outputs``.

    ``save_aux_arg`` names the launcher's ``bool`` parameter that enables the aux stores
    (``"save_aux"`` in the template); it is set per call from ``needs_backward``. ``None``
    for launchers without aux tensors.

    ``recompute_arg`` names a keyword the *returned callable* accepts (``"recompute"`` in the
    template). It is not a launcher parameter: it is popped before the launch, forces
    ``save_aux=False`` when true, and is stored as ``ctx.recompute`` so ``setup_context``
    saves the inputs the backward recomputes from and ``backward`` picks the ``RECOMPUTE``
    kernel variant. ``None`` for ops without a recompute path (the keyword is then rejected).

    ``setup_context(ctx, inputs, output)`` saves whatever ``backward`` needs; ``inputs`` is
    every launcher argument in signature order, keyword-only ones included, on both paths.
    ``backward(ctx, *grad_outputs)`` receives one gradient per forward output (aux
    included; ignore those) and returns one gradient (or ``None``) per forward input, in the
    same order (``None`` for non-tensors).
    """

    def with_aux_flag(args: Tuple[Any, ...], kwargs: dict) -> Tuple[dict, bool]:
        """Return ``(launcher kwargs, record)``: aux is written only when a backward follows
        and the caller did not ask to recompute it."""
        kwargs = dict(kwargs)
        recompute = bool(kwargs.pop(recompute_arg, False)) if recompute_arg is not None else False
        record = needs_backward(*args, *kwargs.values())
        if save_aux_arg is not None:
            kwargs[save_aux_arg] = record and not recompute
        return kwargs, record

    def _mark(ctx: Any, inputs: Sequence[Any], arg_names: Sequence[str]) -> None:
        # setup_context only runs when a backward follows, so "aux was not written" means
        # "the caller asked to recompute". Reading it off the launcher inputs (not a Python
        # side channel) keeps the flag visible to torch.compile, which specialises on it.
        if save_aux_arg is not None and save_aux_arg in arg_names:
            ctx.recompute = not bool(inputs[list(arg_names).index(save_aux_arg)])
        else:
            ctx.recompute = False

    if USE_CUSTOM_OP:
        # torch hands keyword-only launcher arguments to setup_context separately
        # (``keyword_only_inputs``) and wants gradients for the positional ones only; hide
        # that so package code follows one convention.
        schema_args = getattr(getattr(fwd, "_opoverload", None), "_schema", None)
        schema_args = list(schema_args.arguments) if schema_args is not None else []
        kw_names = [a.name for a in schema_args if a.kwarg_only]
        arg_names = [a.name for a in schema_args if not a.kwarg_only] + kw_names
        n_positional = len(schema_args) - len(kw_names) if schema_args else None

        def _setup(ctx: Any, inputs: Sequence[Any], output: Any, keyword_only_inputs: Any = None) -> None:
            kw = keyword_only_inputs or {}
            full = (*inputs, *[kw[n] for n in kw_names if n in kw])
            _mark(ctx, full, arg_names)
            setup_context(ctx, full, output)

        def _backward(ctx: Any, *grad_outputs: Any) -> Tuple[Any, ...]:
            grads = tuple(backward(ctx, *grad_outputs))
            return grads if n_positional is None else grads[:n_positional]

        fwd.register_autograd(_backward, setup_context=_setup)

        def call(*args: Any, **kwargs: Any) -> Tuple[torch.Tensor, ...]:
            kwargs, _ = with_aux_flag(args, kwargs)
            return tuple(fwd(*args, **kwargs))[:n_outputs]

        return call

    # autograd.Function.apply is positional-only: flatten kwargs in signature order.
    names, flatten = _binder(fwd)

    class _Fn(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, record: bool, *inputs: Any) -> Tuple[torch.Tensor, ...]:
            output = tuple(fwd(**dict(zip(names, inputs))))
            # Grad mode is off inside forward, so `record` (decided by the caller) says whether a
            # backward can follow; like the custom-op path, setup_context runs only then.
            if record:
                _mark(ctx, inputs, names)
                setup_context(ctx, inputs, output)
            return output

        @staticmethod
        def backward(ctx: Any, *grad_outputs: Any) -> Tuple[Any, ...]:
            return (None, *backward(ctx, *grad_outputs))

    def call(*args: Any, **kwargs: Any) -> Tuple[torch.Tensor, ...]:
        kwargs, record = with_aux_flag(args, kwargs)
        return tuple(_Fn.apply(record, *flatten(*args, **kwargs)))[:n_outputs]

    return call


def _binder(fn: Callable[..., Any]) -> Tuple[Tuple[str, ...], Callable[..., Tuple[Any, ...]]]:
    """Parameter names of ``fn`` and a function flattening ``(*args, **kwargs)`` into that order."""
    sig = inspect.signature(fn)
    names = tuple(sig.parameters)

    def flatten(*args: Any, **kwargs: Any) -> Tuple[Any, ...]:
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        return tuple(bound.arguments[n] for n in names)

    return names, flatten


def with_eager_backward(
    fwd: Callable[..., Tuple[torch.Tensor, ...]],
    eager: Callable[..., Tuple[torch.Tensor, ...]],
    *,
    n_outputs: int,
    save_aux_arg: Optional[str] = None,
    recompute_arg: Optional[str] = None,
) -> Callable[..., Tuple[torch.Tensor, ...]]:
    """Kernel forward with the backward taken from autograd through ``eager``.

    This is the ``*_fwd_only`` debugging backend: if it matches the reference but the
    fully fused backend does not, the bug is in the backward kernel. ``fwd`` and ``eager``
    take identical arguments (plus ``save_aux_arg`` on ``fwd``, passed as ``False`` here
    since the eager backward needs none of the kernel's aux tensors); only the first
    ``n_outputs`` of ``fwd`` are returned. A ``recompute_arg`` keyword is accepted and
    ignored (the eager backward recomputes everything anyway).
    """
    # Bind through ``eager``'s signature: ``fwd`` may be an opaque custom-op object.
    names, flatten = _binder(eager)
    extra = {} if save_aux_arg is None else {save_aux_arg: False}

    class _Fn(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, *inputs: Any) -> Tuple[torch.Tensor, ...]:
            ctx.is_tensor = [isinstance(a, torch.Tensor) for a in inputs]
            ctx.non_tensors = [None if t else a for a, t in zip(inputs, ctx.is_tensor)]
            ctx.save_for_backward(*[a for a in inputs if isinstance(a, torch.Tensor)])
            with torch.no_grad():
                return tuple(fwd(**dict(zip(names, inputs)), **extra))[:n_outputs]

        @staticmethod
        def backward(ctx: Any, *grad_outputs: Any) -> Tuple[Any, ...]:
            saved = iter(ctx.saved_tensors)
            inputs = [next(saved) if t else a for a, t in zip(ctx.non_tensors, ctx.is_tensor)]
            with torch.enable_grad():
                leaves = [
                    a.detach().requires_grad_(need) if t else a
                    for a, t, need in zip(inputs, ctx.is_tensor, ctx.needs_input_grad)
                ]
                outputs = tuple(eager(**dict(zip(names, leaves))))[:n_outputs]
                wrt = [a for a, t, need in zip(leaves, ctx.is_tensor, ctx.needs_input_grad) if t and need]
                grads = iter(torch.autograd.grad(outputs, wrt, grad_outputs, allow_unused=True)) if wrt else iter(())
            return tuple(next(grads) if (t and need) else None for t, need in zip(ctx.is_tensor, ctx.needs_input_grad))

    def call(*args: Any, **kwargs: Any) -> Tuple[torch.Tensor, ...]:
        if recompute_arg is not None:
            kwargs = dict(kwargs)
            kwargs.pop(recompute_arg, None)
        return tuple(_Fn.apply(*flatten(*args, **kwargs)))

    return call


__all__ = [
    "TORCH_VERSION",
    "USE_CUSTOM_OP",
    "check_versions",
    "register_kernel",
    "needs_backward",
    "make_differentiable",
    "with_eager_backward",
]
