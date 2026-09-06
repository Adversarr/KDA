"""CUDA workloads and phase-specific SoL accounting for vggt_layerscale_residual_layernorm."""

import math
import torch
from kernel import layerscale_residual_layernorm as kernel_fn
from reference import layerscale_residual_layernorm as reference_fn

ROOF = "fp32"
PRIMARY = (1, 5496, 1024)


def _tensor(shape, dtype=torch.bfloat16, strided=False, grad=True):
    if strided:
        storage = torch.randn((*shape[:-1], shape[-1] * 2), device="cuda", dtype=dtype)
        result = storage[..., : shape[-1]].detach()
    else:
        result = torch.randn(shape, device="cuda", dtype=dtype)
    return result.requires_grad_(grad)


def make(shape, strided=False, special=None):
    """Create independent differentiable leaves and fixed positional/mask inputs."""
    torch.manual_seed(731)
    x = _tensor(shape, strided=strided)
    d = shape[-1]
    weight = lambda: (1.0 + 0.05 * torch.randn(d, device="cuda")).requires_grad_()
    bias = lambda: (0.05 * torch.randn(d, device="cuda")).requires_grad_()
    valid = torch.rand(shape[:-1], device="cuda") > 0.3
    if special == "all_invalid":
        valid.zero_()
    return (
        (
            _tensor(shape, torch.float32, strided),
            x,
            (0.01 * torch.ones(d, device="cuda")).requires_grad_(),
            weight(),
            bias(),
            1e-05,
            valid,
            torch.bfloat16,
        ),
        {},
    )


def _reductions(shape):
    rows = math.prod(shape[:-1])
    return [1, 1, rows, rows, rows]


def cases():
    """Primary training shape plus required small/tail and strided coverage."""
    small = (2, 17, 128)
    result = []
    for label, shape, strided, special in [
        ("user_train", PRIMARY, False, None),
        ("small_tail", small, False, None),
        ("strided", small, True, None),
    ]:
        result.append(
            {
                "name": label,
                "required": True,
                "make": lambda shape=shape, strided=strided, special=special: make(
                    shape, strided, special
                ),
                "grad_reductions": _reductions(shape),
            }
        )
    special = "all_invalid"
    result.append(
        {
            "name": special,
            "required": True,
            "make": lambda: make(small, special=special),
            "grad_reductions": _reductions(small),
        }
    )
    empty = (0,) + PRIMARY[1:]
    result.append(
        {
            "name": "zero_rows",
            "required": True,
            "benchmark": False,
            "make": lambda: make(empty),
            "grad_reductions": [max(1, r) for r in _reductions(empty)],
        }
    )
    frame_shape = (4, 1374, 1024)
    result.append(
        {
            "name": "frame_shape",
            "required": True,
            "make": lambda: make(frame_shape),
            "grad_reductions": _reductions(frame_shape),
        }
    )
    return result


def roof(phase, args, kwargs):
    """Compulsory bytes and arithmetic; parameter partials count writes and reads.

    Parameters/tables count once, inference excludes aux. Scalar FLOPs count each
    add/multiply separately; transcendental instructions are not assigned FLOPs.
    """
    x = args[0]
    (c, d) = (x.numel(), x.shape[-1])
    r = c // d
    forward = phase in ("fwd", "infer")
    sms = torch.cuda.get_device_properties(x.device).multi_processor_count
    p = max(1, min((r + 3) // 4, 2 * sms))
    aux = 8 * r if phase != "infer" else 0
    return (
        (12 * c + 12 * d + r + aux, 10 * c + 3 * r)
        if forward
        else (18 * c + 9 * r + 20 * d + 24 * p * d, 17 * c + 2 * r)
    )


__all__ = ["kernel_fn", "reference_fn", "ROOF", "cases", "roof"]
