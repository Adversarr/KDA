"""CUDA workloads and phase-specific SoL accounting for fused_residual_layernorm."""

import math
import torch
from kernel import add_layernorm as kernel_fn
from reference import add_layernorm as reference_fn

ROOF = "fp32"
PRIMARY = (4, 2048, 2048)


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
    return ((x, _tensor(shape, torch.float32, strided), weight(), bias(), 1e-05), {})


def _reductions(shape):
    rows = math.prod(shape[:-1])
    return [1, 1, rows, rows]


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
    p = 0 if r <= 128 and d <= 1024 else max(1, min((r + 3) // 4, 2 * sms))
    aux = 8 * r if phase != "infer" else 0
    return (
        (12 * c + 8 * d + aux, 9 * c + 3 * r)
        if forward
        else (16 * c + 8 * r + 12 * d + 16 * p * d, 14 * c + 2 * r)
    )


__all__ = ["kernel_fn", "reference_fn", "ROOF", "cases", "roof"]
