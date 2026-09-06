"""CUDA workloads and phase-specific SoL accounting for h3_qk_norm_mmrope."""

import math
import torch
from kernel import qk_prep as kernel_fn
from reference import qk_prep as reference_fn

ROOF = "fp32"
PRIMARY = (1, 2048, 56, 128)


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
    token = torch.arange(shape[1], device="cuda")
    positions = torch.stack((token // 512, token // 32 % 16, token % 32), -1).float()
    inv = 1.0 / 10000 ** (torch.arange(16, device="cuda", dtype=torch.float32) / 16)
    angle = (positions[:, :, None] * inv).flatten(1)
    (cos, sin) = (angle.cos(), angle.sin())
    (cos, sin) = (torch.cat((cos, cos), -1), torch.cat((sin, sin), -1))
    return (
        (x, _tensor(shape, strided=strided), weight(), weight(), cos, sin, 1e-05),
        {},
    )


def _reductions(shape):
    rows = math.prod(shape[:-1])
    return [1, 1, rows, rows]


def cases():
    """Primary training shape plus required small/tail and strided coverage."""
    small = (2, 17, 3, PRIMARY[-1])
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
    long_shape = (1, 21840, 56, 128)
    result.append(
        {
            "name": "long_sequence",
            "required": True,
            "make": lambda: make(long_shape),
            "grad_reductions": _reductions(long_shape),
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
    (b, tokens, h, _) = x.shape
    tables = tokens * 96 * 8
    saved = 8 * r
    p = max(1, min(b * tokens, 2 * sms))
    p *= (h + 7) // 8
    if r <= 128:
        p = 0
    if forward:
        return (
            8 * c + 8 * d + tables + (saved if phase != "infer" else 0),
            2 * (7 * c + r),
        )
    return (12 * c + saved + tables + 16 * d + 16 * p * d, 2 * 12 * c)


__all__ = ["kernel_fn", "reference_fn", "ROOF", "cases", "roof"]
