"""CUDA workloads and phase-specific SoL accounting for fused_l2_norm_scale."""

import math
import torch
from kernel import l2_norm_scale as kernel_fn
from reference import l2_norm_scale as reference_fn

ROOF = "fp32"
PRIMARY = (8, 4096, 2048)


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
    if special == "qkv_view":
        (b, tokens, heads, width) = shape
        projection = torch.randn(
            (b, tokens, 3, heads, width), device="cuda", dtype=torch.bfloat16
        )
        x = projection.unbind(2)[0].detach().requires_grad_()
    if special == "zeros":
        with torch.no_grad():
            x.zero_()
    return (
        (
            x,
            torch.full(
                (d,), math.sqrt(d), device="cuda", dtype=torch.float32
            ).requires_grad_(),
            1e-06,
        ),
        {},
    )


def _reductions(shape):
    rows = math.prod(shape[:-1])
    return [1, rows]


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
    for label, shape, special in [
        ("per_head", (2, 31, 8, 128), None),
        ("qkv_view", (2, 31, 8, 128), "qkv_view"),
        ("zero_norm", small, "zeros"),
    ]:
        result.append(
            {
                "name": label,
                "required": True,
                "make": lambda shape=shape, special=special: make(
                    shape, special=special
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
    model_heads = (8, 4096, 16, 128)
    result.append(
        {
            "name": "model_per_head",
            "required": True,
            "make": lambda: make(model_heads, special="qkv_view"),
            "grad_reductions": _reductions(model_heads),
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
    row_tile = 32 if d <= 256 else 4
    p = (
        0
        if r <= 128 and d <= 1024
        else max(1, min((r + row_tile - 1) // row_tile, 2 * sms))
    )
    aux = 8 * r if phase != "infer" else 0
    return (
        (4 * c + 4 * d + aux, 4 * c + r)
        if forward
        else (6 * c + 8 * r + 8 * d + 8 * p * d, 9 * c)
    )


__all__ = ["kernel_fn", "reference_fn", "ROOF", "cases", "roof"]
