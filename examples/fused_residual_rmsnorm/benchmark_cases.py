"""User and layout workloads; roof counts include backward partial reduction traffic."""

import math
import torch
from kernel import add_rmsnorm as kernel_fn
from reference import add_rmsnorm_ref as reference_fn

ROOF = "fp32"


def make(shape, pad=0):
    d = shape[-1]
    x = (
        torch.randn(*shape[:-1], d + pad, device="cuda", dtype=torch.bfloat16)[..., :d]
        .detach()
        .requires_grad_()
    )
    residual = (
        torch.randn(*shape[:-1], d + 2 * pad, device="cuda")[..., :d]
        .detach()
        .requires_grad_()
    )
    weight = (1 + 0.1 * torch.randn(d, device="cuda")).requires_grad_()
    return (x, residual, weight), {"eps": 1e-6}


def cases():
    shapes = [
        ("user_b8_s512_d1024", (8, 512, 1024), 0),
        ("zero_rows", (0, 7, 1024), 0),
        ("small_rows", (1, 7, 1024), 0),
        ("strided_rows", (2, 33, 1024), 16),
    ]
    shapes += [(f"width_{d}", (2, 33, d), 0) for d in (128, 768, 2048, 8192)]
    return [
        dict(
            name=name,
            benchmark=name != "zero_rows",
            make=lambda shape=shape, pad=pad: make(shape, pad),
            grad_reductions=[1, 1, math.prod(shape[:-1])],
        )
        for name, shape, pad in shapes
    ]


def roof(phase, args, kwargs):
    x, residual, w = args
    n = x.numel()
    d = x.shape[-1]
    rows = n // d
    es = x.element_size()
    if phase in ("fwd", "infer"):
        return n * (2 * es + 8) + 4 * d + (4 * rows if phase == "fwd" else 0), 6 * n
    # h, dy, dh, weight, rstd -> dx, dresidual, dw plus write/read dw partials.
    from kernel import _bwd_geometry

    _, r, _ = _bwd_geometry(d)
    programs = max(
        1,
        min(
            (rows + r - 1) // r,
            2 * torch.cuda.get_device_properties(x.device).multi_processor_count,
        ),
    )
    partial_bytes = 0 if 0 < rows <= 128 and d <= 8192 else 8 * programs * d
    return n * (2 * es + 12) + 8 * d + 4 * rows + partial_bytes, 12 * n


__all__ = ["kernel_fn", "reference_fn", "ROOF", "cases", "roof"]
