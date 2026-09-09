"""CUDA workloads and phase-specific SoL accounting for f32_adaln."""

import torch
from kernel import adaln as kernel_fn
from h20_adaln import supported as h20_supported, partial_count as h20_partials
from reference import adaln as reference_fn

ROOF = "fp32"
PRIMARY = (16, 4096, 1152)


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
    chunks = 2 if special in ("silu", "two_way_modulation") else 6
    modulation = 0.05 * torch.randn(
        (shape[0], chunks * d), device="cuda", dtype=torch.float32
    )
    (shift, scale) = modulation.chunk(chunks, -1)[:2]
    shift = shift.detach().requires_grad_()
    scale = scale.detach().requires_grad_()
    return ((x, scale, shift, 1e-06), {"silu": special == "silu"})


def _reductions(shape):
    return [1, shape[1], shape[1]]


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
    special = "silu"
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
    result.append(
        {
            "name": "two_way_modulation",
            "required": True,
            "make": lambda: make(small, special="two_way_modulation"),
            "grad_reductions": _reductions(small),
        }
    )
    result.append(
        {
            "name": "model_silu",
            "required": True,
            "make": lambda: make(PRIMARY, special="silu"),
            "grad_reductions": _reductions(PRIMARY),
        }
    )
    # Parameterized H20 coverage: power-of-two, masked tails, wide rows and SiLU.
    for label, shape, strided, special in [
        ("width_128", (3, 257, 128), False, None),
        ("width_768", (3, 257, 768), True, None),
        ("width_1300", (3, 257, 1300), True, None),
        ("width_2048", (3, 257, 2048), False, None),
        ("width_4096", (3, 257, 4096), True, None),
        ("width_8192", (3, 257, 8192), False, None),
        ("silu_tail", (3, 257, 1300), True, "silu"),
        ("silu_wide", (3, 257, 4096), False, "silu"),
    ]:
        result.append(
            dict(
                name=label,
                required=True,
                make=lambda shape=shape, strided=strided, special=special: make(
                    shape, strided, special
                ),
                grad_reductions=_reductions(shape),
            )
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
    gpu = torch.cuda.get_device_properties(x.device)
    sms = gpu.multi_processor_count
    aux = 8 * r if phase != "infer" else 0
    (b, tokens, _) = x.shape
    silu = kwargs.get("silu", False)
    split_plain = (
        not silu
        and d == 1152
        and tokens >= 1024
        and r >= 8192
        and x.dtype == torch.bfloat16
        and "A800" in gpu.name
    )
    program_scale = 12 if silu and d == 1152 else (8 if split_plain else 2)
    p = max(1, min((tokens + 3) // 4, max(1, program_scale * sms // max(1, b))))
    if h20_supported(x):
        p = h20_partials(x, silu)
    if tokens <= 32 and d <= 256:
        p = 0  # One program writes final parameter gradients directly.
    if forward:
        return (4 * c + 8 * b * d + aux, (10 if silu else 8) * c + 3 * r)
    return (
        6 * c + 8 * r + (16 if silu else 12) * b * d + 16 * b * p * d,
        (19 if silu else 13) * c + 2 * r,
    )


__all__ = ["kernel_fn", "reference_fn", "ROOF", "cases", "roof"]
