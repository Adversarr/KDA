"""CUDA workloads and phase-specific SoL accounting for vggt_qkv_layernorm_rope2d."""

import torch
from kernel import qkv_prep as kernel_fn
from reference import qkv_prep as reference_fn

ROOF = "fp32"
PRIMARY = (1, 5496, 3, 16, 64)


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
    positions = torch.randint(0, 37, (shape[0], shape[1], 2), device="cuda")
    positions[:, 0] = 0
    if shape[1] in (1374, 5496):
        (yy, xx) = torch.meshgrid(
            torch.arange(1, 38, device="cuda"),
            torch.arange(1, 38, device="cuda"),
            indexing="ij",
        )
        grid = torch.cat(
            (
                torch.zeros((5, 2), device="cuda", dtype=torch.int64),
                torch.stack((yy.flatten(), xx.flatten()), -1),
            )
        )
        positions = (
            grid.repeat(shape[1] // 1374, 1)[None].expand(shape[0], -1, -1).contiguous()
        )
    inv = 1.0 / 100 ** (
        torch.arange(0, d // 2, 2, device="cuda", dtype=torch.float32) / (d // 2)
    )
    return ((x, weight(), bias(), weight(), bias(), positions, inv, 1e-05), {})


def _reductions(shape):
    return [1] + [shape[0] * shape[1] * shape[3]] * 4


def cases():
    """Primary training shape plus required small/tail and strided coverage."""
    small = (2, 17, 3, 3, 64)
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
    frame_shape = (4, 1374, 3, 16, 64)
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
    forward = phase in ("fwd", "infer")
    sms = torch.cuda.get_device_properties(x.device).multi_processor_count
    (b, tokens, _, h, d) = x.shape
    qrows = b * tokens * h
    p = max(1, min(b * tokens, 8 * sms))
    (positions, frequencies) = (b * tokens * 2 * 8, d)
    # Training caches 64 fp32 rotary coefficients per token once in Q forward.
    # Q and K backward each read that cache; they no longer read positions/frequencies.
    if forward:
        return (
            4 * c
            + 16 * d
            + positions
            + frequencies
            + ((16 * qrows + 256 * b * tokens) if phase != "infer" else 0),
            2 * qrows * d * 11,
        )
    return (
        16 * c // 3 + 16 * qrows + 512 * b * tokens + 24 * d + 32 * p * d,
        2 * qrows * d * 16,
    )


__all__ = ["kernel_fn", "reference_fn", "ROOF", "cases", "roof"]
