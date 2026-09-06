"""Required MLP shapes and work accounting including parameter casts and bias partials."""

import torch
from kernel import mlp_fc1_gelu as kernel_fn
from reference import mlp_fc1_gelu as eager_mlp

ROOF = "bf16"


def reference_fn(x, w, b, *, recompute=False):
    """Recomputation changes storage, not the eager mathematical contract."""
    return eager_mlp(x, w, b)


def _make(M, K, N, strided=False):
    torch.manual_seed(1729)
    x = torch.randn(
        (1, M, K * (2 if strided else 1)), device="cuda", dtype=torch.bfloat16
    )
    x = x[..., :K].requires_grad_()
    w = (torch.randn((N, K), device="cuda") / K**0.5).requires_grad_()
    b = (torch.randn((N,), device="cuda") * 0.1).requires_grad_()
    return (x, w, b), {}


def cases():
    specs = [
        ("user", 5496, 1024, 4096, False),
        ("real_24_views", 65952, 1024, 4096, False),
        ("tail_strided", 129, 127, 255, True),
        ("small", 7, 64, 128, False),
    ]
    rows = [
        {
            "name": r[0],
            "required": r[0] in ("user", "real_24_views"),
            "make": lambda r=r: _make(*r[1:]),
            "grad_reductions": [1, r[1], r[1]],
        }
        for r in specs
    ]
    for spec in specs:
        def make_recompute(spec=spec):
            args, kwargs = _make(*spec[1:])
            return args, {**kwargs, "recompute": True}

        rows.append({
            "name": spec[0] + "_recompute",
            "required": spec[0] in ("user", "real_24_views"),
            "benchmark": False,
            "make": make_recompute,
            "grad_reductions": [1, spec[1], spec[1]],
        })
    return rows


def gemm_shapes(phase, args, kwargs):
    x, w, b = args
    M, K, N = x.numel() // x.shape[-1], x.shape[-1], w.shape[0]
    shapes = [(M, K, N), (N, K, M)] if phase == "bwd" else [(M, N, K)]
    if phase == "bwd" and kwargs.get("recompute", False):
        shapes.append((M, N, K))
    return shapes


def roof(phase, args, kwargs):
    x, w, b = args
    M, K, N = x.numel() // x.shape[-1], x.shape[-1], w.shape[0]
    # Parameters are fp32 in user code, cast to bf16 before the linear.
    if M <= 256 and N <= 512:
        forward = 2 * M * K + 4 * N * K + 4 * N + 2 * M * N
        if phase != "bwd":
            return forward + (2 * M * N if phase == "fwd" and not kwargs.get("recompute", False) else 0), 2 * M * N * K
        byte_count = 10 * M * N + 4 * M * K + 8 * N * K + 4 * N
        flops = 4 * M * N * K
        if kwargs.get("recompute", False):
            byte_count += forward + 2 * M * N
            flops += 2 * M * N * K
        return byte_count, flops
    if phase == "bwd":
        ncol = (N + 127) // 128
        sms = torch.cuda.get_device_properties(x.device).multi_processor_count
        nrow = max(1, min((M + 31) // 32, 8 * sms // ncol))
        # dY, Z read; dZ write and read twice; X/W read; dX/dW/db writes;
        # fp32 bias partial write/read, and fp32 parameter-gradient conversion.
        byte_count = 10 * M * N + 4 * M * K + 10 * N * K + 8 * nrow * N + 16 * N
        flops = 4 * M * N * K
        if kwargs.get("recompute", False):
            byte_count += 6 * N + 2 * (M * K + N * K + N + M * N)
            flops += 2 * M * N * K
        return byte_count, flops
    # Cast fp32 W/b, addmm reads X/W/b and writes Z; GELU reads Z and writes Y.
    return (6 * (N * K + N) + 2 * (M * K + N * K + N + 3 * M * N), 2 * M * N * K)


__all__ = ["kernel_fn", "reference_fn", "ROOF", "cases", "roof", "gemm_shapes"]
