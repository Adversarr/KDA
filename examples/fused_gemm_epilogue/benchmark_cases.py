"""Language-model GEMM shapes; fp32 parameters cast as under autocast."""

import torch
from kernel import fc1_gelu as kernel_fn, BACKEND
from reference import fc1_gelu_ref

ROOF = "bf16"
__all__ = ["kernel_fn", "reference_fn", "ROOF", "cases", "roof", "gemm_shapes"]


def make(m, n=8192, k=2048, bias_scale=None):
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = (0.02 * torch.randn(n, k, device="cuda")).requires_grad_()
    # The fixture owns an explicit zero-initialized fc1_bias parameter.
    b = (
        torch.zeros(n, device="cuda")
        if bias_scale is None
        else torch.empty(n, device="cuda").uniform_(-bias_scale, bias_scale)
    )
    b = b.requires_grad_()
    return (x, w, b), {}


def reference_fn(x, weight, bias, *, recompute=False):
    """Recompute changes kernel storage policy, not the eager operation."""
    return fc1_gelu_ref(x, weight, bias)


def _recompute(make):
    args, kwargs = make()
    return args, dict(kwargs, recompute=True)


def cases():
    rows = [
        dict(name=name, make=lambda m=m: make(m), grad_reductions=[1, m, m])
        for name, m in [
            ("user_b8_s1024_d2048_h8192", 8192),
            ("small_rows", 16),
            ("tail_rows", 129),
        ]
    ] + [
        dict(
            name="nonzero_bias",
            make=lambda: make(33, bias_scale=0.5),
            grad_reductions=[1, 33, 33],
        )
    ]
    for row in rows:
        row["required"] = row["name"] not in ("tail_rows", "nonzero_bias")
    return rows + [
        dict(
            name=row["name"] + "_recompute",
            required=row["required"],
            benchmark=False,
            make=lambda make=row["make"]: _recompute(make),
            grad_reductions=row["grad_reductions"],
        )
        for row in rows
    ]


def gemm_shapes(phase, args, kwargs):
    x, w, b = args
    m = x.numel() // x.shape[-1]
    n, k = w.shape
    shapes = [(m, n, k)] if phase in ("fwd", "infer") else [(m, k, n), (n, k, m)]
    if phase == "bwd" and kwargs.get("recompute", False):
        shapes.append((m, n, k))
    return shapes


def roof(phase, args, kwargs):
    x, w, b = args
    m = x.numel() // x.shape[-1]
    n, k = w.shape
    if phase in ("fwd", "infer"):
        # fp32 parameter read/cast, bf16 operands, output, optional preactivation aux.
        if BACKEND == "triton" and m > 256:
            # The wide-output fallback materializes unbiased Z, then reads it
            # in a separate epilogue, even for inference or recomputation.
            return 2 * m * k + 8 * n * k + 8 * n + 6 * m * n, 2 * m * n * k
        return 2 * m * k + 8 * n * k + 8 * n + 2 * m * n + (
            2 * m * n if phase == "fwd" and not kwargs.get("recompute", False) else 0
        ), 2 * m * n * k
    # Epilogue reads dY/Z and writes dZ; EACH of the two GEMMs reads dZ.
    if 0 < m <= 256:
        # Small-batch dW stores rounded bf16 values directly to fp32; db is
        # reduced in the GELU adjoint, without an intermediate partial buffer.
        byte_count = 10 * m * n + 4 * m * k + 6 * n * k + 4 * n
        flops = 4 * m * n * k
        if kwargs.get("recompute", False):
            byte_count += 2 * (m * k + n * k + n + m * n)
            flops += 2 * m * n * k
        return byte_count, flops
    # dW is written in bf16, then read/cast to the fp32 input parameter gradient.
    # Bias partials are fp32; the wrapper's bf16 bias cast adds two gradient casts.
    sms = torch.cuda.get_device_properties(x.device).multi_processor_count
    programs = max(1, min((m + 31) // 32, 2 * sms // ((n + 127) // 128)))
    partials = 8 * programs * n
    bias_traffic = 16 * n + (2 * n if BACKEND == "triton" else 0)
    byte_count = 10 * m * n + 4 * m * k + 10 * n * k + bias_traffic + partials
    flops = 4 * m * n * k
    if kwargs.get("recompute", False):
        byte_count += 2 * (m * k + n * k + m * n)
        if BACKEND != "triton":
            byte_count += 2 * n  # cuBLASLt reconstructs biased Z; Triton does not.
        flops += 2 * m * n * k
    return byte_count, flops
