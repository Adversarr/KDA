"""GQA projection views, real rotary tables and independently counted phase traffic."""

import torch
from kernel import qk_prep as kernel_fn
from reference import qk_prep_ref as reference_fn

ROOF = "fp32"


def make(b, s, hq=32, hk=8, d=128, strided=False):
    if strided:
        packed = torch.randn(
            b, s, (hq + 2 * hk) * d, device="cuda", dtype=torch.bfloat16
        )
        q, k, _ = packed.split((hq * d, hk * d, hk * d), dim=-1)
        q = q.view(b, s, hq, d).detach().requires_grad_()
        k = k.view(b, s, hk, d).detach().requires_grad_()
    else:
        q = torch.randn(
            b, s, hq, d, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        k = torch.randn(
            b, s, hk, d, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
    qw = (1 + 0.1 * torch.randn(hq, d, device="cuda")).requires_grad_()
    kw = (1 + 0.1 * torch.randn(hk, d, device="cuda")).requires_grad_()
    phase = (
        torch.arange(s, device="cuda")[:, None]
        * (10000.0 ** (-torch.arange(0, d, 2, device="cuda").float() / d))[None]
    )
    return (q, k, qw, kw, phase.cos(), phase.sin(), 1e-6), {}


def cases():
    return [
        dict(
            name=name,
            benchmark=name != "zero_rows",
            make=lambda b=b, s=s, st=st: make(b, s, strided=st),
            grad_reductions=[1, 1, b * s, b * s],
        )
        for name, b, s, st in [
            ("user_b4_s4096_hq32_hk8_d128", 4, 4096, True),
            ("small_rows", 1, 7, False),
            ("zero_rows", 0, 7, False),
            ("strided_rows", 2, 33, True),
        ]
    ]


def roof(phase, args, kwargs):
    q, k, qw, kw, c, s, _ = args
    n = q.numel() + k.numel()
    d = q.shape[-1]
    rows = n // d
    tables = (c.numel() + s.numel()) * 4
    weights = (qw.numel() + kw.numel()) * 4
    if phase in ("fwd", "infer"):
        return n * 4 + tables + weights + (rows * 4 if phase == "fwd" else 0), n * 10
    programs = min(
        q.shape[0] * q.shape[1],
        2 * torch.cuda.get_device_properties(q.device).multi_processor_count,
    )
    if 0 < q.shape[0] * q.shape[1] <= 32:
        programs = 0  # The short-sequence backward writes final per-head gradients.
    return n * 6 + tables + weights * 2 + rows * 4 + 2 * programs * weights, n * 18


__all__ = ["kernel_fn", "reference_fn", "ROOF", "cases", "roof"]
