"""Required workloads and independent analytical work accounting."""

import torch
from kernel import padded_attention as kernel_fn
from reference import padded_attention as eager_attention, sdpa as baseline_fn

ROOF = "bf16"
ROOF_METHOD = "attention"


def reference_fn(*args, recompute=False):
    """The eager contract is independent of saved-versus-recomputed auxiliaries."""
    return eager_attention(*args)


def _make(B, H, S, D, maskarg, strided=False):
    torch.manual_seed(1729)
    if strided:
        args = [
            torch.randn((B, S, H, D), device="cuda", dtype=torch.bfloat16)
            .transpose(1, 2)
            .requires_grad_()
            for _ in range(3)
        ]
    else:
        args = [
            torch.randn(
                (B, H, S, D), device="cuda", dtype=torch.bfloat16
            ).requires_grad_()
            for _ in range(3)
        ]
    valid = torch.rand((B, S), device="cuda") >= maskarg
    valid[:, 0] = True
    return (*args, valid), {}


def cases():
    specs = [
        ("frame", 4, 16, 1374, 64, 0.15, False),
        ("global", 1, 16, 5496, 64, 0.15, False),
        ("tail_strided", 2, 3, 137, 64, 0.7, True),
        ("single_valid", 2, 2, 129, 64, 1.0, False),
        ("all_valid", 1, 2, 129, 64, 0.0, False),
        ("real_24_views", 2, 16, 32976, 64, 0.3, False),
    ]
    normal = ("frame", "global", "real_24_views")
    rows = [
        {
            "name": row[0],
            "required": row[0] in normal,
            "make": lambda row=row: _make(*row[1:]),
            "grad_reductions": [1, 1, 1],
        }
        for row in specs
    ]


    for spec in specs:
        def make_recompute(spec=spec):
            args, kwargs = _make(*spec[1:])
            return args, {**kwargs, "recompute": True}
        rows.append(dict(name=spec[0] + "_recompute", required=spec[0] in normal, benchmark=False,
                         make=make_recompute, grad_reductions=[1, 1, 1]))
    return rows


def attention_work(phase, args, kwargs):
    """Useful allowed pairs; calibration retains the original attention geometry."""
    q, k, v, valid = args
    _, h, s, _ = q.shape
    pairs = h * s * int(valid.sum().item())
    return dict(q_shape=tuple(q.shape), k_shape=tuple(k.shape), v_shape=tuple(v.shape), pairs=pairs)


def roof(phase, args, kwargs):
    q, k, v, mask = args
    B, H, S, D = q.shape
    es = q.element_size()
    pairs = attention_work(phase, args, kwargs)["pairs"]
    # Required reads/writes; attention GEMMs dominate, score planes are not materialized.
    if phase == "bwd":
        byte_count, flops = B * H * S * (13 * D * es + 20) + 2 * B * S, pairs * 10 * D
        if kwargs.get("recompute", False):
            byte_count += B * H * S * (4 * D * es + 4) + B * S
            flops += pairs * 4 * D
        return byte_count, flops
    aux = 4 * B * H * S if phase == "fwd" and not kwargs.get("recompute", False) else 0
    return (4 * B * H * S * D * es + aux + B * S, pairs * 4 * D)


__all__ = ["kernel_fn", "reference_fn", "ROOF", "cases", "roof", "baseline_fn", "compile_fn"]


def compile_fn(*args, **kwargs):
    """Compile only reusable chunks for sequences whose reference is bounded."""
    if args[0].shape[2] > 8192:
        from reference import compiled_reference
        return compiled_reference()
    from _common import bench
    return bench.compile_or_none(reference_fn, *args, mode="max-autotune-no-cudagraphs", **kwargs)
