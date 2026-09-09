"""Prefix-per-view cases, including the all-valid control and precision regression."""

import torch
from kernel import segmented_attention as kernel_fn
from reference import segmented_attention as reference_fn, sdpa as baseline_fn

ROOF = "bf16"
ROOF_METHOD = "attention"


def make(lengths, tokens_per_view, heads=16, dim=64, strided=False, cancel=False):
    torch.manual_seed(1729)
    shape = (len(lengths), heads, len(lengths[0]) * tokens_per_view, dim)
    if strided:
        tensors = [
            torch.randn(*shape[:-1], dim * 2, device="cuda", dtype=torch.bfloat16)[
                ..., :dim
            ].detach()
            for _ in range(3)
        ]
    else:
        tensors = [
            torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3)
        ]
    if cancel:
        q, k, v = tensors
        q.zero_()
        q[..., 0] = 1
        k.zero_()
        k[:, :, 1, 0] = 0.015625
        v.zero_()
        v[:, :, 0, :] = 32
        v[:, :, 1, :] = -32
    return (*[x.requires_grad_() for x in tensors], lengths, tokens_per_view), {}


def cases():
    specs = [
        ("frame", ((1374,), (1337,), (1300,), (0,)), 1374, 16, 64, False, False),
        ("global", ((1374, 1337, 1300, 0),), 1374, 16, 64, False, False),
        ("all_valid", ((1374, 1374, 1374, 1374),), 1374, 16, 64, False, False),
        ("real_24_views", ((1374, 1337, 1300, 0) * 6,), 1374, 16, 64, False, False),
        ("tail_strided", ((17, 3, 0), (0, 11, 5)), 17, 2, 64, True, False),
        ("single_key", ((0, 1, 0),), 17, 2, 64, False, False),
        ("empty_scene", ((0, 0, 0),), 17, 2, 64, False, False),
        ("width_96", ((17, 11),), 17, 2, 96, False, False),
        ("probability_cancellation", ((2,),), 129, 2, 64, False, True),
    ]
    rows = [
        dict(
            name=s[0],
            make=lambda s=s: make(*s[1:]),
            grad_reductions=[1, 1, 1],
            benchmark=s[0] in ("frame", "global", "all_valid", "real_24_views"),
            required=True,
        )
        for s in specs
    ]
    for s in specs:

        def recompute(s=s):
            args, kwargs = make(*s[1:])
            return args, dict(kwargs, recompute=True)

        rows.append(
            dict(
                name=s[0] + "_recompute",
                make=recompute,
                grad_reductions=[1, 1, 1],
                benchmark=False,
                required=True,
            )
        )
    return rows


def attention_work(phase, args, kwargs):
    q, k, v, lengths, t = args
    _, h, n, _ = q.shape
    return dict(
        q_shape=tuple(q.shape),
        k_shape=tuple(k.shape),
        v_shape=tuple(v.shape),
        pairs=h * n * sum(max(1, sum(row)) for row in lengths),
    )


def roof(phase, args, kwargs):
    q, k, v, lengths, t = args
    b, h, n, d = q.shape
    rk = h * sum(max(1, sum(row)) for row in lengths)
    rq = b * h * n
    pairs = attention_work(phase, args, kwargs)["pairs"]
    es = q.element_size()
    # Two-input packing reads/writes valid KV. Backward scatter reads compact
    # gradients and writes the full original KV planes, including padding zeros.
    packing = 4 * rk * d * es
    metadata = 4 * b * (len(lengths[0]) + 1)
    if phase == "bwd":
        traffic = (7 * rq + 6 * rk) * d * es + 20 * rq
        traffic += (2 * rk + 2 * rq) * d * es + metadata
        flops = 10 * pairs * d
        if kwargs.get("recompute"):
            traffic += (2 * rq + 2 * rk) * d * es + 4 * rq
            flops += 4 * pairs * d
        return traffic, flops
    return (2 * rq + 2 * rk) * d * es + packing + metadata + (
        4 * rq if phase == "fwd" and not kwargs.get("recompute") else 0
    ), 4 * pairs * d


__all__ = [
    "kernel_fn",
    "reference_fn",
    "ROOF",
    "ROOF_METHOD",
    "cases",
    "roof",
    "attention_work",
    "baseline_fn",
    "compile_fn",
]


def compile_fn(*args, **kwargs):
    if args[0].shape[2] > 8192:
        from reference import compiled_reference
        return compiled_reference()
    from _common import bench
    return bench.compile_or_none(reference_fn, *args, mode="max-autotune-no-cudagraphs", **kwargs)
