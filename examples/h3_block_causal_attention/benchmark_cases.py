"""Required workloads and independent analytical work accounting."""

import torch
from kernel import block_causal_attention as kernel_fn
from reference import block_causal_attention as eager_attention, sdpa as baseline_fn

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
    return (*args, maskarg), {}


def cases():
    specs = [
        ("user", 1, 56, 3072, 128, 512, False),
        ("tail_frame", 1, 2, 257, 128, 70, True),
        ("causal", 1, 2, 129, 128, 1, False),
        ("dense", 1, 2, 129, 128, 256, False),
        ("real_14_frames", 1, 56, 21840, 128, 1560, False),
        ("real_40_frames", 1, 56, 62400, 128, 1560, False),
    ]
    normal = ("user", "real_14_frames", "real_40_frames")
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
    q, k, v, block = args
    b, h, s, _ = q.shape
    full, tail = divmod(s, block)
    pairs = b * h * (block * block * full * (full + 1) // 2 + tail * s)
    return dict(q_shape=tuple(q.shape), k_shape=tuple(k.shape), v_shape=tuple(v.shape), pairs=pairs)


def roof(phase, args, kwargs):
    q, k, v, mask = args
    B, H, S, D = q.shape
    es = q.element_size()
    pairs = attention_work(phase, args, kwargs)["pairs"]
    # Required reads/writes; attention GEMMs dominate, score planes are not materialized.
    if phase == "bwd":
        byte_count, flops = B * H * S * (13 * D * es + 20), pairs * 10 * D
        if kwargs.get("recompute", False):
            byte_count += B * H * S * (4 * D * es + 4)
            flops += pairs * 4 * D
        return byte_count, flops
    aux = 4 * B * H * S if phase == "fwd" and not kwargs.get("recompute", False) else 0
    return (4 * B * H * S * D * es + aux, pairs * 4 * D)


__all__ = ["kernel_fn", "reference_fn", "ROOF", "cases", "roof", "baseline_fn", "compile_fn"]


def compile_fn(*args, **kwargs):
    """Compile only reusable chunks for sequences whose reference is bounded."""
    if args[0].shape[2] > 8192:
        from reference import compiled_reference
        return compiled_reference()
    from _common import bench
    return bench.compile_or_none(reference_fn, *args, mode="max-autotune-no-cudagraphs", **kwargs)
