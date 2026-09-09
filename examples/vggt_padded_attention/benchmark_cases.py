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


def _probability_cancellation():
    """Exercise the adopted Flash-style probability rounding boundary.

    The strict FP32-probability reference returned -0.03125. BF16 Flash and
    the current eager probability cast return zero for this case. Keep the
    case to expose the precision convention, not to enforce the retired one.
    """
    q = torch.zeros((1, 2, 129, 64), device="cuda", dtype=torch.bfloat16)
    q[..., 0] = 1
    k = torch.zeros_like(q)
    k[:, :, 1, 0] = 0.015625
    v = torch.zeros_like(q)
    v[:, :, 0, :] = 32
    v[:, :, 1, :] = -32
    valid = torch.zeros((1, 129), device="cuda", dtype=torch.bool)
    valid[:, :2] = True
    return (q.requires_grad_(), k.requires_grad_(), v.requires_grad_(), valid), {}


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
    rows.append(dict(name="probability_cancellation", required=True, benchmark=False,
                     make=_probability_cancellation, grad_reductions=[1, 1, 1]))

    def cancellation_recompute():
        args, kwargs = _probability_cancellation()
        return args, {**kwargs, "recompute": True}

    rows.append(dict(name="probability_cancellation_recompute", required=True, benchmark=False,
                     make=cancellation_recompute, grad_reductions=[1, 1, 1]))
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
    rq = B * H * S
    rk = H * int(mask.sum().item())
    # Scan mask/block totals, then gather valid KV entirely on the GPU.
    # Prefix/block-start lookups are read by each head's pack/scatter pass.
    blocks = (S + 1023) // 1024
    scan = 5 * B * S + 16 * B * blocks + 16 * B
    indices = 8 * rq
    if phase == "bwd":
        byte_count = (7 * rq + 6 * rk) * D * es + 20 * rq
        byte_count += (2 * rk + 2 * rq) * D * es + indices + 8 * B
        flops = pairs * 10 * D
        if kwargs.get("recompute", False):
            byte_count += (2 * rq + 2 * rk) * D * es + 4 * rq + 8 * B
            flops += pairs * 4 * D
        return byte_count, flops
    aux = 4 * rq if phase == "fwd" and not kwargs.get("recompute", False) else 0
    return ((2 * rq + 6 * rk) * D * es + aux + scan + indices, pairs * 4 * D)



__all__ = ["kernel_fn", "reference_fn", "ROOF", "cases", "roof", "baseline_fn", "compile_fn"]


def compile_fn(*args, **kwargs):
    """Compile only reusable chunks for sequences whose reference is bounded."""
    if args[0].shape[2] > 8192:
        from reference import compiled_reference
        return compiled_reference()
    from _common import bench
    return bench.compile_or_none(reference_fn, *args, mode="max-autotune-no-cudagraphs", **kwargs)
