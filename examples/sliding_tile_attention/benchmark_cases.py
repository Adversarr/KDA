"""Tile attention reference rows and attended-area compute roof."""

import torch
from kernel import sliding_tile_attention as kernel_fn, select_config
from reference import (
    sliding_tile_attention_ref as reference_fn,
    sliding_tile_attention_sdpa as baseline_fn,
    sta_tile_mask,
)

ROOF = "bf16"
ROOF_METHOD = "attention"


def make(small=False):
    tiles = (3, 3, 3)
    ts = 16 if small else 384
    text = 7 if small else 128
    windows = [
        (3, 3, 3),
        (1, 3, 3),
        (3, 1, 3),
        (3, 3, 1),
        (1, 1, 3),
        (1, 3, 1),
        (3, 1, 1),
        (1, 1, 1),
    ]
    mask = sta_tile_mask(tiles, windows, torch.device("cuda"))
    s = 27 * ts + text
    tensors = tuple(
        torch.randn(
            1, 8, s, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        for _ in range(3)
    )
    return (*tensors, mask, ts, text), {}


def _real_tile_mask(device):
    """Confirmed real assignment: eight heads for each of the three TASK windows.

    Contiguous window starts extend the odd-window clamping rule to even sizes:
    clamp(query - window//2, 0, grid-window). No user model builder is changed.
    """
    grid = (5, 6, 10)
    windows = [(3, 3, 3)] * 8 + [(3, 6, 10)] * 8 + [(5, 6, 10)] * 8
    coords = torch.stack(torch.meshgrid(*[torch.arange(n, device=device) for n in grid],
                                       indexing="ij"), -1).reshape(-1, 3)
    masks = []
    for window in windows:
        allowed = torch.ones((300, 300), dtype=torch.bool, device=device)
        for axis, (size, width) in enumerate(zip(grid, window)):
            start = (coords[:, axis] - width // 2).clamp(0, size - width)
            allowed &= ((coords[None, :, axis] >= start[:, None]) &
                        (coords[None, :, axis] < start[:, None] + width))
        masks.append(allowed)
    return torch.stack(masks)


def make_real():
    torch.manual_seed(1729)
    mask = _real_tile_mask(torch.device("cuda"))
    tensors = tuple(torch.randn(1, 24, 115456, 128, device="cuda", dtype=torch.bfloat16,
                                requires_grad=True) for _ in range(3))
    return (*tensors, mask, 384, 256), {}


def cases():
    return [
        dict(name="user_train", make=make, grad_reductions=[1, 1, 1]),
        dict(name="real_model", required=True, make=make_real, grad_reductions=[1, 1, 1]),
        dict(name="text_tail", required=False, make=lambda: make(True), grad_reductions=[1, 1, 1]),
    ]


def _metadata_bytes(mask, tile_size, text_len, sequence, query_block, key_block):
    """Count consumed list entries and unique boundary-mask reads independently."""
    masks = mask.detach().cpu().tolist()
    video = len(masks[0]) * tile_size
    queries = (sequence + query_block - 1) // query_block
    entries = 0
    mask_reads = 0
    for head in masks:
        touched = set()
        for q0 in range(0, sequence, query_block):
            q1 = min(q0 + query_block, sequence)
            qt = range(q0 // tile_size, (min(q1, video) + tile_size - 1) // tile_size)
            for k0 in range(0, sequence, key_block):
                k1 = min(k0 + key_block, sequence)
                kt = range(k0 // tile_size, (min(k1, video) + tile_size - 1) // tile_size)
                coordinates = [(q, k) for q in qt for k in kt]
                values = [head[q][k] for q, k in coordinates]
                allowed = q1 > video or k1 > video or any(values)
                if not allowed:
                    continue
                entries += 1
                full = all(values) and k0 + key_block <= sequence
                if not full:
                    touched.update(coordinates)
        mask_reads += len(touched)
    # Each row reads n_full/n_any and its scheduling permutation once.
    return 4 * entries + 12 * len(masks) * queries + mask_reads


def attention_work(phase, args, kwargs):
    """Useful allowed pairs; calibration retains the original attention geometry."""
    q, k, v, mask, ts, text = args
    b, h, _, _ = q.shape
    video = mask.shape[1] * ts
    pairs = b * (int(mask.sum().item()) * ts * ts + h * (2 * video * text + text * text))
    return dict(q_shape=tuple(q.shape), k_shape=tuple(k.shape), v_shape=tuple(v.shape), pairs=pairs)


def roof(phase, args, kwargs):
    q, k, v, mask, ts, text = args
    b, h, s, d = q.shape
    pairs = attention_work(phase, args, kwargs)["pairs"]
    # Q/K/V + O and logsumexp; backward reads Q/K/V/O/dO/LSE,
    # writes dQ/dK/dV and writes/reads preprocessing delta.
    n = q.numel()
    nbytes = (
        8 * n + (4 * b * h * s if phase == "fwd" else 0)
        if phase in ("fwd", "infer")
        else 26 * n + 20 * b * h * s
    )
    cfg = select_config(d, "bwd" if phase == "bwd" else "fwd", s)
    if phase == "bwd":
        nbytes += _metadata_bytes(mask, ts, text, s, cfg["BLOCK_M2"], cfg["BLOCK_N2"])
        nbytes += _metadata_bytes(mask.transpose(1, 2), ts, text, s, cfg["BLOCK_N1"], cfg["BLOCK_M1"])
        nbytes += 2 * mask.numel()  # The timed backward materializes a transposed mask.
    else:
        nbytes += _metadata_bytes(mask, ts, text, s, cfg["BLOCK_M"], cfg["BLOCK_N"])
    return nbytes, pairs * d * (10 if phase == "bwd" else 4)


__all__ = ["kernel_fn", "reference_fn", "ROOF", "cases", "roof", "baseline_fn"]


def compile_fn(*args, **kwargs):
    """Keep the large sequence's Python loops outside compiled chunk graphs."""
    if args[0].shape[2] > 16384:
        from reference import compiled_reference
        return compiled_reference()
    from _common import bench
    return bench.compile_or_none(reference_fn, *args, mode="max-autotune-no-cudagraphs", **kwargs)
