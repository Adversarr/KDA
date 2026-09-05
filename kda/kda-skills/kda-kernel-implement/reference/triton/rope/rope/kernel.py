"""Rotary position embedding (RoPE) in Triton: half and interleaved pairings, fused backward.

For every (batch, position, head) row of ``D`` elements, pairs ``(x1, x2)`` are rotated by the
position's angle::

    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin

* ``half`` (LLaMA / GPT-NeoX ``rotate_half``): ``x1 = x[:D/2]``, ``x2 = x[D/2:]``.
* ``interleaved`` (GPT-J): ``x1 = x[0::2]``, ``x2 = x[1::2]``.

The backward is the rotation by ``-angle`` (``sin -> -sin``), so one kernel serves both
directions through the ``BACKWARD`` constexpr. ``cos``/``sin`` are fp32 tables of shape
``(S, D/2)`` indexed by position; an optional ``positions (B, S)`` int tensor overrides the
implicit ``0..S-1`` (packed sequences, KV-cache offsets).

``x`` may have arbitrary strides (a head slice of a fused QKV projection is the common case);
``y`` is written contiguous in the same ``(B, S, H, D)`` order.
"""

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _rope_kernel(
    x_ptr, cos_ptr, sin_ptr, pos_ptr, y_ptr,
    S, H, D_HALF,
    sx_b, sx_s, sx_h,
    sy_b, sy_s, sy_h,
    BLOCK_H: tl.constexpr, BLOCK_DH: tl.constexpr,
    INTERLEAVED: tl.constexpr, HAS_POS: tl.constexpr, BACKWARD: tl.constexpr, COMPUTE: tl.constexpr,
):
    # One program per (batch, position) and a block of heads: the cos/sin row is loaded once
    # and broadcast over the head tile.
    pid_bs = tl.program_id(0).to(tl.int64)
    b = pid_bs // S
    s = pid_bs % S
    offs_h = (tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)).to(tl.int64)
    offs_d = tl.arange(0, BLOCK_DH)
    hmask = offs_h[:, None] < H

    if HAS_POS:
        pos = tl.load(pos_ptr + pid_bs).to(tl.int64)
    else:
        pos = s
    dmask = offs_d < D_HALF
    cos = tl.load(cos_ptr + pos * D_HALF + offs_d, mask=dmask, other=0.0).to(COMPUTE)[None, :]
    sin = tl.load(sin_ptr + pos * D_HALF + offs_d, mask=dmask, other=0.0).to(COMPUTE)[None, :]
    if BACKWARD:
        sin = -sin

    x_base = x_ptr + b * sx_b + s * sx_s + offs_h[:, None] * sx_h
    y_base = y_ptr + b * sy_b + s * sy_s + offs_h[:, None] * sy_h
    if INTERLEAVED:
        # Load the whole contiguous row and separate even/odd lanes in registers. Indexing
        # memory with 2*offs_d instead defeats vectorisation (13x slower on A800).
        offs_full = tl.arange(0, 2 * BLOCK_DH)
        fmask = hmask & (offs_full[None, :] < 2 * D_HALF)
        x = tl.load(x_base + offs_full[None, :], mask=fmask, other=0.0).to(COMPUTE)
        x1, x2 = tl.split(tl.reshape(x, [BLOCK_H, BLOCK_DH, 2]))
        y1 = x1 * cos - x2 * sin
        y2 = x2 * cos + x1 * sin
        y = tl.reshape(tl.join(y1, y2), [BLOCK_H, 2 * BLOCK_DH])
        tl.store(y_base + offs_full[None, :], y.to(y_ptr.dtype.element_ty), mask=fmask)
    else:
        mask = hmask & dmask[None, :]
        x1 = tl.load(x_base + offs_d[None, :], mask=mask, other=0.0).to(COMPUTE)
        x2 = tl.load(x_base + (offs_d + D_HALF)[None, :], mask=mask, other=0.0).to(COMPUTE)
        y1 = x1 * cos - x2 * sin
        y2 = x2 * cos + x1 * sin
        tl.store(y_base + offs_d[None, :], y1.to(y_ptr.dtype.element_ty), mask=mask)
        tl.store(y_base + (offs_d + D_HALF)[None, :], y2.to(y_ptr.dtype.element_ty), mask=mask)


def _launch(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, positions: Optional[torch.Tensor], interleaved: bool, backward: bool) -> torch.Tensor:
    B, S, H, D = x.shape
    assert D % 2 == 0 and x.stride(3) == 1, "rope: D must be even and the head dim contiguous"
    assert cos.shape == sin.shape == (cos.shape[0], D // 2) and cos.is_contiguous() and sin.is_contiguous()
    d_half = D // 2
    y = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    block_dh = triton.next_power_of_2(d_half)
    block_h = max(1, min(triton.next_power_of_2(H), 2048 // block_dh))  # ~2K-element tiles
    grid = (B * S, triton.cdiv(H, block_h))
    _rope_kernel[grid](
        x, cos, sin, positions if positions is not None else x, y,
        S, H, d_half,
        x.stride(0), x.stride(1), x.stride(2),
        y.stride(0), y.stride(1), y.stride(2),
        BLOCK_H=block_h, BLOCK_DH=block_dh,
        INTERLEAVED=interleaved, HAS_POS=positions is not None, BACKWARD=backward,
        COMPUTE=tl.float32, num_warps=4,
    )
    return y


def rope_fwd(x, cos, sin, positions=None, interleaved=False):
    """``x (B, S, H, D)`` any strides -> rotated ``y`` (contiguous)."""
    return _launch(x, cos, sin, positions, interleaved, backward=False)


def rope_bwd(dy, cos, sin, positions=None, interleaved=False):
    """Adjoint: rotate ``dy`` by the negative angle."""
    return _launch(dy, cos, sin, positions, interleaved, backward=True)


class _Rope(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cos, sin, positions, interleaved):
        ctx.save_for_backward(cos, sin)
        ctx.positions = positions
        ctx.interleaved = interleaved
        return rope_fwd(x, cos, sin, positions, interleaved)

    @staticmethod
    def backward(ctx, dy):
        cos, sin = ctx.saved_tensors
        return rope_bwd(dy, cos, sin, ctx.positions, ctx.interleaved), None, None, None, None


def rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, positions: Optional[torch.Tensor] = None, interleaved: bool = False) -> torch.Tensor:
    """RoPE over the last dim of ``x (B, S, H, D)``; ``cos``/``sin`` are ``(S_max, D/2)`` fp32. Differentiable in ``x``."""
    return _Rope.apply(x, cos, sin, positions, interleaved)


__all__ = ["rope", "rope_fwd", "rope_bwd"]
