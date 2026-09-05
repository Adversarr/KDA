"""Fusion exemplar: per-head RMSNorm -> RoPE (half pairing) -> permuted store, fwd + bwd.

The QK-norm pattern of Qwen3-style attention::

    x (B, S, H, D)  -->  rmsnorm over D with weight w (D,)  -->  rope at position s  -->  y (B, H, S, D)

Three eager ops and two full passes over the tensor (three with the `.contiguous()`) become
one read of ``x`` and one write of ``y``. Read `FUSION.md` for the walkthrough; the primitives
this is built from live in ``../../fp32_norms/rmsnorm``, ``../../rope/rope`` and
``../../permute/bnhd_to_bhnd``.

Saved for the backward: ``rstd (B, S, H)`` fp32 (4 bytes per row vs recomputing a row
reduction from a second read of ``x``).
"""

from typing import Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_fwd_kernel(
    x_ptr, w_ptr, cos_ptr, sin_ptr, y_ptr, rstd_ptr,
    S, H, D_HALF, eps,
    sx_b, sx_s, sx_h,
    BLOCK_H: tl.constexpr, BLOCK_DH: tl.constexpr, COMPUTE: tl.constexpr, SAVE_RSTD: tl.constexpr,
):
    # One program per (b, s) and a block of heads. The row is loaded as two half tiles because
    # RoPE pairs element d with d + D/2; the RMS reduction simply sums over both halves.
    pid_bs = tl.program_id(0).to(tl.int64)
    b = pid_bs // S
    s = pid_bs % S
    offs_h = (tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)).to(tl.int64)
    offs_d = tl.arange(0, BLOCK_DH)
    hmask = offs_h < H
    dmask = offs_d < D_HALF
    mask = hmask[:, None] & dmask[None, :]
    D = 2 * D_HALF

    # --- load once (primitive: rmsnorm) ---
    x_base = x_ptr + b * sx_b + s * sx_s + offs_h[:, None] * sx_h
    x1 = tl.load(x_base + offs_d[None, :], mask=mask, other=0.0).to(COMPUTE)
    x2 = tl.load(x_base + (offs_d + D_HALF)[None, :], mask=mask, other=0.0).to(COMPUTE)
    w1 = tl.load(w_ptr + offs_d, mask=dmask, other=0.0).to(COMPUTE)[None, :]
    w2 = tl.load(w_ptr + offs_d + D_HALF, mask=dmask, other=0.0).to(COMPUTE)[None, :]
    rstd = tl.rsqrt((tl.sum(x1 * x1, axis=1) + tl.sum(x2 * x2, axis=1)) / D + eps)  # [BLOCK_H]
    n1 = x1 * rstd[:, None] * w1
    n2 = x2 * rstd[:, None] * w2

    # --- rotate in registers (primitive: rope) ---
    cos = tl.load(cos_ptr + s * D_HALF + offs_d, mask=dmask, other=0.0).to(COMPUTE)[None, :]
    sin = tl.load(sin_ptr + s * D_HALF + offs_d, mask=dmask, other=0.0).to(COMPUTE)[None, :]
    y1 = n1 * cos - n2 * sin
    y2 = n2 * cos + n1 * sin

    # --- store permuted (primitive: bnhd_to_bhnd): y[b, h, s, :] ---
    y_base = y_ptr + (b * H * S + offs_h[:, None] * S + s) * D
    tl.store(y_base + offs_d[None, :], y1.to(y_ptr.dtype.element_ty), mask=mask)
    tl.store(y_base + (offs_d + D_HALF)[None, :], y2.to(y_ptr.dtype.element_ty), mask=mask)
    if SAVE_RSTD:  # the only saved-for-backward store; skipped when no backward can follow
        tl.store(rstd_ptr + (b * S + s) * H + offs_h, rstd, mask=hmask)


@triton.jit
def _fused_bwd_kernel(
    dy_ptr, x_ptr, w_ptr, cos_ptr, sin_ptr, rstd_ptr, dx_ptr, dw_partial_ptr,
    T, S, H, D_HALF,
    sx_b, sx_s, sx_h,
    sdy_b, sdy_h, sdy_s,
    BLOCK_H: tl.constexpr, BLOCK_DH: tl.constexpr, COMPUTE: tl.constexpr,
):
    # A fixed grid strides over the T = B*S tokens (all heads per step) so dw accumulates in
    # registers; one fp32 partial row per program is reduced on the host.
    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)
    offs_h = tl.arange(0, BLOCK_H).to(tl.int64)
    offs_d = tl.arange(0, BLOCK_DH)
    hmask = offs_h < H
    dmask = offs_d < D_HALF
    mask = hmask[:, None] & dmask[None, :]
    D = 2 * D_HALF
    w1 = tl.load(w_ptr + offs_d, mask=dmask, other=0.0).to(COMPUTE)[None, :]
    w2 = tl.load(w_ptr + offs_d + D_HALF, mask=dmask, other=0.0).to(COMPUTE)[None, :]
    dw1 = tl.zeros([BLOCK_DH], dtype=COMPUTE)
    dw2 = tl.zeros([BLOCK_DH], dtype=COMPUTE)

    for t in range(pid, T, n_programs):
        t = t.to(tl.int64)
        b = t // S
        s = t % S

        # --- gather dy from the permuted (B, H, S, D) layout through its strides: the adjoint of
        # the permuted store, and no .contiguous() when autograd hands us a non-contiguous dy ---
        dy_base = dy_ptr + b * sdy_b + offs_h[:, None] * sdy_h + s * sdy_s
        dy1 = tl.load(dy_base + offs_d[None, :], mask=mask, other=0.0).to(COMPUTE)
        dy2 = tl.load(dy_base + (offs_d + D_HALF)[None, :], mask=mask, other=0.0).to(COMPUTE)

        # --- inverse rotation (adjoint of rope: sin -> -sin) ---
        cos = tl.load(cos_ptr + s * D_HALF + offs_d, mask=dmask, other=0.0).to(COMPUTE)[None, :]
        sin = tl.load(sin_ptr + s * D_HALF + offs_d, mask=dmask, other=0.0).to(COMPUTE)[None, :]
        dn1 = dy1 * cos + dy2 * sin
        dn2 = dy2 * cos - dy1 * sin

        # --- rmsnorm backward with saved rstd ---
        x_base = x_ptr + b * sx_b + s * sx_s + offs_h[:, None] * sx_h
        x1 = tl.load(x_base + offs_d[None, :], mask=mask, other=0.0).to(COMPUTE)
        x2 = tl.load(x_base + (offs_d + D_HALF)[None, :], mask=mask, other=0.0).to(COMPUTE)
        rstd = tl.load(rstd_ptr + t * H + offs_h, mask=hmask, other=0.0)[:, None]
        xhat1 = x1 * rstd
        xhat2 = x2 * rstd
        dxhat1 = dn1 * w1
        dxhat2 = dn2 * w2
        c = (tl.sum(dxhat1 * xhat1, axis=1) + tl.sum(dxhat2 * xhat2, axis=1))[:, None] / D
        dx1 = rstd * (dxhat1 - xhat1 * c)
        dx2 = rstd * (dxhat2 - xhat2 * c)
        dx_base = dx_ptr + ((b * S + s) * H + offs_h[:, None]) * D  # dx is contiguous (B, S, H, D)
        tl.store(dx_base + offs_d[None, :], dx1.to(dx_ptr.dtype.element_ty), mask=mask)
        tl.store(dx_base + (offs_d + D_HALF)[None, :], dx2.to(dx_ptr.dtype.element_ty), mask=mask)
        dw1 += tl.sum(dn1 * xhat1, axis=0)
        dw2 += tl.sum(dn2 * xhat2, axis=0)

    dw_base = dw_partial_ptr + pid.to(tl.int64) * D
    tl.store(dw_base + offs_d, dw1, mask=dmask)
    tl.store(dw_base + offs_d + D_HALF, dw2, mask=dmask)


def _geometry(H: int, D: int) -> Tuple[int, int, int]:
    """(BLOCK_H, BLOCK_DH, num_warps). The backward holds every head of a token in one tile."""
    assert D % 2 == 0, "head dim must be even"
    block_dh = triton.next_power_of_2(D // 2)
    block_h = triton.next_power_of_2(H)
    if block_h * block_dh > 8192:
        raise ValueError(f"rmsnorm_rope_permute: H*D/2 = {H * D // 2} > 8192 needs a head loop in the backward")
    return block_h, block_dh, (4 if block_h * block_dh <= 2048 else 8)


def rmsnorm_rope_permute_fwd(
    x: torch.Tensor, w: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, eps: float, save_rstd: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``x (B, S, H, D)`` any strides -> ``(y (B, H, S, D) contiguous, rstd (B, S, H) fp32)``.

    ``rstd`` is a 0-element placeholder when ``save_rstd`` is false (no backward will follow).
    """
    B, S, H, D = x.shape
    assert x.stride(3) == 1 and w.is_contiguous() and cos.is_contiguous() and sin.is_contiguous()
    block_h, block_dh, warps = _geometry(H, D)
    # forward tiles may be smaller than a full token: split heads to keep ~2K elements per program
    fwd_block_h = max(1, min(block_h, 2048 // block_dh))
    y = torch.empty(B, H, S, D, dtype=x.dtype, device=x.device)
    rstd = torch.empty((B, S, H) if save_rstd else (0,), dtype=torch.float32, device=x.device)
    grid = (B * S, triton.cdiv(H, fwd_block_h))
    _fused_fwd_kernel[grid](
        x, w, cos, sin, y, rstd, S, H, D // 2, eps,
        x.stride(0), x.stride(1), x.stride(2),
        BLOCK_H=fwd_block_h, BLOCK_DH=block_dh, COMPUTE=tl.float32, SAVE_RSTD=save_rstd, num_warps=4,
    )
    return y, rstd


def rmsnorm_rope_permute_bwd(dy: torch.Tensor, x: torch.Tensor, w: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rstd: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """``dy (B, H, S, D)`` -> ``(dx (B, S, H, D) contiguous, dw (D,) in w.dtype)``."""
    B, S, H, D = x.shape
    assert dy.stride(3) == 1, "dy must be contiguous along D; other strides are passed through"
    block_h, block_dh, warps = _geometry(H, D)
    T = B * S
    n_programs = max(1, min(T, 2 * torch.cuda.get_device_properties(x.device).multi_processor_count))
    dx = torch.empty(B, S, H, D, dtype=x.dtype, device=x.device)
    dw_partial = torch.empty(n_programs, D, dtype=torch.float32, device=x.device)
    _fused_bwd_kernel[(n_programs,)](
        dy, x, w, cos, sin, rstd, dx, dw_partial, T, S, H, D // 2,
        x.stride(0), x.stride(1), x.stride(2),
        dy.stride(0), dy.stride(1), dy.stride(2),
        BLOCK_H=block_h, BLOCK_DH=block_dh, COMPUTE=tl.float32, num_warps=warps,
    )
    return dx, dw_partial.sum(0).to(w.dtype)


class _Fused(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, cos, sin, eps, save_rstd):
        y, rstd = rmsnorm_rope_permute_fwd(x, w, cos, sin, eps, save_rstd)
        if save_rstd:
            ctx.save_for_backward(x, w, cos, sin, rstd)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, w, cos, sin, rstd = ctx.saved_tensors
        dx, dw = rmsnorm_rope_permute_bwd(dy, x, w, cos, sin, rstd)
        return dx, dw, None, None, None, None


def rmsnorm_rope_permute(x: torch.Tensor, w: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-head RMSNorm, RoPE (half pairing) and ``(B, S, H, D) -> (B, H, S, D)`` in one kernel. Differentiable in ``x`` and ``w``."""
    # Decided here because grad mode is off inside Function.forward.
    save_rstd = torch.is_grad_enabled() and (x.requires_grad or w.requires_grad)
    return _Fused.apply(x, w, cos, sin, eps, save_rstd)


__all__ = ["rmsnorm_rope_permute", "rmsnorm_rope_permute_fwd", "rmsnorm_rope_permute_bwd"]
