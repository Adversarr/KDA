"""Golden kernel for ``qk_prep``: per-head RMSNorm -> interleaved RoPE -> permuted store, fwd + bwd.

The `fusion-exemplar/rmsnorm_rope_permute` reference with two changes the user's model needs:

* the norm weight is ``(H, D)``, one row per head (Qwen3 q_norm / k_norm per head), so the
  weight tile is ``[BLOCK_H, D]`` and its gradient is accumulated per head;
* RoPE pairs interleaved lanes ``(x[2i], x[2i+1])`` (GPT-J style), so the row is loaded whole
  and split in registers (``tl.split`` / ``tl.join``), as the rope primitive does; indexing
  memory with ``2 * offs_d`` would defeat vectorisation.

``qk_prep`` launches the same kernel twice (q with ``Hq`` heads, k with ``Hk``): the two tensors
are views into one fused ``qkv`` buffer (row stride ``(Hq + 2 Hk) * D``), passed through their
strides. Saved for the backward: ``rstd (B, S, H)`` fp32 per tensor.
"""

from typing import Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _rotate(x, cos, sin, BLOCK_H: tl.constexpr, BLOCK_DH: tl.constexpr):
    # Interleaved RoPE on a [BLOCK_H, 2 * BLOCK_DH] fp32 tile: even lanes pair with odd lanes.
    x1, x2 = tl.split(tl.reshape(x, [BLOCK_H, BLOCK_DH, 2]))
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return tl.reshape(tl.join(y1, y2), [BLOCK_H, 2 * BLOCK_DH])


@triton.jit
def _fused_fwd_kernel(
    x_ptr, w_ptr, cos_ptr, sin_ptr, y_ptr, rstd_ptr,
    S, H, D, eps,
    sx_b, sx_s, sx_h,
    BLOCK_H: tl.constexpr, BLOCK_DH: tl.constexpr, COMPUTE: tl.constexpr, SAVE_RSTD: tl.constexpr,
):
    # One program per (b, s) and a block of heads; the cos/sin row of position s is loaded once.
    pid_bs = tl.program_id(0).to(tl.int64)
    b = pid_bs // S
    s = pid_bs % S
    offs_h = (tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)).to(tl.int64)
    offs_d = tl.arange(0, 2 * BLOCK_DH)
    offs_p = tl.arange(0, BLOCK_DH)
    hmask = offs_h < H
    mask = hmask[:, None] & (offs_d < D)[None, :]
    pmask = offs_p < D // 2

    # --- rmsnorm over the whole row, weight row per head ---
    x = tl.load(x_ptr + b * sx_b + s * sx_s + offs_h[:, None] * sx_h + offs_d[None, :], mask=mask, other=0.0).to(COMPUTE)
    w = tl.load(w_ptr + offs_h[:, None] * D + offs_d[None, :], mask=mask, other=0.0).to(COMPUTE)
    rstd = tl.rsqrt(tl.sum(x * x, axis=1) / D + eps)  # [BLOCK_H]
    # Preserve the eager normalization output boundary before rotary arithmetic.
    n = (x * rstd[:, None] * w).to(x_ptr.dtype.element_ty).to(COMPUTE)

    # --- interleaved rope in registers ---
    cos = tl.load(cos_ptr + s * (D // 2) + offs_p, mask=pmask, other=0.0).to(COMPUTE)[None, :]
    sin = tl.load(sin_ptr + s * (D // 2) + offs_p, mask=pmask, other=0.0).to(COMPUTE)[None, :]
    y = _rotate(n, cos, sin, BLOCK_H, BLOCK_DH)

    # --- permuted store: y[b, h, s, :] ---
    tl.store(y_ptr + (b * H * S + offs_h[:, None] * S + s) * D + offs_d[None, :], y.to(y_ptr.dtype.element_ty), mask=mask)
    if SAVE_RSTD:
        tl.store(rstd_ptr + (b * S + s) * H + offs_h, rstd, mask=hmask)


@triton.jit
def _fused_bwd_kernel(
    dy_ptr, x_ptr, w_ptr, cos_ptr, sin_ptr, rstd_ptr, dx_ptr, dw_partial_ptr,
    T, S, H, D,
    sx_b, sx_s, sx_h,
    sdy_b, sdy_h, sdy_s, sdy_d,
    BLOCK_H: tl.constexpr, BLOCK_DH: tl.constexpr, COMPUTE: tl.constexpr,
):
    # A fixed grid strides over the T = B*S tokens, every head of a token per step, so the
    # per-head dw [H, D] accumulates in registers; one fp32 partial per program, reduced on the host.
    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)
    offs_h = tl.arange(0, BLOCK_H).to(tl.int64)
    offs_d = tl.arange(0, 2 * BLOCK_DH)
    offs_p = tl.arange(0, BLOCK_DH)
    hmask = offs_h < H
    mask = hmask[:, None] & (offs_d < D)[None, :]
    pmask = offs_p < D // 2
    w = tl.load(w_ptr + offs_h[:, None] * D + offs_d[None, :], mask=mask, other=0.0).to(COMPUTE)
    dw = tl.zeros([BLOCK_H, 2 * BLOCK_DH], dtype=COMPUTE)

    for t in range(pid, T, n_programs):
        t = t.to(tl.int64)
        b = t // S
        s = t % S
        # adjoint of the permuted store: gather dy from (B, H, S, D) through its strides
        dy = tl.load(dy_ptr + b * sdy_b + offs_h[:, None] * sdy_h + s * sdy_s + offs_d[None, :] * sdy_d, mask=mask, other=0.0).to(COMPUTE)
        # adjoint of rope: rotate by -angle
        cos = tl.load(cos_ptr + s * (D // 2) + offs_p, mask=pmask, other=0.0).to(COMPUTE)[None, :]
        sin = tl.load(sin_ptr + s * (D // 2) + offs_p, mask=pmask, other=0.0).to(COMPUTE)[None, :]
        dn = _rotate(dy, cos, -sin, BLOCK_H, BLOCK_DH).to(x_ptr.dtype.element_ty).to(COMPUTE)
        # rmsnorm backward with the saved rstd
        x = tl.load(x_ptr + b * sx_b + s * sx_s + offs_h[:, None] * sx_h + offs_d[None, :], mask=mask, other=0.0).to(COMPUTE)
        rstd = tl.load(rstd_ptr + t * H + offs_h, mask=hmask, other=0.0)[:, None]
        xhat = x * rstd
        dxhat = dn * w
        c = tl.sum(dxhat * xhat, axis=1)[:, None] / D
        dx = rstd * (dxhat - xhat * c)
        tl.store(dx_ptr + ((b * S + s) * H + offs_h[:, None]) * D + offs_d[None, :], dx.to(dx_ptr.dtype.element_ty), mask=mask)
        dw += dn * xhat

    tl.store(dw_partial_ptr + (pid.to(tl.int64) * H + offs_h[:, None]) * D + offs_d[None, :], dw, mask=mask)


@triton.jit
def _small_bwd_kernel(
    DY, X, W, COS, SIN, RSTD, DX, DW,
    T: tl.constexpr, S: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
    XB, XS, XH, YB, YH, YS, YD,
    BT: tl.constexpr, BDH: tl.constexpr,
):
    # One head owns every short-sequence token and writes its final weight gradient.
    head = tl.program_id(0).to(tl.int64)
    token = tl.arange(0, BT).to(tl.int64)
    channel = tl.arange(0, 2 * BDH)
    pair = tl.arange(0, BDH)
    batch, pos = token // S, token % S
    mask = (token[:, None] < T) & (channel[None, :] < D)
    dy = tl.load(DY + batch[:, None] * YB + head * YH + pos[:, None] * YS + channel[None, :] * YD, mask, 0).to(tl.float32)
    pmask = (token[:, None] < T) & (pair[None, :] < D // 2)
    cos = tl.load(COS + pos[:, None] * (D // 2) + pair[None, :], pmask, 0)
    sin = tl.load(SIN + pos[:, None] * (D // 2) + pair[None, :], pmask, 0)
    dn = _rotate(dy, cos, -sin, BT, BDH).to(X.dtype.element_ty).to(tl.float32)
    x = tl.load(X + batch[:, None] * XB + pos[:, None] * XS + head * XH + channel[None, :], mask, 0).to(tl.float32)
    inv = tl.load(RSTD + token * H + head, token < T, 0)
    w = tl.load(W + head * D + channel, channel < D, 0).to(tl.float32)
    normalized = x * inv[:, None]
    dxhat = dn * w[None, :]
    dot = tl.sum(dxhat * normalized, 1) / D
    dx = inv[:, None] * (dxhat - normalized * dot[:, None])
    tl.store(DX + (token[:, None] * H + head) * D + channel[None, :], dx, mask)
    dw = tl.sum(tl.where(mask, dn * normalized, 0.), 0)
    tl.store(DW + head * D + channel, dw, channel < D)


def _geometry(H: int, D: int) -> Tuple[int, int, int]:
    """(BLOCK_H, BLOCK_DH, num_warps). The backward holds every head of a token, and its dw, in one tile."""
    assert D % 2 == 0, "head dim must be even"
    block_dh = triton.next_power_of_2(D // 2)
    block_h = triton.next_power_of_2(H)
    if block_h * block_dh > 4096:
        raise ValueError(f"qk_prep: H*D/2 = {H * D // 2} > 4096 needs a head loop in the backward")
    return block_h, block_dh, (4 if block_h * block_dh <= 1024 else 8)


def _num_sms(device: torch.device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


def rmsnorm_rope_permute_fwd(
    x: torch.Tensor, w: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, eps: float, save_rstd: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``x (B, S, H, D)`` any strides, ``w (H, D)`` -> ``(y (B, H, S, D) contiguous, rstd (B, S, H) fp32)``.

    ``rstd`` is a 0-element placeholder when ``save_rstd`` is false (no backward will follow).
    """
    B, S, H, D = x.shape
    assert x.stride(3) == 1 and w.shape == (H, D) and w.is_contiguous() and cos.is_contiguous() and sin.is_contiguous()
    block_h, block_dh, _ = _geometry(H, D)
    fwd_block_h = max(1, min(block_h, 1024 // block_dh))  # ~2K-element tiles: split the heads of a token
    y = torch.empty(B, H, S, D, dtype=x.dtype, device=x.device)
    rstd = torch.empty((B, S, H) if save_rstd else (0,), dtype=torch.float32, device=x.device)
    _fused_fwd_kernel[(B * S, triton.cdiv(H, fwd_block_h))](
        x, w, cos, sin, y, rstd, S, H, D, eps,
        x.stride(0), x.stride(1), x.stride(2),
        BLOCK_H=fwd_block_h, BLOCK_DH=block_dh, COMPUTE=tl.float32, SAVE_RSTD=save_rstd, num_warps=4,
    )
    return y, rstd


def rmsnorm_rope_permute_bwd(
    dy: torch.Tensor, x: torch.Tensor, w: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rstd: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``dy (B, H, S, D)`` any strides, including expanded scalar gradients -> ``(dx (B, S, H, D) contiguous, dw (H, D) in w.dtype)``."""
    B, S, H, D = x.shape
    block_h, block_dh, warps = _geometry(H, D)
    T = B * S
    n_programs = max(1, min(T, 2 * _num_sms(x.device)))
    dx = torch.empty(B, S, H, D, dtype=x.dtype, device=x.device)
    if 0 < T <= 32:
        dw = torch.empty_like(w)
        _small_bwd_kernel[(H,)](
            dy, x, w, cos, sin, rstd, dx, dw, T, S, H, D,
            *x.stride()[:3], *dy.stride(),
            BT=triton.next_power_of_2(T), BDH=block_dh, num_warps=4,
        )
        return dx, dw
    dw_partial = torch.empty(n_programs, H, D, dtype=torch.float32, device=x.device)
    _fused_bwd_kernel[(n_programs,)](
        dy, x, w, cos, sin, rstd, dx, dw_partial, T, S, H, D,
        x.stride(0), x.stride(1), x.stride(2),
        dy.stride(0), dy.stride(1), dy.stride(2), dy.stride(3),
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
    """Per-head RMSNorm (``w (H, D)``), interleaved RoPE and ``(B, S, H, D) -> (B, H, S, D)`` in one kernel."""
    save_rstd = torch.is_grad_enabled() and (x.requires_grad or w.requires_grad)  # grad mode is off inside Function.forward
    return _Fused.apply(x, w, cos, sin, eps, save_rstd)


def qk_prep(
    q: torch.Tensor, k: torch.Tensor, q_norm_w: torch.Tensor, k_norm_w: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, eps: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Drop-in for ``minilm/attention.py::qk_prep``: one launch for q (``Hq`` heads) and one for k (``Hk``)."""
    return rmsnorm_rope_permute(q, q_norm_w, cos, sin, eps), rmsnorm_rope_permute(k, k_norm_w, cos, sin, eps)


__all__ = ["qk_prep", "rmsnorm_rope_permute", "rmsnorm_rope_permute_fwd", "rmsnorm_rope_permute_bwd"]
