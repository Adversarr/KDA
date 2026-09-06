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
    n1 = (x1 * rstd[:, None] * w1).to(x_ptr.dtype.element_ty).to(COMPUTE)
    n2 = (x2 * rstd[:, None] * w2).to(x_ptr.dtype.element_ty).to(COMPUTE)

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
    sdy_b, sdy_h, sdy_s, sdy_d,
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
        dy1 = tl.load(dy_base + offs_d[None, :] * sdy_d, mask=mask, other=0.0).to(COMPUTE)
        dy2 = tl.load(dy_base + (offs_d + D_HALF)[None, :] * sdy_d, mask=mask, other=0.0).to(COMPUTE)

        # --- inverse rotation (adjoint of rope: sin -> -sin) ---
        cos = tl.load(cos_ptr + s * D_HALF + offs_d, mask=dmask, other=0.0).to(COMPUTE)[None, :]
        sin = tl.load(sin_ptr + s * D_HALF + offs_d, mask=dmask, other=0.0).to(COMPUTE)[None, :]
        dn1 = (dy1 * cos + dy2 * sin).to(x_ptr.dtype.element_ty).to(COMPUTE)
        dn2 = (dy2 * cos - dy1 * sin).to(x_ptr.dtype.element_ty).to(COMPUTE)

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
    if B * S * H == 0:
        return y, rstd
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

    block_h, block_dh, warps = _geometry(H, D)
    T = B * S
    n_programs = max(1, min(T, 2 * torch.cuda.get_device_properties(x.device).multi_processor_count))
    dx = torch.empty(B, S, H, D, dtype=x.dtype, device=x.device)
    if T * H == 0:
        return dx, torch.zeros_like(w)
    dw_partial = torch.empty(n_programs, D, dtype=torch.float32, device=x.device)
    _fused_bwd_kernel[(n_programs,)](
        dy, x, w, cos, sin, rstd, dx, dw_partial, T, S, H, D // 2,
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
    """Per-head RMSNorm, RoPE (half pairing) and ``(B, S, H, D) -> (B, H, S, D)`` in one kernel. Differentiable in ``x`` and ``w``."""
    # Decided here because grad mode is off inside Function.forward.
    save_rstd = torch.is_grad_enabled() and (x.requires_grad or w.requires_grad)
    return _Fused.apply(x, w, cos, sin, eps, save_rstd)


__all__ = ["rmsnorm_rope_permute", "rmsnorm_rope_permute_fwd", "rmsnorm_rope_permute_bwd"]



@triton.jit
def _small_pair(Q, K, WQ, WK, C, SINE, YQ, YK, RQ, RK, GQ, GK, DQ, DK, DWQ, DWK,
                S: tl.constexpr, H: tl.constexpr, D: tl.constexpr, ROWS: tl.constexpr,
                QB, QS, QH, KB, KS, KH, GQB, GQH, GQS, GQD, GKB, GKH, GKS, GKD,
                EPS, BACKWARD: tl.constexpr, SAVE: tl.constexpr,
                HALF: tl.constexpr, BN: tl.constexpr, BC: tl.constexpr):
    """Fuse q/k launches; each CTA owns one head row and final weight columns."""
    row = tl.program_id(0).to(tl.int64)
    which = tl.program_id(1)
    X = tl.where(which == 0, Q, K)
    W = tl.where(which == 0, WQ, WK)
    RS = tl.where(which == 0, RQ, RK)
    xb, xs, xh = tl.where(which == 0, QB, KB), tl.where(which == 0, QS, KS), tl.where(which == 0, QH, KH)
    b, token, head = row // (S*H), (row//H)%S, row%H
    d = tl.arange(0, HALF)
    base = X + b*xb + token*xs + head*xh
    x1 = tl.load(base+d).to(tl.float32)
    x2 = tl.load(base+d+HALF).to(tl.float32)
    w1 = tl.load(W+d).to(tl.float32)
    w2 = tl.load(W+d+HALF).to(tl.float32)
    c = tl.load(C+token*HALF+d)
    sn = tl.load(SINE+token*HALF+d)
    if BACKWARD:
        G = tl.where(which == 0, GQ, GK)
        DX = tl.where(which == 0, DQ, DK)
        DW = tl.where(which == 0, DWQ, DWK)
        gb, gh, gs, gd = tl.where(which==0,GQB,GKB),tl.where(which==0,GQH,GKH),tl.where(which==0,GQS,GKS),tl.where(which==0,GQD,GKD)
        gbase = G+b*gb+head*gh+token*gs
        g1 = tl.load(gbase+d*gd).to(tl.float32)
        g2 = tl.load(gbase+(d+HALF)*gd).to(tl.float32)
        dn1 = (g1*c+g2*sn).to(X.dtype.element_ty).to(tl.float32)
        dn2 = (g2*c-g1*sn).to(X.dtype.element_ty).to(tl.float32)
        inv = tl.load(RS+row)
        n1, n2 = x1*inv, x2*inv
        a1, a2 = dn1*w1, dn2*w2
        proj = (tl.sum(a1*n1,0)+tl.sum(a2*n2,0))/D
        tl.store(DX+row*D+d,inv*(a1-n1*proj))
        tl.store(DX+row*D+d+HALF,inv*(a2-n2*proj))
        # Disjoint parameter columns remove both partial storage and reduction launches.
        cols = row*BC+tl.arange(0,BC)
        rr = tl.arange(0,BN).to(tl.int64)
        bb, tt, hh = rr//(S*H),(rr//H)%S,rr%H
        mask = (rr[:,None]<ROWS)&(cols[None,:]<D)
        partner = tl.where(cols<HALF,cols+HALF,cols-HALF)
        gx = G+bb[:,None]*gb+hh[:,None]*gh+tt[:,None]*gs
        dc = tl.load(gx+cols[None,:]*gd,mask,0).to(tl.float32)
        dp = tl.load(gx+partner[None,:]*gd,mask,0).to(tl.float32)
        cc = tl.load(C+tt[:,None]*HALF+(cols%HALF)[None,:])
        ss = tl.load(SINE+tt[:,None]*HALF+(cols%HALF)[None,:])
        dn = (dc*cc+tl.where(cols<HALF,1.,-1.)[None,:]*dp*ss).to(X.dtype.element_ty).to(tl.float32)
        xx = tl.load(X+bb[:,None]*xb+tt[:,None]*xs+hh[:,None]*xh+cols[None,:],mask,0).to(tl.float32)
        ri = tl.load(RS+rr,rr<ROWS,0)
        tl.store(DW+cols,tl.sum(dn*xx*ri[:,None],0),cols<D)
    else:
        inv = tl.rsqrt((tl.sum(x1*x1,0)+tl.sum(x2*x2,0))/D+EPS)
        n1 = (x1*inv*w1).to(X.dtype.element_ty).to(tl.float32)
        n2 = (x2*inv*w2).to(X.dtype.element_ty).to(tl.float32)
        Y = tl.where(which == 0, YQ, YK)
        out = Y+(b*H*S+head*S+token)*D
        tl.store(out+d,n1*c-n2*sn)
        tl.store(out+d+HALF,n2*c+n1*sn)
        if SAVE:
            tl.store(RS+row,inv)


class _SmallPair(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,qw,kw,cos,sin,eps,save):
        b,s,h,d=q.shape
        rows=b*s*h
        yq=torch.empty((b,h,s,d),dtype=q.dtype,device=q.device)
        yk=torch.empty_like(yq)
        rq=torch.empty(rows if save else 0,dtype=torch.float32,device=q.device)
        rk=torch.empty_like(rq)
        _small_pair[(rows,2)](q,k,qw,kw,cos,sin,yq,yk,rq,rk,q,k,q,k,qw,kw,
            s,h,d,rows,*q.stride()[:3],*k.stride()[:3],*q.stride(),*k.stride(),
            eps,False,save,d//2,triton.next_power_of_2(rows),triton.next_power_of_2(triton.cdiv(d,rows)),num_warps=4)
        if save:
            ctx.save_for_backward(q,k,qw,kw,cos,sin,rq,rk)
        return yq,yk

    @staticmethod
    def backward(ctx,gq,gk):
        q,k,qw,kw,cos,sin,rq,rk=ctx.saved_tensors
        b,s,h,d=q.shape
        rows=b*s*h
        dq=torch.empty(q.shape,dtype=q.dtype,device=q.device)
        dk=torch.empty_like(dq)
        dwq,dwk=torch.empty_like(qw),torch.empty_like(kw)
        _small_pair[(rows,2)](q,k,qw,kw,cos,sin,q,k,rq,rk,gq,gk,dq,dk,dwq,dwk,
            s,h,d,rows,*q.stride()[:3],*k.stride()[:3],*gq.stride(),*gk.stride(),
            0.,True,True,d//2,triton.next_power_of_2(rows),triton.next_power_of_2(triton.cdiv(d,rows)),num_warps=4)
        return dq,dk,dwq,dwk,None,None,None,None

def qk_prep(q, k, q_norm_w, k_norm_w, cos, sin, eps=1e-6):
    """Prepare q and k with shared head normalization and half-pairing RoPE."""
    if q.ndim != 4 or k.shape != q.shape or q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16:
        raise ValueError("q and k must be matching bf16 (B,S,H,D) tensors")
    if not q.is_cuda or any(t.device != q.device for t in (k, q_norm_w, k_norm_w, cos, sin)):
        raise ValueError("all tensors must share one CUDA device")
    d = q.shape[-1]
    if d < 2 or d % 2 or min(q.stride(-1), k.stride(-1)) != 1:
        raise ValueError("head dimensions must be even with unit last stride")
    if any(w.shape != (d,) or w.dtype != torch.float32 or not w.is_contiguous() for w in (q_norm_w, k_norm_w)):
        raise ValueError("norm weights must be fp32 (D,)")
    if cos.shape != (q.shape[1], d//2) or sin.shape != cos.shape or cos.dtype != torch.float32 or sin.dtype != torch.float32 or not cos.is_contiguous() or not sin.is_contiguous():
        raise ValueError("rotary tables must be fp32 (S,D/2)")
    rows=q.numel()//d
    if 0 < rows <= 128 and d == 128:
        save=torch.is_grad_enabled() and any(t.requires_grad for t in (q,k,q_norm_w,k_norm_w))
        return _SmallPair.apply(q,k,q_norm_w,k_norm_w,cos,sin,eps,save)
    return rmsnorm_rope_permute(q, q_norm_w, cos, sin, eps), rmsnorm_rope_permute(k, k_norm_w, cos, sin, eps)
