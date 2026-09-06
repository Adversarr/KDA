"""H3 shared RMSNorm, partial 48+48 RoPE and head-major output.

Token/head tiles share rotary tables and weights. The persistent adjoint reduces
heads in registers and writes one bounded fp32 weight partial per program. Only
rstd is saved; normalization and rotary-adjoint bf16 boundaries match eager.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _fwd(X, W, C, SINE, Y, INV, S: tl.constexpr, H: tl.constexpr,
         XB, XS, XH, EPS, SAVE: tl.constexpr, BLOCK_H: tl.constexpr):
    token_id = tl.program_id(0).to(tl.int64)
    batch, token = token_id // S, token_id % S
    head = (tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)).to(tl.int64)
    d, tail = tl.arange(0, 64), tl.arange(0, 32)
    mask = (head[:, None] < H) & (d[None, :] < 48)
    tail_mask = head[:, None] < H
    base = X + batch * XB + token * XS + head[:, None] * XH
    a = tl.load(base + d[None, :], mask, 0).to(tl.float32)
    b = tl.load(base + 48 + d[None, :], mask, 0).to(tl.float32)
    z = tl.load(base + 96 + tail[None, :], tail_mask, 0).to(tl.float32)
    wa = tl.load(W + d, d < 48, 0)[None, :]
    wb = tl.load(W + 48 + d, d < 48, 0)[None, :]
    wz = tl.load(W + 96 + tail)[None, :]
    inv = tl.rsqrt((tl.sum(a*a, 1) + tl.sum(b*b, 1) + tl.sum(z*z, 1)) / 128 + EPS)
    na = (a * inv[:, None] * wa).to(X.dtype.element_ty).to(tl.float32)
    nb = (b * inv[:, None] * wb).to(X.dtype.element_ty).to(tl.float32)
    nz = (z * inv[:, None] * wz).to(X.dtype.element_ty)
    c = tl.load(C + token * 96 + d, d < 48, 0)[None, :]
    s = tl.load(SINE + token * 96 + d, d < 48, 0)[None, :]
    output = Y + (batch * H * S + head[:, None] * S + token) * 128
    tl.store(output + d[None, :], na*c - nb*s, mask)
    tl.store(output + 48 + d[None, :], nb*c + na*s, mask)
    tl.store(output + 96 + tail[None, :], nz, tail_mask)
    if SAVE:
        tl.store(INV + token_id * H + head, inv, head < H)


@triton.jit
def _bwd(DY, X, W, C, SINE, INV, DX, PARTIAL,
         T, S: tl.constexpr, H: tl.constexpr,
         XB, XS, XH, YB, YH, YS, YD,
         BLOCK_H: tl.constexpr):
    pid = tl.program_id(0)
    programs = tl.num_programs(0)
    group = tl.program_id(1)
    head = (group * BLOCK_H + tl.arange(0, BLOCK_H)).to(tl.int64)
    d, tail = tl.arange(0, 64), tl.arange(0, 32)
    mask = (head[:, None] < H) & (d[None, :] < 48)
    tail_mask = head[:, None] < H
    wa = tl.load(W + d, d < 48, 0)[None, :]
    wb = tl.load(W + 48 + d, d < 48, 0)[None, :]
    wz = tl.load(W + 96 + tail)[None, :]
    dwa = tl.zeros((BLOCK_H, 64), tl.float32)
    dwb = tl.zeros((BLOCK_H, 64), tl.float32)
    dwz = tl.zeros((BLOCK_H, 32), tl.float32)
    for token_id in range(pid, T, programs):
        token_id = token_id.to(tl.int64)
        batch, token = token_id // S, token_id % S
        upstream = DY + batch * YB + head[:, None] * YH + token * YS
        da = tl.load(upstream + d[None, :] * YD, mask, 0).to(tl.float32)
        db = tl.load(upstream + (48 + d[None, :]) * YD, mask, 0).to(tl.float32)
        dz = tl.load(upstream + (96 + tail[None, :]) * YD, tail_mask, 0).to(tl.float32)
        c = tl.load(C + token * 96 + d, d < 48, 0)[None, :]
        s = tl.load(SINE + token * 96 + d, d < 48, 0)[None, :]
        # The eager norm output was bf16; its incoming adjoint is rounded there too.
        dna = (da*c + db*s).to(X.dtype.element_ty).to(tl.float32)
        dnb = (db*c - da*s).to(X.dtype.element_ty).to(tl.float32)
        source = X + batch * XB + token * XS + head[:, None] * XH
        a = tl.load(source + d[None, :], mask, 0).to(tl.float32)
        b = tl.load(source + 48 + d[None, :], mask, 0).to(tl.float32)
        z = tl.load(source + 96 + tail[None, :], tail_mask, 0).to(tl.float32)
        inv = tl.load(INV + token_id * H + head, head < H, 0)[:, None]
        a, b, z = a*inv, b*inv, z*inv
        ga, gb, gz = dna*wa, dnb*wb, dz*wz
        projection = (tl.sum(ga*a, 1) + tl.sum(gb*b, 1) + tl.sum(gz*z, 1))[:, None] / 128
        output = DX + (token_id * H + head[:, None]) * 128
        tl.store(output + d[None, :], inv * (ga - a*projection), mask)
        tl.store(output + 48 + d[None, :], inv * (gb - b*projection), mask)
        tl.store(output + 96 + tail[None, :], inv * (gz - z*projection), tail_mask)
        dwa += dna*a
        dwb += dnb*b
        dwz += dz*z
    partial = PARTIAL + (group.to(tl.int64) * programs + pid) * 128
    tl.store(partial + d, tl.sum(dwa, 0), d < 48)
    tl.store(partial + 48 + d, tl.sum(dwb, 0), d < 48)
    tl.store(partial + 96 + tail, tl.sum(dwz, 0))


def _geometry(tokens: int, heads: int, device: torch.device):
    """Bound shared-gradient partials while covering arbitrary head counts."""
    block_h = min(triton.next_power_of_2(max(heads, 1)), 8)
    programs = max(1, min(tokens, 2 * torch.cuda.get_device_properties(device).multi_processor_count))
    return block_h, programs, triton.cdiv(heads, block_h)


def _forward(x, w, cos, sin, eps, save):
    """One token and up to sixteen heads share each forward program."""
    batch, tokens, heads, _ = x.shape
    output = torch.empty((batch, heads, tokens, 128), device=x.device, dtype=x.dtype)
    inv = torch.empty((batch, tokens, heads) if save else (0,), device=x.device, dtype=torch.float32)
    if x.numel():
        block_h = min(triton.next_power_of_2(heads), 16)
        _fwd[(batch*tokens, triton.cdiv(heads, block_h))](
            x, w, cos, sin, output, inv, tokens, heads, *x.stride()[:3], eps, save,
            block_h, num_warps=4)
    return output, inv


def _backward(dy, x, w, cos, sin, inv):
    """Read every upstream stride, including autograd's zero-stride expanded sums."""
    batch, tokens, heads, _ = x.shape
    dx = torch.empty_like(x, memory_format=torch.contiguous_format)
    if not x.numel():
        return dx, torch.zeros_like(w)
    block_h, programs, groups = _geometry(batch*tokens, heads, x.device)
    partial = torch.empty((groups*programs, 128), device=x.device, dtype=torch.float32)
    _bwd[(programs, groups)](dy, x, w, cos, sin, inv, dx, partial,
        batch*tokens, tokens, heads, *x.stride()[:3], *dy.stride(),
        block_h, num_warps=8 if block_h > 16 else 4)
    return dx, partial


@triton.jit
def _reduce_pair(PQ, PK, DQ, DK, P: tl.constexpr, BP: tl.constexpr):
    cols = tl.program_id(0) * 16 + tl.arange(0, 16)
    rows = tl.arange(0, BP)
    offset = rows[:, None] * 128 + cols[None, :]
    mask = rows[:, None] < P
    tl.store(DQ + cols, tl.sum(tl.load(PQ + offset, mask, 0), 0))
    tl.store(DK + cols, tl.sum(tl.load(PK + offset, mask, 0), 0))


class _H3Pair(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, qw, kw, cos, sin, eps, save):
        yq, iq = _forward(q, qw, cos, sin, eps, save)
        yk, ik = _forward(k, kw, cos, sin, eps, save)
        if save:
            ctx.save_for_backward(q, k, qw, kw, cos, sin, iq, ik)
        return yq, yk

    @staticmethod
    def backward(ctx, dyq, dyk):
        q, k, qw, kw, cos, sin, iq, ik = ctx.saved_tensors
        if not q.numel():
            return torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(qw), torch.zeros_like(kw), None, None, None, None
        dxq, pq = _backward(dyq, q, qw, cos, sin, iq)
        dxk, pk = _backward(dyk, k, kw, cos, sin, ik)
        dwq, dwk = torch.empty_like(qw), torch.empty_like(kw)
        _reduce_pair[(8,)](pq, pk, dwq, dwk, pq.shape[0], triton.next_power_of_2(pq.shape[0]), num_warps=4)
        return dxq, dxk, dwq, dwk, None, None, None, None


@triton.jit
def _small_pair(Q, K, WQ, WK, C, SINE, YQ, YK, RQ, RK, GQ, GK, DQ, DK, DWQ, DWK,
                S: tl.constexpr, H: tl.constexpr, D: tl.constexpr, ROWS: tl.constexpr,
                QB, QS, QH, KB, KS, KH, GQB, GQH, GQS, GQD, GKB, GKH, GKS, GKD,
                EPS, BACKWARD: tl.constexpr, SAVE: tl.constexpr,
                HALF: tl.constexpr, BN: tl.constexpr, BC: tl.constexpr):
    """One fused q/k launch; token rows own disjoint final weight columns."""
    row, which = tl.program_id(0).to(tl.int64), tl.program_id(1)
    X, W = tl.where(which == 0, Q, K), tl.where(which == 0, WQ, WK)
    RS = tl.where(which == 0, RQ, RK)
    xb, xs, xh = tl.where(which == 0, QB, KB), tl.where(which == 0, QS, KS), tl.where(which == 0, QH, KH)
    batch, token, head = row // (S*H), (row//H)%S, row%H
    d = tl.arange(0, 128)
    partner = tl.where(d < 48, d + 48, tl.where(d < 96, d - 48, d))
    base = X + batch*xb + token*xs + head*xh
    x = tl.load(base + d).to(tl.float32)
    w = tl.load(W + d)
    c = tl.load(C + token*96 + d%48, d < 96, 1.)
    sn = tl.load(SINE + token*96 + d%48, d < 96, 0.)
    if BACKWARD:
        G = tl.where(which == 0, GQ, GK)
        DX, DW = tl.where(which == 0, DQ, DK), tl.where(which == 0, DWQ, DWK)
        gb, gh, gs, gd = tl.where(which==0,GQB,GKB),tl.where(which==0,GQH,GKH),tl.where(which==0,GQS,GKS),tl.where(which==0,GQD,GKD)
        gbase = G + batch*gb + head*gh + token*gs
        dy = tl.load(gbase + d*gd).to(tl.float32)
        dp = tl.load(gbase + partner*gd).to(tl.float32)
        dn = (dy*c + tl.where(d<48,1.,-1.)*dp*sn).to(X.dtype.element_ty).to(tl.float32)
        inv = tl.load(RS + row)
        n, g = x*inv, dn*w
        tl.store(DX + row*128+d, inv*(g - n*tl.sum(g*n,0)/128))
        cols = row*BC + tl.arange(0,BC)
        rr = tl.arange(0,BN).to(tl.int64)
        bb, tt, hh = rr//(S*H),(rr//H)%S,rr%H
        mask = (rr[:,None]<ROWS)&(cols[None,:]<128)
        partner = tl.where(cols<48,cols+48,tl.where(cols<96,cols-48,cols))
        gx = G+bb[:,None]*gb+hh[:,None]*gh+tt[:,None]*gs
        dc = tl.load(gx+cols[None,:]*gd,mask,0).to(tl.float32)
        dp = tl.load(gx+partner[None,:]*gd,mask,0).to(tl.float32)
        rotary = (rr[:,None]<ROWS)&(cols[None,:]<96)
        cc = tl.load(C+tt[:,None]*96+(cols%48)[None,:],rotary,1.)
        ss = tl.load(SINE+tt[:,None]*96+(cols%48)[None,:],rotary,0.)
        dn = (dc*cc+tl.where(cols<48,1.,-1.)[None,:]*dp*ss).to(X.dtype.element_ty).to(tl.float32)
        xx = tl.load(X+bb[:,None]*xb+tt[:,None]*xs+hh[:,None]*xh+cols[None,:],mask,0).to(tl.float32)
        ri = tl.load(RS+rr,rr<ROWS,0)
        tl.store(DW+cols,tl.sum(dn*xx*ri[:,None],0),cols<128)
    else:
        inv = tl.rsqrt(tl.sum(x*x,0)/128+EPS)
        n = (x*inv*w).to(X.dtype.element_ty).to(tl.float32)
        xp = tl.load(base+partner).to(tl.float32)
        wp = tl.load(W+partner)
        np = (xp*inv*wp).to(X.dtype.element_ty).to(tl.float32)
        Y = tl.where(which == 0, YQ, YK)
        tl.store(Y+(batch*H*S+head*S+token)*128+d,n*c+tl.where(d<48,-1.,1.)*np*sn)
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
    """Shared RMSNorm, partial 96-channel MM-RoPE and head-major output."""
    if q.ndim != 4 or k.shape != q.shape or q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16:
        raise ValueError("q/k must be matching bf16 (B,S,H,128)")
    if not q.is_cuda or any(t.device != q.device for t in (k,q_norm_w,k_norm_w,cos,sin)):
        raise ValueError("all inputs must share one CUDA device")
    if q.stride(-1) != 1 or k.stride(-1) != 1:
        raise ValueError("q/k need unit last stride")
    if any(w.shape != (128,) or w.dtype != torch.float32 or not w.is_contiguous() for w in (q_norm_w,k_norm_w)):
        raise ValueError("shared weights must be contiguous fp32 (128,)")
    if cos.shape != (q.shape[1],96) or sin.shape != cos.shape or any(t.dtype != torch.float32 or not t.is_contiguous() for t in (cos,sin)):
        raise ValueError("rotary tables must be contiguous fp32 (S,96)")
    if q.shape[-1] != 128 or k.shape[-1] != 128:
        raise ValueError("H3 partial MM-RoPE requires head dimension 128")
    rows = q.numel() // 128
    if 0 < rows <= 128:
        save = torch.is_grad_enabled() and any(t.requires_grad for t in (q,k,q_norm_w,k_norm_w))
        return _SmallPair.apply(q,k,q_norm_w,k_norm_w,cos,sin,eps,save)
    save_q = torch.is_grad_enabled() and (q.requires_grad or q_norm_w.requires_grad)
    save_k = torch.is_grad_enabled() and (k.requires_grad or k_norm_w.requires_grad)
    return _H3Pair.apply(q, k, q_norm_w, k_norm_w, cos, sin, eps, save_q or save_k)
