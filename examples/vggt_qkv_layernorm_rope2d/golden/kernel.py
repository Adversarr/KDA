"""Standalone fused head normalization, rotary embedding and permuted store.

The forward reads projection rows directly. Backward owns disjoint input rows and
accumulates shared norm parameter gradients into bounded deterministic partials.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _rotation(values, POS, FREQ, batch, token, S: tl.constexpr, D: tl.constexpr,
              ADJOINT: tl.constexpr, HEADS: tl.constexpr, BLOCK: tl.constexpr):
    # Positions and trig are computed once per token, then broadcast over heads.
    d = tl.arange(0, BLOCK)
    quarter, axis_width = D // 4, D // 2
    partner = (d // axis_width) * axis_width + (d + quarter) % axis_width
    py = tl.load(POS + (batch * S + token) * 2)
    px = tl.load(POS + (batch * S + token) * 2 + 1)
    position = tl.where(d < axis_width, py, px).to(tl.float32)
    freq = tl.load(FREQ + d % quarter, d < D, 0)
    angle = position * freq
    c, sn = tl.cos(angle), tl.sin(angle)
    sign = tl.where(d % axis_width < quarter, -1., 1.)
    if ADJOINT:
        sign = -sign
    other = tl.gather(values, tl.broadcast_to(partner[None, :], (HEADS, BLOCK)), 1)
    return values * c[None, :] + other * (sign * sn)[None, :], c, sn


@triton.jit
def _fwd(ROT, V, VO, COPY_V: tl.constexpr, X, W, BIAS, POS, FREQ, Y, MEAN, INV,
         S: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
         XB, XS, XH, EPS, SAVE: tl.constexpr, HEADS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    batch, token = row // S, row % S
    h, d = tl.arange(0, HEADS), tl.arange(0, BLOCK)
    mask = (h[:, None] < H) & (d[None, :] < D)
    x = tl.load(X + batch * XB + token * XS + h[:, None] * XH + d[None, :], mask, 0).to(tl.float32)
    mean = tl.sum(x, 1) / D
    center = tl.where(d[None, :] < D, x - mean[:, None], 0.)
    inv = tl.rsqrt(tl.sum(center * center, 1) / D + EPS)
    w = tl.load(W + d, d < D, 0).to(tl.float32)
    bias = tl.load(BIAS + d, d < D, 0).to(tl.float32)
    n = ((x - mean[:, None]) * inv[:, None] * w[None, :] + bias[None, :]).to(X.dtype.element_ty).to(tl.float32)
    y, cosine, sine = _rotation(n, POS, FREQ, batch, token, S, D, False, HEADS, BLOCK)
    tl.store(Y + (batch * H * S + h[:, None].to(tl.int64) * S + token) * D + d[None, :], y, mask)
    if COPY_V:
        value = tl.load(V + batch * XB + token * XS + h[:, None] * XH + d[None, :], mask, 0)
        tl.store(VO + (batch * H * S + h[:, None].to(tl.int64) * S + token) * D + d[None, :], value, mask)
    if SAVE and COPY_V:
        # Save the two unique 16-channel axis tables once for both adjoints.
        unique = d % 32 < 16
        offset = row * 64 + (d // 32) * 32 + d % 16
        tl.store(ROT + offset, cosine, unique)
        tl.store(ROT + offset + 16, sine, unique)
    if SAVE:
        tl.store(MEAN + row * H + h, mean, h < H)
        tl.store(INV + row * H + h, inv, h < H)


@triton.jit
def _bwd(ROT, DV, DXV, COPY_V: tl.constexpr, VB, VH, VS, VD, DY, X, W, POS, FREQ, MEAN, INV, DX, PW, PB,
         TOKENS, S: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
         XB, XS, XH, YB, YH, YS, YD, DB, DS, DH,
         HEADS: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    programs = tl.num_programs(0)
    h, d = tl.arange(0, HEADS), tl.arange(0, 16)
    mask = h[:, None] < H
    w0 = tl.load(W + d + 0)[None, :]
    dw0 = tl.full((HEADS, 16), 0., tl.float32)
    db0 = tl.full((HEADS, 16), 0., tl.float32)
    w1 = tl.load(W + d + 16)[None, :]
    dw1 = tl.full((HEADS, 16), 0., tl.float32)
    db1 = tl.full((HEADS, 16), 0., tl.float32)
    w2 = tl.load(W + d + 32)[None, :]
    dw2 = tl.full((HEADS, 16), 0., tl.float32)
    db2 = tl.full((HEADS, 16), 0., tl.float32)
    w3 = tl.load(W + d + 48)[None, :]
    dw3 = tl.full((HEADS, 16), 0., tl.float32)
    db3 = tl.full((HEADS, 16), 0., tl.float32)
    for t in range(pid, TOKENS, programs):
        row = t.to(tl.int64)
        batch, token = row // S, row % S
        upstream = DY + batch * YB + h[:, None].to(tl.int64) * YH + token * YS + d[None, :] * YD
        y0 = tl.load(upstream + 0 * YD, mask, 0).to(tl.float32)
        y1 = tl.load(upstream + 16 * YD, mask, 0).to(tl.float32)
        y2 = tl.load(upstream + 32 * YD, mask, 0).to(tl.float32)
        y3 = tl.load(upstream + 48 * YD, mask, 0).to(tl.float32)
        cy = tl.load(ROT + row * 64 + d)[None, :]
        sy = tl.load(ROT + row * 64 + 16 + d)[None, :]
        cx = tl.load(ROT + row * 64 + 32 + d)[None, :]
        sx = tl.load(ROT + row * 64 + 48 + d)[None, :]
        dn0 = (y0*cy + y1*sy).to(X.dtype.element_ty).to(tl.float32)
        dn1 = (y1*cy - y0*sy).to(X.dtype.element_ty).to(tl.float32)
        dn2 = (y2*cx + y3*sx).to(X.dtype.element_ty).to(tl.float32)
        dn3 = (y3*cx - y2*sx).to(X.dtype.element_ty).to(tl.float32)
        source = X + batch * XB + token * XS + h[:, None] * XH + d[None, :]
        mean = tl.load(MEAN + row * H + h, h < H, 0)[:, None]
        inv = tl.load(INV + row * H + h, h < H, 0)[:, None]
        n0 = (tl.load(source + 0, mask, 0).to(tl.float32) - mean) * inv
        g0 = dn0 * w0
        n1 = (tl.load(source + 16, mask, 0).to(tl.float32) - mean) * inv
        g1 = dn1 * w1
        n2 = (tl.load(source + 32, mask, 0).to(tl.float32) - mean) * inv
        g2 = dn2 * w2
        n3 = (tl.load(source + 48, mask, 0).to(tl.float32) - mean) * inv
        g3 = dn3 * w3
        avg = (tl.sum(g0, 1) + tl.sum(g1, 1) + tl.sum(g2, 1) + tl.sum(g3, 1))[:, None] / 64
        proj = (tl.sum(g0*n0, 1) + tl.sum(g1*n1, 1) + tl.sum(g2*n2, 1) + tl.sum(g3*n3, 1))[:, None] / 64
        output = DX + batch * DB + token * DS + h[:, None] * DH + d[None, :]
        tl.store(output + 0, inv * (g0 - avg - n0 * proj), mask)
        if COPY_V:
            for quadrant in tl.static_range(4):
                col = d[None, :] + quadrant * 16
                value = tl.load(DV + batch * VB + h[:, None].to(tl.int64) * VH + token * VS + col * VD, mask, 0)
                tl.store(DXV + batch * DB + token * DS + h[:, None] * DH + col, value, mask)
        dw0 += tl.where(mask, dn0 * n0, 0.)
        db0 += tl.where(mask, dn0, 0.)
        tl.store(output + 16, inv * (g1 - avg - n1 * proj), mask)
        dw1 += tl.where(mask, dn1 * n1, 0.)
        db1 += tl.where(mask, dn1, 0.)
        tl.store(output + 32, inv * (g2 - avg - n2 * proj), mask)
        dw2 += tl.where(mask, dn2 * n2, 0.)
        db2 += tl.where(mask, dn2, 0.)
        tl.store(output + 48, inv * (g3 - avg - n3 * proj), mask)
        dw3 += tl.where(mask, dn3 * n3, 0.)
        db3 += tl.where(mask, dn3, 0.)
    tl.store(PW + pid.to(tl.int64) * 64 + d + 0, tl.sum(dw0, 0))
    tl.store(PB + pid.to(tl.int64) * 64 + d + 0, tl.sum(db0, 0))
    tl.store(PW + pid.to(tl.int64) * 64 + d + 16, tl.sum(dw1, 0))
    tl.store(PB + pid.to(tl.int64) * 64 + d + 16, tl.sum(db1, 0))
    tl.store(PW + pid.to(tl.int64) * 64 + d + 32, tl.sum(dw2, 0))
    tl.store(PB + pid.to(tl.int64) * 64 + d + 32, tl.sum(db2, 0))
    tl.store(PW + pid.to(tl.int64) * 64 + d + 48, tl.sum(dw3, 0))
    tl.store(PB + pid.to(tl.int64) * 64 + d + 48, tl.sum(db3, 0))


@triton.jit
def _bwd_pair(ROT, DQ, DK, DV, XQ, XK, QW, KW, QM, QI, KM, KI,
              DXQ, DXK, DXV, PQW, PQB, PKW, PKB,
              TOKENS, S: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
              XB, XS, XH, QB, QH, QS, QD, KB, KH, KS, KD, VB, VH, VS, VD,
              DB, DS, DH, HEADS: tl.constexpr, BLOCK: tl.constexpr):
    # Disjoint Q and K CTA groups share a launch; only the Q group writes dV.
    if tl.program_id(1) == 0:
        _bwd(ROT, DV, DXV, True, VB, VH, VS, VD, DQ, XQ, QW, ROT, ROT,
             QM, QI, DXQ, PQW, PQB, TOKENS, S, H, D, XB, XS, XH,
             QB, QH, QS, QD, DB, DS, DH, HEADS, BLOCK)
    else:
        _bwd(ROT, DV, DXV, False, VB, VH, VS, VD, DK, XK, KW, ROT, ROT,
             KM, KI, DXK, PKW, PKB, TOKENS, S, H, D, XB, XS, XH,
             KB, KH, KS, KD, DB, DS, DH, HEADS, BLOCK)


def _forward(x, w, bias, positions, freq, eps, save, v=None, vo=None, rot=None):
    """Launch one program per token; share positions/trig across all heads."""
    b, tokens, heads, d = x.shape
    y = torch.empty((b, heads, tokens, d), device=x.device, dtype=x.dtype)
    mean = torch.empty(b * tokens * heads if save else 0, device=x.device, dtype=torch.float32)
    inv = torch.empty_like(mean)
    if x.numel():
        _fwd[(b * tokens,)](rot if rot is not None else x, v if v is not None else x, vo if vo is not None else y, v is not None, x, w, bias, positions, freq, y, mean, inv,
            tokens, heads, d, *x.stride()[:3], eps, save, triton.next_power_of_2(heads), triton.next_power_of_2(d), num_warps=4)
    return y, mean, inv


def _backward(dy, x, w, positions, freq, mean, inv, dx, dv=None, dxv=None, rot=None):
    """Fused adjoint writing directly into the requested projection gradient view."""
    b, tokens, heads, d = x.shape
    rows = b * tokens * heads
    programs = max(1, min(b * tokens, 8 * torch.cuda.get_device_properties(x.device).multi_processor_count))
    if rows == 0:
        return torch.zeros((programs, d), device=w.device), torch.zeros((programs, d), device=w.device)
    pw = torch.empty((programs, d), device=x.device, dtype=torch.float32)
    pb = torch.empty_like(pw)
    if rows:
        _bwd[(programs,)](rot, dv if dv is not None else dy, dxv if dxv is not None else dx, dv is not None, *(dv.stride() if dv is not None else dy.stride()), dy, x, w, positions, freq, mean, inv, dx, pw, pb,
            b * tokens, tokens, heads, d, *x.stride()[:3], *dy.stride(), *dx.stride()[:3],
            triton.next_power_of_2(heads), triton.next_power_of_2(d), num_warps=2)
    return pw, pb


@triton.jit
def _reduce_parameters(QW, QB, KW, KB, OQW, OQB, OKW, OKB,
                       P: tl.constexpr, D: tl.constexpr, BP: tl.constexpr):
    """Parallelize across parameter kind and columns instead of four large blocks."""
    field = tl.program_id(1)
    source = tl.where(field == 0, QW, tl.where(field == 1, QB, tl.where(field == 2, KW, KB)))
    output = tl.where(field == 0, OQW, tl.where(field == 1, OQB, tl.where(field == 2, OKW, OKB)))
    d = tl.program_id(0) * 4 + tl.arange(0, 4)
    r = tl.arange(0, BP)
    offsets = r[:, None] * D + d[None, :]
    mask = (r[:, None] < P) & (d[None, :] < D)
    value = tl.sum(tl.load(source + offsets, mask, 0), 0)
    tl.store(output + d, value, d < D)


class _QKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qkv, qw, qb, kw, kb, positions, inv_freq, eps, save):
        q, k, v = qkv.unbind(2)
        rot = torch.empty((q.shape[0] * q.shape[1], 64) if save else (0,), device=q.device, dtype=torch.float32)
        vo = torch.empty((v.shape[0], v.shape[2], v.shape[1], v.shape[3]), device=v.device, dtype=v.dtype)
        qo, qm, qi = _forward(q, qw, qb, positions, inv_freq, eps, save, v, vo, rot)
        ko, km, ki = _forward(k, kw, kb, positions, inv_freq, eps, save)
        if save:
            ctx.save_for_backward(qkv, qw, kw, positions, inv_freq, qm, qi, km, ki, rot)
        return qo, ko, vo

    @staticmethod
    def backward(ctx, dq, dk, dv):
        qkv, qw, kw, positions, inv_freq, qm, qi, km, ki, rot = ctx.saved_tensors
        dx = torch.empty_like(qkv)
        q, k, v = qkv.unbind(2)
        dxq, dxk, dxv = dx.unbind(2)
        batches, tokens, heads, d = q.shape
        programs = max(1, min(batches * tokens, 8 * torch.cuda.get_device_properties(q.device).multi_processor_count))
        partial = torch.empty((4, programs, d), device=q.device, dtype=torch.float32)
        pqw, pqb, pkw, pkb = partial.unbind(0)
        if batches * tokens:
            _bwd_pair[(programs, 2)](
                rot, dq, dk, dv, q, k, qw, kw, qm, qi, km, ki,
                dxq, dxk, dxv, pqw, pqb, pkw, pkb,
                batches * tokens, tokens, heads, d, *q.stride()[:3],
                *dq.stride(), *dk.stride(), *dv.stride(), *dxq.stride()[:3],
                triton.next_power_of_2(heads), triton.next_power_of_2(d), num_warps=2)
        else:
            partial.zero_()
        dqw, dqb, dkw, dkb = (torch.empty_like(qw) for _ in range(4))
        _reduce_parameters[(triton.cdiv(qw.numel(), 4), 4)](
            pqw, pqb, pkw, pkb, dqw, dqb, dkw, dkb, pqw.shape[0], qw.numel(),
            triton.next_power_of_2(pqw.shape[0]), num_warps=4)
        return dx, dqw, dqb, dkw, dkb, None, None, None, None


def qkv_prep(qkv, q_norm_w, q_norm_b, k_norm_w, k_norm_b, positions, inv_freq, eps=1e-6):
    """Affine head LayerNorm, 2-D RoPE, and one interleaved QKV input gradient."""
    if qkv.ndim != 5 or qkv.shape[2] != 3 or qkv.shape[-1] != 64 or qkv.dtype != torch.bfloat16 or qkv.stride(-1) != 1:
        raise ValueError("qkv must be bf16 (B,N,3,H,64) with unit last stride")
    parameters = (q_norm_w,q_norm_b,k_norm_w,k_norm_b)
    if not qkv.is_cuda or any(t.device != qkv.device for t in (*parameters,positions,inv_freq)):
        raise ValueError("all tensors must share one CUDA device")
    if any(w.shape != (64,) or w.dtype != torch.float32 or not w.is_contiguous() for w in parameters):
        raise ValueError("norm parameters must be contiguous fp32 (64,)")
    if positions.shape != (*qkv.shape[:2],2) or positions.dtype != torch.int64 or not positions.is_contiguous():
        raise ValueError("positions must be contiguous int64 (B,N,2)")
    if inv_freq.shape != (16,) or inv_freq.dtype != torch.float32 or not inv_freq.is_contiguous():
        raise ValueError("frequencies must be contiguous fp32 (16,)")
    save = torch.is_grad_enabled() and any(t.requires_grad for t in (qkv, q_norm_w, q_norm_b, k_norm_w, k_norm_b))
    return _QKV.apply(qkv, q_norm_w, q_norm_b, k_norm_w, k_norm_b, positions, inv_freq, eps, save)
