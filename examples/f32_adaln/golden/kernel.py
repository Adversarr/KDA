"""Standalone fused row-normalization reference; fp32 math and deterministic gradients.

One forward program owns each row. Backward programs stride over row tiles,
accumulate parameter gradients in registers, then reduce bounded fp32 partials.
Auxiliary mean/rstd are written only when autograd needs them.
"""
import torch
import triton
import triton.language as tl

MODE = 2


def _stride(x):
    """Innermost row stride; remaining leading strides are passed separately."""
    if x.stride(-1) != 1:
        raise ValueError("last dimension must have unit stride")
    return x.stride(-2)


def _outer(x):
    """Outer two strides for rank 2-4 tensors, including broadcast gradients."""
    return (x.stride(-3) if x.ndim >= 3 else 0, x.stride(-4) if x.ndim == 4 else 0)


@triton.jit
def _offset(row, N1: tl.constexpr, N2: tl.constexpr, S1, S2, S3):
    return (row % N1) * S1 + ((row // N1) % N2) * S2 + (row // (N1 * N2)) * S3


@triton.jit
def _forward(X, R, W, BIAS, G, VALID, OUT, STREAM, MEAN, INV,
             D: tl.constexpr, TOKENS: tl.constexpr, SW, SB, SX, SR, SX2, SX3, SR2, SR3, N1: tl.constexpr, N2: tl.constexpr, EPS,
             MODE: tl.constexpr, SILU: tl.constexpr, SAVE: tl.constexpr,
             BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    d = tl.arange(0, BLOCK)
    mask = d < D
    x = tl.load(X + _offset(row, N1, N2, SX, SX2, SX3) + d, mask, 0).to(tl.float32)
    if MODE == 0:
        h = x + tl.load(R + _offset(row, N1, N2, SR, SR2, SR3) + d, mask, 0).to(tl.float32)
    elif MODE == 3:
        branch = tl.load(R + _offset(row, N1, N2, SR, SR2, SR3) + d, mask, 0).to(tl.float32)
        gamma = tl.load(G + d, mask, 0).to(tl.float32)
        valid = tl.load(VALID + row)
        h = tl.where(valid, x + branch * gamma, 0.)
    else:
        h = x
    if MODE == 1:
        mean = 0.
        inv = tl.rsqrt(tl.sum(h * h, 0) + EPS)
        n = h * inv
    else:
        mean = tl.sum(h, 0) / D
        centered = tl.where(mask, h - mean, 0.)
        inv = tl.rsqrt(tl.sum(centered * centered, 0) / D + EPS)
        n = (h - mean) * inv
    if MODE == 2:
        w = 1. + tl.load(W + (row // TOKENS) * SW + d, mask, 0).to(tl.float32)
        bias = tl.load(BIAS + (row // TOKENS) * SB + d, mask, 0).to(tl.float32)
    else:
        w = tl.load(W + d, mask, 0).to(tl.float32)
        if MODE == 1:
            bias = 0.
        else:
            bias = tl.load(BIAS + d, mask, 0).to(tl.float32)
    y = n * w + bias
    if SILU:
        # The optional activation follows the original storage-dtype AdaLN output.
        y = y.to(OUT.dtype.element_ty).to(tl.float32)
        y = y / (1. + tl.exp(-y))
    tl.store(OUT + row * D + d, y, mask)
    if MODE == 0 or MODE == 3:
        tl.store(STREAM + row * D + d, h, mask)
    if SAVE:
        tl.store(MEAN + row, mean)
        tl.store(INV + row, inv)


@triton.jit
def _backward(DY, DH, H, R, W, BIAS, G, VALID, MEAN, INV,
              DX, DR, PW, PB, PG, ROWS, D: tl.constexpr,
              TOKENS: tl.constexpr, SW, SB, SH, SR, SY, SYD, SDH, SDHD,
              SH2, SH3, SR2, SR3, SY2, SY3, SDH2, SDH3, N1: tl.constexpr, N2: tl.constexpr,
              MODE: tl.constexpr, SILU: tl.constexpr,
              BLOCK: tl.constexpr, ROW_TILE: tl.constexpr):
    pid = tl.program_id(0)
    batch = tl.program_id(1)
    programs = tl.num_programs(0)
    d = tl.arange(0, BLOCK)
    rr = tl.arange(0, ROW_TILE)
    dw = tl.full((ROW_TILE, BLOCK), 0., tl.float32)
    db = tl.full((ROW_TILE, BLOCK), 0., tl.float32)
    dg = tl.full((ROW_TILE, BLOCK), 0., tl.float32)
    if MODE == 2:
        w = 1. + tl.load(W + batch * SW + d, d < D, 0).to(tl.float32)
        bias = tl.load(BIAS + batch * SB + d, d < D, 0).to(tl.float32)
        count = TOKENS
    else:
        w = tl.load(W + d, d < D, 0).to(tl.float32)
        count = ROWS
    for start in range(pid * ROW_TILE, count, programs * ROW_TILE):
        local = start + rr
        row = local.to(tl.int64)
        if MODE == 2:
            row += batch.to(tl.int64) * TOKENS
        mask = (local[:, None] < count) & (d[None, :] < D)
        h = tl.load(H + _offset(row, N1, N2, SH, SH2, SH3)[:, None] + d[None, :], mask, 0).to(tl.float32)
        mean = tl.load(MEAN + row, local < count, 0)
        inv = tl.load(INV + row, local < count, 0)
        n = (h - mean[:, None]) * inv[:, None]
        dy = tl.load(DY + _offset(row, N1, N2, SY, SY2, SY3)[:, None] + d[None, :] * SYD, mask, 0).to(tl.float32)
        if SILU:
            pre = (n * w[None, :] + bias[None, :]).to(DY.dtype.element_ty).to(tl.float32)
            sigmoid = 1. / (1. + tl.exp(-pre))
            dy = (dy * sigmoid * (1. + pre * (1. - sigmoid))).to(DY.dtype.element_ty).to(tl.float32)
        dn = dy * w[None, :]
        projection = tl.sum(tl.where(mask, dn * n, 0.), 1)
        if MODE == 1:
            grad = inv[:, None] * (dn - n * projection[:, None])
        else:
            avg = tl.sum(tl.where(mask, dn, 0.), 1) / D
            grad = inv[:, None] * (dn - avg[:, None] - n * (projection / D)[:, None])
        dw += tl.where(mask, dy * n, 0.)
        db += tl.where(mask, dy, 0.)
        if MODE == 0 or MODE == 3:
            dh = tl.load(DH + _offset(row, N1, N2, SDH, SDH2, SDH3)[:, None] + d[None, :] * SDHD, mask, 0).to(tl.float32)
            grad += dh
        if MODE == 3:
            valid = tl.load(VALID + row, local < count, 0)
            grad = tl.where(valid[:, None], grad, 0.)
            branch = tl.load(R + _offset(row, N1, N2, SR, SR2, SR3)[:, None] + d[None, :], mask, 0).to(tl.float32)
            gamma = tl.load(G + d, d < D, 0).to(tl.float32)
            dg += tl.where(mask, grad * branch, 0.)
            tl.store(DR + row[:, None] * D + d[None, :], grad * gamma[None, :], mask)
        elif MODE == 0:
            tl.store(DR + row[:, None] * D + d[None, :], grad, mask)
        tl.store(DX + row[:, None] * D + d[None, :], grad, mask)
    off = (batch.to(tl.int64) * programs + pid) * D + d
    tl.store(PW + off, tl.sum(dw, 0), d < D)
    if MODE != 1:
        tl.store(PB + off, tl.sum(db, 0), d < D)
    if MODE == 3:
        tl.store(PG + off, tl.sum(dg, 0), d < D)


@triton.jit
def _parameter_reduce(PW, PB, DW, DB, P: tl.constexpr, D: tl.constexpr,
                      BP: tl.constexpr, BC: tl.constexpr):
    """Combine scale and bias partials in one launch, independently per batch."""
    columns = tl.program_id(0) * BC + tl.arange(0, BC)
    rows = tl.arange(0, BP)
    batch = tl.program_id(1)
    offset = (batch * P + rows[:, None]) * D + columns[None, :]
    mask = (rows[:, None] < P) & (columns[None, :] < D)
    dw = tl.sum(tl.load(PW + offset, mask, 0), 0)
    db = tl.sum(tl.load(PB + offset, mask, 0), 0)
    tl.store(DW + batch * D + columns, dw, columns < D)
    tl.store(DB + batch * D + columns, db, columns < D)


@triton.jit
def _split_forward(X, W, BIAS, Y, MEAN, INV, TOKENS: tl.constexpr,
                   SW, SB, SX, SX2, SX3, N1: tl.constexpr, N2: tl.constexpr,
                   EPS, SILU: tl.constexpr, SAVE: tl.constexpr):
    """The model width is 1024+128; avoid padding its arithmetic to 2048."""
    row = tl.program_id(0).to(tl.int64)
    a, b = tl.arange(0, 1024), tl.arange(0, 128)
    source = X + _offset(row, N1, N2, SX, SX2, SX3)
    xa = tl.load(source + a).to(tl.float32)
    xb = tl.load(source + 1024 + b).to(tl.float32)
    mean = (tl.sum(xa, 0) + tl.sum(xb, 0)) / 1152
    ca, cb = xa - mean, xb - mean
    inv = tl.rsqrt((tl.sum(ca*ca, 0) + tl.sum(cb*cb, 0)) / 1152 + EPS)
    batch = row // TOKENS
    wa = 1. + tl.load(W + batch * SW + a)
    wb = 1. + tl.load(W + batch * SW + 1024 + b)
    ya = ca * inv * wa + tl.load(BIAS + batch * SB + a)
    yb = cb * inv * wb + tl.load(BIAS + batch * SB + 1024 + b)
    if SILU:
        ya = ya.to(Y.dtype.element_ty).to(tl.float32)
        yb = yb.to(Y.dtype.element_ty).to(tl.float32)
        ya = ya / (1. + tl.exp(-ya))
        yb = yb / (1. + tl.exp(-yb))
    tl.store(Y + row * 1152 + a, ya)
    tl.store(Y + row * 1152 + 1024 + b, yb)
    if SAVE:
        tl.store(MEAN + row, mean)
        tl.store(INV + row, inv)


@triton.jit
def _split_silu_backward(DY, X, W, BIAS, MEAN, INV, DX, PW, PB,
                         TOKENS: tl.constexpr, SW, SB, SX, SX2, SX3,
                         SY, SY2, SY3, SYD, N1: tl.constexpr, N2: tl.constexpr):
    """1152 live channels, split as 1024+128 without masked SFU arithmetic."""
    pid, batch = tl.program_id(0), tl.program_id(1)
    programs = tl.num_programs(0)
    a, b = tl.arange(0, 1024), tl.arange(0, 128)
    wa = 1. + tl.load(W + batch * SW + a)
    wb = 1. + tl.load(W + batch * SW + 1024 + b)
    ba = tl.load(BIAS + batch * SB + a)
    bb = tl.load(BIAS + batch * SB + 1024 + b)
    dwa, dba = tl.full((1024,), 0., tl.float32), tl.full((1024,), 0., tl.float32)
    dwb, dbb = tl.full((128,), 0., tl.float32), tl.full((128,), 0., tl.float32)
    for token in range(pid, TOKENS, programs):
        row = batch.to(tl.int64) * TOKENS + token
        source = X + _offset(row, N1, N2, SX, SX2, SX3)
        adjoint = DY + _offset(row, N1, N2, SY, SY2, SY3)
        mean, inv = tl.load(MEAN + row), tl.load(INV + row)
        na = (tl.load(source + a).to(tl.float32) - mean) * inv
        nb = (tl.load(source + 1024 + b).to(tl.float32) - mean) * inv
        pa = (na * wa + ba).to(DY.dtype.element_ty).to(tl.float32)
        pb = (nb * wb + bb).to(DY.dtype.element_ty).to(tl.float32)
        sa, sb = 1. / (1. + tl.exp(-pa)), 1. / (1. + tl.exp(-pb))
        da = tl.load(adjoint + a * SYD).to(tl.float32)
        db = tl.load(adjoint + (1024 + b) * SYD).to(tl.float32)
        da = (da * sa * (1. + pa * (1. - sa))).to(DY.dtype.element_ty).to(tl.float32)
        db = (db * sb * (1. + pb * (1. - sb))).to(DY.dtype.element_ty).to(tl.float32)
        dna, dnb = da * wa, db * wb
        avg = (tl.sum(dna, 0) + tl.sum(dnb, 0)) / 1152
        projection = (tl.sum(dna * na, 0) + tl.sum(dnb * nb, 0)) / 1152
        tl.store(DX + row * 1152 + a, inv * (dna - avg - na * projection))
        tl.store(DX + row * 1152 + 1024 + b, inv * (dnb - avg - nb * projection))
        dwa += da * na
        dwb += db * nb
        dba += da
        dbb += db
    target = (batch.to(tl.int64) * programs + pid) * 1152
    tl.store(PW + target + a, dwa)
    tl.store(PW + target + 1024 + b, dwb)
    tl.store(PB + target + a, dba)
    tl.store(PB + target + 1024 + b, dbb)


@triton.jit
def _small_silu_backward(DY, X, W, BIAS, MEAN, INV, DX, DW, DB,
                         T: tl.constexpr, D: tl.constexpr, SW, SB,
                         SX, SX2, SX3, SY, SY2, SY3, SYD,
                         N1: tl.constexpr, N2: tl.constexpr,
                         BD: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr):
    """Each token computes dx and owns disjoint modulation-gradient columns."""
    token, batch = tl.program_id(0), tl.program_id(1)
    row = batch.to(tl.int64) * T + token
    d = tl.arange(0, BD)
    x = tl.load(X + _offset(row, N1, N2, SX, SX2, SX3) + d, d < D, 0).to(tl.float32)
    mean, inv = tl.load(MEAN + row), tl.load(INV + row)
    n = (x - mean) * inv
    w = 1. + tl.load(W + batch * SW + d, d < D, 0)
    bias = tl.load(BIAS + batch * SB + d, d < D, 0)
    pre = (n * w + bias).to(DY.dtype.element_ty).to(tl.float32)
    sigmoid = 1. / (1. + tl.exp(-pre))
    dy = tl.load(DY + _offset(row, N1, N2, SY, SY2, SY3) + d * SYD, d < D, 0).to(tl.float32)
    dy = (dy * sigmoid * (1. + pre * (1. - sigmoid))).to(DY.dtype.element_ty).to(tl.float32)
    g = dy * w
    avg = tl.sum(tl.where(d < D, g, 0.), 0) / D
    proj = tl.sum(tl.where(d < D, g * n, 0.), 0) / D
    tl.store(DX + row * D + d, inv * (g - avg - n * proj), d < D)
    cols = token * BC + tl.arange(0, BC)
    tokens = tl.arange(0, BT)
    rows = batch.to(tl.int64) * T + tokens
    mask = (tokens[:, None] < T) & (cols[None, :] < D)
    x = tl.load(X + _offset(rows, N1, N2, SX, SX2, SX3)[:, None] + cols[None, :], mask, 0).to(tl.float32)
    m = tl.load(MEAN + rows, tokens < T, 0)[:, None]
    r = tl.load(INV + rows, tokens < T, 0)[:, None]
    n = (x - m) * r
    w = 1. + tl.load(W + batch * SW + cols, cols < D, 0)
    bias = tl.load(BIAS + batch * SB + cols, cols < D, 0)
    pre = (n * w[None, :] + bias[None, :]).to(DY.dtype.element_ty).to(tl.float32)
    sigmoid = 1. / (1. + tl.exp(-pre))
    dy = tl.load(DY + _offset(rows, N1, N2, SY, SY2, SY3)[:, None] + cols[None, :] * SYD, mask, 0).to(tl.float32)
    dy = (dy * sigmoid * (1. + pre * (1. - sigmoid))).to(DY.dtype.element_ty).to(tl.float32)
    tl.store(DW + batch * D + cols, tl.sum(tl.where(mask, dy*n, 0.), 0), cols < D)
    tl.store(DB + batch * D + cols, tl.sum(tl.where(mask, dy, 0.), 0), cols < D)


class _Norm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, residual, weight, bias, gamma, valid, eps, silu, out_dtype, save):
        rows, d = x.numel() // x.shape[-1], x.shape[-1]
        tokens = x.shape[1] if MODE == 2 else rows
        y = torch.empty(x.shape, device=x.device, dtype=out_dtype)
        stream = torch.empty(x.shape, device=x.device, dtype=torch.float32) if MODE in (0, 3) else x
        mean = torch.empty(rows if save else 0, device=x.device, dtype=torch.float32)
        inv = torch.empty_like(mean)
        if rows and d == 1152:
            _split_forward[(rows,)](x, weight, bias, y, mean, inv, tokens,
                weight.stride(0), bias.stride(0), _stride(x), *_outer(x),
                x.shape[-2], x.shape[-3] if x.ndim >= 3 else 1, eps, silu, save, num_warps=4)
        elif rows:
            _forward[(rows,)](x, residual, weight, bias, gamma, valid, y, stream, mean, inv,
                d, tokens, weight.stride(0), bias.stride(0), _stride(x), _stride(residual), *_outer(x), *_outer(residual),
                x.shape[-2], x.shape[-3] if x.ndim >= 3 else 1, eps, MODE, silu, save,
                triton.next_power_of_2(d), num_warps=4 if d <= 2048 else 8)
        if save:
            ctx.save_for_backward(stream, residual, weight, bias, gamma, valid, mean, inv)
            ctx.silu, ctx.x_dtype = silu, x.dtype
        return (stream, y) if MODE in (0, 3) else y

    @staticmethod
    def backward(ctx, *grads):
        h, residual, weight, bias, gamma, valid, mean, inv = ctx.saved_tensors
        rows, d = h.numel() // h.shape[-1], h.shape[-1]
        tokens = h.shape[1] if MODE == 2 else rows
        batches = h.shape[0] if MODE == 2 else 1
        program_scale = 12 if ctx.silu and d == 1152 else 2
        programs = max(1, min(triton.cdiv(tokens, 4), max(1, program_scale * torch.cuda.get_device_properties(h.device).multi_processor_count // max(1, batches))))
        small = tokens <= 32 and d <= 256
        if small:
            programs = 1
        if rows == 0:
            return (torch.zeros_like(h, dtype=ctx.x_dtype), torch.zeros_like(residual) if MODE in (0, 3) else None,
                    torch.zeros_like(weight), torch.zeros_like(bias) if MODE != 1 else None,
                    torch.zeros_like(gamma) if MODE == 3 else None, None, None, None, None, None)
        dy = grads[-1]
        dh = grads[0] if MODE in (0, 3) else dy
        dx = torch.empty(h.shape, dtype=ctx.x_dtype, device=h.device)
        dr = torch.empty_like(residual)
        pw = torch.empty((batches, programs, d), dtype=torch.float32, device=h.device)
        pb, pg = torch.empty_like(pw), torch.empty_like(pw)
        # Autograd may pass a broadcast gradient (including last-dimension stride zero).
        def gradient_stride(g):
            return g.stride(-2) if g.ndim > 1 else 0
        if small and ctx.silu:
            dw = torch.empty(weight.shape, device=weight.device, dtype=weight.dtype)
            db = torch.empty(bias.shape, device=bias.device, dtype=bias.dtype)
            _small_silu_backward[(tokens, batches)](dy, h, weight, bias, mean, inv, dx, dw, db,
                tokens, d, weight.stride(0), bias.stride(0), _stride(h), *_outer(h),
                gradient_stride(dy), *_outer(dy), dy.stride(-1),
                h.shape[-2], h.shape[-3] if h.ndim >= 3 else 1,
                triton.next_power_of_2(d), triton.next_power_of_2(tokens),
                triton.next_power_of_2(triton.cdiv(d, tokens)), num_warps=4)
            return dx, None, dw, db, None, None, None, None, None, None
        if rows and ctx.silu and d == 1152:
            _split_silu_backward[(programs, batches)](
                dy, h, weight, bias, mean, inv, dx, pw, pb,
                tokens, weight.stride(0), bias.stride(0), _stride(h), *_outer(h),
                gradient_stride(dy), *_outer(dy), dy.stride(-1),
                h.shape[-2], h.shape[-3] if h.ndim >= 3 else 1, num_warps=4)
        elif rows:
            _backward[(programs, batches)](dy, dh, h, residual, weight, bias, gamma, valid,
                mean, inv, dx, dr, pw, pb, pg, rows, d, tokens, weight.stride(0), bias.stride(0),
                _stride(h), _stride(residual), gradient_stride(dy), dy.stride(-1),
                gradient_stride(dh), dh.stride(-1), *_outer(h), *_outer(residual), *_outer(dy), *_outer(dh),
                h.shape[-2], h.shape[-3] if h.ndim >= 3 else 1, MODE, ctx.silu, triton.next_power_of_2(d), triton.next_power_of_2(tokens) if small else (1 if ctx.silu and d == 1152 else 2),
                num_warps=2 if ctx.silu and d == 1152 else (4 if d <= 2048 else 8))
        if small:
            dw, db = pw.view(weight.shape), pb.view(bias.shape)
        else:
            dw = torch.empty(weight.shape, device=weight.device, dtype=weight.dtype)
            db = torch.empty(bias.shape, device=bias.device, dtype=bias.dtype)
            _parameter_reduce[(triton.cdiv(d, 32), batches)](
                pw, pb, dw, db, programs, d, triton.next_power_of_2(programs), 32, num_warps=4)
        dg = pg.sum((0, 1)).to(gamma.dtype) if MODE == 3 else None
        return dx, dr if MODE in (0, 3) else None, dw, db, dg, None, None, None, None, None


def _call(x, residual, weight, bias, gamma, valid, eps, silu=False, out_dtype=None):
    """Dispatch the fused kernel with gradient-mode-aware auxiliary allocation."""
    if x.ndim < 2 or x.ndim > 4 or x.shape[-1] < 1 or x.shape[-1] > 8192:
        raise ValueError("normalization requires rank >= 2 and 1 <= D <= 8192")
    if not x.is_cuda or x.dtype != (torch.float32 if MODE == 3 else torch.bfloat16):
        raise ValueError("activation must be CUDA with the contract dtype")
    tensors = (x, residual, weight, bias, gamma, valid)
    if any(t.device != x.device for t in tensors):
        raise ValueError("all inputs must share one CUDA device")
    _stride(x)
    d = x.shape[-1]
    if MODE in (0, 3):
        if residual.shape != x.shape or residual.dtype != (torch.float32 if MODE == 0 else torch.bfloat16):
            raise ValueError("residual/branch shape or dtype mismatch")
        _stride(residual)
    expected = (x.shape[0], d) if MODE == 2 else (d,)
    for tensor in ((weight,) if MODE == 1 else (weight, bias)):
        if tensor.shape != expected or tensor.dtype != torch.float32 or tensor.stride(-1) != 1 or (MODE != 2 and not tensor.is_contiguous()):
            raise ValueError("normalization parameters must be contiguous fp32 with the contract shape")
    if MODE == 2 and x.ndim != 3:
        raise ValueError("AdaLN requires (B, N, D)")
    if MODE == 3:
        if gamma.shape != (d,) or gamma.dtype != torch.float32 or not gamma.is_contiguous():
            raise ValueError("LayerScale must be contiguous fp32 (D,)")
        if valid.shape != x.shape[:-1] or valid.dtype != torch.bool or not valid.is_contiguous():
            raise ValueError("validity must be contiguous bool with the row shape")
    save = torch.is_grad_enabled() and any(t.requires_grad for t in (x, residual, weight, bias, gamma))
    return _Norm.apply(x, residual, weight, bias, gamma, valid, eps, silu, out_dtype or x.dtype, save)


def adaln(x, scale, shift, eps=1e-6, silu=False):
    """Per-sample adaptive LayerNorm; optional SiLU follows the bf16 cast."""
    return _call(x, x, scale, shift, scale, scale, eps, silu)
