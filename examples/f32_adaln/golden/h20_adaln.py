"""Parameterized H20 AdaLN for the public bf16 / fp32-modulation contract.

D drives reduction sizes, masks and addresses. Decomposed kernels derive their
head and tail from D; there is no fixed channel width. Configuration rules select
row tiles, warps and reduction waves. Small/empty workloads retain kernel.py's path.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _offset(row, N1: tl.constexpr, N2: tl.constexpr, S1, S2, S3):
    return row % N1 * S1 + row // N1 % N2 * S2 + row // (N1 * N2) * S3


@triton.jit
def _split_backward(
    DY,
    X,
    W,
    BIAS,
    MEAN,
    INV,
    DX,
    PW,
    PB,
    TOKENS: tl.constexpr,
    SW,
    SB,
    SX,
    SX2,
    SX3,
    SY,
    SY2,
    SY3,
    SYD,
    N1: tl.constexpr,
    N2: tl.constexpr,
    SILU: tl.constexpr,
    D: tl.constexpr,
    HEAD: tl.constexpr,
    TAIL: tl.constexpr,
):
    """Split at the largest power of two <= D, with a masked tail."""
    pid = tl.program_id(0).to(tl.int64)
    batch = tl.program_id(1).to(tl.int64)
    programs = tl.num_programs(0)
    (a, b) = (tl.arange(0, HEAD), tl.arange(0, TAIL))
    wa = 1.0 + tl.load(W + batch * SW + a)
    wb = 1.0 + tl.load(W + batch * SW + HEAD + b, b < D - HEAD, 0)
    if SILU:
        ba = tl.load(BIAS + batch * SB + a)
        bb = tl.load(BIAS + batch * SB + HEAD + b, b < D - HEAD, 0)
    (dwa, dba) = (tl.full((HEAD,), 0.0, tl.float32), tl.full((HEAD,), 0.0, tl.float32))
    (dwb, dbb) = (tl.full((TAIL,), 0.0, tl.float32), tl.full((TAIL,), 0.0, tl.float32))
    # Keep loop indices and memory offsets int64 across all widths and row counts.
    for token in range(pid, TOKENS, programs):
        row = batch.to(tl.int64) * TOKENS + token
        source = X + _offset(row, N1, N2, SX, SX2, SX3)
        adjoint = DY + _offset(row, N1, N2, SY, SY2, SY3)
        (mean, inv) = (tl.load(MEAN + row), tl.load(INV + row))
        na = (tl.load(source + a).to(tl.float32) - mean) * inv
        nb = (tl.load(source + HEAD + b, b < D - HEAD, 0).to(tl.float32) - mean) * inv
        da = tl.load(adjoint + a * SYD).to(tl.float32)
        db = tl.load(adjoint + (HEAD + b) * SYD, b < D - HEAD, 0).to(tl.float32)
        if SILU:
            pa = (na * wa + ba).to(DY.dtype.element_ty).to(tl.float32)
            pb = (nb * wb + bb).to(DY.dtype.element_ty).to(tl.float32)
            (sa, sb) = (1.0 / (1.0 + tl.exp(-pa)), 1.0 / (1.0 + tl.exp(-pb)))
            da = (
                (da * sa * (1.0 + pa * (1.0 - sa)))
                .to(DY.dtype.element_ty)
                .to(tl.float32)
            )
            db = (
                (db * sb * (1.0 + pb * (1.0 - sb)))
                .to(DY.dtype.element_ty)
                .to(tl.float32)
            )
        (dna, dnb) = (da * wa, db * wb)
        avg = (tl.sum(dna, 0) + tl.sum(dnb, 0)) / D
        projection = (tl.sum(dna * na, 0) + tl.sum(dnb * nb, 0)) / D
        tl.store(DX + row * D + a, inv * (dna - avg - na * projection))
        tl.store(
            DX + row * D + HEAD + b, inv * (dnb - avg - nb * projection), b < D - HEAD
        )
        dwa += da * na
        dwb += db * nb
        dba += da
        dbb += db
    target = (batch * programs + pid.to(tl.int64)) * D
    tl.store(PW + target + a, dwa)
    tl.store(PW + target + HEAD + b, dwb, b < D - HEAD)
    tl.store(PB + target + a, dba)
    tl.store(PB + target + HEAD + b, dbb, b < D - HEAD)


@triton.jit
def _parameter_reduce(
    PW, PB, DW, DB, P: tl.constexpr, D: tl.constexpr, BP: tl.constexpr, BC: tl.constexpr
):
    """Combine scale and bias partials in one launch, independently per batch."""
    columns = tl.program_id(0).to(tl.int64) * BC + tl.arange(0, BC)
    rows = tl.arange(0, BP)
    batch = tl.program_id(1).to(tl.int64)
    offset = (batch * P + rows[:, None]) * D + columns[None, :]
    mask = (rows[:, None] < P) & (columns[None, :] < D)
    dw = tl.sum(tl.load(PW + offset, mask, 0), 0)
    db = tl.sum(tl.load(PB + offset, mask, 0), 0)
    tl.store(DW + batch * D + columns, dw, columns < D)
    tl.store(DB + batch * D + columns, db, columns < D)


@triton.jit
def _rows_forward(
    X,
    W,
    S,
    Y,
    M,
    I,
    T: tl.constexpr,
    SW,
    SS,
    XB,
    XT,
    EPS,
    SILU: tl.constexpr,
    SAVE: tl.constexpr,
    BT: tl.constexpr,
    D: tl.constexpr,
    HEAD: tl.constexpr,
    TAIL: tl.constexpr,
):
    b = tl.program_id(1).to(tl.int64)
    t = tl.program_id(0).to(tl.int64) * BT + tl.arange(0, BT)
    a = tl.arange(0, HEAD)
    d = tl.arange(0, TAIL)
    xa = tl.load(
        X + b * XB + t[:, None].to(tl.int64) * XT + a[None, :], t[:, None] < T, 0
    ).to(tl.float32)
    xb = tl.load(
        X + b * XB + t[:, None].to(tl.int64) * XT + HEAD + d[None, :],
        (t[:, None] < T) & (d[None, :] < D - HEAD),
        0,
    ).to(tl.float32)
    mean = (tl.sum(xa, 1) + tl.sum(xb, 1)) / D
    ca = xa - mean[:, None]
    cb = tl.where(d[None, :] < D - HEAD, xb - mean[:, None], 0.0)
    inv = tl.rsqrt((tl.sum(ca * ca, 1) + tl.sum(cb * cb, 1)) / D + EPS)
    wa = 1.0 + tl.load(W + b * SW + a)
    wb = 1.0 + tl.load(W + b * SW + HEAD + d, d < D - HEAD, 0)
    sa = tl.load(S + b * SS + a)
    sb = tl.load(S + b * SS + HEAD + d, d < D - HEAD, 0)
    ya = ca * inv[:, None] * wa[None, :] + sa[None, :]
    yb = cb * inv[:, None] * wb[None, :] + sb[None, :]
    if SILU:
        ya = ya.to(Y.dtype.element_ty).to(tl.float32)
        yb = yb.to(Y.dtype.element_ty).to(tl.float32)
        ya = ya / (1.0 + tl.exp(-ya))
        yb = yb / (1.0 + tl.exp(-yb))
    row = b * T + t.to(tl.int64)
    tl.store(Y + row[:, None] * D + a[None, :], ya, t[:, None] < T)
    tl.store(
        Y + row[:, None] * D + HEAD + d[None, :],
        yb,
        (t[:, None] < T) & (d[None, :] < D - HEAD),
    )
    if SAVE:
        tl.store(M + row, mean, t < T)
        tl.store(I + row, inv, t < T)


@triton.jit
def _padded(
    X,
    W,
    S,
    Y,
    M,
    I,
    T: tl.constexpr,
    SW,
    SS,
    XB,
    XT,
    EPS,
    SILU: tl.constexpr,
    SAVE: tl.constexpr,
    BT: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(1).to(tl.int64)
    t = tl.program_id(0).to(tl.int64) * BT + tl.arange(0, BT)
    d = tl.arange(0, BLOCK)
    x = tl.load(
        X + b * XB + t[:, None] * XT + d[None, :],
        (t[:, None] < T) & (d[None, :] < D),
        0,
    ).to(tl.float32)
    mean = tl.sum(x, 1) / D
    centered = tl.where(d[None, :] < D, x - mean[:, None], 0.0)
    inv = tl.rsqrt(tl.sum(centered * centered, 1) / D + EPS)
    w = tl.load(W + b * SW + d, d < D, 0)
    s = tl.load(S + b * SS + d, d < D, 0)
    y = centered * inv[:, None] * (1.0 + w[None, :]) + s[None, :]
    if SILU:
        y = y.to(Y.dtype.element_ty).to(tl.float32)
        y = y / (1.0 + tl.exp(-y))
    row = b * T + t
    tl.store(Y + row[:, None] * D + d[None, :], y, (t[:, None] < T) & (d[None, :] < D))
    if SAVE:
        tl.store(M + row, mean, t < T)
        tl.store(I + row, inv, t < T)


def select_config(d, tokens, silu, save=True):
    """Size-derived rules, shared by all widths; no shape lookup table."""
    block = triton.next_power_of_2(d)
    if block <= 1024:
        bt = 8 if silu or d == block else 4
        fw, bw = (4 if silu else 2), 2
        waves = 16 if silu or block <= 256 else 8
    elif block <= 2048:
        bt = (
            (8 if save else 4)
            if silu and d < block
            else (4 if silu or d < block else 8)
        )
        fw = 4 if silu or d == block else 2
        bw = 4 if silu else 2
        waves = (8 if d == block else 12) if silu else (4 if d >= block * 3 // 4 else 8)
    elif block <= 4096:
        bt, fw, bw, waves = (2 if silu else 4), 4, 4, 8
    else:
        bt = 1 if silu or tokens < 512 else 2
        fw, bw, waves = (8 if bt == 1 else 4), 8, 8
    head = 1 << (d.bit_length() - 1)
    tail = triton.next_power_of_2(max(1, d - head))
    tail_padding = tail - (d - head) if d > head else 0
    split = silu and block <= 2048
    if split:
        # Keep the SiLU live-value footprint bounded; inefficient masked tails
        # instead use the regular padded tile and a coarser gradient grid.
        if tail_padding > d // 8:
            bt, fw, bw, waves, split = 4, 4, 2, 4, False
        else:
            row_budget = max(1, 12288 // (head + (tail if d > head else 0)))
            bt = min(bt, 1 << (row_budget.bit_length() - 1))
    return dict(bt=bt, fw=fw, split=split, bw=bw, waves=waves)


def fwd(x, w, s, eps, silu, save):
    B, T, D = x.shape
    cfg = select_config(D, T, silu, save)
    y = torch.empty_like(x, memory_format=torch.contiguous_format)
    m = torch.empty(B * T if save else 0, device=x.device, dtype=torch.float32)
    inv = torch.empty_like(m)
    if B * T:
        block = triton.next_power_of_2(D)
        head = 1 << (D.bit_length() - 1)
        tail = triton.next_power_of_2(max(1, D - head))
        if cfg["split"]:
            _rows_forward[(triton.cdiv(T, cfg["bt"]), B)](
                x,
                w,
                s,
                y,
                m,
                inv,
                T,
                w.stride(0),
                s.stride(0),
                x.stride(0),
                x.stride(1),
                eps,
                silu,
                save,
                cfg["bt"],
                D,
                head,
                tail,
                num_warps=cfg["fw"],
            )
        else:
            kernel = _single_forward if cfg["bt"] == 1 else _padded
            kernel[(triton.cdiv(T, cfg["bt"]), B)](
                x,
                w,
                s,
                y,
                m,
                inv,
                T,
                w.stride(0),
                s.stride(0),
                x.stride(0),
                x.stride(1),
                eps,
                silu,
                save,
                cfg["bt"],
                D,
                block,
                num_warps=cfg["fw"],
            )
    return y, m, inv


def partial_count(x, silu):
    B, T, D = x.shape
    cfg = select_config(D, T, silu)
    return max(
        1,
        min(
            triton.cdiv(T, 4),
            cfg["waves"]
            * torch.cuda.get_device_properties(x.device).multi_processor_count
            // max(1, B),
        ),
    )


def bwd(dy, x, w, s, m, inv, silu):
    B, T, D = x.shape
    cfg = select_config(D, T, silu)
    if B * T == 0:
        return torch.zeros_like(x), torch.zeros_like(w), torch.zeros_like(s)
    P = partial_count(x, silu)
    head = 1 << (D.bit_length() - 1)
    tail = triton.next_power_of_2(max(1, D - head))
    dx = torch.empty_like(x, memory_format=torch.contiguous_format)
    pw = torch.empty((B, P, D), device=x.device, dtype=torch.float32)
    ps = torch.empty_like(pw)
    _split_backward[(P, B)](
        dy,
        x,
        w,
        s,
        m,
        inv,
        dx,
        pw,
        ps,
        T,
        w.stride(0),
        s.stride(0),
        x.stride(1),
        x.stride(0),
        0,
        dy.stride(1),
        dy.stride(0),
        0,
        dy.stride(2),
        T,
        B,
        silu,
        D,
        head,
        tail,
        num_warps=cfg["bw"],
    )
    dw = torch.empty_like(w)
    ds = torch.empty_like(s)
    _parameter_reduce[(triton.cdiv(D, 32), B)](
        pw, ps, dw, ds, P, D, triton.next_power_of_2(P), 32, num_warps=4
    )
    return dx, dw, ds


def supported(x):
    return (
        x.ndim == 3
        and 1 <= x.shape[-1] <= 8192
        and x.shape[0] * x.shape[1] > 128
        and x.shape[1] > 32
        and x.dtype == torch.bfloat16
        and x.is_cuda
        and "H20" in torch.cuda.get_device_properties(x.device).name
    )


@triton.jit
def _single_forward(
    X,
    W,
    S,
    Y,
    M,
    I,
    T: tl.constexpr,
    SW,
    SS,
    XB,
    XT,
    EPS,
    SILU: tl.constexpr,
    SAVE: tl.constexpr,
    BT: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(1).to(tl.int64)
    t = tl.program_id(0).to(tl.int64)
    d = tl.arange(0, BLOCK)
    x = tl.load(X + b * XB + t * XT + d, (t < T) & (d < D), 0).to(tl.float32)
    mean = tl.sum(x, 0) / D
    centered = tl.where(d < D, x - mean, 0.0)
    inv = tl.rsqrt(tl.sum(centered * centered, 0) / D + EPS)
    w = tl.load(W + b * SW + d, d < D, 0)
    s = tl.load(S + b * SS + d, d < D, 0)
    y = centered * inv * (1.0 + w) + s
    if SILU:
        y = y.to(Y.dtype.element_ty).to(tl.float32)
        y = y / (1.0 + tl.exp(-y))
    row = b * T + t
    tl.store(Y + row * D + d, y, (t < T) & (d < D))
    if SAVE:
        tl.store(M + row, mean, t < T)
        tl.store(I + row, inv, t < T)
