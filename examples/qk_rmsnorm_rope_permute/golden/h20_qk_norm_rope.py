"""Parameterized H20 shared-weight RMSNorm/RoPE with token tiles per head.

H and D are compile-time parameters, with masked half-channel tails. Frozen
size-derived configurations preserve bf16 norm/adjoint rounding and fp32 partials.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _forward(
    X,
    W,
    C,
    S,
    Y,
    R,
    T: tl.constexpr,
    SEQ: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    EPS,
    XB,
    XT,
    XH,
    BT: tl.constexpr,
    BD: tl.constexpr,
    SAVE: tl.constexpr,
):
    head = tl.program_id(1).to(tl.int64)
    t = tl.program_id(0).to(tl.int64) * BT + tl.arange(0, BT)
    b = t // SEQ
    s = t % SEQ
    d = tl.arange(0, BD)
    mask = (t[:, None] < T) & (d[None, :] < D // 2)
    base = X + b[:, None] * XB + s[:, None] * XT + head * XH
    x1 = tl.load(base + d[None, :], mask, 0).to(tl.float32)
    x2 = tl.load(base + D // 2 + d[None, :], mask, 0).to(tl.float32)
    inv = tl.rsqrt((tl.sum(x1 * x1, 1) + tl.sum(x2 * x2, 1)) / D + EPS)
    w1 = tl.load(W + d, d < D // 2, 0)
    w2 = tl.load(W + D // 2 + d, d < D // 2, 0)
    n1 = (x1 * inv[:, None] * w1[None, :]).to(X.dtype.element_ty).to(tl.float32)
    n2 = (x2 * inv[:, None] * w2[None, :]).to(X.dtype.element_ty).to(tl.float32)
    c = tl.load(C + s[:, None] * (D // 2) + d[None, :], mask, 0)
    sn = tl.load(S + s[:, None] * (D // 2) + d[None, :], mask, 0)
    y1 = n1 * c - n2 * sn
    y2 = n2 * c + n1 * sn
    dest = Y + ((b[:, None] * H + head) * SEQ + s[:, None]) * D
    tl.store(dest + d[None, :], y1, mask)
    tl.store(dest + D // 2 + d[None, :], y2, mask)
    if SAVE:
        tl.store(R + t * H + head, inv, t < T)


@triton.jit
def _backward(
    G,
    X,
    W,
    C,
    S,
    R,
    DX,
    P,
    T: tl.constexpr,
    SEQ: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    XB,
    XT,
    XH,
    GB,
    GH,
    GT,
    GD,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)
    nprog = tl.num_programs(0)
    d = tl.arange(0, BD)
    rr = tl.arange(0, BT)
    w1 = tl.load(W + d, d < D // 2, 0)[None, :]
    w2 = tl.load(W + D // 2 + d, d < D // 2, 0)[None, :]
    dw1 = tl.full((BT, BD), 0.0, tl.float32)
    dw2 = tl.full((BT, BD), 0.0, tl.float32)
    for start in range(pid * BT, T, nprog * BT):
        t = start.to(tl.int64) + rr
        b = t // SEQ
        s = t % SEQ
        mask = (t[:, None] < T) & (d[None, :] < D // 2)
        gb = G + b[:, None] * GB + head * GH + s[:, None] * GT
        g1 = tl.load(gb + d[None, :] * GD, mask, 0).to(tl.float32)
        g2 = tl.load(gb + (D // 2 + d[None, :]) * GD, mask, 0).to(tl.float32)
        c = tl.load(C + s[:, None] * (D // 2) + d[None, :], mask, 0)
        sn = tl.load(S + s[:, None] * (D // 2) + d[None, :], mask, 0)
        dn1 = (g1 * c + g2 * sn).to(X.dtype.element_ty).to(tl.float32)
        dn2 = (g2 * c - g1 * sn).to(X.dtype.element_ty).to(tl.float32)
        xb = X + b[:, None] * XB + s[:, None] * XT + head * XH
        x1 = tl.load(xb + d[None, :], mask, 0).to(tl.float32)
        x2 = tl.load(xb + D // 2 + d[None, :], mask, 0).to(tl.float32)
        inv = tl.load(R + t * H + head, t < T, 0)[:, None]
        n1 = x1 * inv
        n2 = x2 * inv
        a1 = dn1 * w1
        a2 = dn2 * w2
        proj = (tl.sum(a1 * n1, 1) + tl.sum(a2 * n2, 1))[:, None] / D
        dx1 = inv * (a1 - n1 * proj)
        dx2 = inv * (a2 - n2 * proj)
        dst = DX + (t[:, None] * H + head) * D
        tl.store(dst + d[None, :], dx1, mask)
        tl.store(dst + D // 2 + d[None, :], dx2, mask)
        dw1 += dn1 * n1
        dw2 += dn2 * n2
    target = P + (pid.to(tl.int64) * H + head) * D
    tl.store(target + d, tl.sum(dw1, 0), d < D // 2)
    tl.store(target + D // 2 + d, tl.sum(dw2, 0), d < D // 2)


def fwd(x, w, c, s, eps, save):
    cfg = select_config(x.shape[2], x.shape[3], x.shape[0] * x.shape[1])
    bt, warps = cfg["ft"], cfg["fw"]
    (B, S, H, D) = x.shape
    y = torch.empty((B, H, S, D), device=x.device, dtype=x.dtype)
    rs = torch.empty((B, S, H) if save else (0,), device=x.device, dtype=torch.float32)
    _forward[triton.cdiv(B * S, bt), H](
        x,
        w,
        c,
        s,
        y,
        rs,
        B * S,
        S,
        H,
        D,
        eps,
        *x.stride()[:3],
        bt,
        triton.next_power_of_2(D // 2),
        save,
        num_warps=warps
    )
    return (y, rs)


def bwd(dy, x, w, c, s, rs):
    cfg = select_config(x.shape[2], x.shape[3], x.shape[0] * x.shape[1])
    bt, scale, warps = cfg["bt"], cfg["waves"], cfg["bw"]
    (B, S, H, D) = x.shape
    P = max(
        1,
        min(
            triton.cdiv(B * S, bt),
            scale
            * torch.cuda.get_device_properties(x.device).multi_processor_count
            // H,
        ),
    )
    dx = torch.empty_like(x, memory_format=torch.contiguous_format)
    part = torch.empty((P * H, D), device=x.device, dtype=torch.float32)
    dw = torch.empty_like(w)
    _backward[P, H](
        dy,
        x,
        w,
        c,
        s,
        rs,
        dx,
        part,
        B * S,
        S,
        H,
        D,
        *x.stride()[:3],
        *dy.stride(),
        bt,
        triton.next_power_of_2(D // 2),
        num_warps=warps
    )
    _reduce[triton.cdiv(D, 16),](
        part, dw, P * H, D, triton.next_power_of_2(P * H), 16, num_warps=4
    )
    return (dx, dw)


@triton.jit
def _reduce(P, W, N: tl.constexpr, D: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr):
    r = tl.arange(0, BN).to(tl.int64)
    d = tl.program_id(0).to(tl.int64) * BD + tl.arange(0, BD)
    v = tl.load(P + r[:, None] * D + d[None, :], (r[:, None] < N) & (d[None, :] < D), 0)
    tl.store(W + d, tl.sum(v, 0), d < D)


def select_config(h, d, tokens):
    """Token tiling based on head width and the number of parallel heads."""
    half = triton.next_power_of_2(d // 2)
    if half <= 32:
        ft, fw, bt, waves, bw = 16, 4, 8, (16 if h >= 8 else 8), 2
    elif half <= 64:
        ft, fw = 32, 4
        bt, waves, bw = (4 if h >= 32 else 8), (16 if h >= 32 else 8), 2
    elif half <= 128:
        ft, fw, bt, waves, bw = (
            8,
            (4 if d // 2 == half else 2),
            4,
            (4 if tokens < 1024 else 8),
            2,
        )
    else:
        ft, fw = max(1, 1024 // half), (2 if half <= 256 else 4)
        bt, waves, bw = 1, 16, (2 if half <= 1024 else (4 if half <= 2048 else 8))
    return dict(ft=ft, fw=fw, bt=bt, waves=waves, bw=bw)


def partial_count(x):
    b, s, h, d = x.shape
    cfg = select_config(h, d, b * s)
    return (
        max(
            1,
            min(
                triton.cdiv(b * s, cfg["bt"]),
                cfg["waves"]
                * torch.cuda.get_device_properties(x.device).multi_processor_count
                // max(1, h),
            ),
        )
        * h
    )


def supported(x):
    return (
        x.ndim == 4
        and 2 <= x.shape[-1] <= 8192
        and x.shape[-1] % 2 == 0
        and x.numel() // x.shape[-1] > 128
        and x.shape[0] * x.shape[1] > 32
        and x.dtype == torch.bfloat16
        and x.is_cuda
        and "H20" in torch.cuda.get_device_properties(x.device).name
    )


@triton.jit
def _pair_forward(
    Q,
    K,
    WQ,
    WK,
    C,
    S,
    YQ,
    YK,
    RQ,
    RK,
    T: tl.constexpr,
    SEQ: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    EPS,
    QB,
    QT,
    QH,
    KB,
    KT,
    KH,
    BT: tl.constexpr,
    BD: tl.constexpr,
    SAVE: tl.constexpr,
):
    which = tl.program_id(2).to(tl.int64)
    _forward(
        tl.where(which == 0, Q, K),
        tl.where(which == 0, WQ, WK),
        C,
        S,
        tl.where(which == 0, YQ, YK),
        tl.where(which == 0, RQ, RK),
        T,
        SEQ,
        H,
        D,
        EPS,
        tl.where(which == 0, QB, KB),
        tl.where(which == 0, QT, KT),
        tl.where(which == 0, QH, KH),
        BT,
        BD,
        SAVE,
    )


@triton.jit
def _pair_backward(
    GQ,
    GK,
    Q,
    K,
    WQ,
    WK,
    C,
    S,
    RQ,
    RK,
    DQ,
    DK,
    PQ,
    PK,
    T: tl.constexpr,
    SEQ: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    QB,
    QT,
    QH,
    KB,
    KT,
    KH,
    GQB,
    GQH,
    GQT,
    GQD,
    GKB,
    GKH,
    GKT,
    GKD,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    which = tl.program_id(2).to(tl.int64)
    _backward(
        tl.where(which == 0, GQ, GK),
        tl.where(which == 0, Q, K),
        tl.where(which == 0, WQ, WK),
        C,
        S,
        tl.where(which == 0, RQ, RK),
        tl.where(which == 0, DQ, DK),
        tl.where(which == 0, PQ, PK),
        T,
        SEQ,
        H,
        D,
        tl.where(which == 0, QB, KB),
        tl.where(which == 0, QT, KT),
        tl.where(which == 0, QH, KH),
        tl.where(which == 0, GQB, GKB),
        tl.where(which == 0, GQH, GKH),
        tl.where(which == 0, GQT, GKT),
        tl.where(which == 0, GQD, GKD),
        BT,
        BD,
    )


@triton.jit
def _pair_reduce(
    PQ, PK, WQ, WK, N: tl.constexpr, D: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr
):
    which = tl.program_id(1).to(tl.int64)
    _reduce(tl.where(which == 0, PQ, PK), tl.where(which == 0, WQ, WK), N, D, BN, BD)


class _Pair(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, wq, wk, cos, sin, eps, save):
        b, s, h, d = q.shape
        cfg = select_config(h, d, b * s)
        yq = torch.empty((b, h, s, d), device=q.device, dtype=q.dtype)
        yk = torch.empty_like(yq)
        rq = torch.empty(
            (b, s, h) if save else (0,), device=q.device, dtype=torch.float32
        )
        rk = torch.empty_like(rq)
        _pair_forward[(triton.cdiv(b * s, cfg["ft"]), h, 2)](
            q,
            k,
            wq,
            wk,
            cos,
            sin,
            yq,
            yk,
            rq,
            rk,
            b * s,
            s,
            h,
            d,
            eps,
            *q.stride()[:3],
            *k.stride()[:3],
            cfg["ft"],
            triton.next_power_of_2(d // 2),
            save,
            num_warps=cfg["fw"]
        )
        if save:
            ctx.save_for_backward(q, k, wq, wk, cos, sin, rq, rk)
        return yq, yk

    @staticmethod
    def backward(ctx, gq, gk):
        q, k, wq, wk, cos, sin, rq, rk = ctx.saved_tensors
        b, s, h, d = q.shape
        cfg = select_config(h, d, b * s)
        p = partial_count(q) // h
        dq = torch.empty_like(q, memory_format=torch.contiguous_format)
        dk = torch.empty_like(k, memory_format=torch.contiguous_format)
        pq = torch.empty((p * h, d), device=q.device, dtype=torch.float32)
        pk = torch.empty_like(pq)
        dwq, dwk = torch.empty_like(wq), torch.empty_like(wk)
        _pair_backward[(p, h, 2)](
            gq,
            gk,
            q,
            k,
            wq,
            wk,
            cos,
            sin,
            rq,
            rk,
            dq,
            dk,
            pq,
            pk,
            b * s,
            s,
            h,
            d,
            *q.stride()[:3],
            *k.stride()[:3],
            *gq.stride(),
            *gk.stride(),
            cfg["bt"],
            triton.next_power_of_2(d // 2),
            num_warps=cfg["bw"]
        )
        _pair_reduce[(triton.cdiv(d, 16), 2)](
            pq, pk, dwq, dwk, p * h, d, triton.next_power_of_2(p * h), 16, num_warps=4
        )
        return dq, dk, dwq, dwk, None, None, None, None


def pair(q, k, wq, wk, cos, sin, eps):
    save = torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, wq, wk))
    return _Pair.apply(q, k, wq, wk, cos, sin, eps, save)
