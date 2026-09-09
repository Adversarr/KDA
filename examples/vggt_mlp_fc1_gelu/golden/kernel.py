"""Standalone exact-GELU MLP candidate with eager-compatible bf16 rounding."""
from typing import Dict

import torch
import triton
import triton.language as tl

ACTS: Dict[str, int] = {"none": 0, "relu": 1, "gelu": 2, "silu": 3, "gelu_tanh": 4}  # gelu = erf form; gelu_tanh = the approximation cuBLASLt/nvmath fuse
_MMA_DTYPES = (torch.bfloat16, torch.float16)


@triton.jit
def _act(z, ACT: tl.constexpr):
    """Activation on an fp32 tile. ACT codes follow ``ACTS``; gelu is the exact erf form."""
    if ACT == 1:
        return tl.maximum(z, 0.0)
    if ACT == 2:
        return 0.5 * z * (1.0 + tl.erf(z * 0.7071067811865476))
    if ACT == 3:
        return z * tl.sigmoid(z)
    if ACT == 4:  # tanh(u) = 2 sigmoid(2u) - 1
        u = 0.7978845608028654 * (z + 0.044715 * z * z * z)
        return z * tl.sigmoid(2.0 * u)
    return z


@triton.jit
def _act_grad(z, ACT: tl.constexpr):
    """``d act(z) / dz`` on an fp32 tile."""
    if ACT == 1:
        return tl.where(z > 0, 1.0, 0.0)
    if ACT == 2:
        cdf = 0.5 * (1.0 + tl.erf(z * 0.7071067811865476))
        pdf = tl.exp(-0.5 * z * z) * 0.3989422804014327
        return cdf + z * pdf
    if ACT == 3:
        s = tl.sigmoid(z)
        return s * (1.0 + z * (1.0 - s))
    if ACT == 4:
        u = 0.7978845608028654 * (z + 0.044715 * z * z * z)
        t = 2.0 * tl.sigmoid(2.0 * u) - 1.0
        return 0.5 * (1.0 + t) + 0.5 * z * (1.0 - t * t) * 0.7978845608028654 * (1.0 + 3.0 * 0.044715 * z * z)
    return tl.zeros_like(z) + 1.0


@triton.jit
def _gelu_forward(Z, Y, COUNT, BLOCK: tl.constexpr):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    z = tl.load(Z + offsets, offsets < COUNT, other=0).to(tl.float32)
    tl.store(Y + offsets, _act(z, 2), offsets < COUNT)


@triton.jit
def _epilogue_bwd_kernel(
    dy_ptr, z_ptr, bias_ptr, dz_ptr, db_partial_ptr, M, N, stride_dym, stride_dyn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, ACT: tl.constexpr, HAS_BIAS: tl.constexpr, Z_HAS_BIAS: tl.constexpr,
):
    # Adjoint of the epilogue: dZ = dY * act'(Z + b), streamed in [BLOCK_M, BLOCK_N] tiles (Z_HAS_BIAS:
    # the saved Z already contains b, as cuBLASLt's `*_AUX_BIAS` aux does; db is still reduced). A fixed
    # number of programs along M strides over the row tiles and keeps a register accumulator for
    # db, writing one fp32 partial row per program (host reduces (n_prog_m, N)), the same scheme
    # as the rmsnorm dw reduction.
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1).to(tl.int64)
    n_prog_m = tl.num_programs(0)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    nmask = rn < N
    if HAS_BIAS and not Z_HAS_BIAS:
        bias = tl.load(bias_ptr + rn, mask=nmask, other=0.0).to(tl.float32)
    db = tl.zeros([BLOCK_N], dtype=tl.float32)
    for m0 in range(pid_m * BLOCK_M, M, n_prog_m * BLOCK_M):
        rm = (m0 + tl.arange(0, BLOCK_M)).to(tl.int64)
        mask = (rm[:, None] < M) & nmask[None, :]
        dy = tl.load(dy_ptr + rm[:, None] * stride_dym + rn[None, :] * stride_dyn, mask=mask, other=0.0).to(tl.float32)
        if ACT == 0:
            dz = dy
        else:
            z = tl.load(z_ptr + rm[:, None] * N + rn[None, :], mask=mask, other=0.0).to(tl.float32)
            if HAS_BIAS and not Z_HAS_BIAS:
                z += bias[None, :]
            dz = dy * _act_grad(z, ACT)
        tl.store(dz_ptr + rm[:, None] * N + rn[None, :], dz.to(dz_ptr.dtype.element_ty), mask=mask)
        if HAS_BIAS:
            db += tl.sum(dz.to(dz_ptr.dtype.element_ty).to(tl.float32), axis=0)
    if HAS_BIAS:
        tl.store(db_partial_ptr + pid_m.to(tl.int64) * N + rn, db, mask=nmask)


def _rows(t: torch.Tensor) -> torch.Tensor:
    """Flatten leading dense dimensions without a hidden data copy."""
    if t.stride(-1) != 1:
        raise ValueError("last dimension must have unit stride")
    return t.view(-1, t.shape[-1])


def mlp_fwd(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, save_aux: bool = True):
    """Exact GELU over a bf16 biased linear output, matching eager autocast."""
    x2 = _rows(x)
    wlow, blow = w.to(x.dtype), b.to(x.dtype)
    z = torch.addmm(blow, x2, wlow.t())
    y = torch.empty_like(z)
    if z.numel():
        _gelu_forward[(triton.cdiv(z.numel(), 4096),)](z, y, z.numel(), BLOCK=4096)
    return y.view(*x.shape[:-1], w.shape[0]), z if save_aux else z.new_empty(0), wlow


def mlp_bwd(dy: torch.Tensor, x: torch.Tensor, wlow: torch.Tensor, b: torch.Tensor, z: torch.Tensor):
    """Fused GELU adjoint and bias partials followed by two cuBLAS GEMMs."""
    x2 = _rows(x)
    dy2 = dy.reshape(-1, dy.shape[-1])
    M, N = dy2.shape
    if z.numel() == 0:
        z = torch.addmm(b.to(x.dtype), x2, wlow.t())
    dz = torch.empty_like(z)
    if M == 0:
        return torch.zeros_like(x), torch.zeros_like(wlow), torch.zeros_like(b)
    ncol = triton.cdiv(N, 128)
    sms = torch.cuda.get_device_properties(x.device).multi_processor_count
    nrow = max(1, min(triton.cdiv(M, 32), 8 * sms // ncol))
    partial = torch.empty((nrow, N), device=x.device, dtype=torch.float32)
    _epilogue_bwd_kernel[(nrow, ncol)](
        dy2, z, b, dz, partial, M, N, dy2.stride(0), dy2.stride(1),
        BLOCK_M=32, BLOCK_N=128, ACT=2, HAS_BIAS=True, Z_HAS_BIAS=True, num_warps=4)
    dx = torch.matmul(dz, wlow).view(x.shape)
    # Finer output tiles and a fused dtype boundary help moderate reduction lengths.
    dw = _weight_gradient(dz, x2) if M <= 8192 else torch.matmul(dz.t(), x2)
    db = partial.sum(0).to(x.dtype).to(b.dtype)
    return dx, dw, db


class _MLP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b, recompute):
        y, z, wlow = mlp_fwd(x, w, b, not recompute)
        ctx.save_for_backward(x, wlow, b, z)
        ctx.weight_dtype = w.dtype
        return y

    @staticmethod
    def backward(ctx, dy):
        x, wlow, b, z = ctx.saved_tensors
        dx, dw, db = mlp_bwd(dy, x, wlow, b, z)
        return dx, dw.to(ctx.weight_dtype), db, None


@triton.jit
def _small_linear(X, W, B, Y, Z, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                  SX, SAVE: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for block in range(tl.cdiv(K, BK)):
        kk = block * BK + k
        x = tl.load(X + m[:, None] * SX + kk[None, :], (m[:, None] < M) & (kk[None, :] < K), 0)
        w = tl.load(W + n[None, :] * K + kk[:, None], (n[None, :] < N) & (kk[:, None] < K), 0).to(x.dtype)
        acc = tl.dot(x, w, acc)
    bias = tl.load(B + n, n < N, 0).to(X.dtype.element_ty).to(tl.float32)
    z = (acc + bias[None, :]).to(X.dtype.element_ty)
    mask = (m[:, None] < M) & (n[None, :] < N)
    tl.store(Y + m[:, None] * N + n[None, :], _act(z.to(tl.float32), 2), mask)
    if SAVE:
        tl.store(Z + m[:, None] * N + n[None, :], z, mask)


@triton.jit
def _small_adjoint(DY, Z, DZ, DB, M: tl.constexpr, N: tl.constexpr,
                   SYM, SYN, BM: tl.constexpr):
    m = tl.arange(0, BM)
    n = tl.program_id(0) * 16 + tl.arange(0, 16)
    mask = (m[:, None] < M) & (n[None, :] < N)
    z = tl.load(Z + m[:, None] * N + n[None, :], mask, 0).to(tl.float32)
    dy = tl.load(DY + m[:, None] * SYM + n[None, :] * SYN, mask, 0).to(tl.float32)
    dz = (dy * _act_grad(z, 2)).to(DZ.dtype.element_ty)
    tl.store(DZ + m[:, None] * N + n[None, :], dz, mask)
    tl.store(DB + n, tl.sum(dz.to(tl.float32), 0).to(DZ.dtype.element_ty), n < N)


@triton.jit
def _small_grad_tile(A, B, OUT, P: tl.constexpr, Q: tl.constexpr, R: tl.constexpr,
                   AP, AR, BR, BQ, pid_p, pid_q, BM: tl.constexpr = 32, BN: tl.constexpr = 32, BK: tl.constexpr = 32):
    """Small gradient GEMM, including autocast input and output rounding."""
    p = pid_p * BM + tl.arange(0, BM)
    q = pid_q * BN + tl.arange(0, BN)
    r = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for block in range(tl.cdiv(R, BK)):
        rr = block * BK + r
        a = tl.load(A + p[:, None] * AP + rr[None, :] * AR, (p[:, None] < P) & (rr[None, :] < R), 0)
        b = tl.load(B + rr[:, None] * BR + q[None, :] * BQ, (rr[:, None] < R) & (q[None, :] < Q), 0).to(a.dtype)
        acc = tl.dot(a, b, acc)
    tl.store(OUT + p[:, None] * Q + q[None, :], acc.to(A.dtype.element_ty), (p[:, None] < P) & (q[None, :] < Q))


@triton.jit
def _small_grad_pair(DZ, W, X, DX, DW, M: tl.constexpr, N: tl.constexpr,
                     K: tl.constexpr, SX):
    """Disjoint dX and dW tiles share one launch; each owns its output reduction."""
    pid = tl.program_id(0)
    nk = tl.cdiv(K, 32)
    nx = tl.cdiv(M, 32) * nk
    if pid < nx:
        _small_grad_tile(DZ, W, DX, M, K, N, N, 1, K, 1, pid // nk, pid % nk)
    else:
        pw = pid - nx
        _small_grad_tile(DZ, X, DW, N, K, M, 1, N, SX, 1, pw // nk, pw % nk)


class _SmallMLP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b, recompute):
        y, z = _small_forward(x, w, b, not recompute)
        ctx.save_for_backward(x, w, b, z)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, w, b, z = ctx.saved_tensors
        x2, dy2 = _rows(x), dy.reshape(-1, dy.shape[-1])
        m, n = dy2.shape
        k = x2.shape[1]
        if not z.numel():
            _, z = _small_forward(x, w, b, True)
        dz = torch.empty((m, n), dtype=x.dtype, device=x.device)
        db = torch.empty_like(b)
        _small_adjoint[(triton.cdiv(n, 16),)](dy2, z, dz, db, m, n, *dy2.stride(),
            BM=triton.next_power_of_2(m), num_warps=4)
        dx = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        dw = torch.empty_like(w)
        grid = (triton.cdiv(m, 32) + triton.cdiv(n, 32)) * triton.cdiv(k, 32)
        _small_grad_pair[(grid,)](dz, w, x2, dx, dw, m, n, k, x2.stride(0), num_warps=4)
        return dx, dw, db, None


def _small_forward(x, w, b, save):
    x2 = _rows(x)
    m, k = x2.shape
    n = w.shape[0]
    y = torch.empty((*x.shape[:-1], n), dtype=x.dtype, device=x.device)
    z = torch.empty_like(y) if save else x.new_empty(0)
    _small_linear[(triton.cdiv(m, 16), triton.cdiv(n, 64))](
        x2, w, b, y, z, m, n, k, x2.stride(0), save, BM=16, BN=64, BK=32, num_warps=4)
    return y, z


def mlp_fc1_gelu(x: torch.Tensor, w1: torch.Tensor, b1: torch.Tensor, *, recompute: bool = False):
    """Training uses saved preactivation by default; recompute trades an extra GEMM for it."""
    training = torch.is_grad_enabled() and any(t.requires_grad for t in (x, w1, b1))
    if x.numel() // x.shape[-1] <= 256 and w1.shape[0] <= 512:
        return _SmallMLP.apply(x, w1, b1, recompute) if training else _small_forward(x, w1, b1, False)[0]
    if not training:
        return mlp_fwd(x, w1, b1, False)[0]
    return _MLP.apply(x, w1, b1, recompute)

@triton.jit
def _weight_grad(DZ, X, DW, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                 SX, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    n = tl.program_id(0) * BM + tl.arange(0, BM)
    k = tl.program_id(1) * BN + tl.arange(0, BN)
    r = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for block in range(tl.cdiv(M, BK)):
        rows = block * BK + r
        a = tl.load(DZ + rows[None, :] * N + n[:, None], (rows[None, :] < M) & (n[:, None] < N), 0)
        b = tl.load(X + rows[:, None] * SX + k[None, :], (rows[:, None] < M) & (k[None, :] < K), 0)
        acc = tl.dot(a, b, acc)
    # Preserve autocast's bf16 parameter-adjoint boundary without a conversion pass.
    tl.store(DW + n[:, None] * K + k[None, :], acc.to(DZ.dtype.element_ty).to(tl.float32), (n[:, None] < N) & (k[None, :] < K))


def _weight_gradient(dz, x):
    m, n = dz.shape
    k = x.shape[1]
    out = torch.empty((n, k), device=x.device, dtype=torch.float32)
    _weight_grad[(triton.cdiv(n, 128), triton.cdiv(k, 64))](
        dz, x, out, m, n, k, x.stride(0), BM=128, BN=64, BK=64, num_warps=4, num_stages=3)
    return out
