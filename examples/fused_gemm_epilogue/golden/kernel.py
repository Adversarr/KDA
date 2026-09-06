"""Tanh-GELU GEMM with a fused Triton path for at most 256 rows.

Large matrices use the shared cuBLASLt epilogue reference when nvmath is available,
or the Triton reference's cuBLAS-plus-epilogue path otherwise. Small matrices fuse
bias and rounded GELU in Triton and reduce parameter gradients without partials.
Parameters arrive in fp32 and preserve the eager bf16 cast boundaries.
"""

import importlib.util
import sys
from pathlib import Path
from typing import Optional

import torch
import triton
import triton.language as tl

_REF = Path(__file__).resolve().parents[3] / "kda" / "kda-skills" / "kda-kernel-implement" / "reference"


_LOADED = {}


def _load(backend: str):
    if backend not in _LOADED:
        path = _REF / backend / "epilogue" / "gemm" / "kernel.py"
        spec = importlib.util.spec_from_file_location(f"kda_ref_{backend}_gemm_epilogue", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        _LOADED[backend] = mod
    return _LOADED[backend]


try:
    import nvmath  # noqa: F401

    BACKEND = "nvmath"
except ImportError:
    BACKEND = "triton"

_impl = _load(BACKEND)
gemm_epilogue = _impl.gemm_epilogue
gemm_epilogue_fwd = _impl.gemm_epilogue_fwd


@triton.jit
def _small_dz(DY, Z, DZ, DB, M: tl.constexpr, N: tl.constexpr,
              SYM, SYN, BM: tl.constexpr, BN: tl.constexpr):
    """One column tile reduces all small-batch rows without partial buffers."""
    m = tl.arange(0, BM)
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    mask = (m[:, None] < M) & (n[None, :] < N)
    z = tl.load(Z + m[:, None] * N + n[None, :], mask, 0).to(tl.float32)
    dy = tl.load(DY + m[:, None] * SYM + n[None, :] * SYN, mask, 0).to(tl.float32)
    u = 0.7978845608028654 * (z + 0.044715 * z * z * z)
    t = 2.0 * tl.sigmoid(2.0 * u) - 1.0
    grad = 0.5 * (1.0 + t) + 0.5 * z * (1.0 - t * t) * 0.7978845608028654 * (1.0 + 0.134145 * z * z)
    dz = (dy * grad).to(DZ.dtype.element_ty)
    tl.store(DZ + m[:, None] * N + n[None, :], dz, mask)
    db = tl.sum(dz.to(tl.float32), 0).to(DZ.dtype.element_ty)
    tl.store(DB + n, db, n < N)


@triton.jit
def _small_dw(DZ, X, DW, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
              SXM, BN: tl.constexpr, BK: tl.constexpr, BR: tl.constexpr):
    """Write the rounded low-precision GEMM result directly into fp32 parameters."""
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    k = tl.program_id(1) * BK + tl.arange(0, BK)
    r = tl.arange(0, BR)
    acc = tl.zeros((BN, BK), tl.float32)
    for start in range(tl.cdiv(M, BR)):
        rows = start * BR + r
        a = tl.load(DZ + rows[None, :] * N + n[:, None], (n[:, None] < N) & (rows[None, :] < M), 0)
        b = tl.load(X + rows[:, None] * SXM + k[None, :], (rows[:, None] < M) & (k[None, :] < K), 0)
        acc = tl.dot(a, b, acc)
    tl.store(DW + n[:, None] * K + k[None, :], acc.to(X.dtype.element_ty).to(tl.float32),
             (n[:, None] < N) & (k[None, :] < K))


@triton.jit
def _small_forward_kernel(X, W, B, Y, Z, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                          SX, SAVE: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for block in range(tl.cdiv(K, BK)):
        kk = block * BK + k
        x = tl.load(X + m[:, None] * SX + kk[None, :], (m[:, None] < M) & (kk[None, :] < K), 0)
        w = tl.load(W + n[None, :] * K + kk[:, None], (n[None, :] < N) & (kk[:, None] < K), 0)
        acc = tl.dot(x, w, acc)
    bias = tl.load(B + n, n < N, 0).to(tl.float32)
    z = (acc + bias[None, :]).to(X.dtype.element_ty).to(tl.float32)
    y = z * tl.sigmoid(1.5957691216057308 * (z + 0.044715 * z * z * z))
    mask = (m[:, None] < M) & (n[None, :] < N)
    tl.store(Y + m[:, None] * N + n[None, :], y, mask)
    if SAVE:
        tl.store(Z + m[:, None] * N + n[None, :], z, mask)


def _small_forward(x, w, b, save):
    m, k = x.shape
    n = w.shape[0]
    y = torch.empty((m, n), device=x.device, dtype=x.dtype)
    z = torch.empty_like(y) if save else x.new_empty(0)
    bm = 16 if m <= 64 else 32
    _small_forward_kernel[(triton.cdiv(m, bm), triton.cdiv(n, 128))](
        x, w, b, y, z, m, n, k, x.stride(0), save, BM=bm, BN=128, BK=(128 if m <= 64 else 64), num_warps=4, num_stages=3)
    return y, z


class _SmallGemm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, recompute):
        x2 = x.reshape(-1, x.shape[-1])
        w = weight.to(x.dtype)
        b = bias.to(x.dtype)
        y, z = _small_forward(x2, w, b, not recompute)
        ctx.save_for_backward(x2, w, b, z)
        ctx.input_shape, ctx.weight_dtype, ctx.bias_dtype = x.shape, weight.dtype, bias.dtype
        return y.view(*x.shape[:-1], weight.shape[0])

    @staticmethod
    def backward(ctx, dy):
        x, w, b, z = ctx.saved_tensors
        dy = dy.reshape(-1, dy.shape[-1])
        m, n = dy.shape
        if not z.numel():
            # Match the saved auxiliary's biased GEMM rounding under recompute.
            z = torch.addmm(b, x, w.t())
        dz = torch.empty((m, n), device=x.device, dtype=x.dtype)
        db = torch.empty((n,), device=x.device, dtype=ctx.bias_dtype)
        _small_dz[(triton.cdiv(n, 32),)](dy, z, dz, db, m, n, *dy.stride(),
            BM=triton.next_power_of_2(m), BN=32, num_warps=4)
        dx = torch.matmul(dz, w).view(ctx.input_shape)
        dw = torch.empty(w.shape, device=x.device, dtype=ctx.weight_dtype)
        _small_dw[(triton.cdiv(n, 64), triton.cdiv(x.shape[1], 64))](
            dz, x, dw, m, n, x.shape[1], x.stride(0), BN=64, BK=64, BR=32, num_warps=4)
        return dx, dw, db, None


def fc1_gelu(
    x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor], *, recompute: bool = False, backend: Optional[str] = None
) -> torch.Tensor:
    """Drop-in for ``language.feedforward.fc1_gelu``; ``backend`` forces ``"nvmath"`` or ``"triton"`` (tests, bench)."""
    if backend is None and bias is not None and 0 < x.numel() // x.shape[-1] <= 256:
        if torch.is_grad_enabled() and any(t.requires_grad for t in (x, weight, bias)):
            return _SmallGemm.apply(x, weight, bias, recompute)
        y, _ = _small_forward(x.reshape(-1, x.shape[-1]), weight.to(x.dtype), bias.to(x.dtype), False)
        return y.view(*x.shape[:-1], weight.shape[0])
    impl = _impl if backend in (None, BACKEND) else _load(backend)
    return impl.gemm_epilogue(x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype), "gelu_tanh", recompute=recompute)


__all__ = ["BACKEND", "fc1_gelu", "gemm_epilogue", "gemm_epilogue_fwd"]
