# GEMM + fused epilogue through nvmath-python (`gemm_tensorcore`, preferred)

Files: `kernel.py` (nvmath `Matmul` plans + the Triton twin's streaming pieces + autograd),
`reference.py` (eager `F.linear` + act). The table below compares cuBLAS, nvmath,
the Triton twin, eager and `torch.compile` side by side. Same math, host
signatures, saved-for-backward / recompute contract and tolerances as
[`triton/epilogue/gemm/SNIPPET.md`](../../../triton/epilogue/gemm/SNIPPET.md).

**Rule: if the epilogue is in cuBLASLt's set, use this and write no GEMM.** A GEMM written in
Triton or TileLang starts 5-50% behind cuBLAS's mainloop on sm80 (measured across the tables in
the two twins), and no epilogue fusion pays that back on the wide shapes; cuBLASLt applies the
epilogue inside the cuBLAS kernel for free. Fall back to the Triton twin only when `nvmath` is
not importable, the activation is outside the set (erf-GELU, SiLU, a SwiGLU gate, a quantised
store) or the GEMM is so short that nvmath's host overhead shows (below).

## Math

```
Z = X W^T              X: (M, K) any row stride, W: (N, K) contiguous (nn.Linear)
Y = act(Z + b)         b: (N,); act in {none, relu, gelu_tanh} fused; {gelu (erf), silu} as one streaming pass
```

Backward: the Triton twin's adjoint kernel (`dZ = dY * act'(Z)`, `db` from fp32 partials; here
`Z` is saved *with* the bias, `z_has_bias=True`), then `dX = dZ W`, `dW = dZ^T X` in cuBLAS.
`recompute=True` rebuilds `Z + b` with one `BIAS` GEMM.

## How nvmath is driven

- **Orientation.** cuBLASLt's `BIAS` vector runs along the rows of `a @ b`. For the
  per-output-feature bias of `nn.Linear` compute `Y^T = W X^T` (`a = w`, `b = x.t()`): the result
  is `Y^T` column-major, so `.t()` is the row-major `Y` for free, and the `gelu_aux` tensor
  (rows padded to a multiple of 8, slice `[:N]`) transposes the same way.
- **Epilogs used**: `BIAS`, `RELU[_BIAS]`, `GELU[_BIAS]`, `GELU_AUX[_BIAS]` (stores `Z + b` as the
  aux in storage dtype). `RELU_AUX` returns a bitmask, so ReLU with a backward runs `BIAS` plus
  one streaming ReLU. `MatmulEpilog.DEFAULT` is not accepted by `plan()`: pass `epilog=None`.
  cuBLASLt's GELU is the **tanh approximation**, which is what models call GELU today; the erf
  form differs by up to 2e-3 and is not fusable here.
- **Plan once.** A `Matmul` object per `(shapes, strides, dtype, epilog, has_bias, device)` in a
  dict; `plan()` once (~1.6 ms wall), `reset_operands()` + `execute()` per call. The function
  form `nvmath.linalg.advanced.matmul()` replans every call. A row-strided `x` is its own plan.
- **Host cost.** `execute()` plus the transposes cost ~175 us of Python per call against 18 us
  for `torch.matmul` and 75 us for `torch.matmul` + the Triton pointwise pass. A GEMM shorter
  than that on the device is CPU-bound through nvmath: `select_backend(M, N, K)` returns
  `"triton"` below `2MNK = 6e10` FLOP (~200 us on A800). CUDA graphs or a `torch.compile`d
  training step hide the overhead entirely, so a package used under `torch.compile` can ignore
  the boundary.
- The `Matmul` object holds its operands until the next `reset_operands` (one pinned activation
  per plan); `release_operands()` if that matters. It runs on the current torch stream.

## Measured (A800, bf16, `torch.profiler` device time, 2026-09-04)

Min of three interleaved rounds; the GPU is unlocked, compare within a table. `eager` is
`F.linear` + `F.gelu(approximate="tanh")`; `compile` is `torch.compile` of it.

| shape (M x N x K) | phase | cuBLAS alone | eager | compile | nvmath (Z saved) | nvmath infer | triton fused | triton cuBLAS + epi |
|---|---|---|---|---|---|---|---|---|
| 8192 x 4096 x 1024 (up-proj) | fwd | 327 us | 427 | 424 | **352** | 340 | 480 | 405 |
| | bwd | | 941 | 912 | 928 | | 949 | |
| 8192 x 1024 x 4096 (down-proj) | fwd | 369 | 400 | 401 | **315** | 311 | 387 | 391 |
| | bwd | | 772 | 758 | 766 | | 770 | |
| 4096 x 1024 x 1024 (D x D) | fwd | 75 | 93 | 96 | 74 | 72 | **68** | 88 |
| | bwd | | 172 | 167 | 171 | | 173 | |
| 8192 x 8192 x 2048 (`examples/fused_gemm_epilogue`) | fwd | 1029 | 1213 | 1198 | **1045** | 1018 | 1274 | 1171 |
| | bwd | | 2300 | 2347 | 2285 | | 2318 | |

Host time per call (4096 x 1024 x 1024, us): `torch.matmul` 18, Triton cuBLAS + epilogue 75,
nvmath 176.

What the table says:

- nvmath's forward is **1.16-1.27x eager and 1.15-1.27x `torch.compile`** on the three large
  shapes, where the fused Triton kernel is 0.89-1.03x eager. On the down-projection nvmath is
  even faster than `torch.matmul` alone (315 vs 369 us): cuBLASLt's heuristic picks a better
  kernel for that shape than torch's cuBLAS call does.
- The aux store (`GELU_AUX_BIAS` vs `GELU_BIAS`) costs 1-7% of the forward, the price of a
  backward-ready `Z`; infer mode (`save_aux=False`) skips it.
- On the small `D x D` shape the fused Triton kernel wins by 8% on device time, and nvmath's
  host cost (176 us against a 74 us kernel) makes it CPU-bound in eager mode: this is the case
  `select_backend` sends to Triton.
- The backward is a wash everywhere (two cuBLAS GEMMs dominate; the adjoint is bandwidth-bound).

## In a kernel package

SPEC `kernel_backend: nvmath`; `_nvmath/_impl_fwd.py` holds the plan cache and the forward,
`_impl_bwd.py` imports the adjoint from the package's Triton file or reimplements the 40-line
streaming kernel. The lint rules about `tl.dot` operands do not apply (there is no DSL kernel);
the roofline's `gemm_shapes` still lists the GEMM so the harness times cuBLAS on it, and the
verdict compares against `torch.compile` as for every other backend.
