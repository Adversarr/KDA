# Golden: GEMM + fused epilogue (`fc1_gelu`)

Hand-written target for `fc1_gelu` in `user_repo/minilm/model.py` (`gelu_tanh(x @ W1.T + b1)`,
the MLP's first projection with its epilogue), with recorded A800 reference measurements.

The golden is two lines of policy in `kernel.py`:

```
if nvmath-python is importable -> reference/nvmath/epilogue/gemm  (cuBLASLt GELU_AUX_BIAS epilogue, no kernel)
else                            -> reference/triton/epilogue/gemm  (select_mainloop: cuBLAS + one epilogue kernel here)
```

**nvmath path.** `Matmul(w, x.t())` planned once per shape with the
`GELU_AUX_BIAS` epilog: the GEMM, the bias, the tanh-GELU and the store of the pre-activation
`Z + b` (the aux for the backward) run inside the cuBLAS kernel; the result comes back as `Y^T`
column-major and `.t()` is the row-major `Y` for free. Infer mode uses `GELU_BIAS` (no aux).
The backward is the Triton twin's adjoint (`dZ = dY * gelu'(Z + b)`, fp32 `db` partials) and
two cuBLAS GEMMs. Recompute rebuilds `Z + b` with one `BIAS` GEMM.

**Triton fallback.** Same host API; on this 8192-wide shape `select_mainloop` runs the GEMM in
`torch.matmul` and the epilogue as one pointwise kernel; the fused `tl.dot` mainloop is 9%
slower than that and 22% slower than nvmath.

Contract: `x` keeps its row stride (for example, `x` as a slice of a wider
activation); no `.contiguous()` on inputs, `view` only for the `(..., K) -> (M, K)` flattening;
the fp32 parameters are cast to `x.dtype` once, as the eager code does; `Z` is stored only when
a backward can follow. `gelu_tanh` is what the user's code calls (`approximate="tanh"`), and it
is the only GELU cuBLASLt fuses; a kernel that implements the erf form is off by up to 2e-3.

## A800, user workload M 8192 (batch 8 x seq 1024) x N 8192 x K 2048, bf16

Device time from `torch.profiler`, min of three interleaved rounds; the
GPU is unlocked, compare within the table. `eager` is the user's function (cuBLAS + two
pointwise kernels), `compile` is `torch.compile` of it (cuBLAS + one fused pointwise kernel).
FLOP rate is `2*M*N*K / t` (backward `4*M*N*K`, the two GEMMs). 2026-09-04.

| | time | TFLOP/s | vs eager | vs compile |
|---|---|---|---|---|
| cuBLAS `matmul` alone | 1.017 ms | 270 | | |
| forward eager (user code) | 1.212 ms | 227 | | |
| forward `torch.compile` | 1.198 ms | 229 | 1.01x | |
| **forward golden (nvmath, Z saved)** | **1.042 ms** | 264 | **1.16x** | **1.15x** |
| inference golden (nvmath, no Z) | 1.017 ms | 270 | 1.19x | 1.18x |
| forward Triton fallback (cuBLAS + epilogue kernel, Z saved) | 1.172 ms | 235 | 1.03x | 1.02x |
| forward Triton fused `tl.dot` mainloop | 1.275 ms | 216 | 0.95x | 0.94x |
| forward TileLang twin (Z saved) | 1.470 ms | 187 | 0.82x | 0.81x |
| backward eager | 2.281 ms | 241 | | |
| backward `torch.compile` | 2.348 ms | 234 | 0.97x | |
| backward golden (saved Z) | 2.286 ms | 240 | 1.00x | 1.03x |
| backward golden (recompute Z) | 3.348 ms | 164 | 0.68x | |
| backward Triton fallback (saved Z) | 2.318 ms | 237 | 0.98x | |
| backward TileLang twin (saved Z) | 2.738 ms | 201 | 0.83x | |

The golden forward sits 2.5% above the bare cuBLAS GEMM (the aux store of `Z`), and the
inference forward *is* the bare GEMM: there is nothing left to fuse in this forward. The
backward is a wash by construction (two cuBLAS GEMMs dominate; the adjoint is bandwidth-bound
and matches inductor's).

What to expect from a candidate: a package that drives nvmath lands at 1.04-1.05 ms (1.15x
compile) and is the intended answer; a hand-written Triton GEMM lands at 1.27-1.30 ms with the
reference's tile table and at 1.45-1.60 with the tutorial matmul's `% M` wrap or an 8-warp
tile, i.e. *slower than eager*; a TileLang GEMM at 1.47. A kernel that upcasts an operand to
fp32 (`mma_operand`) is 8-16x slower. A `db` computed with per-row atomics or an unfused `dZ`
shows as a backward above 2.5 ms. The activation-memory question in the TASK has one answer:
`recompute` drops the `M*N*es` = 128 MiB `Z` per layer for a 1.06 ms extra GEMM in the
backward (0.68x), or `Z` is saved and the forward is the 1.04 ms above.

Bytes and FLOPs for the roofline: forward `2*M*N*K` FLOPs at the achievable cuBLAS rate plus
`M*(K+N)*es + N*es` bytes (x, y, bias) and `M*N*es` more for `Z`. Backward: adjoint
`3*M*N*es + n_prog*N*4` bytes, GEMMs `4*M*N*K` FLOPs.
