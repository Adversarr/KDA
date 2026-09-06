# Golden: GEMM + fused epilogue (`fc1_gelu`)

The final golden passes numerical and performance acceptance. All eight saved/recomputed
numerical workloads, four timed shapes and independent adjoint checks pass in the fresh
final-source run.

Hand-written target for `fc1_gelu` in `user_repo/language/feedforward.py` (`gelu_tanh(x @ W1.T +
b1)`, the MLP's first projection with its epilogue), with recorded A800 reference measurements.

The large-row backend policy in `kernel.py` is:

```
if nvmath-python is importable -> reference/nvmath/epilogue/gemm  (cuBLASLt GELU_AUX_BIAS epilogue, no kernel)
else                            -> reference/triton/epilogue/gemm  (select_mainloop: cuBLAS + one epilogue kernel here)
```

**nvmath path.** `Matmul(w, x.t())` planned once per shape with the `GELU_AUX_BIAS` epilog: the
GEMM, the bias, the tanh-GELU and the store of the pre-activation `Z + b` (the aux for the
backward) run inside the cuBLAS kernel; the result comes back as `Y^T` column-major and `.t()`
is the row-major `Y` for free. Infer mode uses `GELU_BIAS` (no aux). The backward is the Triton
twin's adjoint (`dZ = dY * gelu'(Z + b)`, fp32 `db` partials) and two cuBLAS GEMMs. Recompute
rebuilds `Z + b` with one `BIAS` GEMM.

**Triton fallback.** Same host API; on this 8192-wide shape `select_mainloop` runs the GEMM in
`torch.matmul` and the epilogue as one pointwise kernel; the fused `tl.dot` mainloop is 9%
slower than that and 22% slower than nvmath.

Contract: `x` keeps its row stride (for example, `x` as a slice of a wider activation); the
large path uses `view` for `(..., K) -> (M, K)` flattening; the small path uses `reshape`, which
can copy an incompatible leading layout; the fp32 parameters are cast to `x.dtype` once, as the
eager code does; `Z` is retained for backward only when a backward can follow; the Triton
fallback still materializes its temporary preactivation. `gelu_tanh` is what the user's code
calls (`approximate="tanh"`), and it is the only GELU cuBLASLt fuses; a kernel that implements
the erf form is off by up to 2e-3.

## Historical A800 reference, user workload M 8192 (batch 8 x seq 1024) x N 8192 x K 2048, bf16

Device time from `torch.profiler`, min of three interleaved rounds; the GPU is unlocked, compare
within the table. `eager` is the user's function (cuBLAS + two pointwise kernels), `compile` is
`torch.compile` of it (cuBLAS + one fused pointwise kernel). FLOP rate is `2*M*N*K / t`
(backward `4*M*N*K`, the two GEMMs). 2026-09-04.

| Phase | Golden | Eager | Compiled | SoL |
|---|---:|---:|---:|---:|
| forward | 1.0877 ms | 1.2748 ms | 1.2888 ms | 93.5% |
| inference | 1.0675 ms | 1.2748 ms | 1.2979 ms | 95.4% |
| backward | 2.3525 ms | 2.3661 ms | 2.3426 ms | 84.0% |

The golden forward sits 2.5% above the bare cuBLAS GEMM (the aux store of `Z`), and the
inference forward *is* the bare GEMM: there is nothing left to fuse in this forward. The
backward is a wash by construction (two cuBLAS GEMMs dominate; the adjoint is bandwidth-bound
and matches inductor's).

What to expect from a candidate: a package that drives nvmath lands at 1.04-1.05 ms (1.15x
compile) and is the intended answer; a hand-written Triton GEMM lands at 1.27-1.30 ms with the
reference's tile table and at 1.45-1.60 with the tutorial matmul's `% M` wrap or an 8-warp tile,
i.e. *slower than eager*; a TileLang GEMM at 1.47. A kernel that upcasts an operand to fp32
(`mma_operand`) is 8-16x slower. A `db` computed with per-row atomics or an unfused `dZ` shows
as a backward above 2.5 ms. The activation-memory question in the TASK has one answer:
`recompute` drops the `M*N*es` = 128 MiB `Z` per layer for a 1.06 ms extra GEMM in the backward
(0.68x), or `Z` is saved and the forward is the 1.04 ms above.

Bytes and FLOPs for the measured cuBLASLt path: forward is `2*M*N*K` useful FLOPs and `2*M*K +
8*N*K + 8*N + 2*M*N` bytes, plus `2*M*N` when training saves Z. Include the fp32 parameter
reads, bf16 cast writes and operand reads. Backward is `4*M*N*K` FLOPs and `10*M*N + 4*M*K +
10*N*K + 16*N + 8*p*N` bytes, with p bias-partial rows. Partials are written and read once.
Recompute adds one biased GEMM's reads and Z write. The Triton fallback has a separate Z
write/read and Y write in every forward mode; its precise alternate count is kept beside the
benchmark in `benchmark_cases.py`.

## Small-row tuning and full coverage

For at most 256 rows, a tiled Triton GEMM fuses the biased preactivation and tanh-GELU,
retaining the bf16 boundary required by eager `F.linear`. The small backward reduces bias inside
the activation adjoint and writes the bf16-rounded weight product directly to fp32 storage. This
removes a separate weight-gradient cast and the bias partial buffer. The large workload retains
the cuBLASLt path.

Fresh full-workload measurements on 2026-09-06, minimum of three interleaved profiler rounds
with fresh compiler state per workload:

| Workload | Forward | Backward | Inference | Forward / backward / inference SoL |
|---|---:|---:|---:|---|
| user_b8_s1024_d2048_h8192 | 1.086368 ms | 2.352282 ms | 1.068417 ms | 0.937 / 0.839 / 0.951 |
| small_rows | 0.087856 ms | 0.074176 ms | 0.087792 ms | 0.933 / 0.917 / 0.937 |
| tail_rows | 0.110896 ms | 0.107647 ms | 0.111920 ms | 0.785 / 1.400 / 0.771 |
| nonzero_bias | 0.094927 ms | 0.080320 ms | 0.097151 ms | 0.904 / 1.045 / 0.877 |

The extra round increases the small forward reduction tile from 64 to 128 channels for at most
64 rows; larger tails retain 64. A uniform 128-channel tile regressed the 129-row case and was
rejected. The final configuration passes all four timed cases, all four recomputed variants and
independent adjoints. The 16-row inference is 0.087792 ms versus 0.093855 ms before this round.
The odd-row cases remain visible diagnostics under the accepted suite scope; this final run also
passes their gates. The status is **pass**.

For small backward, compulsory traffic is `10*M*N + 4*M*K + 6*N*K + 4*N` bytes and work is
`4*M*N*K` FLOPs. The two GEMM roof shapes remain `(M,K,N)` and `(N,K,M)`. The 129-row efficiency
exceeds one because the measured cuBLAS roof includes a slow small-reduction GEMM; the custom
product beats this library measurement. It is not a datasheet efficiency or evidence that the
FLOP count should be raised. Both bounds and raw samples are preserved.

## Reproduce the checks

From the checkout root, using the GPU Python interpreter:

```bash
python examples/fused_gemm_epilogue/benchmark.py --verify --json tmp/golden-benchmarks/fused_gemm_epilogue/verify.json
python examples/fused_gemm_epilogue/benchmark.py --bench --json tmp/golden-benchmarks/fused_gemm_epilogue/report.json
python examples/fused_gemm_epilogue/benchmark.py --adjoint --json tmp/golden-benchmarks/fused_gemm_epilogue/adjoint.json
```

[Workloads and SoL accounting](../benchmark_cases.py) define the inputs, gradient reductions,
and per-phase bytes/FLOPs. The [shared runner](../../_benchmark.py) uses KDA's product runtime
directly. Reports are generated outside `golden/`; new numerical and timing reports require an
independent acceptance audit. Use a distinct output filename for scoped `--workload` or
`--fwd-only` runs.
