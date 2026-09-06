# Golden: fused residual add + RMSNorm

Final verdict: **pass**. All required numerical and performance rows pass. Full output/gradient
checks, phase times, measured copy roofs, separate datasheet bounds, baseline ratios, round
samples and available repeats are recorded in `golden.json`. Source-matched full required
numerical and performance coverage, eligible baselines, independent adjoints and near-gate
repeats audited; all pass.

Hand-written target for `add_rmsnorm` in `user_repo/minilm/model.py`, with recorded A800
reference measurements below.

Fusion: one program per row loads `x` (bf16) and `residual` (fp32) once, writes the fp32 stream
`h`, the bf16 normalised view `y` and, only when a backward can follow (`save_rstd`), `rstd`.
The backward is the rmsnorm snippet's backward with `dh` added to the row gradient before the
store and a second, fp32 store for `dresidual`; `h` is saved because it is an output anyway (no
extra traffic), so nothing is recomputed. A candidate kernel that writes `rstd` unconditionally
is a review finding even when its numbers match.

Contract (the v1.1 lint rules, all mechanical): every input keeps its own row stride into the
kernel (`x`, `residual`, and `dy`/`dh` as autograd hands them over); there is no `.contiguous()`
and no `.reshape` anywhere (`view` / `as_strided` for the `(..., D) -> (N, D)` flattening, which
raises instead of copying on a layout it cannot express); row offsets are int64; `rstd` is
stored under a constexpr. Forward geometry follows the row width rule: one program per row for
`D <= 32768` whatever the row count (the 32-row case runs the same kernel), split-D (stage 1
writes `h` and a partial sum of squares per chunk, stage 2 re-reads the fp32 `h`) beyond that,
forward-only since the fused backward tile caps at `D = 8192`.

For at most 128 rows and widths through 8192, each backward block owns one activation row and
disjoint weight-gradient columns. It writes the final weight gradient directly, removing partial
storage and the second reduction launch. Larger inputs retain the persistent backward.

## A800 measurements

4096 x 1024 (batch 8, seq 512), bf16 x / fp32 residual and weight.

Device time from `torch.profiler`; roof = same-size `copy_`.

| Phase | Golden | Eager | Compiled | SoL |
|---|---:|---:|---:|---:|
| forward (training) | 0.03232 ms | 0.15344 ms | 0.03386 ms | 116.1% |
| forward (inference) | 0.03242 ms | 0.15384 ms | 0.03312 ms | 116.7% |
| backward | 0.05774 ms | 0.34690 ms | 0.07282 ms | 76.9% |

All phase entries were timed round-robin (min of three rounds), so compare the ratios within the
table on the measured GPU.

Bytes counted: forward `N*D*(2*es+8) + 4*D + 4*N` (inference omits `4*N`); backward
`N*D*(2*es+12) + 8*D + 4*N + 8*P*D`. `P` is the actual partial-program count, or zero for the
direct small-input backward.

What to expect from a candidate kernel: forward at copy speed (there is nothing else to gain);
backward between 70% and 80% of copy with a strided grid of ~2 programs per SM and one fp32 `dw`
partial per program. Per-row `dw` partials or `atomic_add` for `dw` show up as a backward below
50%. Forgetting `dh` (the gradient into the residual output) shows up as `dresidual` and `dx`
mismatches on every workload.

Note on the forward: `torch.compile` fuses this chain equally well, so the forward speedup over
the compiled baseline is ~0; the kernel's value is the fused backward (the compiled backward of
this chain is 1.3x slower than the golden: Inductor splits it and reduces `dw` separately) and,
for users not on `torch.compile`, the 4-6x over eager on both passes. Under the verdict's 5%
baseline tolerance the forward is "parity", not "slower".

## Reproduce the checks

From the checkout root, using the GPU Python interpreter:

```bash
python examples/fused_residual_rmsnorm/benchmark.py --verify --json tmp/golden-benchmarks/fused_residual_rmsnorm/verify.json
python examples/fused_residual_rmsnorm/benchmark.py --bench --json tmp/golden-benchmarks/fused_residual_rmsnorm/report.json
python examples/fused_residual_rmsnorm/benchmark.py --adjoint --json tmp/golden-benchmarks/fused_residual_rmsnorm/adjoint.json
```

[Workloads and SoL accounting](../benchmark_cases.py) define the inputs, gradient reductions,
and per-phase bytes/FLOPs. The [shared runner](../../_benchmark.py) uses KDA's product runtime
directly. Reports are generated outside `golden/`; new numerical and timing reports require an
independent acceptance audit. Use a distinct output filename for scoped `--workload` or
`--fwd-only` runs.
