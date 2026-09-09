# Golden: VGGT linear + exact GELU (`mlp_fc1_gelu`)

Hand-written target for `mlp_fc1_gelu` in `user_repo/models/mlp.py`: the first `1024 -> 4096`
MLP projection, its bias, and exact erf GELU. The second projection and residual addition remain
separate. This is the withheld reference solution for evaluating an agent's output. Numerical
coverage is complete; performance acceptance is **pass** under the documented GEMM compiled-parity policy.

The forward uses cuBLAS `addmm` for the biased linear product, followed by one 4096-element
streaming Triton GELU kernel. The choice follows the existing wide-output GEMM finding that a
custom mainloop can lose more throughput than a separate epilogue costs. It is an initial
implementation choice: this golden has no new measurement proving that choice beats a fused
alternative, and does not claim to execute GEMM and GELU in a single launch.

Contract: the fp32 weight and bias are cast to bf16 as eager autocast does. The *biased* linear
result then rounds to bf16 before exact GELU. `matmul` followed by a separately rounded bias
addition is not the same boundary. Neither is applying GELU directly to an unrounded
accumulator. The tanh approximation supported by cuBLASLt's GELU epilogue is a different
activation; substituting it does not implement this task. Inputs keep regular row strides
through `view`, which raises instead of copying when the leading dimensions cannot be flattened
without a copy.

Backward streams the saved biased preactivation Z and dY through one Triton kernel: `dZ = dY *
(Phi(Z) + Z*phi(Z))`. It writes bf16 dZ and accumulates deterministic fp32 bias partials over
row tiles. Each loop reduces its tile into a column vector, shortening the accumulator lifetime.
Eight program waves distribute the row reduction across the GPU. The host launches their
reduction and `dX = dZ @ W` through cuBLAS. For at most 8192 reduction rows, a 128-by-64, K64 Triton
GEMM computes `dW = dZ.T @ X` and rounds its accumulator to bf16 before storing fp32
parameter gradients. This improves output-tile parallelism and removes the separate dtype
conversion pass. Longer reductions retain cuBLAS. Parameter gradients observe
the eager bf16 autocast boundary and are returned in fp32. Bias reduction is included in the
backward cost; there are no per-element parameter atomics.

For small inputs (at most 256 rows and 512 output channels), one Triton GEMM loads and rounds
fp32 parameters in registers, adds bias, rounds the biased linear output to bf16, and applies
exact GELU. Backward reduces bias directly inside the GELU adjoint and launches disjoint dX/dW
tile groups together, writing parameter gradients directly in fp32 after the required bf16
rounding. Broadcast output gradients are supported through explicit strides.

Default training retains bf16 Z. At the user workload this is 5496*4096*2 = 45,023,232 bytes,
four times the input activation volume. `recompute=True` saves no Z and rebuilds it with one
additional biased GEMM in backward, adding `2*M*N*K` FLOPs plus its traffic. Inference retains
no backward auxiliary, although this two-kernel forward still materializes a temporary Z for
GELU to read. Saving an existing GEMM result for backward adds lifetime, not a second forward
store.

## A800 measurements

Fresh full-workload measurements on 2026-09-08 use profiler device times and three interleaved
rounds, resetting compiler state per workload. Both required model shapes and both diagnostic
shapes pass output, inference and gradient comparisons, as do all four recomputed variants.

| Workload | Forward | Backward | Inference | Forward / backward / inference SoL |
|---|---:|---:|---:|---|
| user | 0.264145 ms | 0.564016 ms | 0.264496 ms | 0.701 / 0.856 / 0.700 |
| real_24_views | 2.793211 ms | 5.229622 ms | 2.814011 ms | 0.796 / 0.848 / 0.773 |

Status is **pass** under the explicitly authorized GEMM policy: numerical correctness and
eligible compiled parity >=0.95x determine acceptance; SoL remains visible as a diagnostic.
Primary backward is 0.981x compiled in the full run and remains above 0.95x in both independent
repeats. All required forward/inference comparisons also meet the compiled gate. One repeat's
inference SoL is 0.692; this would miss the old strict 70% gate and is retained rather than
rounded up. The original captures and subsequent policy review are distinguishable in the JSON.

Independent adjoints pass on small, strided-tail and full primary workloads. Optional small/tail
numerics pass, but their timing captures lost GPU events; those missing timings remain explicit
and do not waive any required model shape. The older tuning capture is preserved under
`historical_capture`. No numerical tolerance or eager rounding boundary was relaxed.

Bytes and FLOPs for the roofline: forward executes `2*M*N*K` tensor-product FLOPs. For this
two-kernel design its compulsory traffic is `6*(N*K+N) + 2*(M*K+N*K+N+3*M*N)` bytes:
fp32-to-bf16 parameter casts, addmm inputs/output, then the Z read and Y write. The saved and
inference paths have the same streaming traffic; they differ in Z lifetime.

Backward executes `4*M*N*K` tensor-product FLOPs. Count dY and Z reads, dZ's write and its two
GEMM reads, X/W reads, dX/dW writes, fp32 bias partial writes and reads, and both
parameter-gradient dtype conversions. With p row programs, these accesses sum to `10*M*N + 4*M*K
+ C*N*K + 8*p*N + 16*N` bytes, with C=6 for the direct fp32 dW store (M<=8192) and C=10
for the cuBLAS-plus-conversion path, excluding cache reuse. For the small fused path, forward traffic
is `2*M*K + 4*N*K + 4*N + 2*M*N`, plus `2*M*N` for saved Z during training. Backward traffic is
`10*M*N + 4*M*K + 8*N*K + 4*N`; there are no low-precision parameter buffers or bias partials.
Recompute adds the actual rebuilding forward traffic and `2*M*N*K` FLOPs. The large path adds
its biased GEMM and bias cast. Compare the measured cuBLAS roof on the forward and two backward
GEMM shapes with the copy floor; do not infer achievable throughput from datasheet peak alone.

What to expect from a candidate: the exact activation and rounding contract, a mainloop selected
by measured cost, complete gradients, and an explicit saved-versus- recomputed trade-off. A
hand-written GEMM slower than cuBLAS can erase the benefit of fusion; tanh-GELU can look fast
while solving the wrong problem. Per-row bias partials increase traffic, atomics compromise
deterministic reduction, and excluding casts or bias reduction from the timed region produces an
invalid comparison.

## Reproduce the checks

From the checkout root, using the GPU Python interpreter:

```bash
python examples/vggt_mlp_fc1_gelu/benchmark.py --verify --json tmp/golden-benchmarks/vggt_mlp_fc1_gelu/verify.json
python examples/vggt_mlp_fc1_gelu/benchmark.py --bench --json tmp/golden-benchmarks/vggt_mlp_fc1_gelu/report.json
python examples/vggt_mlp_fc1_gelu/benchmark.py --adjoint --json tmp/golden-benchmarks/vggt_mlp_fc1_gelu/adjoint.json
```

[Workloads and SoL accounting](../benchmark_cases.py) define the inputs, gradient reductions,
and per-phase bytes/FLOPs. The [shared runner](../../_benchmark.py) uses KDA's product runtime
directly. Reports are generated outside `golden/`; new numerical and timing reports require an
independent acceptance audit. Use a distinct output filename for scoped `--workload` or
`--fwd-only` runs.
