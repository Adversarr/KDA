# Golden verification and benchmarking

Each example keeps its workloads and operation-specific byte/FLOP formulas in
`benchmark_cases.py`. Its `benchmark.py` entrypoint uses the shared example runner, which
imports KDA's product runtime directly. Goldens contain kernels, independent eager references, a
readable result summary and structured evidence.

## Run the checks

From the checkout root, using the GPU Python interpreter:

```bash
python examples/fused_residual_rmsnorm/benchmark.py --verify --json tmp/golden-benchmarks/rmsnorm/verify.json
python examples/fused_residual_rmsnorm/benchmark.py --bench --json tmp/golden-benchmarks/rmsnorm/report.json
python examples/fused_residual_rmsnorm/benchmark.py --adjoint --json tmp/golden-benchmarks/rmsnorm/adjoint.json
```

Replace the example directory to select another operation. `--verify` checks outputs, gradients
and inference across all declared workloads, including recompute where supported. `--adjoint`
independently checks isolated outputs and broadcast upstream gradients on a small workload.
`--workload NAME` selects one case; `--fwd-only` omits backward. Scoped runs require an explicit
output path.

`--bench` verifies the kernel and each comparison baseline before timing. It uses profiler
device times, three warmup iterations and ten measured iterations in each of three interleaved
rounds. The reported time is the minimum round median. Eager, compiled and applicable attention
baselines must preserve the numerical contract. Acceptance compares against the fastest eligible
baseline, including SDPA when available. Raw product speedups compare eager/compiled only;
attention summaries also record the audit that includes SDPA.

Generated reports belong outside `golden/`. Preserve failed captures and diagnostic repeats;
never replace a complete record with a scoped result. `golden.json` retains measurement-time
source hashes and validation hashes. Kernel hashes identify the measured implementation;
validation hashes describe the runner used for that capture and can predate documentation or CLI
cleanup.

When documentation paths or unused standard-library imports are cleaned after measurement,
`published_source_hashes` identify the published files and `source_cleanup` records the
nonfunctional changes. Executable function bodies and JIT source remain unchanged. The original
`source_hashes` are preserved; cleanup is not presented as a new measurement.

## Performance gates

The [product roof
rules](../kda/kda-skills/kda-kernel-implement/reference/common/speed-of-light.md) and runtime
decisions are authoritative:

- Require achievable `roof_ms / kernel_ms >= 0.70` where SoL is gated.
- Require `best_baseline_ms / kernel_ms >= 0.95`.
- When the achievable roof is below 10 microseconds, report SoL without gating it
and retain the existing 1-microsecond absolute baseline parity allowance.
- Gate training forward, backward and inference as applicable. Recompute numerics
always count; recompute performance gates only when recompute is the default.
- Repeat near-gate failures twice as prescribed by the runtime. Conflicting results
remain incomplete; selecting the favorable capture is not acceptance.

Numerically correct kernels that miss a performance gate remain `tune`. Missing coverage,
unavailable calibration and unresolved measurement disagreement remain `incomplete`. Numerical
or contract failures remain `fail`.

## SoL accounting

Count compulsory reads and writes per kernel pass, including outputs, saved statistics,
reduction partials, parameter gradients and recomputation. This is a work model, not a
measurement of actual DRAM traffic: caches and repeated tile loads can change physical traffic.
Keep the datasheet bound separate from the measured achievable roof.

Attention uses `max(copy_ms, density * dense_flash_ms[phase])`, where density is allowed
query/key pairs divided by the full dense pair count. Calibration forces FlashAttention on
independent seeded tensors with the same batch, head, sequence, head-width and storage-dtype
geometry. GQA retains distinct query/KV head counts. Forward, backward and inference have
separate measurements; recompute adds forward and backward. Unsupported calibration has no GEMM
or math-attention fallback.

This is a useful-work throughput model. Dense FlashAttention can have different masking and
intermediate rounding, so calibration neither certifies numerical correctness nor predicts exact
sparse latency. A same-FLOP square GEMM does not measure achievable complete-attention
throughput.

For `P` allowed pairs and head width `D`, useful attention work is `4*P*D` forward and `10*P*D`
backward. Split backward can execute `14*P*D` because it reconstructs QK and dP in both gradient
passes. Precision emulation and duplicated work do not increase the useful-work numerator. With
`R=B*H*S` rows and `es` bytes per element, split backward traffic is `13*R*D*es + 20*R` before
operation-specific metadata:

| Pass | Activation vectors read or written | fp32 auxiliary bytes |
|---|---:|---:|
| Preprocess | O and dO: 2 | delta write: 4R |
| dK/dV | Q/K/V/dO reads, dK/dV writes: 6 | LSE and delta reads: 8R |
| dQ | Q/K/V/dO reads, dQ write: 5 | LSE and delta reads: 8R |

Padded attention counts `P=H*S*sum(key_valid)`, plus validity-mask reads. Sliding-tile attention
includes scheduling lists, counts, permutations, boundary-mask reads and the backward
mask-transpose copy. Cached list construction is cold setup.

GEMM/MLP products contribute `2*M*N*K` forward and `4*M*N*K` backward; recomputation adds
`2*M*N*K`. Retain casts, saved preactivations, activation-gradient traffic and parameter
reductions. A GEMM roof omits sequential activation/SFU costs and some layout effects;
efficiency above one requires an accounting audit, not an inflated roof estimate.

## Workload scope

Normal model workloads remain required even when expensive to validate. Deliberate odd-row and
abnormal stress cases remain visible as optional diagnostics in these examples. Their
mathematical tolerances and performance thresholds are unchanged.

| Example | Required model workloads | Optional diagnostics |
|---|---|---|
| GEMM epilogue | 8192 rows, 16 rows | 129-row tail, nonzero-bias 33 rows |
| Exact-GELU MLP | 5496 rows, 24-view 65952 rows | Strided 129-row tail, 7 rows |
| H3 attention | Primary, 14 frames, 40 frames | Strided tail, token-causal, single block |
| Padded attention | Frame, global, 24 views | Strided tail, single-valid, all-valid |
| Sliding-tile attention | Primary and 24-head real model; eight heads per window size | 439-token text tail |

The same designation applies to recompute variants. Other examples declare their required cases
directly in `benchmark_cases.py`. Historical records preserve the scope used when measured.

## Profiler limitations

The runtime reads raw Kineto GPU events and CPU iteration annotations, excluding synthetic GPU
annotations and cache flushes. Every iteration must have finite, positive device time. One retry
is allowed for incomplete collection; reports retain both attempts in `measurement_diagnostics`.
Kernel exceptions are not retried and explicit profiler measurements never fall back to
CUDA-event timing.

Large bounded-memory references can still produce expensive traces or lose GPU events.
Interrupted captures and partial samples cannot establish performance correctness. Each affected
golden records its specific limitation separately from its numerical results.
