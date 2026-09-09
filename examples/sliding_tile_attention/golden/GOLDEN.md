# Golden: Sliding Tile Attention (`sliding_tile_attention`)

Hand-written target for `sliding_tile_attention` in `user_repo/sta/attention.py`: video tokens
in tile order plus text, with a per-head tile mask. Text queries see all keys and text keys are
visible to every query. This withheld solution remains **incomplete** because real-model
performance evidence is missing.

## Design

A streaming FA2 kernel traverses cached per-head lists of allowed key blocks. Fully visible
blocks come first and need no element mask; partially visible blocks reconstruct the tile
predicate. A heavy-first permutation schedules long lists before short ones so heterogeneous
head windows do not leave a long GPU tail. The cache retains the mask owner and its version;
in-place mutation invalidates lists, and inference tensors without version counters are not
cached.

Forward uses 128 query rows, 128 keys, 8 warps and 2 stages. Three stages exceed the
shared-memory limit with indirect K/V addresses. A fully denied initial tile keeps a finite
running softmax state until a visible tile arrives. Inference omits normalizer storage. Backward
consists of preprocessing and deterministic dK/dV and dQ kernels: 64 owned rows by 64 streamed
rows, 4 warps and 2 stages. The extra tuning round selects 32-by-32 tiles and 4 warps for
sequences of at most 512 tokens; the normal model geometry is unchanged.

Contract: the eager bf16 QK product rounds before fp32 scaling and softmax; its probability
gradient also rounds before the softmax adjoint. Both boundaries are explicit. An independent
sum/random/broadcast adjoint audit exposed the missing rounding in the earlier version; the
corrected version passes that audit. Inputs preserve batch/head/row strides and require a unit
final stride. Outputs and input gradients retain their storage dtype; the mask has no gradient.
No gradient atomics or token-by-token mask allocation are used.

The bounded large reference now groups 512 queries per head, without changing its
rounding boundaries or FP32 cross-chunk gradient accumulation. Large-workload
revalidation uses CUDA-only CUPTI collection to avoid CPU trace materialization.
Bounded/dense equivalence checks pass on A800 and H20. The interrupted CPU+CUDA
capture remains preserved. The 2026-09-08 CUDA-only H20 run advanced through all three
forward/inference rounds and reached backward round 3, then hit its 5400-second timeout
before writing a complete report. No timing or acceptance is inferred from those progress logs.

## A800 measurements

Profiler device times on 2026-09-06, minimum of three interleaved rounds with fresh compiler
state per workload. All eligible baselines are checked numerically before timing. Full
comparisons and raw samples are in `golden.json`.

| Workload / phase | Golden | Eager | Compiled | SDPA | Achievable roof | SoL |
|---|---:|---:|---:|---:|---:|---:|
| user_train / fwd | 0.880432 ms | 35.304652 ms | 7.210048 ms | 15.549838 ms | 0.768652 ms | 0.873 |
| user_train / bwd | 2.884544 ms | 37.518259 ms | 11.211778 ms | 20.040211 ms | 2.128463 ms | 0.738 |
| user_train / infer | 0.885902 ms | 35.305677 ms | 6.472415 ms | 15.550059 ms | 0.768567 ms | 0.868 |

The primary passes output, inference, gradient, SoL and baseline checks. Compiled eager is its
strongest eligible baseline, giving 8.189x forward, 3.887x backward and 7.306x inference
speedups. The required real-model case has 24 heads and 115456 tokens: eight heads each use
`(3,3,3)`, `(3,6,10)` and `(5,6,10)` windows on a `(5,6,10)` grid, with 384 tokens per tile and
256 text tokens. Its allowed-pair density is 56.53%; the earlier rough 5–15% estimate does not
describe this case. All three workloads pass output, inference and gradient checks in a separate
final numerical run, and fresh independent adjoints pass. The 439-token text tail is an optional
diagnostic. Real-model profiling was interrupted in the first forward round after more than ten
minutes without completed round progress: the GPU was idle while raw Kineto event processing
held about 60 GB of host memory. Its performance and calibrated roof remain incomplete; no
partial timing is accepted.

The real model has 180841611264 allowed pairs. Its useful forward/backward work is
92.591/231.477 trillion FLOPs. Compulsory traffic including scheduling metadata is 2.893 GB
training forward, 2.882 GB inference and 9.636 GB backward (decimal units). These static counts
do not substitute for the missing achievable-roof calibration. The additional tuning round is
exhausted.

For `R=B*H*S` and `C=R*D`, activation traffic is `8*C` forward, plus `4*R` for training
normalizers, and `26*C+20*R` backward across its three kernels. The canonical script now adds
consumed sparse-list/count/permutation entries, boundary mask reads and the backward
mask-transpose copy. Useful compute counts allowed pairs only: `4*D` per pair forward and the
existing `10*D` backward roof. Split backward repeats QK and dP products, executing `14*D` per
pair before tail overhead. This overhead does not increase the useful-work roof. Large-workload
coverage is assessed separately from the primary result.

## Roof-model audit

The achievable roof is `max(copy_ms, density * dense_flash_phase_ms)`, calibrated with the same
Q/K/V geometry and storage dtype for each phase. Datasheet bounds remain separate. The
useful-work numerator excludes duplicated products, sparse scheduling overhead and precision
emulation. This calibration estimates throughput; it is not an exact sparse-latency prediction
or a correctness reference. See the [SoL audit](../../BENCHMARKING.md).

Both normal model workloads are required. The approved 68–70% SoL margin and 95% baseline gate apply;
roofs below 10 microseconds retain the SoL waiver and existing 1-microsecond absolute baseline
allowance. Missing profiler evidence remains incomplete, and optional stress failures cannot
waive a required model row.

## Reproduce the checks

From the checkout root, using the GPU Python interpreter:

```bash
python examples/sliding_tile_attention/benchmark.py --verify --json tmp/golden-benchmarks/sliding_tile_attention/verify.json
python examples/sliding_tile_attention/benchmark.py --bench --json tmp/golden-benchmarks/sliding_tile_attention/report.json
python examples/sliding_tile_attention/benchmark.py --adjoint --json tmp/golden-benchmarks/sliding_tile_attention/adjoint.json
```

[Workloads and SoL accounting](../benchmark_cases.py) define the inputs, gradient reductions,
and per-phase bytes/FLOPs. The [shared runner](../../_benchmark.py) uses KDA's product runtime
directly. Reports are generated outside `golden/`; new numerical and timing reports require an
independent acceptance audit. Use a distinct output filename for scoped `--workload` or
`--fwd-only` runs.
