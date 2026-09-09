# Golden: segmented multi-view attention

Status: **pass**, with the two H20 backward acceptance exceptions recorded below.
This is a synthetic prefix-length extension inspired by VGGT-style workloads, not an
interface claimed to exist in official VGGT. The complete 18-case numerical run,
all required performance workloads and independent tail-strided adjoints are recorded
in `golden.json`.

## Implementation and precision

Immutable CPU lengths describe each view's valid K/V prefix. One batched GPU gather
packs those prefixes; Q retains its full original length. Each scene reads its own
actual key count from a small offset table, while storage uses the batch maximum.
An empty scene attends to original key zero. Rectangular FA2 streams the compact keys;
no token-pair mask or full score plane is instantiated. Batched gradient scatter
restores original positions and explicitly zeros padding gradients. Packing, offset
transfer and scatter are included in the appropriate measured phases.

Scores, softmax statistics and output accumulators use FP32. Following the explicitly
approved FlashAttention precision convention, block probabilities round to BF16
before PV. Backward computes dP and the output/upstream row reduction in FP32, then
uses BF16 P/dS operands and FP32 tensor-core accumulation. The independent PyTorch
reference implements that mixed-precision adjoint explicitly; ordinary autograd
through a probability cast would insert a different dP rounding boundary. Numerical
tolerances remain unchanged.

D64 forward uses 128 query rows by 64 keys, four warps and three stages. Backward
uses 64-by-64 key-owned tiles and 128-by-32 query-owned tiles, four warps and two
stages. Full forward/dQ key tiles avoid tail predicates; the final tile stays masked.
Batch/head addresses are int64. Leading strides, width tails, broadcast upstream
and optional recompute remain supported. Recompute retains packed K/V and rebuilds
attention auxiliaries, without repeating the gather.

## H20 measurements and acceptance

CUDA-only CUPTI device times, three interleaved rounds of ten measured calls; each
reported value is the minimum round median. Complete callable timing includes every
packing/scatter kernel. All listed SDPA baselines passed numerical checks.

| Case | Forward (ms) | Backward (ms) | Inference (ms) | Bwd SoL | SDPA / golden bwd |
|---|---:|---:|---:|---:|---:|
| frame | 0.224513 | 0.880036 | 0.224449 | 95.1% | 0.9476x |
| global | 0.758974 | 2.655630 | 0.761457 | 103.9% | 1.0561x |
| all_valid | 1.025775 | 3.525643 | 1.032912 | 107.2% | 0.7958x |
| real_24_views | 24.591358 | 83.015440 | 24.598770 | 113.2% | 1.1700x |

On 2026-09-08 the user explicitly accepted the measured frame and all-valid backward
rows because their achieved SoL was already high, and requested no further tuning.
Frame is near the 0.95 baseline boundary. All-valid's 0.7958x SDPA result is a specific
accepted exception, not a near-gate result. Both gaps remain visible in
`acceptance_exceptions`; no timing, tolerance or global baseline threshold was changed
to obtain this status. The original raw report ignored SDPA in its product verdict;
the separate baseline-policy audit includes SDPA and preserves its `tune` finding
before applying these explicit exceptions.

The attention roof is a density-scaled dense Flash throughput proxy, so a ratio above
100% is possible and is not a claim to exceed hardware peak. The A800 frame capture
also passed, at 0.182848 / 0.698271 / 0.179968 ms for forward/backward/inference;
this is a scoped supporting result, not a complete A800 acceptance matrix.

## Roof accounting

Let `Rq=B*H*N`, `Rk=H*sum(max(1,sum(scene_lengths)))` and `P=N*Rk`. With `es=2`,
packing costs `4*Rk*D*es`; forward attention costs `(2*Rq+2*Rk)*D*es`, plus
saved normalizers and offset metadata. Backward counts `(7*Rq+6*Rk)*D*es+20*Rq`
for attention and `(2*Rk+2*Rq)*D*es` for scatter, including padding zeros, plus
metadata. Recompute adds attention forward traffic/work without repacking.
Useful FLOPs remain `4*P*D` forward and `10*P*D` backward. Extra implementation
work does not raise the useful-work roof. Exact formulas are in `benchmark_cases.py`.

The former FP32-probability captures remain under `historical_capture`; they do not
establish acceptance of the revised precision contract.

## Reproduce the checks

From the checkout root, using the GPU Python interpreter:

```bash
python examples/vggt_segmented_attention/benchmark.py --verify --json tmp/golden-benchmarks/vggt_segmented_attention/verify.json
KDA_PROFILER_ACTIVITIES=cuda python examples/vggt_segmented_attention/benchmark.py --bench --json tmp/golden-benchmarks/vggt_segmented_attention/report.json
python examples/vggt_segmented_attention/benchmark.py --adjoint --json tmp/golden-benchmarks/vggt_segmented_attention/adjoint.json
```

Raw benchmark gates remain visible; the capture-specific user exceptions above are
part of the published audit. New captures require review. See
[benchmark policy](../../BENCHMARKING.md). Outputs belong outside `golden/`, and
scoped runs must use distinct filenames.
