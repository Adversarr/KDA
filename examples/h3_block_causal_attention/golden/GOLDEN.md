# Golden: frame-block causal attention

Hand-written target for `block_causal_attention` in `user_repo/video/attention.py`: `softmax(Q
K.T / sqrt(D), frame(key) <= frame(query)) V`. A query sees its entire frame, including later
keys. This withheld solution remains unfinished until all required numerical and performance
gates pass.

Forward owns 128 query rows and streams 128-key tiles with 8 warps and 3 stages. Whole key tiles
visible to the first query use an unmasked loop; only frame-crossing and tail tiles use element
masks. Online softmax keeps the normalizer and output in registers without a quadratic score
allocation. Training saves one fp32 log2 normalizer per query; inference omits it.

Backward launches preprocessing, deterministic dK/dV, and deterministic dQ. The key-owned pass
uses 64 keys and 32 queries per tile; the query-owned pass uses 64 queries and 32 keys, with 8
warps and 2 stages. Both skip invisible frames and avoid mask comparisons for fully visible
tiles. Wider backward tiles and two-stage forward did not improve the first-round result and
were rejected.

Contract: bf16 QK rounds before fp32 scaling and softmax. Probability/value products observe the
bf16 boundary. Token-causal masking is correct only at frame size one; a frame at least as long
as the sequence is dense attention. Row and head strides, partial frames and int64 batch/head
bases are retained. Broadcast dO is materialized when its final stride is not one. Optional
recompute rebuilds O and its normalizer. The independent large reference uses 128-query chunks
and at most four heads, accumulating dK/dV in fp32 before their final bf16 cast. Its
dense-equivalence and outer-autocast checks pass.

## A800 measurements

Profiler device times on 2026-09-06, minimum of three interleaved rounds with fresh compiler
state per workload. All eligible baselines are checked numerically before timing. Full
comparisons and raw samples are in `golden.json`.

| Workload / phase | Golden | Eager | Compiled | SDPA | Achievable roof | SoL |
|---|---:|---:|---:|---:|---:|---:|
| user / fwd | 1.004894 ms | 18.166116 ms | 3.647241 ms | 3.388526 ms | 0.786444 ms | 0.783 |
| user / bwd | 3.267336 ms | 23.075807 ms | 5.183655 ms | 11.133361 ms | 2.356868 ms | 0.721 |
| user / infer | 1.029246 ms | 18.169012 ms | 3.614538 ms | 3.387883 ms | 0.786312 ms | 0.764 |

The primary passes all three SoL and baseline gates. Fresh independent adjoints pass. Required
21840-token numerics pass, but both profiler attempts lose the first five device samples. A
separate final numerical run passes all 12 cases, including 62400 tokens and every recompute
variant. Large-shape performance remains incomplete; primary results cannot establish those
gates.

The extra round accumulates the probability/value dot product directly into the output
accumulator and shortens the softmax statistics' live ranges. Geometry is unchanged. The
additional tuning round is exhausted.

Let `R=B*H*S`, `es=2` and `S=full*frame+tail`. The useful pair count is
`P=B*H*(frame**2*full*(full+1)/2 + tail*S)`. Forward traffic is `4*R*D*es`, plus `4*R` for saved
normalizers. Backward compulsory pass traffic is `13*R*D*es + 20*R`, including preprocessing,
both gradient passes and their auxiliaries. The useful-work roof uses `4*P*D` forward and
`10*P*D` backward FLOPs. The split backward actually executes `14*P*D` tensor-product work
because QK and dP are recomputed in both passes; this redundant work does not raise the
useful-work roof. Recompute adds one forward to backward. Tail padding and repeated tile loads
are overhead, not reasons to relax the gate.

## Roof-model audit

The achievable roof is `max(copy_ms, density * dense_flash_phase_ms)` with same-geometry,
storage-dtype FlashAttention calibration. Its forward/backward useful work is `4*P*D` /
`10*P*D`; datasheet bounds are separate. This throughput proxy replaces the earlier same-FLOP
GEMM estimate. Intermediate rounding and structured scheduling still differ from the
calibration. See the [SoL audit](../../BENCHMARKING.md).

Acceptance remains **incomplete** because required large-workload measurements are missing.
Failed captures remain visible in `golden.json`; no zero or partial sample is accepted. A
bounded collector-start diagnostic did not reproduce the loss, so it does not justify another
runtime change or selecting a favorable capture. The 70% SoL and 95% baseline gates are
unchanged; roofs below 10 microseconds retain the existing SoL waiver and 1-microsecond absolute
baseline allowance.

## Reproduce the checks

From the checkout root, using the GPU Python interpreter:

```bash
python examples/h3_block_causal_attention/benchmark.py --verify --json tmp/golden-benchmarks/h3_block_causal_attention/verify.json
python examples/h3_block_causal_attention/benchmark.py --bench --json tmp/golden-benchmarks/h3_block_causal_attention/report.json
python examples/h3_block_causal_attention/benchmark.py --adjoint --json tmp/golden-benchmarks/h3_block_causal_attention/adjoint.json
```

[Workloads and SoL accounting](../benchmark_cases.py) define the inputs, gradient reductions,
and per-phase bytes/FLOPs. The [shared runner](../../_benchmark.py) uses KDA's product runtime
directly. Reports are generated outside `golden/`; new numerical and timing reports require an
independent acceptance audit. Use a distinct output filename for scoped `--workload` or
`--fwd-only` runs.
