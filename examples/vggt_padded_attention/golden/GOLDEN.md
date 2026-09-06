# Golden: per-batch key-padded attention

Hand-written target for `padded_attention` in `user_repo/multiview/attention.py`:
`softmax(Q.float() @ K.float().T / sqrt(D), key_valid) @ V.float()`, returned in bf16. This
withheld solution is **incomplete**, with a demonstrated frame-attention SoL failure.

Forward streams key tiles and accumulates online softmax in fp32, without an `S*S` score or mask
allocation. The retained D64 tile is 128 queries by 64 keys, 4 warps and 3 stages. Training
stores one fp32 log2 normalizer per query; inference omits it. A fully invalid first key tile
keeps a finite running state.

Probability products retain an fp32 residual: split each probability into a bf16 high component
and a bf16 residual, then accumulate both tensor-core products in fp32. The same decomposition
is used for probability and score-gradient products in backward. It retains roughly 16
significant bits instead of silently dropping the residual. This replaced the initial tf32x3
implementation; numerical evidence, not the precision label, determines acceptance.

Backward launches preprocessing, a key-owned dK/dV pass and a query-owned dQ pass, each using
64-by-64 tiles, 4 warps and 2 stages. No atomics are used. Optional recompute rebuilds O and its
normalizer. Batch/head/ row strides and int64 batch/head bases are retained; broadcast dO is
materialized when needed. Validity masks keys only: padded queries still have outputs and input
gradients. The caller guarantees at least one valid key per batch.

## A800 measurements

Profiler device times on 2026-09-06, minimum of three interleaved rounds with fresh compiler
state per workload. All eligible baselines are checked numerically before timing. Full
comparisons and raw samples are in `golden.json`.

| Workload / phase | Golden | Eager | Compiled | SDPA | Achievable roof | SoL |
|---|---:|---:|---:|---:|---:|---:|
| frame / fwd | 0.494768 ms | 5.295827 ms | 2.718907 ms | 0.628447 ms | 0.187092 ms | 0.378 |
| frame / bwd | 1.574269 ms | 9.869711 ms | 4.751705 ms | 1.536366 ms | 0.574049 ms | 0.365 |
| frame / infer | 0.495310 ms | 5.292470 ms | 2.664105 ms | 0.628366 ms | 0.187133 ms | 0.378 |

All 12 saved-state and recompute numerical cases pass, including the normal frame, global and
24-view shapes. Fresh independent adjoints also pass. Frame attention passes the fastest
eligible baseline gate (1.270x forward, 0.976x backward and 1.269x inference), but reaches only
36–38% SoL. Global and 24-view profiling lose positive device samples even after the single
diagnostic retry; those phases remain incomplete. Partial samples are retained as diagnostics
and are not accepted timings.

The extra eight-warp trial regressed global forward/backward to 2.082/10.040 ms and was
rejected. The retained four-warp kernel is frozen; the additional tuning round is exhausted.
Older global timings do not fill the current profiler evidence gap.

Let `R=B*H*S`, `P=H*S*sum(key_valid)` and `es=2`. Forward compulsory traffic is `4*R*D*es+B*S`,
plus `4*R` for training normalizers. Backward counts `13*R*D*es+20*R+2*B*S` bytes across
preprocessing and both gradient passes. The unchanged useful-work roof uses `4*P*D` forward and
`10*P*D` backward FLOPs. Residual decomposition adds tensor products: nominally `6*P*D` forward
and `20*P*D` across the split backward before masked-tile overhead; this precision and
recomputation overhead does not raise the roof. Optional recompute adds its forward work and
traffic to backward. The measured copy/compute roofs remain distinct from the datasheet bound.

## Roof-model audit

The achievable attention roof is the larger of the measured copy time and the same-geometry
dense FlashAttention phase time multiplied by allowed-pair density. Datasheet bounds are
recorded separately. Storage-dtype FlashAttention calibrates useful-work throughput; it does not
reproduce this operation's fp32 intermediates. Precision decomposition and duplicated backward
work cannot inflate the roof. See the [SoL audit](../../BENCHMARKING.md) for the calibration and
its limits. The 70% SoL and 95% baseline gates are unchanged; only roofs below 10 microseconds
waive SoL and admit the existing 1-microsecond absolute baseline allowance.

Acceptance is **incomplete** because required measurements are missing. The raw runtime reports
timing errors as `fail`; numerical verification passes. The completed frame row independently
remains `tune`. Optional odd/tail diagnostics cannot hide these normal-workload failures.

## Reproduce the checks

From the checkout root, using the GPU Python interpreter:

```bash
python examples/vggt_padded_attention/benchmark.py --verify --json tmp/golden-benchmarks/vggt_padded_attention/verify.json
python examples/vggt_padded_attention/benchmark.py --bench --json tmp/golden-benchmarks/vggt_padded_attention/report.json
python examples/vggt_padded_attention/benchmark.py --adjoint --json tmp/golden-benchmarks/vggt_padded_attention/adjoint.json
```

[Workloads and SoL accounting](../benchmark_cases.py) define the inputs, gradient reductions,
and per-phase bytes/FLOPs. The [shared runner](../../_benchmark.py) uses KDA's product runtime
directly. Reports are generated outside `golden/`; new numerical and timing reports require an
independent acceptance audit. Use a distinct output filename for scoped `--workload` or
`--fwd-only` runs.
