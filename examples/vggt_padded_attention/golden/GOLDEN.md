# Golden: per-batch key-padded attention

Status: **pass**. The complete 14-case A800 run, all required performance workloads
and independent tail-strided adjoints pass under the revised Flash-style precision
contract. H20 also passes all 14 numerical cases and eight forced-native-Flash
forward/gradient comparisons across BF16/FP16, head widths and recompute modes.

Two GPU prefix scans compact arbitrary valid K/V positions into a per-scene prefix.
The rectangular attention loops visit only that prefix; Q keeps its original length.
Forward then streams K/V tiles through online softmax, with FP32 scores, normalization statistics
and output accumulation. The unnormalized block weights round to BF16 before PV, following
FlashAttention's mixed-precision convention. This replaces the former two-component BF16
probability decomposition following explicit user authorization. The default D64 tile is
128 queries by 64 keys, four warps and three stages. Training saves an FP32 log2 normalizer;
inference omits it. No full score or token-pair mask plane is allocated.

Backward computes the row reduction `delta = sum(output * dOutput)` in FP32. Key-owned dK/dV
and query-owned dQ passes reconstruct probabilities, compute dP and dS in FP32, and round P/dS
to BF16 before their tensor-core products. Those products accumulate in FP32.
A GPU scatter restores dK/dV to original positions and writes zero for invalid keys.
The normal training path retains compact K/V and the index map; recompute rebuilds
only attention auxiliaries. Packing, scans and scatter are included in phase timing. The independent
reference spells out this mixed-precision adjoint with PyTorch operations; naive autograd
through a BF16 probability cast inserts a different dP rounding boundary.

The key-validity vector masks keys only. Padded queries retain outputs and gradients. Every
batch entry has at least one valid key, as guaranteed by the caller. Leading strides, tail
blocks, independent gradients, broadcast upstream gradients and optional recompute remain
supported. Numerical tolerances are unchanged.

## Precision regression

The retained `probability_cancellation` cases use logits 0 and 1/512 with values +32 and -32.
The former FP32-probability contract returned -0.03125. Under the adopted BF16 Flash convention,
the weights round before PV and the output is zero. A forced native Flash backend on A800
confirms zero, with output and all three gradients matching the new golden and reference.
This is a deliberate contract revision, not evidence that standard FlashAttention is incorrect.

## A800 measurements

CUDA-only CUPTI device times, three interleaved rounds of ten measured calls;
reported values are the minimum round medians. Scan, gather and scatter costs are
included. All eligible baselines, including SDPA, pass numerical checks before timing.

| Case | Forward (ms) | Backward (ms) | Inference (ms) | Bwd SoL | SDPA / golden bwd |
|---|---:|---:|---:|---:|---:|
| frame | 0.222880 | 0.806686 | 0.264014 | 85.3% | 2.3073x |
| global | 0.723243 | 2.343137 | 0.717565 | 79.2% | 2.4446x |
| real_24_views | 35.403401 | 115.293621 | 35.460821 | 82.3% | 3.3627x |

All required forward, backward and inference SoL/baseline gates pass. Full original
timings, numerical checks, collector mode and source hashes are in `golden.json`.
The independent baseline-policy audit includes SDPA; no performance exception is
needed. Previous FP32-probability evidence remains under `historical_capture` and
does not establish acceptance of this revised contract.

## Roof accounting

For `Rq=B*H*S`, `Rk=H*sum(key_valid)`, `P=S*Rk` and `es=2`, forward
counts `(2*Rq+6*Rk)*D*es` for attention plus packing, `4*Rq` for saved
normalizers, and mask scan/index metadata traffic. Backward counts
`(7*Rq+6*Rk)*D*es+20*Rq` for attention plus `(2*Rk+2*Rq)*D*es` for
scatter, including zero padding, and index reads. The exact scan/index accounting
is in `benchmark_cases.py`. Useful work is `4*P*D` forward and `10*P*D`
backward FLOPs. Optional recompute adds attention forward work/traffic without
repeating compaction.

The achievable roof remains the larger of the measured copy floor and same-geometry dense
BF16 FlashAttention time scaled by allowed-pair density. Original Flash SoL is still the
performance gate, under the suite's approved 68–70% margin. No candidate timing or extra
implementation work inflates the roof. See [benchmark policy](../../BENCHMARKING.md).

## Reproduce the checks

From the checkout root, using the GPU Python interpreter:

```bash
python examples/vggt_padded_attention/benchmark.py --verify --json tmp/golden-benchmarks/vggt_padded_attention/verify.json
KDA_PROFILER_ACTIVITIES=cuda python examples/vggt_padded_attention/benchmark.py --bench --json tmp/golden-benchmarks/vggt_padded_attention/report.json
python examples/vggt_padded_attention/benchmark.py --adjoint --json tmp/golden-benchmarks/vggt_padded_attention/adjoint.json
```

[Workloads and SoL accounting](../benchmark_cases.py) define the inputs, gradient reductions,
and per-phase bytes/FLOPs. The [shared runner](../../_benchmark.py) uses KDA's product runtime
directly. Reports are generated outside `golden/`; new numerical and timing reports require an
independent acceptance audit. Use a distinct output filename for scoped `--workload` or
`--fwd-only` runs.
