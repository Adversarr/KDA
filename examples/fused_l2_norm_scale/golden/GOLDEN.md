# Golden: Fused L2 normalization + learned scale

Hand-written target for `normalized/ops.py::l2_norm_scale` in the example's `user_repo/`.

Forward uses `rsqrt(sum(x*x)+eps)`, without dividing by the last dimension, then applies the
learned channel scale. The same implementation handles wide hidden-state rows and narrow
per-head rows. It accepts rank-two through rank-four layouts, including q/k slices from an
interleaved projection.

Backward uses `rstd * (dy*scale - normalized*sum(dy*scale*normalized))`. It writes `dx` once and
accumulates shared `dscale` in bounded deterministic fp32 partials. The current implementation
retains fp32 mean (zero for L2) and inverse norm per row for a common normalization launcher;
inference saves neither.

## Contract

The standalone kernel imports only PyTorch and Triton. Inputs and upstream gradients use their
own leading strides, including broadcast gradients. Last-dimension activation storage is
contiguous. Row/base offsets use int64, tails are masked, and empty inputs return without a
zero-grid launch. Forward auxiliaries are allocated and written only when a backward can follow.

## A800 measurements

B8 S4096 D2048; 32768 rows, bf16 activation and fp32 scale initialized to sqrt(D).

Device time from `torch.profiler` on NVIDIA A800-SXM4-80GB, torch 2.11.0+cu128. Entries were
timed round-robin, taking the minimum of three rounds. Roof percentages use a measured same-size
copy, with byte accounting below; these are measurements on this device, not a performance
promise elsewhere.

| Phase | Golden | Eager | Compiled | SoL |
|---|---:|---:|---:|---:|
| forward (training) | 0.1562 ms | 1.8644 ms | 0.1575 ms | 100.3% |
| forward (inference) | 0.1549 ms | 1.7750 ms | 0.1568 ms | 101.0% |
| backward | 0.2664 ms | 3.9387 ms | 0.2619 ms | 88.2% |

Final verdict: **pass**. The fresh full report covers 8 workloads, including every required
output, inference and gradient check. Independent adjoints also pass. All numerical results,
per-workload timings, copy roofs and raw rounds are recorded in `golden.json`; the recorded
measurement and published source hashes are recorded separately after nonfunctional cleanup. Small latency-bound rows retain the runtime's
10-us SoL waiver and 1-us absolute baseline parity rule.

## Traffic and arithmetic

`C` denotes activation elements and `R` rows unless defined otherwise; `D` is the channel width
and `P` the reduction programs. Parameters and positional tables count once as compulsory
traffic. Partial buffers count both their write and subsequent reduction read; final
parameter-gradient stores are included. Transcendental instruction cost is not represented by an
invented FLOP multiplier.

Forward bytes: `4*C + 4*D + 8*R`; inference omits `8*R`. Backward bytes: `6*C + 8*R + 8*D +
8*P*D`, `P=min(ceil(R/T),2*SMs)`, with T=32 for D<=256 and T=4 otherwise; the direct small
reduction uses P=0 when R<=128 and D<=1024. The partial term counts one write and one reduction
read. Arithmetic estimates: `4*C+R` forward, `9*C` backward.

## What to check in a candidate

Using the RMSNorm mean-square denominator changes the answer by sqrt(D). The fixture's learned
scale starts at sqrt(D), not one. Per-head q/k views have a different stride between tokens than
between heads; flattening them as contiguous rows is incorrect.

## Reproduce the checks

From the checkout root, using the GPU Python interpreter:

```bash
python examples/fused_l2_norm_scale/benchmark.py --verify --json tmp/golden-benchmarks/fused_l2_norm_scale/verify.json
python examples/fused_l2_norm_scale/benchmark.py --bench --json tmp/golden-benchmarks/fused_l2_norm_scale/report.json
python examples/fused_l2_norm_scale/benchmark.py --adjoint --json tmp/golden-benchmarks/fused_l2_norm_scale/adjoint.json
```

[Workloads and SoL accounting](../benchmark_cases.py) define the inputs, gradient reductions,
and per-phase bytes/FLOPs. The [shared runner](../../_benchmark.py) uses KDA's product runtime
directly. Reports are generated outside `golden/`; new numerical and timing reports require an
independent acceptance audit. Use a distinct output filename for scoped `--workload` or
`--fwd-only` runs.
