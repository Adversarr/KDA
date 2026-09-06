# Golden: Masked LayerScale residual + LayerNorm

Hand-written target for `vision/blocks.py::layerscale_residual_layernorm` in the example's
`user_repo/`.

One program forms `x + gamma*branch`, zeros invalid rows, and produces the fp32 stream plus its
bf16 affine LayerNorm. The stream is an output retained for backward; only mean/rstd are extra
saved tensors.

The fused adjoint adds the direct stream gradient to the LayerNorm gradient before applying the
validity mask. Valid rows produce `dx`, `dbranch=dx*gamma`, and the channel-reduced `dgamma`.
Invalid rows contribute zero to those three gradients and `dnorm_w`, but still contribute to
`dnorm_b`, because their normalized output equals the bias. Three deterministic fp32
parameter-partial buffers are reduced together in one final Triton launch.

## Contract

The standalone kernel imports only PyTorch and Triton. Inputs and upstream gradients use their
own leading strides, including broadcast gradients. Last-dimension activation storage is
contiguous. Row/base offsets use int64, tails are masked, and empty inputs return without a
zero-grid launch. Forward auxiliaries are allocated and written only when a backward can follow.

## A800 measurements

B1 N5496 D1024; fp32 stream, bf16 branch, LayerScale gamma=0.01.

Device time from `torch.profiler` on NVIDIA A800-SXM4-80GB, torch 2.11.0+cu128. Entries were
timed round-robin, taking the minimum of three rounds. Roof percentages use a measured same-size
copy, with byte accounting below; these are measurements on this device, not a performance
promise elsewhere.

| Phase | Golden | Eager | Compiled | SoL |
|---|---:|---:|---:|---:|
| forward (training) | 0.0439 ms | 0.2326 ms | 0.0432 ms | 110.2% |
| forward (inference) | 0.0433 ms | 0.2335 ms | 0.0427 ms | 111.1% |
| backward | 0.0812 ms | 0.3560 ms | 0.1200 ms | 86.2% |

Final verdict: **pass**. The fresh full report covers 6 workloads, including every required
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

Forward bytes: `12*C + 12*D + R + 8*R`; inference omits the last `8*R`. Backward bytes: `18*C +
9*R + 20*D + 24*P*D`, `P=min(ceil(R/4),2*SMs)`. This includes the validity mask, all five output
gradients, and both write/read traffic of three partial reductions. Arithmetic: `10*C+3*R`
forward, `17*C+2*R` backward.

## What to check in a candidate

Masking after LayerNorm revives invalid residual rows. Zeroing every parameter gradient on
invalid rows incorrectly removes the affine-bias gradient. Both returned outputs must contribute
before the residual mask and LayerScale adjoint.

## Reproduce the checks

From the checkout root, using the GPU Python interpreter:

```bash
python examples/vggt_layerscale_residual_layernorm/benchmark.py --verify --json tmp/golden-benchmarks/vggt_layerscale_residual_layernorm/verify.json
python examples/vggt_layerscale_residual_layernorm/benchmark.py --bench --json tmp/golden-benchmarks/vggt_layerscale_residual_layernorm/report.json
python examples/vggt_layerscale_residual_layernorm/benchmark.py --adjoint --json tmp/golden-benchmarks/vggt_layerscale_residual_layernorm/adjoint.json
```

[Workloads and SoL accounting](../benchmark_cases.py) define the inputs, gradient reductions,
and per-phase bytes/FLOPs. The [shared runner](../../_benchmark.py) uses KDA's product runtime
directly. Reports are generated outside `golden/`; new numerical and timing reports require an
independent acceptance audit. Use a distinct output filename for scoped `--workload` or
`--fwd-only` runs.
