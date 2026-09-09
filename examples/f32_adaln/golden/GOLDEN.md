# Golden: FP32 adaptive LayerNorm

Hand-written target for `diffusion/layers.py::adaln` in the example's `user_repo/`.

One forward program computes a row's affine-free LayerNorm and applies the `(B,D)` sample
modulation as `normalized*(1+scale)+shift`. Normalization and modulation use fp32. Mean and
inverse standard deviation are saved only for training; neither normalized activations nor a
per-token modulation expansion is stored.

Backward partitions its reduction grid by sample: `dx` is rowwise, while `dscale` and `dshift`
reduce over that sample's tokens only. The tuned grid distributes approximately two blocks per
SM across all samples, reducing the partial-buffer footprint. Optional SiLU follows the bf16
AdaLN output boundary; its backward also rounds the activation adjoint to bf16 before the
normalization adjoint.

The 1152-wide model forward splits into 1024+128 channels, avoiding padding arithmetic to 2048
lanes. Plain backward uses two-row tiles; model SiLU backward also splits channels into 1024+128,
uses four warps, and distributes twelve program waves across batches. This removes masked
SiLU arithmetic and improves parallelism without changing the bf16 adjoint boundary. Scale/shift partials share a final reduction
launch. Small SiLU inputs use token blocks with disjoint final modulation-gradient columns.

## Contract

The standalone kernel imports only PyTorch and Triton. Inputs and upstream gradients use their
own leading strides, including broadcast gradients. Last-dimension activation storage is
contiguous. Row/base offsets use int64, tails are masked, and empty inputs return without a
zero-grid launch. Forward auxiliaries are allocated and written only when a backward can follow.

## A800 measurements

B16 N4096 D1152; bf16 activation and per-sample fp32 modulation.

Device time from `torch.profiler` on NVIDIA A800-SXM4-80GB, torch 2.11.0+cu128. Entries were
timed round-robin, taking the minimum of three rounds. Roof percentages use a measured same-size
copy, with byte accounting below; these are measurements on this device, not a performance
promise elsewhere.

| Phase | Golden | Eager | Compiled | SoL |
|---|---:|---:|---:|---:|
| forward (training) | 0.19649 ms | 3.33733 ms | 0.32928 ms | 89.5% |
| forward (inference) | 0.19493 ms | 3.33529 ms | 0.27757 ms | 90.0% |
| backward | 0.37411 ms | 6.75783 ms | 0.72678 ms | 70.5% |

Measured full-workload verdict on 2026-09-08: **pass**. All seven numerical workloads,
eligible baselines, and applicable phase gates pass. Model SiLU backward is 0.36779 ms,
75.0% of the measured achievable roof, and 3.227x compiled. Two independent model-SiLU
repeats retain 74.7% SoL and pass every phase. Independent model-sized and small-input
adjoints pass, including broadcast upstream gradients.

Full output/gradient checks, phase times, measured copy roofs, separate datasheet bounds,
baseline ratios, round samples and source-matched repeats are recorded in `golden.json`.
The older tuning capture is preserved under `historical_capture`.

## Traffic and arithmetic

`C` denotes activation elements and `R` rows unless defined otherwise; `D` is the channel width
and `P` the reduction programs. Parameters and positional tables count once as compulsory
traffic. Partial buffers count both their write and subsequent reduction read; final
parameter-gradient stores are included. Transcendental instruction cost is not represented by an
invented FLOP multiplier.

Let `R=B*N`, `C=R*D`, and `P=min(ceil(N/4),max(1,floor(A*SMs/B)))`, with `A=12` for 1152-wide
SiLU and `A=2` otherwise. Direct small-input reductions use `P=0`. Forward bytes: `4*C + 8*B*D +
8*R`; inference omits auxiliaries. Backward bytes: `6*C + 8*R + 12*B*D + 16*B*P*D`; SiLU also
reads the shift (`4*B*D`). Forward arithmetic is `8*C+3*R` (SiLU: `10*C+3*R`); backward is
`13*C+2*R` (SiLU: `19*C+2*R`). Exp/rsqrt instructions are excluded from arithmetic FLOPs.

## What to check in a candidate

Reducing modulation gradients across batches is a semantic error. Scale is fp32, so `1+scale`
must not be prematurely rounded. Optional SiLU must preserve the bf16 AdaLN boundary. At width
1152, forward handles 1024+128 channels explicitly; masked reduction lanes and backward geometry
still affect efficiency.

## Reproduce the checks

From the checkout root, using the GPU Python interpreter:

```bash
python examples/f32_adaln/benchmark.py --verify --json tmp/golden-benchmarks/f32_adaln/verify.json
python examples/f32_adaln/benchmark.py --bench --json tmp/golden-benchmarks/f32_adaln/report.json
python examples/f32_adaln/benchmark.py --adjoint --json tmp/golden-benchmarks/f32_adaln/adjoint.json
```

[Workloads and SoL accounting](../benchmark_cases.py) define the inputs, gradient reductions,
and per-phase bytes/FLOPs. The [shared runner](../../_benchmark.py) uses KDA's product runtime
directly. Reports are generated outside `golden/`; new numerical and timing reports require an
independent acceptance audit. Use a distinct output filename for scoped `--workload` or
`--fwd-only` runs.
