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
lanes. On A800, plain backward also uses this split when there are at least 1024 tokens per
sample and 8192 total rows, with four warps and eight program waves across batches. Other
plain workloads keep two-row tiles. Model SiLU backward uses the split with twelve waves,
preserving its bf16 adjoint boundary. Scale/shift partials share a final reduction launch.
Small SiLU inputs use token blocks with disjoint final modulation-gradient columns.

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
| forward (training) | 0.19650 ms | 3.33858 ms | 0.32936 ms | 89.5% |
| forward (inference) | 0.19558 ms | 3.33874 ms | 0.27754 ms | 89.7% |
| backward | 0.29856 ms | 6.75869 ms | 0.72685 ms | 90.8% |

Measured full-workload verdict on 2026-09-09: **pass**. All seven numerical workloads,
eligible baselines, and applicable phase gates pass. Model SiLU backward is 0.36763 ms,
74.9% of the measured achievable roof, and 3.236x compiled. Small, model-sized and model-SiLU
adjoints pass against the independent eager reference, including broadcast upstream gradients.
Source/coverage audit was performed in the implementation context, without a separate worker.

Full output/gradient checks, phase times, measured copy roofs, separate datasheet bounds,
baseline ratios, round samples and source-matched repeats are recorded in `golden.json`.
Previous complete records are preserved under `historical_capture`.

### Paired optimization evidence

Three interleaved profiler rounds compare the old and new kernels on identical inputs. In the
final source-matched comparison, model plain backward is **0.43685 -> 0.31755 ms (1.376x,
27.3% less device time)**. These isolated kernel comparisons have different surrounding work
from the full eager/compiled/roof run above; compare timings within each capture.

Increasing the old kernel's grid alone gives only 1.025x. Splitting the channels and reducing
the row tile from two rows to one lowers registers from 197 to 64 per thread, with no local
allocation in either kernel. The primary grid grows from 208 to 864 programs. The increased
parameter-partial traffic is included in the new roof accounting.

Six additional shapes cover both dispatch boundaries, strided storage, random/broadcast
adjoints and deterministic repeated gradients. The 8192-row cases improve 1.14–1.17x; smaller
cases retain the old path. A preliminary 1024-row specialization was slower and was excluded.
Lint passes with zero findings after making parameter-offset int64 casts explicit.
Raw sweeps, repeats, resources and boundary checks are in `golden.json:tuning_measurements`.

## Traffic and arithmetic

`C` denotes activation elements and `R` rows unless defined otherwise; `D` is the channel width
and `P` the reduction programs. Parameters and positional tables count once as compulsory
traffic. Partial buffers count both their write and subsequent reduction read; final
parameter-gradient stores are included. Transcendental instruction cost is not represented by an
invented FLOP multiplier.

Let `R=B*N`, `C=R*D`, and `P=min(ceil(N/4),max(1,floor(A*SMs/B)))`, with `A=12` for 1152-wide
SiLU, `A=8` for the A800 plain specialization above, and `A=2` otherwise. Direct small-input
reductions use `P=0`. Forward bytes: `4*C + 8*B*D +
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

## Parameterized H20 implementation (2026-09-09)

`h20_adaln.py` replaces the former narrow model-size specialization. `TUNED.json` records the frozen selector and measured configurations; `h20.json` contains source-matched measurements, independent eager adjoints, boundary checks and the former narrow evidence under `historical_capture`. The A800 section above and `golden.json` describe their historical workload suite; this expanded suite was not remeasured on A800.

The same kernels support bf16 `(B,N,D)` with fp32 `(B,D)` modulation for D=1…8192. All normalization denominators, addresses and masks use D. Decomposed kernels derive `HEAD=floor_power_of_two(D)` and `TAIL=next_power_of_two(max(1,D-HEAD))`; invalid tail channels are masked and excluded from variance. D=1152 is not a selector branch. Plain multirow forward competes with decomposed SiLU tiles; wide single-row tiles avoid the 2D layout overhead. Backward, including SiLU, uses the same parameterized decomposition and int64 loop/address arithmetic.

Four power-of-two width buckets choose row tiles, warps and reduction waves. SiLU tile footprint and tail padding refine the configuration by formulas rather than exact dimensions. The per-sample partial count is `max(1,min(ceil(N/4),waves*SMs//B))`. Training-only statistics and the bf16 SiLU adjoint boundary are preserved.

Automatic dispatch requires H20, more than 128 total head/activation rows and more than 32 tokens per sample (AdaLN) or B*S tokens (QK). Small, empty, other-device and out-of-range widths retain the existing path. There is no runtime autotuning or KDA runtime import.

The expanded required suite has 15 workloads. All output, inference and gradient checks pass, plus 68 boundary/layout cases and 20 independent eager/autograd adjoint scenarios. Boundary coverage includes width and dispatch boundaries, non-power-of-two tails, independently permuted outer axes, channel-strided/broadcast upstreams, empty inputs and a float64 process default dtype. Forced legacy dispatch is bitwise identical to the saved original implementation on all required cases.

Same-input paired comparisons use profiler device time, cold L2, three interleaved rounds, three warmups and ten timed iterations. Each cell below is the geometric mean across the measured matrix, followed by its range; these are sample-matrix summaries, not guarantees for every supported shape.

| Phase | Geometric mean speedup vs original generic golden | Range |
|---|---:|---:|
| fwd | 1.31× | 0.98–3.12× |
| bwd | 1.29× | 0.96–2.07× |
| infer | 1.30× | 0.95–3.13× |

Full benchmark performance status: **`tune`**. Calibration, numerical tolerances and performance thresholds are unchanged; byte accounting follows the actual partial count.
Model SiLU and the added wide SiLU backward remain below the copy-SoL gate. The generalized int64-loop model SiLU backward is about 4% slower than the former narrow implementation; this tradeoff removes a substantial non-power-of-two-tail regression. Plain model forward/backward retain their main speedups.

```sh
python examples/f32_adaln/benchmark.py --verify --bench --json /tmp/adaln-h20-general-fresh.json
python examples/f32_adaln/benchmark.py --adjoint --workload user_train --json /tmp/adaln-h20-general-adjoint-fresh.json
```
