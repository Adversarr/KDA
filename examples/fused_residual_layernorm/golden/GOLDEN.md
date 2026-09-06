# Golden: Fused residual add + LayerNorm

Hand-written target for `model.py::add_layernorm` in the example's `user_repo/`.

One forward program owns each row: it loads bf16 `x` and fp32 `residual`, forms the fp32 stream
`h`, computes its mean and population variance, and writes `h` and the bf16 affine-normalized
view. The returned stream is retained for backward, so the only extra saved tensors are one fp32
mean and inverse standard deviation per row. Inference writes neither auxiliary.

The backward combines the gradient into `h` with the LayerNorm adjoint before storing bf16 `dx`
and fp32 `dresidual`. Programs stride over four-row tiles, retain `dweight` and `dbias`
accumulators in registers, and reduce one fp32 partial per program. It does not materialize the
normalized activation.

The persistent backward now combines weight and bias partial reductions in one launch. Small
inputs use one activation row and disjoint parameter-gradient columns per block, writing final
gradients directly.

## Contract

The standalone kernel imports only PyTorch and Triton. Inputs and upstream gradients use their
own leading strides, including broadcast gradients. Last-dimension activation storage is
contiguous. Row/base offsets use int64, tails are masked, and empty inputs return without a
zero-grid launch. Forward auxiliaries are allocated and written only when a backward can follow.

## A800 measurements

B4 S2048 D2048; 8192 rows, bf16 x and fp32 residual/weight/bias.

Device time from `torch.profiler` on NVIDIA A800-SXM4-80GB, torch 2.11.0+cu128. Entries were
timed round-robin, taking the minimum of three rounds. Roof percentages use a measured same-size
copy, with byte accounting below; these are measurements on this device, not a performance
promise elsewhere.

| Phase | Golden | Eager | Compiled | SoL |
|---|---:|---:|---:|---:|
| forward (training) | 0.11904 ms | 0.97664 ms | 0.12136 ms | 99.7% |
| forward (inference) | 0.11867 ms | 0.97553 ms | 0.12070 ms | 100.1% |
| backward | 0.21296 ms | 1.81180 ms | 0.22515 ms | 75.7% |

Final verdict: **pass**. All required numerical and performance rows pass.

Full output/gradient checks, phase times, measured copy roofs, separate datasheet bounds,
baseline ratios, round samples and available repeats are recorded in `golden.json`.
Source-matched full required numerical and performance coverage, eligible baselines, independent
adjoints and near-gate repeats audited; all pass.

## Traffic and arithmetic

`C` denotes activation elements and `R` rows unless defined otherwise; `D` is the channel width
and `P` the reduction programs. Parameters and positional tables count once as compulsory
traffic. Partial buffers count both their write and subsequent reduction read; final
parameter-gradient stores are included. Transcendental instruction cost is not represented by an
invented FLOP multiplier.

Forward bytes: `12*C + 8*D + 8*R`; inference omits `8*R`. Backward bytes: `16*C + 8*R + 12*D +
16*P*D`, where `P=min(ceil(R/4),2*SMs)` for persistent backward and zero for the direct
small-input path. The last term includes both writes and reads of weight/bias partials.
Arithmetic estimates are `9*C+3*R` forward and `14*C+2*R` backward; rsqrt is an SFU instruction,
not an assigned FLOP.

## What to check in a candidate

Dropping the gradient from either returned output produces an incorrect residual adjoint. Saving
a normalized tensor adds an unnecessary activation-sized allocation. Independent strides must be
retained for `x`, residual, and both upstream gradients.

## Reproduce the checks

From the checkout root, using the GPU Python interpreter:

```bash
python examples/fused_residual_layernorm/benchmark.py --verify --json tmp/golden-benchmarks/fused_residual_layernorm/verify.json
python examples/fused_residual_layernorm/benchmark.py --bench --json tmp/golden-benchmarks/fused_residual_layernorm/report.json
python examples/fused_residual_layernorm/benchmark.py --adjoint --json tmp/golden-benchmarks/fused_residual_layernorm/adjoint.json
```

[Workloads and SoL accounting](../benchmark_cases.py) define the inputs, gradient reductions,
and per-phase bytes/FLOPs. The [shared runner](../../_benchmark.py) uses KDA's product runtime
directly. Reports are generated outside `golden/`; new numerical and timing reports require an
independent acceptance audit. Use a distinct output filename for scoped `--workload` or
`--fwd-only` runs.
