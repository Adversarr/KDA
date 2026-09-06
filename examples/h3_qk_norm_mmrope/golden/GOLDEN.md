# Golden: H3 shared RMSNorm + partial MM-RoPE

Hand-written target for `h3/attention.py::qk_prep` in the example's `user_repo/`.

The norm spans all 128 channels. MM-RoPE rotates only channels 0:96 using two 48-channel halves;
the final 32 channels pass through normalized and unrotated. Tables contain the model's
three-axis positions and base-10000 frequencies. The bf16 normalization boundary is preserved in
both forward and backward.

Forward groups token/head tiles with separate 48+48+32 channel regions. Backward uses eight-head
tiles, accumulates per-head weight contributions through the token loop, and reduces heads only
on exit. Q/K parameter partials share a final reduction launch. Small inputs fuse Q/K in one
forward and one backward launch with direct final weight-gradient ownership. Only inverse RMS is
saved.

## Contract

The standalone kernel imports only PyTorch and Triton. Inputs and upstream gradients use their
own leading strides, including broadcast gradients. Last-dimension activation storage is
contiguous. Row/base offsets use int64, tails are masked, and empty inputs return without a
zero-grid launch. Forward auxiliaries are allocated and written only when a backward can follow.

## A800 measurements

B1 S2048 H56 D128, fp32 shared weights, grid (4,16,32), eps 1e-5.

Device time from `torch.profiler` on NVIDIA A800-SXM4-80GB, torch 2.11.0+cu128. Entries were
timed round-robin, taking the minimum of three rounds. Roof percentages use a measured same-size
copy, with byte accounting below; these are measurements on this device, not a performance
promise elsewhere.

| Phase | Golden | Eager | Compiled | SoL |
|---|---:|---:|---:|---:|
| forward (training) | 0.08694 ms | 1.89732 ms | 0.20478 ms | 88.5% |
| forward (inference) | 0.08453 ms | 1.89723 ms | 0.19000 ms | 90.2% |
| backward | 0.14402 ms | 3.81242 ms | 0.32038 ms | 78.5% |

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

Define per q or k: `C=B*S*H*128`, `R=B*S*H`, `P=min(B*S,2*SMs)*ceil(H/8)`. Combined Q/K forward
bytes: `8*C + 8*D + 768*S + 8*R`; inference omits `8*R`. Backward bytes: `12*C + 8*R + 768*S +
16*D + 16*P*D`. For the direct small-input backward, `P=0`. Scalar arithmetic estimates are
`2*(7*C+R)` forward and `24*C` backward.

## What to check in a candidate

Normalizing only rotated channels, rotating the last 32 channels, or using 64-channel half
pairing is incorrect. Shared weights reduce across all heads. The required long-sequence case
uses 21840 tokens and must retain int64 base offsets.

## Reproduce the checks

From the checkout root, using the GPU Python interpreter:

```bash
python examples/h3_qk_norm_mmrope/benchmark.py --verify --json tmp/golden-benchmarks/h3_qk_norm_mmrope/verify.json
python examples/h3_qk_norm_mmrope/benchmark.py --bench --json tmp/golden-benchmarks/h3_qk_norm_mmrope/report.json
python examples/h3_qk_norm_mmrope/benchmark.py --adjoint --json tmp/golden-benchmarks/h3_qk_norm_mmrope/adjoint.json
```

[Workloads and SoL accounting](../benchmark_cases.py) define the inputs, gradient reductions,
and per-phase bytes/FLOPs. The [shared runner](../../_benchmark.py) uses KDA's product runtime
directly. Reports are generated outside `golden/`; new numerical and timing reports require an
independent acceptance audit. Use a distinct output filename for scoped `--workload` or
`--fwd-only` runs.
