# Golden: Shared RMSNorm + half-pairing RoPE + permute

Hand-written target for `decoder/attention.py::qk_prep` in the example's `user_repo/`.

Derived from the product's standalone RMSNorm/RoPE fusion exemplar. Each launch processes q or k
with a tile of heads from one token, shares that token's rotary-table row across heads, and
stores directly in head-major order. RMSNorm output is rounded to bf16 before RoPE, matching the
eager graph. The saved auxiliary is one fp32 inverse standard deviation per head row.

Backward gathers arbitrary-stride head-major upstream gradients, applies inverse rotation,
rounds that adjoint to bf16, and computes the RMSNorm adjoint. A bounded grid loops over tokens,
reduces shared weight gradients over heads in registers and then reduces per-program partials.
Representative shapes use separate Q and K launches. Small rows use a fused Q/K path and direct
final weight-gradient ownership.

## Contract

The standalone kernel imports only PyTorch and Triton. Inputs and upstream gradients use their
own leading strides, including broadcast gradients. Last-dimension activation storage is
contiguous. Row/base offsets use int64, tails are masked, and empty inputs return without a
zero-grid launch. Forward auxiliaries are allocated and written only when a backward can follow.

## A800 measurements

B8 S2048 H32 D128, bf16 q/k with shared fp32 head weights.

Device time from `torch.profiler` on NVIDIA A800-SXM4-80GB, torch 2.11.0+cu128. Entries were
timed round-robin, taking the minimum of three rounds. Roof percentages use a measured same-size
copy, with byte accounting below; these are measurements on this device, not a performance
promise elsewhere.

| Phase | Golden | Eager | Compiled | SoL |
|---|---:|---:|---:|---:|
| forward (training) | 0.3221 ms | 8.4639 ms | 0.8060 ms | 98.1% |
| forward (inference) | 0.3208 ms | 8.4668 ms | 0.6238 ms | 98.0% |
| backward | 0.5881 ms | 13.7416 ms | 1.1866 ms | 79.8% |

Historical A800 verdict: **pass**. That report covers the original 4 workloads, including every required
output, inference and gradient check. Independent adjoints also pass. All numerical results,
per-workload timings, copy roofs and raw rounds are recorded in `golden.json`; the recorded
kernel and reference hashes identify the sources measured at that time. Small latency-bound rows retain the runtime's
10-us SoL waiver and 1-us absolute baseline parity rule.

## Traffic and arithmetic

`C` denotes activation elements and `R` rows unless defined otherwise; `D` is the channel width
and `P` the reduction programs. Parameters and positional tables count once as compulsory
traffic. Partial buffers count both their write and subsequent reduction read; final
parameter-gradient stores are included. Transcendental instruction cost is not represented by an
invented FLOP multiplier.

Let `C=B*S*H*D` and `R=B*S*H` for one tensor. Combined q+k forward bytes: `8*C + 8*D + 4*S*D +
8*R`; inference omits `8*R`. Backward bytes: `12*C + 8*R + 4*S*D + 16*D + 16*P*D`,
`P=min(B*S,2*SMs)`, or P=0 for the direct small reduction at R<=128 and D=128. Arithmetic
estimates for both tensors: `2*(7*C+R)` forward and `24*C` backward.

## What to check in a candidate

The pairing is rotate-half, not interleaved. Norm weights are shared `(D,)`, unlike the GQA
example's per-head weights. Ignoring either storage-dtype rounding boundary changes the
numerical graph. A pre-permute copy defeats the fused-store advantage.

## Reproduce the checks

From the checkout root, using the GPU Python interpreter:

```bash
python examples/qk_rmsnorm_rope_permute/benchmark.py --verify --json tmp/golden-benchmarks/qk_rmsnorm_rope_permute/verify.json
python examples/qk_rmsnorm_rope_permute/benchmark.py --bench --json tmp/golden-benchmarks/qk_rmsnorm_rope_permute/report.json
python examples/qk_rmsnorm_rope_permute/benchmark.py --adjoint --json tmp/golden-benchmarks/qk_rmsnorm_rope_permute/adjoint.json
```

[Workloads and SoL accounting](../benchmark_cases.py) define the inputs, gradient reductions,
and per-phase bytes/FLOPs. The [shared runner](../../_benchmark.py) uses KDA's product runtime
directly. Reports are generated outside `golden/`; new numerical and timing reports require an
independent acceptance audit. Use a distinct output filename for scoped `--workload` or
`--fwd-only` runs.

## Parameterized H20 implementation (2026-09-09)

`h20_qk_norm_rope.py` replaces the former narrow model-size specialization. `TUNED.json` records the frozen selector and measured configurations; `h20.json` contains source-matched measurements, independent eager adjoints, boundary checks and the former narrow evidence under `historical_capture`. The A800 section above and `golden.json` describe their historical workload suite; this expanded suite was not remeasured on A800.

H and even D=2…8192 are compile-time parameters, including masked non-power-of-two half-channel tails. No H=32/D=128 equality guard remains. The public Q/K path launches both inputs in one grid, reusing the parameterized single-input computation with separate Q/K strides. Backward writes per-head/token-tile fp32 partials and combines both weight gradients in one final reduction launch. Single-input entry points remain available.

Four half-width buckets choose forward/backward token tiles and warps; head count and short token sequences determine reduction waves. Each input has `H*max(1,min(ceil(B*S/BT),waves*SMs//H))` partial rows. The bf16 norm output and inverse-RoPE adjoint boundaries remain explicit.

Automatic dispatch requires H20, more than 128 total head/activation rows and more than 32 tokens per sample (AdaLN) or B*S tokens (QK). Small, empty, other-device and out-of-range widths retain the existing path. There is no runtime autotuning or KDA runtime import.

The expanded required suite has 10 workloads. All output, inference and gradient checks pass, plus 26 boundary/layout cases and 21 independent eager/autograd adjoint scenarios. Boundary coverage includes width and dispatch boundaries, non-power-of-two tails, independently permuted outer axes, channel-strided/broadcast upstreams, empty inputs and a float64 process default dtype. Forced legacy dispatch is bitwise identical to the saved original implementation on all required cases.

Same-input paired comparisons use profiler device time, cold L2, three interleaved rounds, three warmups and ten timed iterations. Each cell below is the geometric mean across the measured matrix, followed by its range; these are sample-matrix summaries, not guarantees for every supported shape.

| Phase | Geometric mean speedup vs original generic golden | Range |
|---|---:|---:|
| fwd | 1.66× | 1.05–6.23× |
| bwd | 3.52× | 2.02–7.07× |
| infer | 1.62× | 1.03–5.22× |

Full benchmark performance status: **`pass`**. Calibration, numerical tolerances and performance thresholds are unchanged; byte accounting follows the actual partial count.
All applicable gates pass in the expanded suite. Small latency-bound cases retain the existing documented gate exceptions.

```sh
python examples/qk_rmsnorm_rope_permute/benchmark.py --verify --bench --json /tmp/qk-h20-general-fresh.json
python examples/qk_rmsnorm_rope_permute/benchmark.py --adjoint --workload user_train --json /tmp/qk-h20-general-adjoint-fresh.json
```
