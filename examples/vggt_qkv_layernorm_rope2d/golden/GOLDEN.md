# Golden: VGGT affine head LayerNorm + 2-D RoPE

Hand-written target for `geometry/qkv.py::qkv_prep` in the example's `user_repo/`.

Q and K are read directly from the interleaved projection, normalized with shared affine
LayerNorm, rotated by their y/x positions, and stored head-major. Each axis uses rotate-half
within 32 channels. Five special tokens per view have zero position; patch positions run from 1
through 37. The bf16 normalization boundary is preserved.

The Q forward also copies V into head-major storage and, during training, saves 64 fp32 rotary
coefficients per token for both adjoints. Backward processes four 16-channel quarters, reuses
the saved coefficients and writes the V adjoint directly into the interleaved projection
gradient. Q and K use disjoint CTA groups in one backward launch; only the Q group writes the V
adjoint. Eight program waves per group and two warps improve parallelism. One final reduction is
parallelized across both parameter kind and channel groups.

## Contract

The standalone kernel imports only PyTorch and Triton. Inputs and upstream gradients use their
own leading strides, including broadcast gradients. Last-dimension activation storage is
contiguous. Row/base offsets use int64, tails are masked, and empty inputs return without a
zero-grid launch. Forward auxiliaries are allocated and written only when a backward can follow.

## A800 measurements

B1 N5496 3 H16 D64; four views, 1374 tokens/view, eps1e-5.

Device time from `torch.profiler` on NVIDIA A800-SXM4-80GB, torch 2.11.0+cu128. Entries were
timed round-robin, taking the minimum of three rounds. Roof percentages use a measured same-size
copy, with byte accounting below; these are measurements on this device, not a performance
promise elsewhere.

| Phase | Golden | Eager | Compiled | SoL |
|---|---:|---:|---:|---:|
| forward (training) | 0.05902 ms | 1.18606 ms | 0.33350 ms | 82.7% |
| forward (inference) | 0.05394 ms | 1.18726 ms | 0.18670 ms | 87.4% |
| backward | 0.07947 ms | 1.67587 ms | 0.23313 ms | 79.6% |

Measured status on 2026-09-08: **pass**. All five full-run numerical workloads and
eligible baselines pass. Two primary repeats retain 80.9% backward SoL; frame repeats retain
81.8% and 76.9%. Every applicable phase passes. Independent small and strided adjoints pass;
H20 also passes all five numerical workloads.

Full output/gradient checks, phase times, measured copy roofs, separate datasheet bounds,
baseline ratios, raw rounds and source-matched repeats are recorded in `golden.json`.
The older conflicting captures remain under `historical_capture`. A first repaired-source run
lost its first copy-calibration GPU event in both profiler attempts. The collector now primes
CUDA activity before the measured markers; the new complete run and repeats use that collector.

## Traffic and arithmetic

`C` denotes activation elements and `R` rows unless defined otherwise; `D` is the channel width
and `P` the reduction programs. Parameters and positional tables count once as compulsory
traffic. Partial buffers count both their write and subsequent reduction read; final
parameter-gradient stores are included. Transcendental instruction cost is not represented by an
invented FLOP multiplier.

Let `C=B*N*3*H*D`, `R=B*N*H` (q rows), and `P=min(B*N,8*SMs)`. Training forward bytes: `4*C +
16*D + 16*B*N + D + 16*R + 256*B*N`; inference omits `16*R + 256*B*N`. The last term is the
saved rotary table, written once. Backward bytes: `(16/3)*C + 16*R + 512*B*N + 24*D + 32*P*D`; Q
and K each read the saved table, so backward no longer reads positions or frequencies. V traffic
and all four partial-buffer writes/reads are included. Arithmetic estimates are `22*R*D` forward
and `32*R*D` backward, excluding transcendental instructions.

## What to check in a candidate

This is affine LayerNorm, not RMSNorm. Axis halves and pairing quarters must not be confused.
Returning three disconnected input gradients violates the one-projection contract. Recomputing
trig independently for every head is a measurable bottleneck in the initial candidate.

## Reproduce the checks

From the checkout root, using the GPU Python interpreter:

```bash
python examples/vggt_qkv_layernorm_rope2d/benchmark.py --verify --json tmp/golden-benchmarks/vggt_qkv_layernorm_rope2d/verify.json
python examples/vggt_qkv_layernorm_rope2d/benchmark.py --bench --json tmp/golden-benchmarks/vggt_qkv_layernorm_rope2d/report.json
python examples/vggt_qkv_layernorm_rope2d/benchmark.py --adjoint --json tmp/golden-benchmarks/vggt_qkv_layernorm_rope2d/adjoint.json
```

[Workloads and SoL accounting](../benchmark_cases.py) define the inputs, gradient reductions,
and per-phase bytes/FLOPs. The [shared runner](../../_benchmark.py) uses KDA's product runtime
directly. Reports are generated outside `golden/`; new numerical and timing reports require an
independent acceptance audit. Use a distinct output filename for scoped `--workload` or
`--fwd-only` runs.
