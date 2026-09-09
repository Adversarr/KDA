# A800 compiler and layout experiments (2026-09-09)

Measured on NVIDIA A800-SXM4-80GB, SM80, 108 SMs, torch 2.11.0+cu128 and Triton 3.6.0.
Use these as hypotheses for that toolchain, not universal launch defaults. The installed
`CUDAOptions` exposes `maxnreg` and `ptx_options`; it does not expose `sched4reg`,
`min_shared_mem` or `clc`. Hopper/Blackwell MMA, TMEM and dynamic register-allocation controls
do not supply an A800 optimization path.

## Measurement

Diagnostics use the product `bench_ms(method="profiler")`, flushing L2 before each measured
call, three warmups, ten iterations and three interleaved rounds. Tables take the minimum
round median. Correctness against the unchanged kernel precedes diagnostic timing; adopting
a candidate additionally requires the independent eager, full workload/baseline/roof checks
and adjoints. Compilation resource counters are recorded with source hashes. The driver's
`n_spills` field is local allocation in 32-bit words, not a dynamic spill count.

## Attention: lowering the register cap did not help

H3 block-causal attention, `(B,H,S,D)=(1,56,3072,128)`, 512-token frames:

| Change from current configuration | Forward ms | Backward ms |
|---|---:|---:|
| Baseline | 1.00515 | 3.25626 |
| `tl.range(..., disable_licm=True)` in the three main loops | 1.04108 | 3.52023 |
| `disallow_acc_multi_buffer=True` | 1.00601 | 3.26827 |
| `maxnreg=128` | 2.17647 | 7.26834 |
| `maxnreg=96` | 3.83017 | 12.19248 |
| `ptx_options="--opt-level=2"` | 1.00508 | 3.26279 |

Baseline forward uses 255 registers/thread, 160 KiB shared memory and 10 local words;
backward dK/dV and dQ use 255 registers with 44 and 14 local words. These observations
justify investigating resource pressure, but do not establish that a register cap improves
occupancy or speed. Twofold loop unrolling requested 288 KiB shared memory and could not run.
One/two-stage forward and eight-warp backward also lost. A 128x32/four-warp forward was only
about 1% faster and was not adopted.

Causal FA2 reference, `(1,32,8192,128)`: baseline 3.18018/10.90501 ms forward/backward;
disabling LICM gives 3.30190/11.18202 ms. Accumulator-buffer and PTXAS O2 changes were within
measurement variation. The 128x32 forward again improved less than 1%. Both attention
implementations retain their existing configuration.

An isolated process with `DISABLE_LLVM_OPT=disable-lsr` gave mixed H3 results and large
round-to-round variation. It lacked a paired same-setting baseline repeat and is inconclusive;
no environment-variable change was adopted. Raw PTX hashes can differ through debug metadata,
so a different hash alone is not evidence of different instructions.

## AdaLN: remove padded reduction work before limiting registers

Plain AdaLN backward, bf16 `(B,N,D)=(16,4096,1152)`, fp32 sample modulation:

| Candidate, same diagnostic session | Backward ms | Registers/thread |
|---|---:|---:|
| Two rows, 2048 padded channels, two program waves | 0.43585 | 197 |
| Same kernel, eight waves only | 0.42502 | 197 |
| One row, 1024+128 channels, eight waves | 0.31726 | 64 |

Both kernels have zero local allocation and 4 KiB shared memory. The split reduces live
values and padded reduction work; a larger grid then supplies more concurrent work. The
primary grid grows from 208 to 864 programs. This is a two-part structural change, not an
isolated compiler-flag benefit. A final-source paired repeat confirms 0.43685 -> 0.31755 ms
(1.376x, 27.3% less device time). Full eager/compiled/roof benchmarking has different
surrounding work and measures 0.29856 ms, 90.8% of its achievable roof; timings from the two
protocols must not be mixed into one speedup.

The new plain path is restricted to A800, `D=1152`, `N>=1024` and `B*N>=8192`. A preliminary
1024-row case regressed, motivating the total-row guard. Validated 8192-row cases improved
1.14–1.17x, including strided input. Smaller cases retain their prior path. SiLU retains its
existing twelve-wave split, its storage-dtype activation/adjoint boundaries and its measured
performance. Parameter partials increase with the grid; the byte model includes their writes
and reduction reads. Neither the numerical tolerances nor performance thresholds changed.

Seven required workloads, all eligible baseline comparisons, small/model/SiLU adjoints and
six additional boundary/stride/determinism cases pass. Source audit was performed in the
implementation context; no independent worker was used. The KDA source repository records
the raw captures in `examples/f32_adaln/golden/golden.json`, under `tuning_measurements`;
the prior complete record is retained under `historical_capture`.

For future rowwise fusions, compare a legal non-padded channel decomposition and a smaller
row tile when padding and live accumulators are substantial. Sweep the partial-reduction
grid separately, account for its traffic, and keep a measured fallback for small inputs.
