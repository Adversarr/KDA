# Speed of light (SOL) for fused training kernels

The single source for roofline reasoning in KDA. `_speed_of_light.py::estimate(phase, **inputs)`
in every op returns `(bytes_moved, flops)`; `_run_dev.py` turns it into a time and an
efficiency, and the M3/M4 gate reads that efficiency.

## Two roofs, and which one you are under

- **Memory roof**: `bytes_moved / bandwidth`. Every fused op in the Level-1 catalogue
  (norms, residuals, gating, RoPE, permutes) is here: a handful of FLOPs per byte.
- **Compute roof**: `flops / peak_flops` of the unit doing the math (`ROOF = "fp32"` for CUDA
  cores, `"bf16"` for tensor cores). Only GEMM epilogues get near it.

SOL time is the larger of the two. Report which roof you are under in SPEC.

## Datasheet peak vs achievable roof

`_common.gpu_info` derives the datasheet HBM bandwidth (A800: 2039 GB/s) and peak TFLOP/s.
Real kernels never see the datasheet number: a plain `copy_` reaches ~73% of it at 64 MB and
~84% at 256 MB on A800 (ramp-up, tail, DRAM refresh). So the gate uses the **achievable roof**:
`_run_dev.py` times a device copy moving the same `bytes_moved` and takes the slower of that
and the compute roof. `sol_eff = roof_ms / kernel_ms`; `>= 0.7` meets the SOL component of the runtime gate; baseline, correctness and coverage still apply. Both numbers appear in
`report.json` (`sol_ms` datasheet, `roof_ms` achievable).

Consequence: a memory-bound kernel at 90% of copy is done. Chasing the datasheet number is
chasing DRAM physics.

## Counting bytes

Count every tensor once, at its own element size, for the *minimum* traffic of the fused op:

- **forward**: each input read once, each output written once, aux saved for the backward written once.
- **backward**: each upstream gradient read once, each saved tensor and each input needed
  for recomputation read once, each gradient written once. "Once" is per *kernel that reads
  it* in the SPEC's reference design: a fused row-wise backward reads `dy` once; a GEMM
  backward that runs `dX = dZ W` and `dW = dZ^T X` as two cuBLAS calls reads `dZ` twice, and
  a `db` column reduction that is not fused into either GEMM's epilogue reads it a third time
  (the `_speed_of_light.py` template comment says the same in GEMM terms). Weight-gradient partials that
  the host reduces count too, **written once and read back once by the reduce**,
  `2 * n_programs * D * 4`, plus the `D * 4` write of `dw` itself (the same both-directions
  rule as the split-K partials below; the table rows abbreviate it as `partials`). Their
  reduce kernel is part of the op. At SPEC time the geometry is not chosen yet: count
  partials with the reference geometry, `n_programs = min(ceil(rows / R), 2 * SM count)` with
  `R = max(1, 4096 // next_pow2(D))` (the row-wise kernels in `triton/fp32_norms/`), and
  `torch.cuda.get_device_properties(x.device).multi_processor_count` for the SM count inside
  `_speed_of_light.py::estimate`; write that assumption into the SPEC roofline. An
  implementer who launches differently does **not** edit `_speed_of_light.py`
  (the roofline is not the implementer's to move; a lowered roof would inflate `sol_eff`):
  it reports the new partial count in `IMPL_NOTES.md` and its 10-line report, and the
  orchestrator updates `_speed_of_light.py` and the SPEC prose before the next verify. For
  the user's shapes the partials are a few percent of the backward bytes, so the choice moves
  `sol_eff` little either way.

Do not count register or L2 traffic, and do not count bytes the eager version moves but the
fusion avoids: SOL is a property of the op, not of an implementation.

`estimate` takes one of four phases; the last two differ from the first two only in what is
*not* moved:

- **`infer`**: the forward under `no_grad`. Inputs read once, user-visible outputs written
  once, **no aux** (nothing is saved). `infer <= fwd` always; for rmsnorm the difference is
  `rows*4`, for a GEMM that saves `Z` it is a whole `M*N*es` write.
- **`bwd_recompute`**: the backward when the aux was not saved. Drop the aux read and add the
  reads needed to rebuild it (for norms `x` is already read, so the aux bytes simply vanish and
  the FLOPs grow; for a GEMM epilogue that recomputes `Z`, add the `X` and `W` reads and the
  `2*M*N*K` FLOPs of the recomputed GEMM). The gate applies to it only when SPEC
  `recompute.default: true`; it is still timed and reported so the user can see the trade.

Worked examples (`n = rows * D`, `es = element size of x`):

| op | fwd bytes | bwd bytes | infer | bwd_recompute |
|---|---|---|---|---|
| rmsnorm | `2*n*es + rows*4` (x, y, rstd) | `3*n*es + rows*4 + partials` (dy, x, dx, rstd; `partials = 2*n_prog*D*4 + D*4`) | `2*n*es` | `3*n*es + partials` |
| residual + rmsnorm, fp32 residual out | `n*es + n*4 + n*4 + n*es + rows*4` (x, r, h_out fp32, y, rstd) | `dy, dh, h, dx, dr, rstd, partials` | drop `rows*4` | drop `rstd` |
| rope | `2*n*es + S*D*4` (tables are tiny) | same | same | same |
| rmsnorm + rope + permute | `2*n*es + rows*4` | `3*n*es + rows*4 + partials` | `2*n*es` | `3*n*es + partials` |

Per-sample parameter gradients (adaLN `dscale`, `dshift` of shape `(B, D)`, each the sum over
sample `b`'s `N` tokens) follow the same partials rule with the per-sample geometry: programs
never straddle a sample, so with `n_pb = ceil(N / R)` programs per sample the partial buffer is
`(B, n_pb, D)` fp32 per gradient, counted written once and read once by the reduce, plus the
`B * D * 4` finals: `2 * (2 * B * n_pb * D * 4) + 2 * B * D * 4` for the pair. When a tune
round changes `R` or the grid cap, `n_pb` changes and the roofline must follow (orchestrator
hand-off, M4).

An op with two weight gradients in one kernel (`q_norm_w` and `k_norm_w` of a fused QK prep)
counts partials per weight: each has its own `(n_programs, D)` buffer written once and read
once, and its own `D * 4` final; `rows` in `n_programs` is the row count of the chain that
produces it (both chains when one program handles a `q` row and a `k` row together).

**SFU work under the copy roof.** On CUDA-core ops the achievable roof is a same-size copy,
and a per-element transcendental in the epilogue (`sigmoid`, `exp`, `erf`: 1-2 MUFU ops each,
16 per clock per SM, about `2.4e12/s` on an A800) is invisible to it. Compare `mufu_ops *
n_elements / 2.4e12` with the copy time: at `D = 1152`, bf16, 65k rows, a SiLU is ~60 us of
MUFU against a ~190 us copy, ~30%, and a `BLOCK_D = 2048` tile issues it on the masked lanes
too (~1.8x). An epilogue that costs a third or more of the copy time does not hide behind the
memory stream at the occupancy a row-wise kernel has, and no FLOP term in
`_speed_of_light.py` can raise the roof (the compute roof is the FMA rate, far below the copy).
Such rows can read `tune` at 0.6-0.7 with nothing defective: a fast sigmoid and chunking away the masked lanes buy 5-10 points, then the gap
is structural. Say so in the SPEC Roofline, and the orchestrator accepts best-so-far after
one tune round rather than two.

Table bytes (`cos`/`sin`, `S * rotary_dim * 4`, re-read by every head from L2) may be counted
or not: they are under 1% of `n * es` at any realistic `H`, so both readings give the same
`sol_eff` to two decimals. State the choice once in SPEC Roofline (the fused row above omits
them, the unfused `rope` row includes them; compute-patterns.md says "do not count") and keep
prose and `_speed_of_light.py` identical; do not spend a gate paragraph on it.

FLOPs for these are `O(10) * n`; at fp32 CUDA-core rates they are 20-50x below the memory
time, which is why the compute roof never binds for them.

## Compute roof: GEMM + epilogue worked example

`Y = act(X W^T + b) + R`, `X: (M, K)`, `W: (N, K)`, bf16 storage (`es = 2`), fp32 accumulate.

- **FLOPs**: `2*M*N*K` for the GEMM; the epilogue adds `O(M*N)` and is ignored.
- **Bytes (fwd)**: `M*K*es + N*K*es + N*4 + M*N*es (R) + M*N*es (Y)`, plus `M*N*es` for the saved
  pre-activation `Z` under `SAVE_AUX`. `ROOF = "bf16"` so `_run_dev.py` reports the datasheet
  `sol_ms` from the tensor-core peak (`gpu_info`: A800 312 TFLOP/s dense, H100 SXM 989, B200
  2250) **and** measures the achievable roof as cuBLAS on `gemm_shapes(phase)` (below).
- **Which roof**: at `M = 8192, N = K = 4096` the FLOPs take `2*8192*4096*4096 / 312e12 = 1.76 ms`
  and the bytes `(8192*4096*2*3 + 4096*4096*2) / 2039e9 = 0.11 ms`: compute-bound by 16x, and
  the epilogue fusion saves only the `R`/`Z`/`Y` round trips. At `M = 64` (decode-style, the
  `4D -> D` down-projection with a small token count) the same arithmetic gives `0.014 ms` compute
  vs `0.017 ms` bytes (the weight read dominates): memory-bound, where split-K exists.
- **Split-K partials**: with `SPLIT_K = s` each split writes an fp32 partial `M*N*4` and the
  reduce reads them back and writes `Y`: add `2*s*M*N*4` to the bytes (partials written, read)
  and count `Y` once as before. For `M = 64, N = 4096, s = 4` that is `8 MB`, a third of the
  weight read: split-K pays for itself only when the tile grid is far below the SM count.
  Choose `s` in `select_config`; the orchestrator counts it in `_speed_of_light.py` with the
  same formula and says so in the SPEC roofline prose (the implementer reports the formula,
  as for `dw` partials above).
- **Backward** (fused adjoint + two cuBLAS GEMMs): adjoint bytes `dY, Z (or recompute), dZ, dR,
  db partials`; the `dX = dZ W` and `dW = dZ^T X` GEMMs are `4*M*N*K` FLOPs at the compute roof.
  `_speed_of_light.py` counts both because the op owns them (the kernel launches them).
- **GEMM count per phase** (the check that catches a doubled roof): one GEMM is `2*M*N*K`;
  `fwd` and `infer` run 1, `bwd` runs 2, `bwd_recompute` runs 3 (the recomputed `Z` plus the
  two above). With a `gemm = 2*M*N*K` helper that is `gemm`, `2*gemm`, `3*gemm`; `4*gemm` for
  the backward (a factor copied from the `4*M*N*K` above) doubles the roof, and the symptom is
  an inflated backward `sol_eff`, potentially above 1. That ratio compares a measured achievable roof, so audit the count separately from the datasheet bound. For
  `M = 8192, N = 8192, K = 2048`: fwd `0.27 TFLOP` = `0.88 ms`, bwd `0.55 TFLOP` = `1.76 ms`,
  bwd_recompute `0.82 TFLOP` = `2.64 ms` on A800.

Reading a compute-roof number: `sol_eff` is `roof_ms / kernel_ms`, and for a tensor-core op
(`ROOF = "bf16"`) the **achievable roof is measured cuBLAS on the phase's GEMMs**
(`_common/bench.py::matmul_ms` on `_speed_of_light.gemm_shapes(phase, ...)`), exactly as the
memory roof is a measured same-size copy. The datasheet peak (312 TFLOP/s on A800) is reported
as `sol_ms` but is reached by nothing: cuBLAS lands at 70-85% of it depending on shape, and
`torch.compile(mode="max-autotune")` (the harness's baseline for tensor-core ops) picks cuBLAS
or a Triton template that ties it with the epilogue fused. That compiled number *is* the
speed of light for a GEMM + epilogue. Against the cuBLAS roof a library mainloop plus
an epilogue pass can read ~0.85-0.95 and
the `SOL_THRESHOLD = 0.7` component means at least 70% of the measured cuBLAS throughput for the same work. A CUDA-core fallback (`< 0.3`) or a poor tile (`0.3-0.6`) can explain a gap; correctness alone does not guarantee that efficiency. Give
`gemm_shapes` the real `(M, N, K)` list per phase (fwd `[(M, N, K)]`, bwd
`[(M, K, N), (N, K, M)]`, bwd_recompute all three): without it the harness times one cube of
the same FLOPs, which cuBLAS runs near its best, so the roof is tighter than the true shapes'.
`torch.matmul` on the same shapes is the baseline
that must not beat the fused kernel by more than the epilogue's bytes explain (`3*M*N*es` at HBM
speed: ~230 us for `8192 x 8192` bf16 on A800). When it does, the fused mainloop is the wrong
tool for that shape and `torch.matmul` + one epilogue kernel is the kernel to ship (the Triton
GEMM SNIPPET's `mainloop="cublas"` path and its table); the pattern is still `gemm_tensorcore`. The bar that
matters is the report's `compile` column: inductor runs cuBLAS or its own Triton GEMM template
and fuses the pointwise tail into one kernel. When that column wins, read the template it
generated (kda-kernel-implement SKILL.md §3, "When the `compile` baseline ... cannot beat")
before inventing a hypothesis; the Triton GEMM SNIPPET's measured table shows what the fusion
buys and costs shape by shape.

## Small grids and the latency floor

Below a few MB the roof is not bandwidth but latency: a device copy of 0.5 MB takes ~3.5 us
on A800 against ~0.25 us at the datasheet rate, and a one-program-per-row kernel over 32-100
rows takes 4-7 us for the same bytes. A forward that is one kernel lands near `sol_eff 1.0`
against that copy; a backward that is a kernel plus a `dw` host reduce plus autograd's grad
buffer fill lands at 0.2-0.5 *while beating eager by 4x*, because at this size the time is
kernel count, not bytes. So the verdict has a **latency floor**: when the achievable roof of a
phase is under `LATENCY_FLOOR_MS` (10 us, `_common/report.py`) its `sol_eff` is reported but
not gated and only the baseline comparison counts (a kernel within `BASELINE_TOLERANCE` = 5%
of the best baseline is parity, not "slower"). Do not chase `sol_eff` on such a row, and do not
split rows across programs to "fill the machine": it makes the kernel slower (rmsnorm SNIPPET,
geometry table). The structural limit worth a second kernel geometry is row *width* (register
file), see `compute-patterns.md`, row width rule.

## Reading an efficiency number

These ranges guide diagnosis only. The finalized verification verdict (including baseline, correctness and coverage) and orchestrator state determine stage transitions.

| `sol_eff` | meaning | next move |
|---|---|---|
| >= 0.9 | near the measured roof | further device-time tuning may have little value |
| 0.7-0.9 | near the runtime SOL threshold; launch geometry or tail effects | possible diagnostic: rows-per-program, `num_warps`, fewer partials |
| 0.4-0.7 | below the usual SOL threshold | check for uncoalesced access (strided index arithmetic), oversized tiles spilling registers, too few programs, a hidden second pass (`.contiguous()`, per-row partials) |
| < 0.4 | something structural | the kernel is not doing what the roofline assumed: extra reads, atomics, scalar loads, or the roofline is wrong |

`ncu` (see the `ncu-report` skill) tells memory- from latency-bound apart when the table above
is not enough: `dram__throughput` near 80%+ can explain limited memory-throughput headroom; it does not override the verdict.
