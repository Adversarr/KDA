# Compute patterns

The closed list of ways a KDA kernel is allowed to solve an op. SPEC `compute_pattern` names one
row; the scaffold picks it at M0 from the user's region, the internal M1 checkpoint checks it, the
implementer follows the *required structure*, and the verifier checks the *grepable signature*
(row "pattern honoured"). `_common/spec.py::PATTERNS` mirrors this table row for row: adding a
pattern means adding it there too.

Why a closed list: a fused op is either memory-bound streaming, a tensor-core GEMM, or an
attention-style tiled reduction. Each has one known-good structure and one dominant failure
mode, and a kernel that mixes them (a row-wise norm written as a tiled GEMM, a GEMM written on
CUDA cores) is slow in ways the numbers alone do not explain.

## Table

Only `_common/report.py` defines machine passing gates (currently SOL 0.7, with its latency,
baseline and recompute rules). The performance signals below are measured reference targets
and diagnostic observations, not additional or pattern-specific acceptance thresholds.
A supported explicit backend choice wins; otherwise choose nvmath for a supported GEMM
epilogue when available, then Triton. Hardware recommendations are scoped to their measurements.

| pattern | op classes | regime | required structure | grepable signature | performance signal | default `kernel_backend` | reference |
|---|---|---|---|---|---|---|---|
| `elementwise_stream` | residual add, gating, scale/shift, casts, dropout masks | memory roof | flat 1-D grid over `numel`, `BLOCK` elements per program, one load per input and one store per output, math in `compute_dtype`, mask on the tail | `tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)`, no reduction (`tl.sum`) | reference target `sol_eff >= 0.85` at >= 64 MB; smaller rows may be latency-bound; runtime baseline/coverage gates still apply | `triton` | [`triton/elementwise/residual_add`](../triton/elementwise/residual_add/SNIPPET.md) |
| `rowwise_onepass` | RMSNorm, LayerNorm, L2 norm, softmax over `D`, per-row scale, adaptive LayerNorm (`(1 + scale[b]) * norm(x) + shift[b]` with per-sample `(B, D)` modulation broadcast over the tokens of sample `b`, optional gate/activation epilogue), any `reduce(D) -> broadcast(D)` fusion with `D <= 8192` | memory roof | one program per row (or `R` rows per program), whole row in registers via `BLOCK_D = next_power_of_2(D)`, reduction and normalisation from the same registers, `x` read once; aux (`rstd`, `mean`) stored under `SAVE_AUX`; backward reads `dy` and `x` once, `dw` partials per program reduced on the GPU via a host-launched reduction; per-sample parameters (adaLN `scale[b]`, `shift[b]`) are gathered by `b = row // n_tokens` and their gradients `dscale`, `dshift` are per-`(b, program)` partials reduced over that sample's programs on the GPU (never `tl.atomic_add` into `(B, D)`; the sum is over `N` tokens and must be deterministic). **Row width rule** below | `tl.arange(0, BLOCK_D)` with `mask = cols < D`, one `tl.load` of `x` per row, `tl.sum(..., axis=0)`; grid `(n_rows,)` or `(cdiv(n_rows, R),)` | reference target `sol_eff >= 0.8` at the user shapes; `< 0.5` can indicate a second read of `x`, per-row atomics or register spills from an oversized row | `triton` | [`triton/fp32_norms/rmsnorm`](../triton/fp32_norms/rmsnorm/SNIPPET.md) (split-D variant for wide rows) |
| `rope_pairs` | rotary embedding, any pairwise rotation of `D/2` element pairs, optionally fused after a row-wise norm; **partial** rotation (`rotary_dim < D`: MiniMax-H3 rotates 96 of 128 head channels, the last 32 pass through) and multi-axis tables (3D MM-RoPE: `cos`/`sin` are `(S, rotary_dim)` already summed over the `(t, h, w)` axes by the model, so the kernel sees one table and gathers by token index; the axis split is not the kernel's business) | memory roof | load the two halves (or `tl.split` interleaved pairs) of the rotated prefix once, `cos`/`sin` tables read per position, both halves written once, the pass-through tail `[rotary_dim:D]` copied (or, fused after a norm, normalised and stored) without touching the tables; position ids via a gather on the row index | two `tl.load` at `+0` and `+D/2` (or `tl.split`/`tl.join`), `cos_ptr`/`sin_ptr` indexed by position, no reduction | reference target `sol_eff >= 0.8`; table size is `S*D*4` bytes in the reference; include required reads in the roofline and account for caching when interpreting it | `triton` | [`triton/rope/rope`](../triton/rope/rope/SNIPPET.md), fused: [`triton/fusion-exemplar/rmsnorm_rope_permute`](../triton/fusion-exemplar/rmsnorm_rope_permute/FUSION.md) |
| `permute_copy` | `(B,N,H,D) -> (B,H,N,D)` and other axis permutations, gather/scatter by index, fused as the *store side* of another pattern | memory roof | compute the permuted destination address from the source row index; the same kernel serves both directions with swapped strides; never materialise the permuted tensor separately when fused | strides of both layouts as launcher arguments, a store address built from `tl.program_id` decomposition (`b, h, n = ...`), no math on the values | reference target `sol_eff >= 0.8`; the fused form avoids a separate permutation pass; address cost is workload-dependent | `triton` | [`triton/permute/bnhd_to_bhnd`](../triton/permute/bnhd_to_bhnd/SNIPPET.md) |
| `gemm_tensorcore` | `Y = act(X W^T + b)` and any GEMM with a fused epilogue (bias, activation, dtype cast, quantisation scale, saved pre-activation). A requested residual epilogue must preserve the actual code boundary; many transformer residuals are added by the next norm, while cuBLASLt only handles its supported epilogue set | compute roof (`ROOF = "bf16"`/`"fp16"`) for `M,N,K >= 1024`; memory roof for skinny shapes | **Without an explicit supported backend choice**, when `nvmath` is importable and the epilogue is in cuBLASLt's set (bias, ReLU, tanh-GELU, each with an aux store of the pre-activation, `BGRAD*` reductions), drive `nvmath.linalg.advanced.Matmul` with the epilog: the GEMM and its epilogue run in one cuBLAS kernel at cuBLAS speed (`kernel_backend: nvmath`, `reference/nvmath/epilogue/gemm/`). Backward = the Triton adjoint (`dZ = dY * act'(Z)`, `db` partials) + `torch.matmul` for `dX`, `dW`; `RECOMPUTE` rebuilds `Z` with one more cuBLASLt GEMM. **Second choice** (no nvmath, an activation outside the set, a GEMM under ~200 us where nvmath's 130-175 us host cost shows): `torch.matmul` for `Z` plus one pointwise epilogue kernel over it (the Triton twin's `mainloop="cublas"`). **Last** (an epilogue neither can express: quantised store, custom gate): the fused Triton kernel, 2-D tile grid over `(M, N)`, `K` loop with `tl.dot` on operands in **storage dtype**, fp32 accumulator, epilogue on the accumulator once, `SPLIT_K` for skinny shapes | `Matmul(w, x.t()).plan(epilog=MatmulEpilog.GELU_AUX_BIAS, epilog_inputs={"bias": b})` planned once per shape, `reset_operands` + `execute` per call; or `torch.matmul` + an epilogue kernel that reads `Z` once; or `tl.dot(a, b, acc)` with `a`,`b` straight from `tl.load` and no `.to(tl.float32)` (lint `mma_operand`) | the verdict gate is `sol_eff >= 0.7` against the **achievable** compute roof, which for a tensor-core `ROOF` is measured cuBLAS on the phase's GEMMs (`_speed_of_light.gemm_shapes`; speed-of-light.md), not the datasheet peak; the compile baseline is `max-autotune`. nvmath reaches 0.97-1.0 of that roof with the epilogue included (A800: 8192x8192x2048 cuBLAS 1017 us, nvmath 1045, Triton fused 1275, TileLang 1470); a hand-written GEMM at `0.3-0.6` warrants checking the tile and mainloop gap to cuBLAS (measure `torch.matmul` alone), `< 0.3` with fp32 operands can indicate a CUDA-core fallback | supported explicit choice; else applicable `nvmath`, else `triton` (`tilelang` experimental; backend and mainloop notes below) | [`nvmath/epilogue/gemm`](../nvmath/epilogue/gemm/SNIPPET.md), [`triton/epilogue/gemm`](../triton/epilogue/gemm/SNIPPET.md), [`tilelang/epilogue/gemm`](../tilelang/epilogue/gemm/SNIPPET.md) |
| `flash_attention_2` | scaled-dot-product attention (MHA/GQA), causal, block-causal or dense, with fused softmax; dense non-causal attention over a packed multimodal sequence (DiT / MiniMax-H3: `H = 56`, `Dh = 128`, `S` in the tens of thousands, softmax in fp32); block-causal attention in video generation (frame chunks of `block_size` tokens attend to all earlier chunks and their own) | compute roof for `head_dim >= 64` and long `S_kv`; the FLOP count is the *attended area* (`S(S+1)/2` pairs causal, `S(S+block)/2` block-causal, `Sq*Skv` dense), not the full plane | FA2 tiling: `Q` block per program, loop over `K,V` blocks, online softmax in the log2 domain (`m`, `l` running statistics, `exp2`), `O` accumulated in fp32, `lse` saved for the backward; the mask lives in two places only, the loop bounds (skip blocks that are fully masked) and a per-element select on the blocks that straddle a boundary, never in a materialised `(S, S)` tensor; backward = preprocess `delta = rowsum(o*do)` + a dK/dV kernel (one program per K,V block, loop over Q blocks) + a dQ kernel (one program per Q block, loop over K,V blocks), `P` recomputed from `lse` in both, no atomics (deterministic, 5 GEMMs vs 2 forward); the score plane `B*H*Sq*Skv` is logical but its index is real: 56 heads x 6200 tokens is past 2^31 (indexing rule below; the reference kernels use int64 for the `(b, h)` base and the program's own tile and int32 inside the K,V loop, with a host guard) | two `T.gemm`/`tl.dot` per `K,V` block (`QK^T`, `PV`), `T.reduce_max`/`tl.max` and running `exp2` rescale | observed FLOP-based `sol_eff >= 0.5` at `S >= 2048` (the reference kernels reach 0.53-0.63 of the datasheet peak, 0.85-0.94 of torch's FA2 kernel; `F.scaled_dot_product_attention` is the baseline to beat, not eager: time it yourself in `_scratch/` and put the number in `IMPL_NOTES.md` / the verifier notes, the harness baseline is the materialised path). Roof: leave `gemm_shapes` unimplemented so the harness times cuBLAS on one same-FLOP cube (stacked skinny `(B*H*S, S, d)` GEMMs run slower in cuBLAS than the fused kernel and report `sol_eff > 1`); FLOPs count the attended area only | `triton` (TileLang is experimental: its dense GQA reference is 7-9% faster, its per-shape JIT adds substantial compilation time) | `reference/{triton,tilelang}/attention/fa2_causal/` (causal MHA) and `fa2_gqa/` (dense GQA, `Sq != Skv`), tested fwd + bwd on A800; block-causal recipe in both SNIPPETs; block-sparse (arbitrary tile/block mask) recipe below |
| `swiglu_dual_gemm` | `silu(X W1^T) * (X W3^T)` followed by the down projection, the MLP block of Llama-style models | compute roof | **not a dual-GEMM kernel**: run `[W1; W3]` as one concatenated `(2H, K)` GEMM (nvmath or `torch.matmul`, cuBLAS speed, `X` read once) and fuse `silu(g) * u` with its adjoint (`dg = dh * u * silu'(g)`, `du = dh * silu(g)`) as one `elementwise_stream` kernel over the `(M, 2H)` product; `SAVE_AUX` keeps the pre-activations or `RECOMPUTE` rebuilds them with the same GEMM; the down GEMM is plain cuBLAS. A hand-written two-accumulator mainloop starts 20-50% behind cuBLAS on sm80 and does not repay the one saved read of `X` | `torch.matmul(x, w13.t())` (or a planned nvmath `Matmul`) + one Triton pointwise kernel that reads `g, u` once and writes `h` | as `gemm_tensorcore`: the GEMM is the roof, the gate kernel is at the copy roof | `triton` for the gate kernel (the GEMM is a library call) | [`triton/epilogue/gemm`](../triton/epilogue/gemm/SNIPPET.md) `_pointwise_epilogue_kernel` for the streaming pass; `tilelang-wiki/examples/fusedmoe/` for the loop structure if a grouped (MoE) variant is ever needed |

## Choosing a row at M0

1. Is there a matmul in the region? No -> one of the four streaming rows (a fusion of several
   streaming primitives takes the row of the one with the reduction, `rowwise_onepass`, else
   the producer's row: a norm + RoPE + permute fusion is `rowwise_onepass` with `rope_pairs`
   and `permute_copy` as its epilogue, whatever the SNIPPET that inspired it was filed under;
   the verifier's "pattern honoured" check reads the reduction, not the epilogue). Yes -> `gemm_tensorcore`, unless the region is an attention block
   (`flash_attention_2`: four tested references, causal and dense-GQA in both backends) or
   the SwiGLU pair (`swiglu_dual_gemm`: one concatenated library GEMM + a gate kernel).
2. Select a supported explicit backend first, otherwise applicable nvmath, otherwise Triton. M1 records
   the choice and the reason. One kernel backend per package.
3. `D <= 8192` stays the contract for a fused `rowwise_onepass` backward (the row tile fits
   registers); forward-only rows go to 32768 in one program and beyond that through split-D
   (row width rule below). Wider training rows are out of scope for this pass and the
   scaffold says so at the gate.

## Row width rule (`rowwise_onepass`, also `elementwise_stream` fusions with a reduction)

The intuition "rows below the SM count (A100/A800 108, H100 132, B200 148) leave the machine
idle, so split the row" is wrong for the sizes this pattern covers, and the measurements are
in the rmsnorm SNIPPET: with 32-100 rows of `D <= 32768`, one program per row is *at or
above* the copy roof (a 5 us kernel against a 4-9 us copy; both are latency-bound), and a
two-stage split-D is slower by a second launch and a second read. What does break the one-row
kernel is **row width**: past `D = 32768` (bf16, 16 warps) the row no longer fits the
register file, the kernel spills and runs 5x slower than copy.

```
next_pow2(D) <= 32768  ->  one program per row, whatever n_rows is
next_pow2(D)  > 32768  ->  split-D: (row, chunk) programs, partial reductions, second pass normalises
```

- The measured reference selects split-D from `dims["d"]`. Use this default first; a
  different decision based on row count or GPU geometry requires measurement on the target.
- `.agents/skills/kda-kernel-scaffold/reference/workloads.md` still makes `user_small_rows` a required workload: it is the
  regression row for grids smaller than the machine (int64 offsets, masks, `dw` partials with
  few programs), not a reason for a second path. Its phases sit under the verdict's 10 us
  latency floor, where `sol_eff` is reported but not gated and only "not slower than
  eager/compile" counts (`speed-of-light.md`, small grids).
- A fused backward keeps `x`, `dy` and the `dw` accumulator of a row tile in registers and
  caps out earlier (`D <= 8192` in the reference); wider rows need a split-D backward too.

## Indexing rule (every pattern): int32 overflows at 2^31 elements, and that is an ordinary size

Triton computes `tl.program_id(0) * stride` in int32 unless the program id is widened first;
the product wraps past `2^31 - 1` elements, which is a **4 GiB bf16 tensor** (8 GiB fp32), not
an exotic one: a `(B, S, D)` activation of a long-context step, the `(M, N)` output plane of a
65k-token GEMM, attention scores at `56 heads x 6200 tokens`. The rule in every launcher is
`row = tl.program_id(0).to(tl.int64)` (and `pid_m.to(tl.int64) * stride_m` for 2-D grids)
*before* the multiply; `tl.arange` offsets within a block may stay int32. The verify skill's
index-safety probe and the `edge_huge` workload exist for this row; size that row as the
**smallest** crossing (`.agents/skills/kda-kernel-scaffold/reference/workloads.md`: `[262145, 8192]` token-wise, an
`M = 131073, N = 16384, K = 256` GEMM plane, a `B = 1, H = 8, S = 16512` score plane) so it
actually runs. A row that every card skips tests nothing; `validate_spec` rejects oversized inputs.

## Backend note: choose by explicit request, capability and measured applicability

Honor a supported explicit backend request. Otherwise choose applicable nvmath for GEMM
epilogues and Triton for the remaining patterns. TileLang stays in
the tree as an experimental backend with tested twins (GEMM epilogue, both attention kernels),
because two costs are not visible in the source: it is hard to
make fast for a GEMM (below), and its JIT compiles one kernel per shape specialisation
(`T.const` strides and lengths) at 4-6 s each (tens of seconds for an attention backward),
multiplied by the package's distinct workloads and phases. No switch fixes either (execution backend, `pass_configs`, tile sweep: all
measured, `reference/tilelang/README.md`); what shortens the compile bill is declaring the
extents that only enter the grid (`M`, `B`, `H`) as `T.dynamic`, which the references now do.

The GEMM measurement (both SNIPPETs, A800)
says Triton: the TileLang GEMM + epilogue reaches 0.8x of the Triton twin on the up- and
down-projection forwards and parity only on the `D x D` and skinny shapes, after the same effort.
**TileLang is very difficult to get to a fast GEMM in this codebase**, for reasons that are not
visible in the source and each cost a measurement to find (details and numbers in
`reference/tilelang/README.md`): a `T.dynamic` stride halves mainloop throughput, a fragment
written straight to global memory costs 6x what a shared-staged store costs, a fragment that is
only cleared or only filled by scalar stores fails layout inference, and the tile configuration
that wins in Triton (`128x128x64`, 8 warps) loses in TileLang. The parts that TileLang makes
short (the K loop is three lines, `T.copy` predicates every tail) are not where the time goes.

The default GEMM choice is nvmath when it imports and supports the requested epilogue,
otherwise Triton. An explicit supported TileLang request remains valid with these measured
limitations disclosed. For `flash_attention_2` the two backends
are at parity on the causal kernel and TileLang is 7-9% ahead on the dense GQA kernel (the
attention mainloop is two `T.gemm`s on shared tiles with a fragment in between, which TileLang
lowers well; the GEMM epilogue's scattered stores are what it lowers badly), but the default is
still `triton`: the 7-9% is smaller than the compile-time cost above, and the Triton
`reference/triton/attention/` snippets are complete. Take `tilelang` only when the user asks
for it, and then budget the JIT time (keep the workload table to the required rows during
M2/M4 diagnostics with `--workload <row> --json <op_dir>/report.diagnostic.json --md <op_dir>/report.diagnostic.md`; complete required verification before stage completion). `swiglu_dual_gemm` is a library GEMM plus a Triton gate kernel, not a
dual-GEMM kernel.

## Mainloop note for `gemm_tensorcore`: the library first, a kernel last

Both Triton and TileLang are known to be slow at a GEMM on sm80 relative to cuBLAS, and
cuBLASLt already fuses the epilogues a transformer needs. The order of preference is therefore
(1) **nvmath-python** (`reference/nvmath/epilogue/gemm/`: bias, ReLU, tanh-GELU, aux store,
`BGRAD` reductions run inside the cuBLAS kernel; measured 1.16-1.27x eager and `torch.compile`
on the large shapes, and on the down-projection faster than `torch.matmul` itself because
cuBLASLt's heuristic picks a better kernel), (2) `torch.matmul` + one pointwise epilogue
kernel, (3) a fused Triton GEMM. Do not create a new fused GEMM kernel when (1) or (2)
expresses the op. The nvmath twin's `select_backend` draws the (1)/(2) line by device time
(~200 us on A800: `execute()` costs 130-175 us of host time per call, hidden under CUDA graphs
or `torch.compile` but not in eager). Two facts to carry into the SPEC: cuBLASLt's GELU is the
**tanh approximation** (the one models use; `F.gelu`'s erf default differs by up to 2e-3, so
the SPEC names the form the user's code calls), and cuBLASLt's addend `C` sits *inside* the
activation, so there is no fused post-activation residual, and none is needed.

When (3) is the choice, the fused-epilogue win is bounded by the epilogue's traffic (`~3 * M * N * es` bytes, one
pointwise pass), and the fused mainloop's loss against cuBLAS is not: on A800 the Triton
mainloop is within 7-14% of `torch.matmul` on the down-projection, `D x D` and skinny shapes
and the fused kernel wins or ties there, but 23-53% behind on wide outputs (the `D -> 4D`
up-projection, an 8192-wide MLP), where **cuBLAS + one fused epilogue kernel** is the fastest
forward (by 18% and 2-4% respectively), level with or ahead of `torch.compile`. The Triton reference therefore has both behind one host
signature (`mainloop="triton" | "cublas"`, `select_mainloop` decides by shape, table in its
SNIPPET.md). An implementer of this pattern times `torch.matmul` alone on the SPEC workloads
first; a fused mainloop further from it than the epilogue traffic costs is replaced by cuBLAS
plus the epilogue kernel, and IMPL_NOTES says which shapes went which way. Both count as the
pattern: the signature to grep is `tl.dot` *or* `torch.matmul` followed by an epilogue kernel
that reads the product once.

**Numerics can decide the mainloop before speed does, and that is an M0/M1 decision.** The
harness compares the kernel's output with the eager reference at a fixed `atol` (bf16 1.6e-2
on a non-reduced output). When the epilogue cancels (a difference or a sum with `y` near 0
while `|z| ~ sqrt(K)`, tens; the old `gelu(z) + R` task had this), a mainloop that rounds `z` differently from cuBLAS is off by a bf16
ulp of `z` (~0.5 absolute), far outside the tolerance, and an accurate `tl.dot` mainloop
*cannot* pass at the user's shape however fast it is: torch's bf16 GEMM uses reduced-precision
accumulation (`allow_bf16_reduced_precision_reduction`) that the fused product does not
reproduce. The kernel that passes is `torch.addmm` with the bias folded in (bit-equal to eager
`F.linear`) plus the epilogue kernel, and its saved aux is the **bias-inclusive**
`round_bf16(Z + b)`, not `round_bf16(X W^T)`. So at scaffold time, when the SPEC math has a
difference or a sum that can cancel on a fixed-atol output, write the SPEC for the cuBLAS
mainloop and the bias-inclusive aux (Math, Contracts, `saved_for_backward`, the workloads'
numerics note), or set SPEC `tolerances` with a justification.

## Block-sparse attention (`flash_attention_2` with a tile / block mask)

Sliding Tile Attention, block-local windows, per-head `(NT, NT)` tile masks, any attention
whose mask is dense inside blocks and structured between them. Four rules, from the STA golden
(`examples/sliding_tile_attention/golden/`, 0.85 ms fwd on 8 heads x 10.5k tokens at 31%
density, 167 TFLOP/s):

1. **Turn the mask into block lists on the host, once.** For the kernel's `(BLOCK_M, BLOCK_N)`
   geometry compute `any[h, i, j]` (some pair allowed) and `full[h, i, j]` (every pair allowed)
   with a few einsums over the tile mask, then per `(head, Q block)` the list of allowed K
   blocks with the full ones first, plus `n_full`, `n_any`. The device loop is then
   `for it in range(0, n_full)` (unmasked) and `for it in range(n_full, n_any)` (masked), reading
   `j = tl.load(idx + it)`. Never loop over all tiles in the kernel testing the mask as you go:
   that is `NT` gather + reduce + branch iterations per program and an element mask on every
   block, including the fully allowed ones. The dK/dV kernel takes the transposed lists. Cache
   the lists by mask pointer and geometry; the `(H, NQB, max_nnz)` int32 table is the biggest
   mask-derived tensor (75 MB at HunyuanVideo size, versus a 320 GB token mask).
2. **Rebuild the element mask from the source structure in the masked loop**: one gather of
   the tile mask at `(m // tile, n // tile)` per partial block, plus the text / tail terms.
   Align blocks with the tile size when you can (`384 = 3 x 128`): then video blocks are never
   partial and the masked loop is empty or one tail block long.
3. **Heavy-first program order.** Heads with different windows do very different amounts of
   work; a natural grid leaves the dense heads to the last wave. Sort the `(head, block)` rows
   by list length descending into a `perm` table and have each program read its row from it
   (grid `(H * NQB, B)`). That alone was 1.22 -> 0.85 ms forward and 3.53 -> 2.70 ms backward.
4. **Fully-denied rows.** A partial block can deny a whole row; before any block contributed
   `m_new = -inf` and `exp2(m - m_new)` is NaN. Substitute 0 for `-inf` in the subtraction on
   the masked path (the causal diagonal never needs this).

Also: indirect K,V addresses cost one more pipelined tile of shared memory; the 128 x 128 x 128
forward tile that fits 3 stages in the causal kernel needs 2 here. FLOPs for the roofline count
the attended pairs (`sum(mask) * tile^2` plus the text rows and columns), not the plane.

## Tensor-core rule (`gemm_tensorcore`, `flash_attention_2`, `swiglu_dual_gemm`)

Operands enter `tl.dot` / `T.gemm` in storage dtype (bf16/fp16); the accumulator is
`compute_dtype` (fp32 by default); one cast happens in the epilogue at the store. An
`.to(tl.float32)` on an operand, or a `T.alloc_shared(..., accum_dtype)` operand buffer, turns a
tensor-core GEMM into a CUDA-core one: 8-16x slower with correct numerics, which is why the lint
makes it a hard finding rather than leaving it to the numbers.
