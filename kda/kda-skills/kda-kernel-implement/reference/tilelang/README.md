# TileLang references (experimental backend)

Tested TileLang 0.1.13 / 0.1.14 snippets for `kernel_backend: tilelang`, the twins of the Triton tree
with the same math and host signatures. TileLang is **experimental** in KDA: the implementer takes this tree when SPEC names
`tilelang`. Backend selection respects a supported explicit user choice, then applicable nvmath, then Triton. Two reasons, both measured: the GEMM below ends at 0.8x of
its Triton twin, and the JIT compiles one kernel per shape specialisation at 4-6 s each (up to
tens of seconds for the attention backward), multiplied by every distinct workload and phase.
Budget for that with scoped diagnostics (`--workload <row> --json <op_dir>/report.diag.<run>.json --md <op_dir>/report.diag.<run>.md`), followed by the complete required correctness coverage, and read
[the flag study](#what-the-flags-do-not-change-and-what-does) before looking for a switch that
fixes it: there is none. General TileLang semantics live in
`general-infra-skills/tilelang-wiki/` (read `references/language_basics.md`,
`references/jit_autotune.md`, `references/python_compat.md` and the `examples/gemm/` family
first); this page is what KDA's kernel packages additionally require, and what the measurements
taught about the cost of TileLang constructs on sm80.

| snippet | pattern | status |
|---|---|---|
| [`epilogue/gemm`](epilogue/gemm/SNIPPET.md) | `gemm_tensorcore` | correct, tested; **0.8x of the Triton twin** on the large forwards, parity elsewhere (see the warning) |
| [`attention/fa2_causal`](attention/fa2_causal/SNIPPET.md) | `flash_attention_2` | correct, tested fwd + bwd, deterministic; **parity with the Triton twin** (0.90x / 0.71x of torch's FA2 kernel fwd / bwd at 8192 tokens); the page lists the six lowering pitfalls met (FullRow policy, pipelined-loop races, 1-D copy tails, staging-tile reuse, power-of-two fragment reductions) |
| [`attention/fa2_gqa`](attention/fa2_gqa/SNIPPET.md) | `flash_attention_2` | dense GQA, `Sq != Skv`; **the fastest attention reference in either backend** (0.94x / 0.76x of torch's FA2, 197 TFLOP/s forward) |

## Attention is where TileLang is at its best

The FA2 mainloop is two `T.gemm`s on shared tiles with a fragment in between, which TileLang
lowers as well as Triton does (tie on the causal kernel, 7-9% ahead on dense GQA). What it
cost to get there is on the [`attention/fa2_causal`](attention/fa2_causal/SNIPPET.md) page;
the short list: `policy=T.GemmWarpPolicy.FullRow` on every GEMM whose accumulator feeds
another GEMM, `num_stages=1` (two stages raced on tail tiles and were slower), no `if`
around shared-memory reads inside a `T.Pipelined` loop, 1-D `T.copy` leaves the tail
uninitialised, one staging tile per store, and fragment reductions only for power-of-two `D`.

## What the flags do not change, and what does

Measurements on A800 with TileLang 0.1.13 and 0.1.14 distinguish launcher overhead
from device execution:

- **Execution backend** (`tvm_ffi`, the `auto` choice; `nvrtc`; `cython`) is the launcher, not
  the compiler. Device time is identical under all three (GEMM 0.330 / 0.332 ms, FA2 forward
  2.01 / 2.15 / 2.02 with clock drift). `nvrtc` compiles one kernel in 4.4 s instead of 5.7
  (no nvcc, no host `cc`) and launches in 21 us instead of 10 (Triton: 25); `cython` is worst
  on both (11 s, 27 us). `nvrtc` also breaks on a tensor argument named `res` (its launcher's
  local for the `CUresult`; also `config`, `kernels`, `stream`), so never name a tensor
  argument `res`; `cython` rejects `T.StridedTensor` with `T.dynamic` extents. Details
  and the table: `tilelang-wiki/references/targets.md`.
- **`pass_configs`** (`TL_ENABLE_FAST_MATH`, `TL_PTXAS_REGISTER_USAGE_LEVEL=10`,
  `TL_DISABLE_SAFE_MEMORY_ACCESS`, all three together): 0.332 -> 0.337 / 0.341 / 0.332 / 0.342 ms
  on the fused GEMM. Noise. The generated CUDA already has what sm80 needs: `cp.async` with
  `commit`/`wait_group` for the `T.Pipelined` stages, `ldmatrix.x4`, `mma.sync.m16n8k16`,
  rasterised block ids (`T.use_swizzle`).
- **Tile sweep, bare bf16 GEMM 4096x4096x1024** (48 configs of `bM/bN in {128, 256}`, `bK in
  {32, 64}`, 2-4 stages, 128/256 threads, then the top 12 re-timed round-robin against cuBLAS,
  min of 3): best `128x128x64`, 2 stages, 128 threads at **1.12x cuBLAS**; the reference's
  `128x128x32` / 3 stages at 1.20-1.23x; every 256-wide tile 1.28x or worse; 128 threads on a
  256-wide tile 3-17x (register spills). Swizzle panel 4 / 8 / 16 / off: within 2%. So the
  TileLang mainloop on sm80 is 0.8-0.9x of cuBLAS (cuBLAS runs this shape at 230 TFLOP/s, 74% of
  peak) at the best tile, and the measured residual-add variant (bias + GELU + Z + residual) adds ~150 us on
  top of a ~180 us mainloop, where the Triton twin adds ~60. The gap is codegen (the epilogue
  lowering and the mainloop's register allocation), not a flag.
- **`T.dynamic` vs `T.const` is the one lever that matters, and it is about compile time.**
  Each distinct `T.const` value is a new 4-6 s compile (TVM passes ~2.2 s, nvcc ~2.7 s, host cc
  0.8 s). Extents that only enter the grid and base offsets are free as `T.dynamic`: GEMM `M`
  (0.323 ms dynamic vs 0.331 static), attention `B, H` (2.09 vs 2.13 ms). Extents that set
  loop trip counts or tile predicates are not: dynamic `S` on the FA2 forward cost +13% device
  time and +50% compile time. The references now declare `M` / `B, H` dynamic and `N, K` /
  `S, D` const, which cuts a multi-shape verify from one compile per workload to one per
  distinct `(N, K)` or `(S, D)`.
- **Clock drift dwarfs all of the above.** The same config measured 164 and 229 us in two
  passes of the sweep (the A800 idles at 1155 MHz and boosts to 1410); only round-robin
  timing against a fixed reference (interleaved rounds) gives usable ratios. A single-pass sweep
  "found" dynamic `M` 40% faster than static; interleaved they tie.

## Warning: TileLang is very difficult for fast GEMM

The GEMM + epilogue reference took the same effort in both backends. Triton matches or beats
`torch.compile` on three of four shapes; TileLang ends at 0.8x of Triton on the two shapes that
matter (`D -> 4D` and `4D -> D` at 8192 tokens) and at parity on the rest. The mainloop itself
(`T.copy` + `T.gemm`, 180 TFLOP/s bare, 0.8-0.9x cuBLAS at the best tile of a 48-config sweep)
is about as fast as Triton's; most of the loss is in the epilogue and in constructs whose cost
the source does not show. Each item below cost a measurement to find:

- **A `T.dynamic` stride halves throughput.** `x: T.StridedTensor((M, K), (T.dynamic("s"), 1))`
  ran the 8192x4096x1024 mainloop at 714 us; the same declaration with the stride as a Python
  `int` argument (a compile-time constant) ran at 399 us. With a dynamic stride the copies
  cannot prove 16-byte alignment and stop vectorising. A stride specialisation adds few
  variants in practice (`N, K` are `T.const` already; `M` is dynamic and a row stride is
  usually `K` or a fixed pad of it).
- **A fragment stored straight to global costs 6x.** `T.copy(acc, z[m0, n0])` from the MMA
  accumulator writes 4-byte pairs across scattered rows: 100 us for the `Z` store of the
  up-projection. Staging through a shared tile (`T.copy(acc, out_s); T.copy(out_s, z[...])`)
  makes the global store 16-byte and coalesced: 16 us. The persistent-GEMM example does this
  (`C_shared`); the quickstart does not.
- **A scalar epilogue loop is slower still.** `for i, j in T.Parallel(...): if m < M and n < N:
  y[m, n] = ...` with the bias/residual read per element: +180 us over the plain kernel versus
  +75 us for the tile-wise form (bias via a shared row, residual via a fragment copy).
- **Layout inference has preconditions.** A fragment that is only `T.clear`ed and never read
  (the `db` accumulator when `has_bias` is false), or one filled only by scalar stores with a
  bounds guard, fails with `The layout for fragment X can not be inferred correctly`. Fill
  fragments with `T.copy`, allocate them under the Python flag that uses them, and let the
  predicated copy zero the tail instead of guarding the accumulation.
- **Triton's tile table does not transfer.** `128x128x64` with 8 warps wins in Triton and is
  ~6% slower in TileLang than `128x128x32` with 4 warps; `64x128x64` is best for the 64-row tile.
  Sweep with interleaved rounds before trusting a config.
- **Split-K needs a padded slab.** Stacking the per-split fp32 partials as `(split_k * M, N)`
  corrupts the M tail when `M % block_M != 0` (the tail tile of split `s` writes into slab
  `s + 1`); pad to `(split_k * ceildiv(M, block_M) * block_M, N)` or use a 3-D buffer.
  `T.atomic_add` into one buffer is the alternative the wiki example uses; it cannot host a
  non-linear epilogue, which is why the reference sums in a second kernel.
- **Annotations cannot sit under `if`.** `bias: T.Tensor(...)` inside a Python branch raises
  `Unexpected type for TIR MatchBuffer`. Declare optional tensors with `T.dynamic` extents so
  a 1-element dummy passes the packed-ABI shape check, and skip the reads under the flag.
- **Helpers with loops are `@T.macro`.** A plain Python function that contains `T.Parallel`
  raises `'ForFrame' object is not iterable`; only the JIT function's own AST is lowered.
  Expression-only helpers (the activation formulas) may stay plain functions.

The `compute-patterns.md` default for `gemm_tensorcore` is therefore `triton`. Use TileLang for a
GEMM only when the user asks or the op needs a TileLang-only primitive, budget for the list
above, and keep the Triton twin's numbers as the bar.

## Conventions for a KDA kernel package (`_tilelang/`)

- **Eager JIT.** `@tilelang.jit` on a function that takes the tensors and the launch parameters
  directly. Extents that set loop trip counts or tile predicates are `T.const("N, K")` /
  `T.const("S, D")` (inferred from the tensors, one compiled variant per value, 4-6 s each);
  extents that only enter the grid and base offsets are `T.dynamic` (`M` rows of a GEMM, `R`
  rows of a token-wise kernel, `B, H` of an attention): measured free, and they are what
  varies between the SPEC's workloads. Never name a tensor argument `res`, `config`, `kernels`
  or `stream` (the `nvrtc` launcher's own locals). Python `bool` / `int` arguments (`save_aux`, `act`, `split_k`, tile sizes)
  specialise the kernel: TileLang caches one variant per distinct value, so `if save_aux:` is
  folded at compile time like a Triton `constexpr`.
- **Strides are Python `int` arguments**, declared with `T.StridedTensor((R, D), (stride, 1),
  dtype)`. Never `.contiguous()` an input on the host (lint `copy`); the packed-ABI check
  rejects a tensor whose strides differ from the declaration, so pass `x.stride(0)` through.
  `interface.py` already rejected a non-unit last stride.
- **Optional tensors** stay positional: the host passes a 1-element dummy when the flag is off,
  the declaration uses `T.dynamic` extents, the kernel never reads it under the flag.
- **Outputs** are `T.empty(...)` inside the kernel and returned (`out_idx` is inferred); an aux
  tensor that is compiled out is `T.empty((0, N), ...)` so the launcher's schema stays fixed.
  A second kernel that writes into the first one's outputs takes them as plain inputs.
- **Dtypes.** Tensor-core operands (`T.gemm`) stay in storage dtype in shared memory; the
  accumulator fragment is `accum_dtype = float32` (SPEC `compute_dtype`); the one cast to storage
  dtype happens in the `T.copy` to the (shared-staged) output. `T.cast(..., accum_dtype)` on a
  `T.gemm` operand buffer is the lint `mma_operand` hard finding.
- **Tails.** `T.copy` predicates out-of-range rows/cols/K with zeros on load and drops them on
  store; `M, N, K` need not divide the tile. Scalar loops need explicit guards.
- **Logging.** TileLang prints an INFO line per compile; set
  `logging.getLogger("tilelang").setLevel(logging.WARNING)` in `_impl_*.py`.
- **M5.** `@tilelang.autotune` is allowed while tuning and is a hard lint finding once STATUS
  ticks M5: freeze the winner into `_configs.py::select_config` (dims -> config dict).
- **torch.compile.** The launcher is wrapped by `_common.compat.register_kernel` like the
  Triton one; the compiled JIT kernel is an opaque call inside the custom op, and the
  `_fake` function describes the outputs (including the `(0, N)` placeholder).

## Measuring

Same rules as the Triton tree: use `torch.profiler` device time and interleaved rounds
for comparisons. The generated package's `_run_dev.py --bench --method profiler` provides
the workflow measurement path. Numbers in SNIPPET.md are
A800 bf16 and only ratios within one table are stable on an unlocked GPU. Compile time is
4-6 s per variant (`execution_backend="nvrtc"` 4.4 s, same device time; the study above);
`TILELANG_CACHE_DIR` keeps them across processes, so the harness's verify and bench passes pay
each variant once; `TILELANG_DISABLE_CACHE=1` when a kernel edit seems ignored (FAQs.md).
Set `TILELANG_*` in the GPU interpreter's environment before importing TileLang.
