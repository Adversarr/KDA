---
name: tilelang-wiki
description: A local, code-grounded TileLang v0.1.13 reference for DSL semantics, kernel authoring, API lookup, tuning, debugging, examples, and compiler/runtime behavior. Use whenever you write, modify, explain, optimize, validate, or troubleshoot TileLang kernels.
---

# TileLang Wiki (v0.1.13)

Pin **`tilelang==0.1.13`**. The source of truth is this skill's
`references/` and `examples/` plus TileLang v0.1.13 language/runtime
behavior. For install and extra internals, use [tilelang.com](https://tilelang.com).
DeepSeek TileKernels production recipes live at
`references/tile_kernels/README.md`; copied kernels are under
`references/tile_kernels/src/`.

Use this skill for TileLang concepts, kernel authoring, optimization,
debugging, operator selection, or compiler/runtime internals. Do not use it
for unrelated CUDA, Triton, or generic TVM work unless the user is clearly
working through TileLang.

## Source of truth

- Default `import tilelang.language as T` is the **CUDA dialect**, not a
  portable surface. Portable names live on `tilelang.language.common`.
  HIP/Metal/CPU use `tilelang.<backend>.language`.
- Two JIT styles: **eager** (`T.const` / `T.empty` / compile+run) vs **lazy**
  (factory + `@T.prim_func` + `out_idx`). `T.const` is eager-only.
- `T.Kernel` no longer accepts `cluster_dims` or `is_cpu`. Cluster launch is
  `T.ClusterKernel(..., cluster_dims=)` (CUDA-only). Passing `cluster_dims` to
  `T.Kernel` is a `TypeError`.
- `T.gemm` is **synchronous** (implicit WGMMA/TCGEN05 wait). Explicit-async is
  `T.wgmma_gemm` / `T.tcgen05_gemm`. SM120 NVF4 is `T.mma_gemm_blockscaled`.
- Target env var is `TILELANG_DEFAULT_TARGET` (JSON dicts allowed).
  `TILELANG_TARGET`, `TILELANG_CLEAR_CACHE`, `tilelang.clear_cache()`,
  `execution_backend="dlpack"`, and CUDA `"torch"` are gone.
- Example-only helpers left the package (`#2761`): sparse compress is
  `examples.gemm_sp.sparse_utils`; dequant helpers are
  `examples/dequantize_gemm/quantize/`.

## When To Invoke

Invoke when the user asks how to install or choose a target/backend; write or
understand a kernel; look up `@tilelang.jit`, `T.Kernel`, `T.ClusterKernel`,
`T.copy`, `T.gemm`, `T.Pipelined`, or memory scopes; check Python-in-kernel
compatibility; tune, profile, or debug; pick a local example; or reason about
compiler/runtime behavior.

## Primary Sources

One-level pointers only:

- [references/README.md](references/README.md)
- [references/language_basics.md](references/language_basics.md)
- [references/jit_autotune.md](references/jit_autotune.md)
- [references/python_compat.md](references/python_compat.md)
- [references/targets.md](references/targets.md)
- [references/debug.md](references/debug.md)
- [references/sm100.md](references/sm100.md)
- [references/enums.md](references/enums.md)
- [references/pass_config.md](references/pass_config.md)
- [examples/README.md](examples/README.md)
- [FAQs.md](FAQs.md)
- [references/tile_kernels/README.md](references/tile_kernels/README.md)

Open `references/README.md`, `references/language_basics.md`, and
`references/jit_autotune.md` early. Use `references/python_compat.md` for
dialects, shapes, tensors, and casts. Use `references/targets.md` for target
vs execution backend. Use `references/debug.md` first for “which pass broke
this?” (`TL_LOWER_TRACE`). Use `references/sm100.md` for Blackwell TMEM /
TCGEN05. Open `FAQs.md` early for compiler errors, profiler surprises,
autotune failures, or “I edited the kernel but it did not recompile”
(cache keys).

**Always check the examples.** The vendored `examples/` tree is often more
current than prose. Prefer `examples/README.md` to pick a family, then open
the matching script.

## Mental Model

1. Choose a dialect: default `T` is CUDA; portable names are on
   `tilelang.language.common`.
2. Choose a JIT style: eager (`T.const` / `T.empty`) or lazy (`@T.prim_func`
   + `out_idx`).
3. Declare shapes with eager-only `T.const(...)` or symbolic `T.dynamic(...)`.
4. Annotate buffers with `T.Tensor(...)` / `T.Tensor[[...]]`; allocate eager
   outputs with `T.empty(...)`.
5. Launch with `T.Kernel(...)`. Clustered CTAs use CUDA-only
   `T.ClusterKernel(..., cluster_dims=)`.
6. Allocate with `T.alloc_shared`, `T.alloc_fragment`, `T.alloc_local`, or
   `T.alloc_var`. SM100 TMEM is `T.alloc_tmem` (rank ≥ 2).
7. Move tiles with `T.copy(...)`. Compute with synchronous `T.gemm(...)`,
   reductions, or elementwise loops.
8. Optimize with `T.Pipelined(...)`, `T.use_swizzle(order="row"|"column"|"mlx")`,
   autotune, or target-specific primitives.
9. Validate against a reference. For “which pass broke this?”, use
   `TL_LOWER_TRACE` before assuming the source edit was ignored.

Distinctions to keep sharp:

- `T.const(...)` is eager-only and inferred from concrete input tensors.
- `T.dynamic(...)` stays symbolic in the compiled kernel.
- `T.Parallel(...)` is the usual elementwise loop; `T.Pipelined(...)` is the
  usual staged copy/compute loop.
- `T.if_then_else(...)` is for value-producing conditionals.
- `T.copy(...)` is the default movement primitive; `T.async_copy(...)` and
  `T.tma_copy(...)` are explicitly managed overlap.

New users: start from `references/README.md` and `examples/quickstart.py`.

## Working Style

Route by user intent, not by file tree.

1. Identify the goal: setup, first kernel, DSL semantics, operator/example
   selection, optimization, debugging, or internals.
2. Open `references/README.md` for the overview and API map.
3. Open `references/language_basics.md` for the kernel skeleton, then the
   matching `references/tilelang_language/<topic>/basic.md`.
4. Open `examples/README.md` when the user needs a runnable match.
5. Open `FAQs.md` for known failure modes.
6. Open deeper pages only for exact semantics or target-specific paths.
7. Prefer the simplest correct example before Hopper/Blackwell variants.

## Answering Rules

- Prefer practical guidance over file dumps.
- Use `references/README.md` for concepts and `examples/README.md` for
  locating an example family.
- If docs and examples overlap, use docs for semantics and examples for
  implementation patterns.
- Surface target assumptions early: CUDA vs HIP/Metal/CPU, and
  Hopper/Blackwell-specific behavior.
- Separate semantic correctness from performance tuning.
- For debugging: reproduce, inspect generated artifacts, compare against a
  reference, then minimize. Lead with `TL_LOWER_TRACE`.
- For autotune or profiler surprises, consider cache reuse before assuming
  the source edit was ignored.
- When recommending an example, say why it matches the operator, dtype, and
  hardware.

Install TileLang with `pip install tilelang==0.1.13`. Extra install and
internals notes are on [tilelang.com](https://tilelang.com), not in this
skill.

## Caveats

- Defaults are the cheatsheets plus `references/tilelang/README.md` and
  `references/tilelang_language/README.md`.
- The local example tree evolves faster than prose. Prefer
  `examples/README.md`, then the specific example directory.
- `FAQs.md` is issue-driven and may mention pitfalls not yet in the guides.
- Autotune and JIT cache behavior can explain a rerun that does not visibly
  recompile. Check cache rules before assuming the kernel body change
  invalidated the key.
- Production recipes for DeepSeek TileKernels live under
  `references/tile_kernels/`; copied kernel sources are in
  `references/tile_kernels/src/`.
