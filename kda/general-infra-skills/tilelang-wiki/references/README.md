# TileLang Reference Guide

This is the main reference entry point. Use it to choose the next document;
use the linked pages for kernel syntax, API details, examples, and debugging.

TileLang is a Python-first DSL for high-performance kernels. Pin
`tilelang==0.1.13`. The files in this skill are the local source of truth.
Use `../examples/README.md` for runnable example selection and `../FAQs.md`
for known failure modes.

`import tilelang.language as T` is the CUDA dialect. Portable names live on
`tilelang.language.common`. HIP/Metal/CPU use `tilelang.<backend>.language`.

## Start Here

- `language_basics.md`: kernel-writing cheatsheet. Eager vs lazy shapes,
  `T.Kernel` / `T.ClusterKernel`, memory scopes, `T.copy`, `T.gemm`, and a
  FlashAttention 2 walkthrough.
- `jit_autotune.md`: compile/tuning cheatsheet. `tilelang.jit`, `.compile`,
  `tilelang.autotune`, `early_stop`, per-config `pass_configs`, and cache
  rules.
- `python_compat.md`: dialects, supported Python, `T.Tensor` / `T.StridedTensor`,
  eager-only `T.const`, and in-body annotations.
- `targets.md`: `target` vs `execution_backend`, auto-detect order, CuteDSL,
  LLVM, Metal M5, WebGPU emit-only.
- `debug.md`: `TL_LOWER_TRACE` first, Pass Visualizer, legacy
  `TILELANG_PASS_DIFF`, and a short checklist.
- `sm100.md`: single-CTA TMEM recipe, then WS / 2CTA / CLC / block-scaled
  pointers.
- `tilelang/README.md`: top-level `import tilelang` API index.
- `tilelang_language/README.md`: `import tilelang.language as T` API index.

## Reading Order

1. `language_basics.md` for the kernel skeleton.
2. `jit_autotune.md` to compile, call, profile, or tune.
3. `python_compat.md` for Python syntax, shapes, tensors, dtypes, or casts.
4. `targets.md` when the question is CUDA vs HIP/Metal/CPU or backend choice.
5. `tilelang_language/<topic>/basic.md` for exact `T.*` usage.
6. `tilelang/<topic>/basic.md` for exact top-level `tilelang.*` usage.
7. Open `advanced.md` only when a basic page points there or the kernel uses
   TMA, WGMMA, TCGEN05, cluster launch, manual sync, AutoDD, grouped autotune
   compile, or low-level debug tooling.

## API Detail Map

Top-level `tilelang.*` APIs:

- `tilelang/jit/basic.md`: `tilelang.jit`, eager calls, lazy factory style,
  `.compile(...)`, `out_idx`, source inspection, and profiling helpers.
- `tilelang/jit/advanced.md`: `tilelang.compile(...)`,
  `tilelang.par_compile(...)`, mode inference, cache keys, pass configs,
  target/backend defaults, and debug output.
- `tilelang/autotune/basic.md`: `tilelang.autotune`,
  `AutoTuner.from_kernel(...)`, compile/profile arguments, captured inputs,
  and common pitfalls.
- `tilelang/autotune/advanced.md`: config binding, result objects, validation,
  tuning caches, worker controls, timeouts, grouped compile, and multi-GPU
  benchmarking.
- `tilelang/env/basic.md`: cache helpers, `TILELANG_DEFAULT_TARGET`, namespaced
  cache paths, and import-time behavior.
- `tilelang/env/advanced.md`: cache directories, env-var precedence, and
  runtime setup debugging.
- `tilelang/autodd_tools/basic.md`: AutoDD and `tilelang.tools`.
- `tilelang/autodd_tools/advanced.md`: layout plotting, generated-source
  callbacks, and lower-level instrumentation.

`tilelang.language as T` APIs:

- `tilelang_language/loop/`: serial, unrolled, parallel, pipelined,
  vectorized, and persistent loop forms.
- `tilelang_language/allocate/`: tensor annotations, eager outputs, shared,
  local, fragment, scalar, barrier, TMEM (rank ≥ 2), descriptor, and reducer
  allocations.
- `tilelang_language/copy_op/`: `T.copy`, `prefer_instruction`, `T.async_copy`,
  TMA, cluster copy, gather/scatter, and transpose.
- `tilelang_language/gemm_op/`: dense GEMM, sparse GEMM, WGMMA, TCGEN05,
  block-scaled GEMM, SM120 `T.mma_gemm_blockscaled`.
- `tilelang_language/basic_operations/`: `T.fill`, `T.clear`, annotations,
  elementwise assignments, pointer views, and math.
- `tilelang_language/kernel_warpgroup_cluster_builtins/`: `T.Kernel`, CUDA-only
  `T.ClusterKernel`, barriers, warpgroup helpers, and low-level MMA builtins.
- `tilelang_language/annotations/`: swizzle (`row|column|mlx`), layout maps,
  restrict-buffer, launch bounds, compile flags, and pass configs.
- `tilelang_language/reduce_op/`: tile reductions, `T.cumsum`, `T.cummax`,
  generic and warp reductions.
- `tilelang_language/misc/`: atomics, debug helpers, dynamic symbols, PDL,
  and raw TIR exports.

Each topic uses `basic.md` for common usage and `advanced.md` for
target-specific or rarely used behavior.

## Other Reference Areas

- `enums.md`: `T.GemmWarpPolicy` and other enums that change lowering.
- `pass_config.md`: `tilelang.PassConfigKey`, including `TL_PASS_PROFILE*`
  and `TL_IF_STMT_BINDING_INLINE_REPLAYABLE_BINDS`.
- `debug.md`: lower-trace, Pass Visualizer, AutoDD, and the debug checklist.
- `sm100.md`: Blackwell TMEM / TCGEN05 starting recipe.
- `tile_kernels/README.md`: DeepSeek TileKernels house style and recipes.

For install and extra internals, use [tilelang.com](https://tilelang.com)
or the distilled pages in this skill.

## Practical Routing

- First kernel: `language_basics.md`, then `../examples/README.md`.
- API lookup: `tilelang/README.md` or `tilelang_language/README.md`, then the
  matching `basic.md`.
- Tuning: `jit_autotune.md`, then `tilelang/autotune/basic.md` only if needed.
- Debugging: `../FAQs.md` and `debug.md`.
- Targets/backends: `targets.md`.
- SM100: `sm100.md`, then `../examples/gemm_sm100/`.
- Production recipes: `tile_kernels/README.md`.
