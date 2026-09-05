# JIT And Autotune Cheatsheet

Use this page to compile with `tilelang.jit` and scan configs with
`tilelang.autotune`. Detail pages: `tilelang/jit/basic.md`,
`tilelang/jit/advanced.md`, `tilelang/autotune/basic.md`,
`tilelang/autotune/advanced.md`.

## Compile With `tilelang.jit`

1. Decorate a Python function with `@tilelang.jit`.
2. Write the program with `tilelang.language as T`.
3. Call `.compile(...)` with shape and specialization arguments.
4. Launch the returned `JITKernel` with runtime tensors.

```python
@tilelang.jit
def add(A, B, block_M: int, block_N: int, threads: int):
    M, N = T.const("M, N")
    A: T.Tensor((M, N), T.float32)
    B: T.Tensor((M, N), T.float32)
    C = T.empty((M, N), T.float32)
    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
        for i, j in T.Parallel(block_M, block_N):
            C[by * block_M + i, bx * block_N + j] = A[by * block_M + i, bx * block_N + j] + B[by * block_M + i, bx * block_N + j]
    return C

kernel = add.compile(M=1024, N=1024, block_M=32, block_N=32, threads=128)
C = kernel(A, B)
print(kernel.get_kernel_source())
latency_ms = kernel.get_profiler().do_bench()
```

`T.const` dimensions are fixed for the compiled specialization. Tile sizes
and `threads` are Python specialization arguments.

Eager functions can also be called directly (`C = add(A, B, 32, 32, 128)`).
Compile first when you want source, profiling, export, or reuse.

Lazy factories return `@T.prim_func` and use `out_idx` to mark outputs.

Default `target` / `execution_backend` / `verbose` come from
`TILELANG_DEFAULT_TARGET`, `TILELANG_EXECUTION_BACKEND`, and
`TILELANG_VERBOSE`. See `targets.md`.

## Tune With `tilelang.autotune`

Place `@tilelang.autotune(...)` **above** `@tilelang.jit(...)`. Bare
`@tilelang.autotune` is not supported. Every config key must match a factory
parameter. `pass_configs` is reserved and may appear **inside** a config
dict; those values overlay the decorator / compile-args pass configs.

```python
configs = [
    {"block_M": 32, "block_N": 32, "threads": 128},
    {"block_M": 64, "block_N": 64, "threads": 256,
     "pass_configs": {tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True}},
]

@tilelang.autotune(configs=configs, warmup=3, rep=20, early_stop=True, early_stop_factor=2.0)
@tilelang.jit
def tuned_add(A, B, block_M=32, block_N=32, threads=128):
    ...

best_kernel = tuned_add.compile(M=1024, N=1024)
```

`early_stop=True` skips a full benchmark when a cheap estimate exceeds
`best_latency * early_stop_factor` (`early_stop_factor` must be ≥ 1.0).

If the caller supplies every tunable value, the decorator skips the search
and compiles that fixed configuration.

## Decorator Versus `AutoTuner.run`

The decorator is the usual example path. Grouped compile and multi-GPU
benchmarking exist **only** on the programmatic path:

```python
result = (
    AutoTuner.from_kernel(kernel=kernel, configs=configs)
    .set_compile_args(out_idx=[-1], target="auto")
    .set_profile_args(ref_prog=reference, skip_check=False)
    .run(
        warmup=3,
        rep=20,
        timeout=30,
        enable_grouped_compile=True,
        group_compile_size=2,
        benchmark_multi_gpu=True,
        early_stop=True,
        early_stop_factor=2.0,
    )
)
best_kernel = result.kernel
```

`AutoTuner.run(...)` also accepts `use_pipeline`, `benchmark_devices`, and
`timeout`. The decorator does not expose grouped compile or multi-GPU.

Use `set_autotune_inputs(...)` when correctness depends on real metadata
(masks, packed offsets, grouped-GEMM tables, varlen).

## Cache Rules

- JIT kernel cache is namespaced
  `$TILELANG_CACHE_DIR/<version>/<os-arch>/kernels/`.
- Bypass with `TILELANG_DISABLE_CACHE=1` and/or `tilelang.disable_cache()`.
- Autotune persist is separate: `TILELANG_AUTO_TUNING_DISABLE_CACHE=1`.
- Passing `ref_prog`, `supply_prog`, or `manual_check_prog` **disables
  autotune persist**. Callbacks have no reliable identity.
- Lib-stamp opt-in: `TILELANG_KERNEL_CACHE_USE_LIB_STAMP=1`.

When debugging stale results, disable **both** cache layers.

## Config Design

Keep the first search small:

- Tile sizes: `block_M`, `block_N`, `block_K`
- Pipeline depth: `num_stages`
- Launch width: `threads`
- Architecture flags only after the baseline is correct

Do not explode a huge Cartesian product. Prefer `tilelang.PassConfigKey`
constants. For kernels that write explicit output buffers, use lazy style
with `out_idx`.

## Where To Go Next

- `tilelang/jit/basic.md` / `advanced.md` — eager vs lazy, cache keys,
  `par_compile`.
- `tilelang/autotune/basic.md` / `advanced.md` — captured inputs, timeouts,
  grouped compile, multi-GPU.
- `tilelang/env/basic.md` — `TILELANG_DEFAULT_TARGET`, namespaced cache.
- `targets.md` — target vs execution backend.
- `debug.md` — `TL_LOWER_TRACE` when a candidate fails to lower.
