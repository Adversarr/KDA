# Pass Config

`pass_configs` carries TileLang compiler and lowering switches. The public
enum is `tilelang.PassConfigKey`. Prefer the enum over raw strings.

```python
@tilelang.jit(
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
)
def kernel(...):
    ...
```

```python
tilelang.compile(
    func,
    out_idx=[2],
    target="cuda",
    pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
)
```

In-body equivalents are `T.annotate_pass_configs({...})` and
`T.annotate_compile_flags([...])`. External `pass_configs=` overlay
function-level configs.

## Frequent Keys

These appear repeatedly in examples:

- `TL_ENABLE_FAST_MATH` — default `False`. Passes `--use_fast_math` to nvcc.
- `TL_DISABLE_WARP_SPECIALIZED` — default `False`. Required on many SM100
  kernels and almost all TileKernels factories.
- `TL_DISABLE_TMA_LOWER` — **deprecated**. Prevents plain `T.copy` from
  auto-lowering to TMA store. Prefer `T.copy(..., disable_tma=True)`.

## TileLang Simplification

`TL_SIMPLIFY` is a dict config, not a bool. Its options are **unprefixed
strings inside that dict**, not top-level `pass_configs` keys and not
`PassConfigKey` members:

```python
pass_configs={
    tilelang.PassConfigKey.TL_SIMPLIFY: {
        "transitively_prove_inequalities": False,
        "convert_boolean_to_and_of_ors": False,
        "apply_constraints_to_boolean_branches": False,
        "propagate_knowns_to_prove_conditional": False,
        "propagate_knowns_to_simplify_expressions": False,
        "enable_let_inline": True,
    },
}
```

Do not pass `"transitively_prove_inequalities"` (or a
`TL_SIMPLIFY_TRANSITIVELY_PROVE_INEQUALITIES` enum) as its own
`pass_configs` key.

## Safety And Semantic Checks

- `TL_DISABLE_DATA_RACE_CHECK` — default `False`
- `TL_DISABLE_PRELOWER_SEMANTIC_CHECK` — default `False`
- `TL_DISABLE_SAFE_MEMORY_ACCESS` — default `False`
- `TL_DISABLE_OUT_OF_BOUND_WARNING` — default `True`

## Lowering And Performance

- `TL_DEVICE_COMPILE_FLAGS` — extra nvcc/NVRTC flags (string or list)
- `TL_CONFIG_INDEX_BITWIDTH` — default `32`
- `TL_DISABLE_VECTORIZE_256` — default `False`
- `TL_ENABLE_ASYNC_COPY` — default `True`; auto cp.async only inside
  `T.Pipelined(..., num_stages>0)` (or `T.Parallel(..., prefer_async=True)`)
- `TL_ENABLE_LOWER_LDGSTG` / `TL_ENABLE_LOWER_LDGSTG_PREDICATED` — default
  `False`
- `TL_ENABLE_VECTORIZE_PLANNER_VERBOSE` — default `False`
- `TL_DISABLE_WGMMA` — default `False`
- `TL_DISABLE_SHUFFLE_ELECT` — default `False`
- `TL_DISABLE_LOOP_UNSWITCHING` — default `False`
- `TL_LOOP_UNSWITCHING_ALLOW_NON_TRIVIAL_ELSE` — default `False`
- `TL_IF_STMT_BINDING_INLINE_REPLAYABLE_BINDS` — default `True`. When True,
  IfStmtBinding may rewrite `if cond: idx = ids[i]; copy(idx); gemm()` into
  separately guarded statements with `idx` substituted, exposing copy/compute
  to pipeline planning.
- `TL_DISABLE_THREAD_STORAGE_SYNC` — default `False`
- `TL_PTXAS_REGISTER_USAGE_LEVEL` — `[0, 10]` or `None`

There is **no** `TL_ENABLE_PTXAS_VERBOSE_OUTPUT`. Use
`TL_DEVICE_COMPILE_FLAGS` with `--ptxas-options=--verbose` if you need that
output.

## Memory Planning

- `TL_DEBUG_MERGE_SHARED_MEMORY_ALLOCATIONS` — default `False`
- `TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE` — default `False`
- `TL_DISABLE_SHARED_MEMORY_REUSE` — default `False`
- `TL_STORAGE_REWRITE_DETECT_INPLACE` — default `False`

## Debugging And IR Inspection

- `TL_FORCE_LET_INLINE` — default `False`
- `TL_AST_PRINT_ENABLE` — default `False`
- `TL_LAYOUT_VISUALIZATION_ENABLE` — default `False`
- `TL_LAYOUT_VISUALIZATION_FORMATS` — `"pdf"`, `"png"`, `"svg"`, or `"all"`
- `TL_ENABLE_DUMP_IR` — default `False`
- `TL_DUMP_IR_DIR` — default `./dump_ir/`
- `TL_PASS_PROFILE` — default `False`. Per-pass timing.
- `TL_PASS_PROFILE_THRESHOLD_MS` — default `0` (show all). Also
  `TILELANG_PASS_PROFILE` / `TILELANG_PASS_PROFILE_THRESHOLD_MS`.

For “which pass broke this?”, prefer `TL_LOWER_TRACE` over dumping every IR.
See `debug.md`.

## TIR Pass Controls

- `TIR_ENABLE_EQUIV_TERMS_IN_CSE` — default `True`
- `TIR_DISABLE_CSE` — default `False`
- `TIR_SIMPLIFY` — default `True`
- `TIR_DISABLE_STORAGE_REWRITE` — default `False`
- `TIR_DISABLE_VECTORIZE` — default `False`
- `TIR_USE_ASYNC_COPY` — default `True`
- `TIR_ENABLE_DEBUG` — default `False`
- `TIR_MERGE_STATIC_SMEM` — default `True`
- `TIR_ADD_LOWER_PASS` — default `None`
- `TIR_NOALIAS` — default `True`

## Output

- `CUDA_KERNELS_OUTPUT_DIR` — default empty string

`TL_DISABLE_FAST_MATH` is not a `PassConfigKey`. Use `TL_ENABLE_FAST_MATH`.
