# Python Compatibility And Dtypes

TileLang kernels are written in Python syntax, but the kernel body is lowered
to TileLang/TIR rather than executed by the CPython interpreter. Use Python
freely for host-side configuration; inside `@tilelang.jit` or `@T.prim_func`
bodies, stay within the syntax that TileLang lowers.

## Dialects

`import tilelang.language as T` is a thin facade over the **CUDA** dialect
(`T.__tilelang_dialect__ == "cuda"`). Portable names live on
`tilelang.language.common`. Backend-specific names are not on `common`.

| Import | Dialect | Notes |
| --- | --- | --- |
| `import tilelang.language as T` | CUDA | Default. Includes `ClusterKernel`, `tma_copy`, `wgmma_gemm`, `tcgen05_gemm`. |
| `import tilelang.language.common as T` | common | Portable surface. No CUDA-only or ROCm-only names. |
| `import tilelang.cuda.language as T` | CUDA | Same as the default facade. |
| `import tilelang.rocm.language as T` | ROCm / HIP | MFMA / WMMA emitters. No `ClusterKernel`. |
| `import tilelang.metal.language as T` | Metal | simdgroup TIR exports. |
| `import tilelang.cpu.language as T` | CPU | Same `__all__` as `common`. |
| `import tilelang.webgpu.language as T` | WebGPU | Same `__all__` as `common`. |

`T.const` is on the eager frontend and is **eager-only**. It is not a
`@T.prim_func` shape helper.

## Two Frontends

- Eager `@tilelang.jit` uses a Python AST frontend around `with T.Kernel(...)`.
- `@T.prim_func` uses the TileLang TVM-script parser.

Both share `T.Tensor`, `T.Kernel`, `T.Parallel`, scoped allocations, `T.copy`,
and typed buffers. `T.empty` and `T.const` are eager-style.

## Tensor Annotations

These forms are equivalent for a contiguous global buffer:

```python
A: T.Tensor((M, N), T.float16)
A: T.Tensor[[M, N], T.float16]
A: T.Tensor((M, N), "float16")
```

`T.Tensor` defaults to global memory and row-major contiguous strides. A
scalar shape is treated as one-dimensional. `T.Buffer` still exists but is
deprecated; new code should use `T.Tensor`.

Non-contiguous global buffers need an explicit stride tuple:

```python
x: T.StridedTensor[(num_tokens, hidden), (token_stride, 1), dtype]
```

`T.ptr` / `T.make_tensor` rehydrate a typed buffer from a pointer. Use them
for pointer tables (`T.Tensor[(L,), T.ptr]` then `T.make_tensor(ptr, shape,
dtype)`). `T.make_tensor` requires a JIT / prim_func builder.

```python
C = T.empty((M, N), T.float16)          # eager output
A_shared = T.alloc_shared((BM, BK), dtype)
C_frag = T.alloc_fragment((BM, BN), T.float32)
flag = T.alloc_var(T.int32, init=0)
```

## Eager-Only `T.const`

```python
M, N = T.const("M, N")          # also T.const("M N")
```

`T.const` declares constexpr dimensions inferred from concrete input tensors.
Calling it outside `@tilelang.jit` eager mode raises `JITNoBuilderError`.
Lazy factories take `M, N, K` as ordinary Python factory arguments instead.

`T.dynamic("m")` keeps a dimension symbolic in the compiled kernel.

## In-Body Compiler Annotations

Place these inside the JIT factory or prim_func body:

```python
T.annotate_pass_configs({tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
T.annotate_compile_flags(["--use_fast_math"])
```

They require a builder. External `pass_configs=` / `compile_flags=` overlay
or extend the function-level values. In eager phase 1 the annotations are
ignored and applied in phase 2.

## Supported Kernel Python

| Pattern | Use it for | Notes |
| --- | --- | --- |
| `for i in range(n)` | Serial loops | Lowered like `T.serial(n)`. |
| `for i in T.serial(...)` | Explicit serial loops | Prefer when clarity matters. |
| `for i, j in T.Parallel(...)` | Parallel tile loops | Common for elementwise work. |
| `for k in T.Pipelined(...)` | Software-pipelined loops | Copy/compute staging. |
| `while condition` | Device loops | Keep loop state in TileLang variables. |
| `if` / `elif` / `else` | Branches | Compile-time booleans fold; symbolic conditions lower to TIR. |
| `x if cond else y` | Value conditionals | Eager/JIT; use `T.if_then_else` when parser behavior is uncertain. |
| `A[i]`, `A[i, j]`, `A[i:j]` | Loads/stores/regions | Regions mainly for `T.copy`. |
| `A[-1]` | Negative indexing | Legalized when extent and sign are provable. |
| `x = expr`, `A[i] += expr` | Bindings and updates |  |
| `a, b = b, a` | Tuple assignment | Swap/unpack is tested. |
| `assert cond, msg` | Assertions | `@T.prim_func` → `T.Assert`. |
| `T.print(...)` | Device print | Ordinary `print` is host-only. |
| `with T.Kernel(...)` | Launch region | Arbitrary Python context managers have no device semantics. |

Treat these as **non-portable** or unsupported in device code:

- Consecutive assignment `a = b = c`.
- `break` / `continue` (eager AST may accept them; `@T.prim_func` does not).
- Iterating lists, `zip`, or `enumerate` as kernel loops.
- Stepped slices `A[i:j:step]`.
- `is`, `in`, `not in`, `type(...)`, `isinstance(...)` as device predicates.
- Ordinary Python functions for device execution; use `@T.macro`.

## Dtypes

Most dtype-taking APIs accept TileLang/TVM dtype objects (`T.float32`),
strings (`"float16"`), Python scalars (`int`, `float`, `bool`), NumPy
dtypes, and Torch dtypes including FP8/FP4 when present.

Aliases: `T.float` → `float32`, `T.half` → `float16`, `T.double` → `float64`,
`T.int` → `int32`, `T.uint` → `uint32`, `T.long` → `int64`, `T.short` →
`int16`.

Dtype objects are callable: `T.float32(1)`, `T.int32(i)`. Use
`T.cast(value, dtype)` for conversion and `T.reinterpret(value, dtype)` for
bit reinterpretation. Keep accumulation dtype explicit for mixed precision.

## Host Versus Kernel

Host Python may use containers and loops freely:

```python
for block_m, block_n in [(64, 64), (128, 64)]:
    kernel = make_kernel(block_m, block_n)
```

Inside the kernel, use TileLang loops and buffers. Build-time values can use
normal Python. Device-time values should use TileLang expressions, buffers,
loops, casts, assertions, and macros.
