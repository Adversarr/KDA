# Language Basics

This page is the short kernel-writing cheatsheet. Exact APIs live under
`tilelang_language/`. Conventional imports:

```python
import tilelang
import tilelang.language as T
```

`import tilelang.language as T` is the CUDA dialect. Portable names live on
`tilelang.language.common`.

## Two Kernel Shapes

**Eager** — `@tilelang.jit` body, `T.const` / `T.empty`, compile+run:

```python
@tilelang.jit
def add(A, B, block_M: int, block_N: int, dtype=T.float32):
    M, N = T.const("M, N")
    A: T.Tensor((M, N), dtype)
    B: T.Tensor((M, N), dtype)
    C = T.empty((M, N), dtype)

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
        for i, j in T.Parallel(block_M, block_N):
            C[by * block_M + i, bx * block_N + j] = A[by * block_M + i, bx * block_N + j] + B[by * block_M + i, bx * block_N + j]
    return C
```

`T.const` is eager-only. Direct calls compile and run: `C = add(A, B, 32, 32)`.

**Lazy** — factory + nested `@T.prim_func` + `out_idx`:

```python
@tilelang.jit(out_idx=[-1])
def matmul_factory(M, N, K, block_M=128, block_N=128, block_K=32, threads=256):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), T.float16),
        B: T.Tensor((K, N), T.float16),
        C: T.Tensor((M, N), T.float16),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            ...
    return main
```

Use `T.dynamic("m")` only when a dimension must stay symbolic. Dtypes are
usually `T.float16`, `T.float32`, or strings such as `"float16"`.

## Launch

`T.Kernel(*blocks, threads=None, prelude=None)` is the portable launch.
`threads=128` means `(128, 1, 1)`. There is no `cluster_dims` and no `is_cpu`.

Clustered CTAs are CUDA-only:

```python
with T.ClusterKernel(grid_x, grid_y, threads=128, cluster_dims=2) as (bx, by):
    rank = T.block_rank_in_cluster()
```

Passing `cluster_dims=` to `T.Kernel` is a `TypeError`.

Persistent SM-resident loops are an advanced path. See
`T.PersistentTileScheduler` in `tilelang_language/loop/advanced.md` and
`../examples/gemm_sm100/gemm_tcgen5mma_ws_persistent.py`. Do not start there.

## Memory And Movement

```python
A_shared = T.alloc_shared((block_M, block_K), dtype)
C_frag = T.alloc_fragment((block_M, block_N), T.float32)
tmp = T.alloc_local((4,), dtype)
scale = T.alloc_var("float32", init=1.0)
```

SM100 TMEM is `T.alloc_tmem(shape, dtype)` with rank ≥ 2. Same warp must
allocate and deallocate.

`T.copy(src, dst)` is the default movement primitive. Use
`T.async_copy` / `T.tma_copy` only when you manage waits yourself.

```python
T.copy(A[by * block_M, k * block_K], A_shared)
T.clear(C_frag)
T.fill(scores_max, -T.infinity("float32"))
```

## Compute

```python
T.gemm(A_shared, B_shared, C_frag)  # synchronous; implicit WGMMA/TCGEN05 wait
T.reduce_max(scores, scores_max, dim=1)
T.reduce_sum(scores, scores_sum, dim=1)
T.exp2(x)
T.if_then_else(cond, true_value, false_value)
```

Explicit-async: `T.wgmma_gemm` / `T.tcgen05_gemm`. SM120 NVF4:
`T.mma_gemm_blockscaled`. Use Python `if`/`else` for control flow.
`T.if_then_else` when the conditional must produce a value.

Do not rely on `break` / `continue` inside kernels. They are not portable
across eager and `@T.prim_func` frontends.

## Minimal Tiled GEMM

```python
@tilelang.jit
def matmul(A, B, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
    M, N, K = T.const("M, N, K")
    A: T.Tensor((M, K), dtype)
    B: T.Tensor((K, N), dtype)
    C = T.empty((M, N), dtype)

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_K), dtype)
        B_shared = T.alloc_shared((block_K, block_N), dtype)
        C_frag = T.alloc_fragment((block_M, block_N), accum_dtype)
        T.clear(C_frag)
        for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
            T.copy(A[by * block_M, k * block_K], A_shared)
            T.copy(B[k * block_K, bx * block_N], B_shared)
            T.gemm(A_shared, B_shared, C_frag)
        T.copy(C_frag, C[by * block_M, bx * block_N])
    return C
```

## Put Together: FlashAttention 2 Forward

The bundled example `../examples/flash_attention/example_mha_fwd_bshd.py` is the
canonical fused-attention walkthrough. It is a **lazy** factory
(`@tilelang.jit(out_idx=[3])` + `@T.prim_func`) that still uses only the
primitives on this page: `T.Kernel`, shared staging, fragment accumulators,
`T.copy`, `T.Pipelined`, `T.gemm`, reductions, `T.exp2`, and
`T.if_then_else` for the causal mask.

Reading path:

- `out_idx=[3]` marks `Output` as the returned buffer.
- Host-side Python chooses shapes, tiles, stages, and
  `TL_ENABLE_FAST_MATH`.
- `T.gemm(..., policy=T.GemmWarpPolicy.FullRow)` is the usual attention
  partition.
- Online softmax state stays in fragments across the K/V pipeline.

Start from that file rather than inventing a new attention skeleton.

## Where To Go Next

- `tilelang_language/loop/` — `T.Parallel`, `T.Pipelined`, persistent
  schedulers.
- `tilelang_language/allocate/` — tensors, TMEM, barriers.
- `tilelang_language/copy_op/` — `T.copy`, TMA, `prefer_instruction`.
- `tilelang_language/gemm_op/` — sync vs explicit-async GEMM.
- `tilelang_language/kernel_warpgroup_cluster_builtins/` — `T.Kernel` /
  `T.ClusterKernel`.
- `sm100.md` — single-CTA TMEM recipe.
- `python_compat.md` — dialects, `T.Tensor[[M, N]]`, eager-only `T.const`.
