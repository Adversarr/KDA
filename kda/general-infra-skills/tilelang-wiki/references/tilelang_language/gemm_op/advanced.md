# GEMM Operations: Advanced

This page covers explicit GEMM variants and shape caveats. Start with
`T.gemm` or `T.gemm_sp` unless you need manual synchronization, sparse paths, or
Blackwell tensor-memory behavior.

## Shape Checks

Dense `T.gemm` / `T.wgmma_gemm` / `T.tcgen05_gemm` check these constraints
before lowering:

- `C` has rank ≥ 2. The last two modes are `(M, N)`; any leading modes must
  have extent `1`.
- `A` and `B` have rank at least 2. Leading extents must be `1`; the final two
  dimensions are the matrix tile.
- With `transpose_A=False`, `A` is `(M, K)`; with `transpose_A=True`, `A` is
  `(K, M)`.
- With `transpose_B=False`, `B` is `(K, N)`; with `transpose_B=True`, `B` is
  `(N, K)`.
- Per-axis strides and final-axis offsets are still serialized for out-of-tree
  consumers; in-tree lowering consumes the full operand regions.

Sparse `T.gemm_sp` follows the same rank and offset style, but its logical K is
`2 * compressed_K` from `A_sparse`.

Block-scaled APIs currently require `C` to be exactly rank 2.

## Explicit Hopper WGMMA

```python
T.wgmma_gemm(
    A,
    B,
    C,
    transpose_A=False,
    transpose_B=False,
    policy=T.GemmWarpPolicy.Square,
    clear_accum=False,
)

T.wgmma_gemm_sp(
    A_sparse,
    E,
    B,
    C,
    transpose_A=False,
    transpose_E=False,
    transpose_B=False,
    policy=T.GemmWarpPolicy.Square,
    clear_accum=False,
)
```

These APIs request Hopper WGMMA lowering and do not emit the implicit
warp-group wait used by the high-level synchronous path. Use them only when the
surrounding schedule explicitly manages WGMMA completion. If the target or
operand pattern cannot use WGMMA, compilation fails instead of falling back.

## Explicit Blackwell TCGEN05

```python
T.tcgen05_gemm(
    A,
    B,
    C,
    transpose_A=False,
    transpose_B=False,
    policy=T.GemmWarpPolicy.Square,
    clear_accum=False,
    *,
    mbar,  # BarrierType | None
    use_2cta=False,
)
```

`T.tcgen05_gemm` requests Blackwell TCGEN05 lowering and never auto-emits
`mbarrier_wait_parity`. The schedule must wait before consuming the
tensor-memory result.

`mbar=None` is legal: it omits the completion arrival for an intermediate
issue. A later TCGEN05 operation in the same issue stream may publish the
completion event for the whole sequence.

With `use_2cta=True`, each CTA provides half of `N` from `B`; the wrapper
checks `N_B * 2 == N_C` and requires `cluster_dims` `(2,1,1)` or `(1,2,1)`.

The sparse form has the same explicit-asynchronous contract:

```python
T.tcgen05_gemm_sp(
    A_sparse,
    E,
    B,
    C,
    transpose_A=False,
    transpose_E=False,
    transpose_B=False,
    policy=T.GemmWarpPolicy.Square,
    clear_accum=False,
)
```

## Block-Scaled TCGEN05

```python
T.tcgen05_gemm_blockscaled(
    A,
    B,
    C,
    SFA_tmem,
    SFB_tmem,
    transpose_A=False,
    transpose_B=False,
    clear_accum=False,
    wg_wait=0,
    mbar=None,
    *,
    k_start,
    sf_a_granularity_k,
    sf_b_granularity_k,
    use_2cta=False,
)
```

This is the explicit Blackwell block-scaled path. `A` and `B` are FP8/FP6/FP4
mxf8f6f4 shared-memory operands, `C` is the tensor-memory accumulator, and
`SFA_tmem` / `SFB_tmem` are E8M0 scale factors already in tensor memory. The
wrapper always uses `GemmWarpPolicy.Square`.

`k_start` is the logical K-axis start offset for this MMA tile.
`sf_a_granularity_k` and `sf_b_granularity_k` say how many K elements one
packed scale factor covers. The compiler derives the PTX scale-factor A/B IDs
for each internal K32 MMA atom from these values. There are no `sf_a_id` /
`sf_b_id` kwargs.

`mbar` is required at the call site (`assert mbar is not None`).
`use_2cta=True` requires `cluster_dims` `(2,1,1)` or `(1,2,1)` and has no
fallback.

When copying a block-scaled accumulator out of tensor memory, annotate the
tensor-memory layout:

```python
layout = T.make_blockscaled_gemm_layout(C_tmem, A_shared, transpose_A=False)
T.annotate_layout({C_tmem: layout})
```

Use the block-scaled path only for kernels already structured around Blackwell
tensor memory, explicit barriers, and scale-factor movement. It is not a drop-in
replacement for ordinary `T.gemm`.

## SM120 Block-Scaled MMA

```python
T.mma_gemm_blockscaled(
    A,
    B,
    C,
    SFA,
    SFB,
    transpose_A=False,
    transpose_B=False,
    policy=T.GemmWarpPolicy.Square,
    clear_accum=False,
    *,
    k_start,
    sf_a_granularity_k,
    sf_b_granularity_k,
    sf_layout=None,
)
```

This is the SM120 warp-level NVF4 path:
`m16n8k64.kind::mxf4nvf4.block_scale.scale_vec::4X` with E2M1 operands, FP32
accumulation, and UE4M3 scale factors. It uses the same `k_start` /
granularity model as `T.tcgen05_gemm_blockscaled`, but it is synchronous
`mma.sync` and does not use tensor memory or mbarriers.

See `examples/gemm_sm120/sm120_nvfp4_blockscaled_gemm.py` and
`maint/gemm/gemm_sm120/benchmark_sm120_nvfp4_blockscaled_gemm.py`.
