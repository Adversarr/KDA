# Enums

Enums that change lowering, not just names.

## `T.GemmWarpPolicy`

`T.GemmWarpPolicy` is the warp-partition policy for `T.gemm`,
`T.wgmma_gemm`, and `T.tcgen05_gemm`. It controls how warps in one CTA split a
GEMM output tile across the `M x N` axes. It does not change correctness; it
changes work partitioning after lowering and therefore affects performance.

Public values:

- `T.GemmWarpPolicy.Square` — default. Balanced across `M` and `N`.
- `T.GemmWarpPolicy.FullRow` — bias toward `M` (`m_warp = num_warps`,
  `n_warp = 1`). Common in attention.
- `T.GemmWarpPolicy.FullCol` — bias toward `N`.

```python
T.gemm(a_shared, w_shared, acc, policy=T.GemmWarpPolicy.FullRow)
```

`T.tcgen05_gemm_blockscaled` always uses `Square`. Starting heuristic: `Square`
when `block_M` ≈ `block_N`, `FullRow` when `block_M` is much larger,
`FullCol` when `block_N` is much larger. For fused epilogues, include `policy`
in autotune instead of hard-coding it.

## Other enums

- `tilelang.PassConfigKey` — compiler switches. See `pass_config.md`.
- `tilelang.TensorSupplyType` — autotune / profiler input generation
  (`Auto`, `Integer`, `Normal`, …).
- Swizzle `order` is a string, not an enum: `"row"`, `"column"`, or `"mlx"`.
