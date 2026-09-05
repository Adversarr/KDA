# GEMM + fused epilogue (`gemm_tensorcore`, TileLang, experimental)

> **Prefer `reference/nvmath/epilogue/gemm/`** (cuBLASLt epilogues at cuBLAS speed), then the
> Triton twin for what cuBLASLt lacks. TileLang and Triton GEMMs are both known to trail cuBLAS
> on sm80; this twin is a TileLang idiom reference, not a kernel to ship.

Files: `kernel.py` (kernels + wrappers + autograd), `reference.py` (eager `F.linear` + act).
The side-by-side table below also times the Triton twin.
The math, host signatures, saved-for-backward / recompute contract, tolerances and fusion notes are
those of [`triton/epilogue/gemm/SNIPPET.md`](../../../triton/epilogue/gemm/SNIPPET.md); this page
only records what is TileLang-specific and how the two compare.

**Status: correct and tested, 0.8x of the Triton twin on the large forwards.** `compute-patterns.md`
sets the `gemm_tensorcore` default backend to `triton` on this measurement; the pitfalls that cost
the difference are listed in [`tilelang/README.md`](../../README.md). Use this snippet when the user
asks for TileLang or a TileLang-only primitive is needed, not as the first choice for a GEMM.

Two differences from the Triton twin as it now stands: the TileLang kernel has **no
`mainloop="cublas"` path** (its whole point is the `T.gemm` mainloop; the cuBLAS + epilogue
kernel alternative and when it wins are in the Triton SNIPPET and `compute-patterns.md`), and it
saves `Z` *with* the bias folded in (the Triton twin saves the raw product and re-adds the bias
in the adjoint, so that the cuBLAS output can be saved as is). The bias is declared in the
storage dtype here; the Triton twin reads any float dtype.

## Structure

```
_gemm_epilogue_kernel      T.copy(x tile, xs); T.copy(w tile, ws); T.gemm(xs, ws, acc, transpose_B=True)
                           -> _epilogue_tile (split_k == 1) or fp32 partial slab (split_k > 1)
_splitk_epilogue_kernel    sums the slabs of one tile, then _epilogue_tile
_epilogue_tile (T.macro)   bias (shared row) -> Z store -> act -> Y store,
                           both stores staged through one shared tile
_epilogue_bwd_kernel       dZ = dY * act'(Z) in [32, 128] tiles, db partials per program
```

- Eager JIT: `N, K` are `T.const`, `M` is `T.dynamic` (grid-only extent, measured free);
  strides, tile sizes and every flag (`act`, `has_bias`, `save_aux`, `split_k`) are Python
  arguments, one compiled variant per distinct tuple.
- `X`, `dY` are `T.StridedTensor` with the row stride as a **compile-time int**
  (`x.stride(0)`); the `T.dynamic` form halves mainloop throughput (README).
- An optional `bias` arrives as a 1-element dummy when absent; its declaration uses a
  `T.dynamic` extent so the packed-ABI shape check passes, and the flag compiles the read out.
  Never name a tensor argument `res` (the `nvrtc` launcher shadows it).
- Outputs are `T.empty` inside the kernel; `z` is `(0, N)` when `save_aux` is false, `partial` is
  `(0, N)` when `split_k == 1`.
- Split-K slabs are padded to whole tiles (`split_k * ceildiv(M, block_M) * block_M` rows) so an
  `M` tail cannot spill into the next split's slab; the epilogue runs once on the sum
  (`T.atomic_add` cannot host a non-linear epilogue).
- `select_config`: same policy as the Triton twin; tiles are `128x128x32` / 3 stages / 128
  threads for >= 4 waves, `64x128x64` below, `64x64x64` / 4 stages with split-K for `M <= 128`.

## Measured (A800, bf16, `torch.profiler` device time)

These measurements are for a residual-add variant of the epilogue, not the current
signature (which has one fragment read less on the forward, ~1 pointwise pass less in
eager/compile). Min over three interleaved rounds; the GPU is unlocked and the measurements used
a lower clock state than the Triton SNIPPET's (cuBLAS 311 us here vs 333 there on the down-proj,
303 vs 274 on the up-proj), so compare ratios, not absolute numbers, across the two pages. Both
fused kernels were timed in the same table here.

| shape (M x N x K) | phase | eager | compile | triton fused | tilelang fused | tilelang / triton | tilelang vs compile |
|---|---|---|---|---|---|---|---|
| 8192 x 4096 x 1024 (up-proj) | fwd (Z saved) | 509 us | 424 | 490 | 582 | 0.84x | 0.73x |
| | infer (no Z) | 509 | 424 | 466 | 542 | 0.86x | 0.78x |
| | bwd (Z saved) | 856 | 917 | 939 | 846 | 1.11x | 1.08x |
| | bwd recompute | | | | 1334 | | |
| 8192 x 1024 x 4096 (down-proj) | fwd | 422 | 401 | 393 | 464 | 0.85x | 0.86x |
| | infer | 422 | 401 | 389 | 436 | 0.89x | 0.92x |
| | bwd | 758 | 762 | 771 | 747 | 1.03x | 1.02x |
| 4096 x 1024 x 1024 (D x D) | fwd | 103 | 94 | 90 | 88 | 1.02x | 1.07x |
| | infer | 103 | 94 | 82 | 84 | 0.98x | 1.12x |
| | bwd | 172 | 169 | 173 | 170 | 1.02x | 0.99x |
| 64 x 1024 x 4096 (skinny, split-K 4) | fwd | 25.8 | 22.4 | 21.7 | 20.6 | 1.05x | 1.09x |
| | infer | 25.8 | 22.4 | 20.2 | 20.6 | 0.98x | 1.09x |
| | bwd | 42.8 | 36.4 | 45.7 | 49.2 | 0.93x | 0.74x |

Split-K sweep, 64 x 1024 x 4096 silu (us): cuBLAS linear 19.0, compile 22.0; `split_k` 1: 36.9,
2: 26.1, **4: 20.8**, 8: 23.1, 16: 29.8. Same knee as Triton (4 splits, 64 programs).

Where the time goes (up-projection, `128x128x32`, one variant at a time, us): bare mainloop 381
(cuBLAS 274, Triton's mainloop ~400); `+ bias` 381; `+ Z` stored from the fragment 483, staged
through shared 397; `+ gelu` 441 (erf on the fp32 fragment, ~60 us, half of Triton's 120);
everything staged 454 vs everything stored from fragments 525. The remaining
~60 us gap to the Triton twin on this shape was not found before the work was stopped
(candidates: the third grid axis kept for split-K, the `T.dynamic` extents of the optional
tensors, the swizzle panel size). The `D x D`, skinny and all backward rows are at parity because
the adjoint kernel and the cuBLAS `dX`/`dW` GEMMs dominate them.

What the table says:

- **TileLang loses where the fused epilogue matters most** (the two long forwards) and ties
  where it does not; against `torch.compile` it is behind on both large forwards, which Triton
  ties or beats on one. A `gemm_tensorcore` kernel therefore starts from the Triton snippet.
- The TileLang **mainloop is not the problem** (180 TFLOP/s bare, above Triton's); the epilogue
  and the constructs around it are, and their cost is invisible in the source. Budget for the
  README's pitfall list before choosing this backend.
- Recompute costs the same 1.5x on the backward as in Triton (one extra GEMM).
