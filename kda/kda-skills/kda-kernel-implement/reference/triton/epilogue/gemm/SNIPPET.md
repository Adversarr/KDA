# GEMM + fused epilogue (`gemm_tensorcore`, Triton)

> **Prefer `reference/nvmath/epilogue/gemm/`.** Bias, ReLU, tanh-GELU and the saved
> pre-activation are cuBLASLt epilogues; nvmath-python runs them inside the cuBLAS GEMM at
> cuBLAS speed (8192 x 8192 x 2048: cuBLAS 1017 us, nvmath 1053, this kernel 1273, TileLang
> ~1600). Hand-written GEMMs in Triton and TileLang are known to trail cuBLAS on sm80; use this
> file when nvmath is not importable, for an activation cuBLASLt lacks (erf-GELU, SiLU, a
> SwiGLU gate, a quantised store) or for GEMMs too short for nvmath's 130 us host cost.

Files: `kernel.py` (kernels + wrappers + autograd), `reference.py` (eager `F.linear` + act).
Same math and workloads as the TileLang
twin in `reference/tilelang/epilogue/gemm/` and the nvmath twin.

**Two mainloops, one host API.** `mainloop="triton"` is the fused GEMM + epilogue kernel this
page is mostly about; `mainloop="cublas"` is `torch.matmul` followed by the same epilogue as one
pointwise kernel (`_pointwise_epilogue_kernel`) over the stored `Z`. `select_mainloop(M, N, K)`
picks per shape from the table below (`N >= 4096` -> cuBLAS), because on A800 the Triton
mainloop trails cuBLAS by 23-53% on wide outputs and the fused epilogue cannot repay that; on
narrow outputs the fused kernel ties or wins. **Measure cuBLAS alone first** (`torch.matmul` on
the shape): if the fused mainloop is further from it than the epilogue's traffic
(`3 * M * N * es` bytes at HBM speed), fuse only the epilogue and say so in IMPL_NOTES.

## Math

```
Z = X W^T              X: (M, K) any row stride, W: (N, K) contiguous (nn.Linear)
Y = act(Z + b)         b: (N,) any float dtype; act in {none, relu, gelu (erf), gelu_tanh, silu}
```

No residual: `act(X W^T + b) + R` is not a transformer pattern (the residual is added by the
next norm, `residual_then_norm`), and cuBLASLt can only add an addend *inside* the activation.
`gelu_tanh` is what models call GELU (GPT-2, ViT, HunyuanVideo, `nn.GELU(approximate="tanh")`)
and what cuBLASLt fuses; `gelu` is the erf form `F.gelu` defaults to. They differ by up to
2e-3, above a bf16 ulp near 1: implement the one the user's code calls.

Backward, with `dZ = dY * act'(Z + b)` (the saved `Z` carries no bias, so the cuBLAS output is
saved as is and both mainloops save the same tensor). With the cuBLAS mainloop the aux *is*
the matmul output, kept instead of freed, so `kernel_infer ~= kernel_fwd` by construction:
the inference path saves nothing device-side on this mainloop (the verifier's
`kernel_infer <= kernel_fwd` row reads equal, which is correct, not a missing `save_aux`
branch); only the fused `tl.dot` mainloop skips a store under `no_grad`. This passes against `reference.py` at
these shapes; a user op whose epilogue cancels (an output near 0 with `|z| ~ sqrt(K)`)
under the default bf16 `atol` needs the bias folded into the GEMM (`torch.addmm(b, x, w.t())`,
bit-equal to eager `F.linear`) and the bias-inclusive `round_bf16(Z + b)` saved instead; the
SPEC decides that at M0/M1 (`compute-patterns.md`, "Numerics can decide the mainloop"):

```
db = sum_rows(dZ)      fp32 partials per program, reduced on the host
dX = dZ @ W            torch.matmul (cuBLAS)
dW = dZ^T @ X          torch.matmul; X may be row-strided
```

## Tensor-core rule

`tl.dot(a, tl.trans(b), acc)` takes `a` and `b` straight from `tl.load` in storage dtype
(bf16/fp16); the accumulator is `tl.float32`; the epilogue (`_epilogue`) runs on the fp32
accumulator and casts once at the store. No `.to(tl.float32)` on an operand: the lint rule
`mma_operand` flags it, and on sm80 it would silently move the GEMM off the tensor cores.
fp32 inputs are not accepted (`_MMA_DTYPES`): an fp32 GEMM is cuBLAS's job.

## Saved-for-backward and recompute

The forward keeps the product `Z` (storage dtype, `M*N*es` bytes): the fused kernel stores it
under `SAVE_AUX`, the cuBLAS path returns the matmul output (or frees it), in both cases only
when grad mode is on, some input requires grad and `act != none` (for `act = none`, `dZ = dY`
and nothing is saved). `gemm_epilogue(..., recompute=True)` saves nothing and the backward
rebuilds `Z` with one `torch.matmul`: one extra GEMM, `2*M*N*K` FLOPs, in exchange for
`M*N*es` bytes of activation memory. For the
`D -> 4D` up-projection `Z` is 4x the size of `X`, which is when the trade is worth offering;
measured cost below (`bwd recompute` ~1.4x the saved-Z backward). In a kernel package the same
switches are `save_aux` (set per call by `_common.compat`) and `KDA_RECOMPUTE` / SPEC `recompute`.

## Layout and geometry

- `X` carries its row stride into the kernel (`stride_xm`); nothing is copied. `Y`, `Z`, `dZ` are allocated contiguous. Any leading shape is flattened with `view`.
- Grouped tile order (`GROUP_M = 8` row tiles walk one column of `W` tiles, L2 reuse).
- Row/col indices for the loads are marked `tl.multiple_of` / `tl.max_contiguous` when the
  tile divides `M` / `N` (`EVEN_M`, `EVEN_N`), which is what lets Triton emit 16-byte vector
  loads; the `% M` wrap used for ragged edges hides that and costs ~20% of GEMM throughput, so
  keep it on the ragged path only. `EVEN_K` drops the K mask when `K % (BLOCK_K * SPLIT_K) == 0`.
  This trick and the tile table were lifted from inductor's own `triton_mm` template (the
  `AUTOTUNE` printout of `torch.compile(..., mode="max-autotune-no-cudagraphs")` names the
  winning config, the source sits under `/tmp/torchinductor_<user>/`); when `torch.compile` is
  the baseline to beat, read what it generated first (kda-kernel-implement SKILL.md §3).
- Offsets are int64 from `tl.program_id(...).to(tl.int64)`: `M*N` and `M*K` pass 2^31 at LLM sizes.
- `select_config(M, N, K, num_sms)`: 128x128x64 / 3 stages / 4 warps (inductor's pick on sm80
  too) when the 128x128 grid has >= 4 waves of tiles, else 64x128x64; skinny `M <= 128` uses
  64x64x64 / 4 stages with **split-K**: `tiles * SPLIT_K ~ num_sms / 2`, at least 8 K blocks per
  split. Split-K writes fp32 partials `(SPLIT_K, M, N)` and `_splitk_epilogue_kernel` sums them
  and runs the same `_epilogue` (count `2 * SPLIT_K * M * N * 4` bytes in the roofline).
- Adjoint kernel: `[32, 128]` tiles, a fixed number of programs along `M` (`2 * SMs /
  n_col_blocks`) strides over the row tiles and keeps a register `db` accumulator, one fp32
  partial row per program (the rmsnorm `dw` scheme).

## Measured (A800, bf16, `torch.profiler` device time)

Numbers are the min over three interleaved rounds: this GPU is unlocked
(1155 MHz application clock, boosting to 1410) and shifts every number by up to 20% between
clock states, so only ratios within one table are meaningful. `eager` is `F.linear` + `F.gelu`
(cuBLAS + two pointwise kernels; the table was measured with a residual add as a third, since
removed, which shifts eager/compile by ~1 pointwise pass); `compile` is `torch.compile` of the same function
(inductor: cuBLAS or its own Triton mm template, plus one fused pointwise kernel) and is the bar
a fused epilogue has to clear.

| shape (M x N x K) | phase | cuBLAS alone | eager | compile | fused triton | cuBLAS + epilogue kernel | selected | vs compile |
|---|---|---|---|---|---|---|---|---|
| 8192 x 4096 x 1024 (up-proj) | fwd (Z saved) | 327 us | 529 | 436 | 541 | **443** | cublas | 0.98x |
| | infer (no Z) | | 529 | 436 | 499 | **443** | | 0.98x |
| | bwd (Z saved) | | 845 | 837 | 870 | (same) | | 0.96x |
| | bwd recompute | | | | 1164 | | | |
| 8192 x 1024 x 4096 (down-proj) | fwd | 330 | 385 | 366 | **359** | 363 | triton | 1.02x |
| | infer | | 385 | 366 | **353** | 363 | | 1.04x |
| | bwd | | 688 | 677 | 688 | | | 0.98x |
| 4096 x 1024 x 1024 (D x D) | fwd | 74 | 103 | 94 | **89** | 95 | triton | 1.06x |
| | infer | | 103 | 94 | **81** | 95 | | 1.16x |
| | bwd | | 172 | 167 | 175 | | | 0.95x |
| 64 x 1024 x 4096 (skinny, split-K 4) | fwd | 17.7 | 25.9 | 22.4 | 24.2 | 24.4 | triton | 0.93x |
| | infer | | 25.9 | 22.4 | **20.2** | 24.2 | | 0.90x |
| | bwd | | 42.7 | 36.3 | 46.1 | | | 0.79x |

The `8192 x 8192 x 2048` MLP of `examples/fused_gemm_epilogue` (its GOLDEN.md, two runs):
cuBLAS 1019, eager 1443, compile 1279, fused triton 1299 / 1250 infer, cuBLAS + epilogue
**1250-1269**.

The Triton mainloop's distance from cuBLAS is the whole story: 7% on the down-projection and
`D x D`, 14% on the skinny shape, but 53% on the up-projection and 23% on the 8192-wide MLP.
Where it is small the fused epilogue (which saves one read of `Z` and the pointwise launch)
nets a win over both eager and `torch.compile`; where it is large no epilogue saving covers
it, and cuBLAS + one epilogue kernel is the fastest forward there is (18% ahead of the fused
kernel on the up-projection, 2-4% on the 8192-wide MLP), level with or ahead of `torch.compile`
(which does the same thing with a less tight pointwise kernel). `select_mainloop` draws the
line at `N >= 4096`, which is where the measured shapes put it; the cause of the wide-N loss
was not isolated (candidates: the 128x128 tile's 16-step K loop at `K = 1024`, L2 behaviour of
2048 tiles over a 16 MB `X`).

Split-K sweep, 64 x 1024 x 4096 silu (us): cuBLAS linear 19.0, compile 21.9; `SPLIT_K` 1: 40.4,
2: 25.3, **4: 19.6**, 8: 20.9, 16: 24.4. Sixteen tiles alone leave 85% of the SMs idle; four
splits (64 programs) is the knee, past it the partial traffic (`2*s*M*N*4` bytes) wins.

What the table says:

- **Pick the mainloop per shape, from a cuBLAS measurement**, not from the pattern name: the
  `gemm_tensorcore` row promises a fused epilogue, and on sm80 that fusion is a win only where
  Triton's mainloop is within ~10% of cuBLAS. The erf-based gelu on the fp32 accumulator is
  also ~120 us of CUDA-core work on the up-projection (`act=none` 399 us vs `gelu` 525 us at
  the same tile) that does not overlap the tensor cores; inductor and the cuBLAS path pay it in
  a memory-bound kernel where it hides under the traffic. `relu` costs nothing, `silu` about a
  third of `gelu`.
- **The backward is a wash by construction**: two cuBLAS GEMMs dominate, the adjoint kernel is
  bandwidth-bound and matches inductor's fused pointwise kernel; it does not depend on the
  mainloop.
- **Infer mode** (no `Z` store) is worth 4-8% on the fused forward: a training-only kernel that
  always writes `Z` gives that away in evaluation. The cuBLAS path has no infer saving (the
  matmul writes `Z` regardless).
- The TileLang twin (`reference/tilelang/epilogue/gemm/SNIPPET.md`, same table timed side by
  side) ends at 0.84x of this kernel on the up-projection forward and 0.85x on the
  down-projection, parity elsewhere; `compute-patterns.md` therefore makes `triton` the default
  backend for `gemm_tensorcore`, and `reference/tilelang/README.md` lists why TileLang is hard
  to make fast for a GEMM.

## Tolerances

`y`, `dx`, `dw`, `db` all meet the fixed defaults with `check(..., tol_dtype=x.dtype,
reduced_over=...)`: `dx` reduces over `N`, `dw` and `db` over `M`. The eager reference rounds
to bf16 three times (`Z`, `+b`, `act`); the fused kernel once, so it is closer to the fp32
value than the reference it is checked against.

## Fusing with neighbours

- **SwiGLU**: not a dual-GEMM kernel. Run `[W1; W3]` as one concatenated cuBLAS GEMM (nvmath or
  `torch.matmul`) and fuse `silu(g) * u` (+ its adjoint) as one pointwise kernel; pattern row
  `swiglu_dual_gemm`.
- **Norm in front**: do not fuse a row norm into the GEMM's `X` load (the row statistic needs the
  whole row, the tile has `BLOCK_K` of it); run the norm kernel first, its output is the GEMM's `X`.
- **Quantised epilogue**: per-row scale + cast to fp8/int8 goes where the `.to(y_ptr.dtype.element_ty)`
  cast is; the scale is a second output of the same program.
