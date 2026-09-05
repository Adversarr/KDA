# RMSNorm (fp32 math, fused forward + backward)

Files: `kernel.py` (kernels + wrappers + autograd), `reference.py` (eager).

## Math

Per row `x` of `D` elements, weight `w` of `D`:

```
rstd = rsqrt(mean(x^2) + eps)                   fp32, saved for the backward  (4 bytes/row)
y    = (x * rstd) * w                           cast to x.dtype on store
```

Backward with `xhat = x * rstd`, `dxhat = dy * w`:

```
dx = rstd * (dxhat - xhat * mean(dxhat * xhat))
dw = sum_rows(dy * xhat)
```

`rstd` is saved rather than recomputed because recomputing costs a full extra read of `x`
in the backward (one row reduction per row), while saving costs 4 bytes per row.

Saved tensors are optional work. The forward takes `save_rstd` and the kernel a
`SAVE_RSTD: tl.constexpr`; when no backward can follow (grad mode off, or neither `x` nor
`w` requires grad) nothing is allocated or stored and a 0-element placeholder is returned.
`rmsnorm()` decides this before `Function.apply` because grad mode is off inside `forward`.
In a kernel package the same flag is called `save_aux` and `_common.compat` sets it per call.
Kernel libraries do this for every saved statistic; a training-only kernel that always writes
them is wrong for evaluation and inference.

## Dtype and layout rules

- Input `x`: fp32/fp16/bf16, shape `(..., D)`, unit stride along `D`, **any row stride**: a
  padded row (`x = storage[:, :D]`) or a slice of a fused `qkv` is passed as `stride_row`,
  never copied. `dy` likewise carries its own row stride into the backward. The outputs `y`
  and `dx` are allocated contiguous. A non-unit last stride is rejected (a kernel package
  rejects it in `interface.py`); the wrapper flattens leading dims with `view`, no copy.
- `w`: any float dtype, contiguous. `dw` is returned in `w.dtype` after an fp32 reduction.
- All arithmetic in fp32 (`COMPUTE = tl.float32`); the only casts are at load and store.
- The reference casts `x` to fp32, does everything in fp32, casts once at the end. HF-style
  modules cast `xhat` back to the input dtype *before* the weight multiply; a user's
  `_eager.py` keeps their order, and the kernel's store must match it (cast `x * rstd`, then
  multiply by `w` in that dtype) or bf16 outputs will differ by 1 ulp in a fraction of elements.

## Launch geometry (A800 measurements, 16K x 1K bf16)

- Forward: one program per row, `BLOCK_D = next_pow2(D)`, 4 warps up to 4K, 8 to 16K, 16
  beyond. Runs at copy speed (~100% of a same-size `copy_`), so nothing is left to tune.
- Backward: a fixed grid of `2 x SMs` programs strides over rows, `R = 4096 // BLOCK_D` rows
  per `[R, BLOCK_D]` tile so several rows' loads are in flight. Each program accumulates
  `dw` in registers and writes one fp32 partial row; the host reduces `(n_programs, D)` with
  `torch.sum`. Kernel alone ~86% of copy, ~77% including the reduce. More programs make the
  kernel faster but the reduce slower; 2/SM is the measured sweet spot.
- Per-row `dw` partials (`(N, D)` fp32) or cross-row atomics are both wrong choices: the
  first adds more traffic than the whole backward, the second is non-deterministic.
- The backward tile caps `D` at 8192 (`BWD_MAX_D`); a wider row needs a split-D backward
  (two-stage `mean(dxhat * xhat)`), not in this snippet.

## Small rows and wide rows: when to split a row across programs

`kernel.py` carries a second forward geometry, **split-D** (`_rmsnorm_fwd_sumsq_kernel` writes
one partial sum of squares per `(row, chunk)`, `_rmsnorm_fwd_normalize_kernel` sums the row's
partials to `rstd` and normalises its chunk; chunk 0 stores `rstd`). `_fwd_geometry` picks it,
and the measured comparison is below (bf16, 108 SMs, `torch.profiler` device time in us,
`copy` = a device copy of the same bytes):

| rows | D | MiB | one program per row | split-D (4 chunks; 8 at 64K) | copy | chosen |
|---|---|---|---|---|---|---|
| 32 | 4096 | 0.25 | 4.2 | 7.2 | 3.3 | one-row |
| 32 | 8192 | 0.50 | 5.0 | 7.9 | 3.6 | one-row |
| 64 | 4096 | 0.50 | 4.7 | 7.7 | 3.6 | one-row |
| 64 | 8192 | 1.00 | 6.1 | 9.0 | 8.0 | one-row |
| 100 | 4096 | 0.78 | 5.4 | 8.3 | 4.0 | one-row |
| 100 | 8192 | 1.56 | 6.7 | 9.8 | 8.9 | one-row |
| 1024 | 8192 | 16.0 | 25.8 | 34.1 | 26.1 | one-row |
| 32 | 32768 | 2.0 | 9.3 | 11.8 | 9.3 | one-row |
| 32 | 65536 | 4.0 | **56.1** | **15.4** | 11.2 | split-D |

Two conclusions, both against the intuition that "rows below the SM count leave the machine
idle":

- **Small rows are not a problem for `D <= 32K`.** With 32-100 rows the one-row kernel is at
  or above the copy roof (a 5 us kernel against a 4-9 us copy): at these sizes both are bound
  by launch and memory latency, not bandwidth, and extra programs only add a second launch
  and a second read. Do not split for small row counts; the verdict does not gate `sol_eff`
  on a phase whose roof is under the 10 us latency floor (`speed-of-light.md`), only the
  comparison with eager/compile.
- **Wide rows are.** Past `ONE_ROW_MAX_D = 32768` a row no longer fits the register file at
  16 warps: the one-row kernel spills and runs 5x slower than copy, while split-D stays within
  1.4x of it (the second read of `x` is an L2 hit). That is the rule `_fwd_geometry` applies:
  `next_pow2(D) > ONE_ROW_MAX_D` -> split-D with chunks of `min(D/4, 8192)`. The fused backward
  is still capped at 8192, so wide rows are forward-only (inference) in this snippet.

The same argument holds for every row-wise op in this family (LayerNorm, L2 norm, softmax):
copy the split-D pair when the row is wide, not when the row count is small.

## Tolerances

`y` and `dx` meet the fixed defaults elementwise. `dw` reduces over every row: its rounding
noise is `~sqrt(N) * eps * |term|` (eps of the lowest precision in the chain) and the
reference's summation order differs, so near-zero elements exceed a fixed atol at thousands
of rows. The KDA verifier scales `atol` by `sqrt(rows reduced)` for gradients smaller than
the output and uses the workload dtype's tolerance row. No SPEC override is needed for this.

## Fusing with neighbours

- **Residual in**: load `residual`, add in fp32, store the fp32/bf16 sum `h` *and* normalize
  it in the same program (`elementwise/residual_add` has the add and its stride handling; the
  fused op is this kernel with that load and store added, two user-visible outputs `(h, y)`
  and `rstd` as the aux; the backward reads `h` back instead of `x`). There is no separate
  snippet for the fusion: these two are what you copy.
- **RoPE / permute out**: normalize per head, rotate, then store to the permuted layout
  (see `fusion-exemplar/rmsnorm_rope_permute`).
- **Gate out**: multiply `y` by a gate tile before the store; the gate's gradient is
  `dy * y_pre_gate`, so keep the pre-gate value in registers for the backward or recompute it.
