# IMPL_NOTES: `{{op}}`

Written by the implementer before coding, appended after every iteration. The next
implementer starts from a fresh context and inherits only what is written here.

## Fusion plan

- Compute pattern (SPEC `compute_pattern`) and how the kernel follows its required structure: TODO
- Reference primitives used (paths under `kda-kernel-implement/reference/`): TODO
- Memory traffic plan: which tensors are loaded once, which are written once, which stay in
  registers between fused steps: TODO
- Compute dtype (SPEC `compute_dtype`): where the single cast to the storage dtype happens;
  tensor-core operands stay in storage dtype: TODO
- Strides passed to the kernel (row stride of every token-wise input; no `.contiguous()`): TODO
- Saved for backward vs recomputed (mirrors SPEC; `RECOMPUTE` variant when `recompute.available`): TODO
- Launch geometry: rows per program, block sizes, `num_warps`, per-workload branches; the
  widest row (register budget, split-D past it) and the small-rows workload (no split): TODO
- Known risks (tile tails, non-power-of-two `D`, int32 overflow, dtype casts): TODO

## Iteration log

### Iteration 1
- Change: TODO
- Result (`report.json` verdict, worst SOL efficiency): TODO
- Next: TODO
