# Reference kernels: table of contents

Complete, tested, standalone snippets, one directory per primitive and kernel backend. Each
holds `SNIPPET.md` (math, dtype/layout rules, measured launch geometry, fusion notes),
`kernel.py` (kernels, host wrappers, autograd) and `reference.py` (eager). Validate the
generated package against its eager reference with `_run_dev.py --verify --bench` at M3;
check reference kernels with `kda-kernel-verify-and-bench/scripts/lint_kernel.py`. Validated pair:
torch 2.11 / triton 3.6 / tilelang 0.1.14 / nvmath-python 1.0 on A800 (sm80), CUDA 12.8; the
numbers in each SNIPPET were measured there.

**Library first.** A GEMM with an epilogue is not written as a kernel when nvmath-python
(cuBLASLt) expresses it: [`nvmath/epilogue/gemm`](nvmath/epilogue/gemm/SNIPPET.md) is the
preferred `gemm_tensorcore` reference and `kernel_backend: nvmath` the preferred SPEC value;
Triton and TileLang GEMMs are known to trail cuBLAS on sm80.

There is no helper library: the implementer reads the primitives that appear in the target
op and writes the fused kernel from their bodies, as `fusion-exemplar/` demonstrates. Which
primitives, and in which backend, is decided by the SPEC `compute_pattern` row of
[`common/compute-patterns.md`](common/compute-patterns.md).

## Triton primitives (`triton/`)

| family | snippet | pattern | fwd | bwd | notes |
|---|---|---|---|---|---|
| `fp32_norms` | [`rmsnorm`](triton/fp32_norms/rmsnorm/SNIPPET.md) | `rowwise_onepass` | yes | yes | saved `rstd`; strided-grid `dw` accumulation; the model for every row-wise norm. Two forward paths chosen by `_fwd_geometry`: one program per row up to `D = 32768`, **split-D two-stage** beyond (row width rule); the SNIPPET's table shows small row counts do not need a split and wide rows do; strided `x`/`dy` without a copy |
| `elementwise` | [`residual_add`](triton/elementwise/residual_add/SNIPPET.md) | `elementwise_stream` | yes | identity | flat 1-D streaming skeleton; fp32 residual stream; gating lives here too |
| `rope` | [`rope`](triton/rope/rope/SNIPPET.md) | `rope_pairs` | yes | yes | half + interleaved pairings, strided input, position ids; `tl.split` for interleaved |
| `permute` | [`bnhd_to_bhnd`](triton/permute/bnhd_to_bhnd/SNIPPET.md) | `permute_copy` | yes | yes | permuted store address; same kernel both directions |
| `epilogue` | [`gemm`](nvmath/epilogue/gemm/SNIPPET.md) (**nvmath, preferred**), [`gemm`](triton/epilogue/gemm/SNIPPET.md) (Triton), [`gemm`](tilelang/epilogue/gemm/SNIPPET.md) (TileLang, experimental) | `gemm_tensorcore` | yes | adjoint + `matmul` | `Y = act(X W^T + b)`, no residual (a transformer adds it in the next norm). nvmath: the epilogue (bias, ReLU, tanh-GELU, aux store) runs inside the cuBLAS kernel via `nvmath.linalg.advanced.Matmul`, planned once per shape; measured 1.16-1.27x eager and `torch.compile`, faster than the Triton and TileLang fused kernels on every large shape. Triton: two mainloops behind one API (`select_mainloop`: fused `tl.dot` kernel, or `torch.matmul` + one pointwise epilogue kernel where cuBLAS is far ahead), `ACT` in {none, relu, gelu (erf), gelu_tanh, silu}, `SPLIT_K`, `SAVE_AUX`/`recompute`; the adjoint kernel and the pointwise pass are what the nvmath twin reuses. `X` row-strided everywhere; measured against eager, cuBLAS and `torch.compile` |
| `attention` | [`fa2_causal`](triton/attention/fa2_causal/SNIPPET.md) | `flash_attention_2` | yes | 3 kernels | FlashAttention-2, causal MHA, `(B, H, S, D)` any strides: log2-domain online softmax, K loaded transposed, mask split (unmasked blocks / diagonal blocks), `lse` saved; backward = preprocess + dK/dV kernel + dQ kernel, `P` recomputed, no atomics (deterministic); int64 base + own tile, int32 in the loop with a host guard (int64 in the loop cost 11%); tile sweep tables; **0.88x / 0.76x of torch's FA2** fwd / bwd at 8192 tokens, 14-22x over eager; block-causal recipe |
| `attention` | [`fa2_gqa`](triton/attention/fa2_gqa/SNIPPET.md) | `flash_attention_2` | yes | 3 kernels | dense (non-causal) GQA, `Sq != Skv`, `H_kv` divides `H`: K,V head `h // GROUP`, tail-only mask, the group reduction of dK/dV inside one program (loop over the group's heads, deterministic); 0.87x / 0.70x of torch's FA2; decode/window/bias fusion notes |

## TileLang primitives (`tilelang/`) — experimental backend

Honor a supported explicit backend choice; otherwise nvmath is preferred for supported
GEMM epilogues and Triton elsewhere. Choose experimental TileLang when explicitly requested. The TileLang twins below are correct and tested (its dense GQA kernel is the fastest of the
four attention references; the causal ones tie), but its per-shape JIT is tens of seconds per
kernel (compute-patterns.md, backend note), and
**TileLang is very difficult to make fast for a GEMM**: the
twin below is correct and ends at 0.8x of the Triton kernel on the large forwards after the
same effort, for reasons the source does not show. [`tilelang/README.md`](tilelang/README.md)
lists those pitfalls with their measured cost, and gives the conventions the `_tilelang/`
package template assumes (eager JIT, `T.StridedTensor` with the stride as a compile-time int,
`accum_dtype`, Python-level `save_aux`/`recompute` specialisation, shared-staged stores).

| family | snippet | pattern | fwd | bwd | notes |
|---|---|---|---|---|---|
| `epilogue` | [`gemm`](tilelang/epilogue/gemm/SNIPPET.md) | `gemm_tensorcore` | yes | adjoint + `matmul` | same contract and workloads as the Triton `epilogue/gemm`; `T.gemm` on shared-memory operands, `T.Pipelined` K loop, `T.macro` epilogue with shared-staged stores, split-K via padded fp32 slabs + a reduce kernel; **0.84x / 0.85x of the Triton twin on the up-/down-projection forwards**, parity elsewhere |
| `attention` | [`fa2_causal`](tilelang/attention/fa2_causal/SNIPPET.md) | `flash_attention_2` | yes | 3 kernels | same contract as the Triton twin; `T.gemm` with `GemmWarpPolicy.FullRow` on every GEMM (the `QK^T` fragment feeds `PV`), mask as the accumulator's initial fill, **`num_stages=1` everywhere** (2 stages race on tail tiles in the backward and are slower), 1-D `T.copy` does not zero-fill (select the tail out), two staging tiles for two stores, row-loop preprocess (fragment reduce needs power-of-two `D`); **parity with Triton** (0.90x / 0.71x of torch's FA2) |
| `attention` | [`fa2_gqa`](tilelang/attention/fa2_gqa/SNIPPET.md) | `flash_attention_2` | yes | 3 kernels | dense GQA twin; two `T.const` groups for `Sq != Skv`, `GROUP` a compile-time int, group loop inside the dK/dV program; **the fastest of the four: 0.94x / 0.76x of torch's FA2**, 197 TFLOP/s forward |
| `mlp` | `swiglu_dual_gemm` | `swiglu_dual_gemm` | - | - | not a kernel: one concatenated library GEMM + a Triton gate kernel (`compute-patterns.md`) |

## Fusion exemplar

| exemplar | primitives fused | walkthrough |
|---|---|---|
| [`rmsnorm_rope_permute`](triton/fusion-exemplar/rmsnorm_rope_permute/FUSION.md) | rmsnorm + rope + bnhd_to_bhnd | load once, two primitives in registers, permuted store, fused adjoint, save-vs-recompute, precision of fusions |

## Example op compositions

Op names listed so you can recognise your op's shape: which primitives compose it and
which pattern row applies. They are **not** shipped with the skill
(no `examples/` directory exists in your repo); the snippets above are the only code to copy.

| op | primitives | pattern |
|---|---|---|
| `fused_residual_rmsnorm` | residual_add + rmsnorm | `rowwise_onepass` |
| `fused_residual_layernorm` | residual_add + rmsnorm (LayerNorm variant: mean and rstd saved) | `rowwise_onepass` |
| `qk_rmsnorm_rope_permute` | rmsnorm + rope + bnhd_to_bhnd (the exemplar, on a real repo) | `rowwise_onepass` |
| `qk_multihead_rmsnorm_rope_permute` | rmsnorm (per head, separate q/k weights) + rope + bnhd_to_bhnd | `rowwise_onepass` |
| `f32_adaln` | rmsnorm/LayerNorm without affine + elementwise scale/shift (gating family) | `rowwise_onepass` |
| `fused_l2_norm_scale` | fp32_norms (L2 over D) + elementwise scale | `rowwise_onepass` |
| `fused_gemm_epilogue` | epilogue: `gelu_tanh(x W^T + b)` (nvmath `GELU_AUX_BIAS` when importable, else the Triton twin whose `select_mainloop` sends the 8192-wide MLP to cuBLAS + epilogue kernel) | `gemm_tensorcore` |
| `h3_block_causal_attention` | attention/fa2_causal with the block-causal mask (video: frame chunks; `block_size` tokens per chunk, dense inside a chunk, causal across chunks), `H = 56`, `Dh = 128` | `flash_attention_2` |

## Common

- [`common/compute-patterns.md`](common/compute-patterns.md): the closed list of patterns, the
  structure each requires, the signature the verifier greps for, the row width and tensor-core
  rules.
- [`common/speed-of-light.md`](common/speed-of-light.md): how to write the roofline for
  `_speed_of_light.py` in all four phases, what the achievable roof is, worked examples
  including the GEMM compute roof and split-K partials.
- [`common/a800-compiler-tuning.md`](common/a800-compiler-tuning.md): measured Triton 3.6
  compiler-option rejections and the 1152-channel AdaLN backward split; resource counts,
  separate grid control, small-input fallback and source-matched validation.
