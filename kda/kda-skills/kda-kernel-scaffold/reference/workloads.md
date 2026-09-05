# Workloads per op class

What the `workloads:` list of `SPEC.md` must contain, by op class. A workload is one set of
shapes, dtypes, strides and scalar params the harness verifies and benches; `required: true`
rows gate the verdict, the others are diagnostics. The user's real shapes always come first.
`D` below is the last (feature) dim, `rows` the product of the leading dims.

Rules that hold for every class:

- The last dim of every tensor input is contiguous (`interface.py` rejects anything else);
  leading layouts must match the op's declared support and be passed to the kernel. The
  default `rows_and_stride` helper requires collapsible leading dimensions; it does not
  support arbitrary transposed leading axes. Do not claim broader layout support from a padded row test. `strides: {x: [...]}` declares a
  workload's layout explicitly (same rank as `shapes.x`, last entry `1`); an input without an
  entry is contiguous.
- Dtype: the user's autocast dtype for activations (`dtype: bf16` typically); `dtypes:` for
  the inputs the user keeps in fp32 (norm weights, gates, biases when the code says so). An
  *activation* the model keeps in fp32 goes there too: a pre-norm block with an fp32 residual
  stream is `dtypes: {residual: fp32, weight: fp32}` with `dtype: bf16` for `x`, and the
  contract then accepts only fp32 for that input (the example below shows the weight-only case).
- Rank: the user rows may keep the model's own rank (`x: [8, 512, 1024]` for a `(B, S, D)`
  model) or be flattened to `(rows, D)`; the kernel maps the declared leading layout to a row index
  without copying; use `rows_and_stride` only for collapsible leading dimensions. Use the model's rank for the `user_*` rows so `strides:` and the
  contract probe describe the tensors the model really passes; the sweep rows are 2-D.
- The ultra-large row `edge_huge` is `source: edge`, `required: false`, and exists for one
  reason: int32 element offsets overflow at 2^31 elements (a 4 GiB bf16 tensor, 8 GiB fp32).
  So it is the **smallest** shape with one tensor of `2^31 + 1` or more elements, every other
  input shrunk to the minimum the op accepts: `[262145, 8192]` for a token-wise op (4.0 GiB),
  an `(M, N)` output plane just above 2^31 with a short `K` for a GEMM, a score plane
  `B * H * Sq * Skv > 2^31` with small `B` and `Dh` for attention. Full forward and backward
  run on it. The harness budgets 6x the input bytes and skips a row that does not fit, with a
  note; `validate_spec` rejects a row above 13 GiB of inputs (6x = 78 GiB, an idle 80 GB card;
  13 GiB is a bf16 activation plus an fp32 stream at 2^31 elements)
  and an `edge_huge` above 2 GiB that crosses nothing. The 6x estimate is a floor, not the
  eager reference's real need: an fp32-upcast norm at 2^31 elements holds ~10 temporaries of
  8 GiB, so on a shared card the row can still run out of memory; the harness then records
  the OOM as a skip on an optional row (an error on a required one). A row skipped for lack
  of memory does not exercise the overflow it was designed to catch. Where
  2^31 is *realistic*, not an edge: attention scores (`56 heads x 6200^2` tokens),
  `(M, N)` GEMM planes of a 65k-token batch, and any `(B, S, D)` activation of a long-context
  step; the kernel rule is `.to(tl.int64)` on the program id before any stride multiply.
- Names: `user_<what>`, `rep_<what>`, `edge_<what>`; the name says which rule the row exists for.

## Token-wise ops (no GEMM): norms, RoPE, gating, residual adds, permutes

Structure: one program per row or per row group, `D <= 8192` is the contract.

The odd row varies the *token* dim (`N` of `(B, N, D)`, `S` of `(B, S, H, D)`), never `B` or
`H`. When positions come from a multi-axis grid the model's config cannot make odd
(`(t, h, w) = (4, 16, 32)`), build the odd row's tables with the model's own table function
over a grid that can (`(1, 1, S)`), and say so in the SPEC; the kernel sees one `(S, R)` table
either way.

| row | why | required |
|---|---|---|
| `user_*` (at least two: one with an odd/uneven token count, one even) | the training shapes; `rows = batch * seq` | yes |
| `user_small_rows` (`rows = 32`, user's `D`) | a grid below every SKU's SM count (A100/A800 108, H100 132, B200 148): the regression row for few-program launches (`dw` partials from few programs, masks, int64 offsets) and for the latency floor; subject to the runtime latency and baseline gates. Measured: one program per row is already at the copy roof here, no split is needed (compute-patterns.md, row width rule) | yes |
| `user_strided` (user's shapes, `strides: {x: [D + pad, 1]}`) | the layout the model actually produces. Read the user's code: `q, k, v = qkv.split(D, -1)` gives row stride `3D`; a `[..., :D]` slice gives the source width. When nothing in the code suggests a stride, use a padded row `D + 128` | yes |
| `rep_d_<n>` sweep, at least four `D`: a small even (256), a medium even that is not a multiple of 16 (1000), a medium odd (1025), a large even (8192) | block sizing and masking: `next_power_of_2`, tail masks, the register budget at `D = 8192`. Most evens should be multiples of 32; the odd one is the mask test. When the op's contract forbids a value (2-D RoPE needs `D % 4 == 0`, a paired rotation needs even `D`), take the nearest legal `D` that is still not a multiple of 16 (`1020`, `1028`) and say why in SPEC; the row tests the tail mask, not the illegal shape. For a head-wise op whose `D` is a head dim the model fixes (128), the wide rows describe a geometry the kernel will never ship: keep them optional, expect `sol_eff` near zero on them (register spills), and say in the SPEC that they test masking and correctness only; or drop `rep_d_8192` when the kernel's contract caps `D` | no |
| `edge_tail` (`rows = 3`, `D` prime such as 5 or 37) | a row that does not fill a warp and a block that does not fill a tile | no |
| `edge_huge` (`rows * D > 2^31`, the smallest such: `[262145, 8192]`, 4.0 GiB bf16) | int64 indexing; runs on an idle 80 GB card (6x = 24 GiB); `[1048576, 8192]` (16 GiB) never does | no |

The two user rows differ in the token dim (one odd count, e.g. `batch * seq = 4 * 1023`,
one even) so a rows-per-program branch is exercised with and without a tail. A training
config yields one shape, so the second row is synthesised from it (the same `D`, a token
count the model could see: a shorter sequence, a last partial batch) and stays `source:
user`, `required: true`: it gates because the tail branch is part of the user's kernel, not
because the loop runs it today. Say so in the SPEC workload prose.

Per-sample parameters (adaptive LayerNorm: `scale`, `shift` of shape `(B, D)` applied to the
`N` tokens of sample `b`) fix the rank: every row keeps `x: [B, N, D]` so the kernel can
derive `b = row // N`, and the two user rows vary `N` (odd and even), not `B`. The small-rows
row is `[2, 16, D]` (two samples, a `dscale`/`dshift` reduction over 16 tokens from few
programs); the `D` sweep keeps `B = 2`; `edge_huge` is `[1, 262145, 8192]`. `scale`/`shift`
stay fp32 (`dtypes:`), as the conditioning MLP produces them (probe it: a modulation `Linear`
under autocast emits bf16, whatever the prose says); they receive gradients, so they are not in
`grad_inputs` exclusions. A gate/activation the user marks optional (adaLN followed by SiLU) is
a `params: {act: silu}` on a second copy of the user row, `required: true` when a call site in
the user's model uses it (the final layer does), else `rep_`. The filled set for the
`f32_adaln` example (`B = 16`, `N = 4096`, `D = 1152`):

```yaml
workloads:
  - {name: user_train_even, source: user, required: true, dtype: bf16,
     shapes: {x: [16, 4096, 1152], scale: [16, 1152], shift: [16, 1152]}, dtypes: {scale: fp32, shift: fp32}, params: {eps: 1.0e-6, act: none}}
  - {name: user_train_odd, source: user, required: true, dtype: bf16,
     shapes: {x: [16, 4095, 1152], scale: [16, 1152], shift: [16, 1152]}, dtypes: {scale: fp32, shift: fp32}, params: {eps: 1.0e-6, act: none}}
  - {name: user_train_silu, source: user, required: true, dtype: bf16,
     shapes: {x: [16, 4096, 1152], scale: [16, 1152], shift: [16, 1152]}, dtypes: {scale: fp32, shift: fp32}, params: {eps: 1.0e-6, act: silu}}
  - {name: user_small_rows, source: user, required: true, dtype: bf16,
     shapes: {x: [2, 16, 1152], scale: [2, 1152], shift: [2, 1152]}, dtypes: {scale: fp32, shift: fp32}, params: {eps: 1.0e-6, act: none}}
  - {name: user_strided, source: user, required: true, dtype: bf16,
     shapes: {x: [16, 4096, 1152], scale: [16, 1152], shift: [16, 1152]}, strides: {scale: [6912, 1], shift: [6912, 1]},
     dtypes: {scale: fp32, shift: fp32}, params: {eps: 1.0e-6, act: none}}
  # the strided tensors are the *modulation* views: Block does `chunk(6, dim=-1)` on a (B, 6D)
  # Linear output (row stride 6D = 6912) and FinalLayer `chunk(2)` (row stride 2D); x itself is
  # contiguous. The probe (scaffold step 1) prints this; a padded x would exercise the wrong contract.
  - {name: rep_d_256, source: representative, required: false, dtype: bf16,
     shapes: {x: [2, 8192, 256], scale: [2, 256], shift: [2, 256]}, dtypes: {scale: fp32, shift: fp32}, params: {eps: 1.0e-6, act: none}}
  # ... rep_d_1000, rep_d_1025, rep_d_8192 alike; edge_tail [3, 1, 37] fp32; edge_huge [1, 262145, 8192]
```

Quote string-valued enum params when a second YAML reader is possible (`act: "none"`,
`act: "silu"`): PyYAML reads bare `none` as the string, other parsers as null; `1.0e-6` with the
dot as everywhere.

```yaml
workloads:
  - {name: user_train, source: user, required: true, dtype: bf16,
     shapes: {x: [8192, 2048], residual: [8192, 2048], weight: [2048]}, dtypes: {weight: fp32}, params: {eps: 1.0e-6}}
  - {name: user_train_odd, source: user, required: true, dtype: bf16,
     shapes: {x: [4092, 2048], residual: [4092, 2048], weight: [2048]}, dtypes: {weight: fp32}, params: {eps: 1.0e-6}}
  - {name: user_small_rows, source: user, required: true, dtype: bf16,
     shapes: {x: [32, 2048], residual: [32, 2048], weight: [2048]}, dtypes: {weight: fp32}, params: {eps: 1.0e-6}}
  - {name: user_strided, source: user, required: true, dtype: bf16,
     shapes: {x: [8192, 2048], residual: [8192, 2048], weight: [2048]}, strides: {x: [6144, 1]},
     dtypes: {weight: fp32}, params: {eps: 1.0e-6}}
  - {name: rep_d_256, source: representative, required: false, dtype: bf16, shapes: {x: [16384, 256], residual: [16384, 256], weight: [256]}, dtypes: {weight: fp32}, params: {eps: 1.0e-6}}
  - {name: rep_d_1000, source: representative, required: false, dtype: bf16, shapes: {x: [8192, 1000], residual: [8192, 1000], weight: [1000]}, dtypes: {weight: fp32}, params: {eps: 1.0e-6}}
  - {name: rep_d_1025, source: representative, required: false, dtype: bf16, shapes: {x: [8192, 1025], residual: [8192, 1025], weight: [1025]}, dtypes: {weight: fp32}, params: {eps: 1.0e-6}}
  - {name: rep_d_8192, source: representative, required: false, dtype: bf16, shapes: {x: [2048, 8192], residual: [2048, 8192], weight: [8192]}, dtypes: {weight: fp32}, params: {eps: 1.0e-6}}
  - {name: edge_tail, source: edge, required: false, dtype: fp32, shapes: {x: [3, 37], residual: [3, 37], weight: [37]}, params: {eps: 1.0e-6}}
  - {name: edge_huge, source: edge, required: false, dtype: bf16, shapes: {x: [262145, 8192], residual: [262145, 8192], weight: [8192]}, dtypes: {weight: fp32}, params: {eps: 1.0e-6}}
```

(`edge_huge` here is 8.0 GiB of inputs because the op has two full-size inputs; with an fp32
residual stream it is 12.0 GiB, the most any row may declare, and it runs only on an idle
card: say so in the SPEC prose. The element count, not `D`, sets the bytes, so no reshaping
helps.)

## GEMM family: linear + epilogue (bias, activation), split-K

Inputs `X (M, K)`, `W (N, K)` (torch `Linear` layout), optional `b (N)`. No residual: a
transformer adds it in the next norm (`fused_residual_*`), and cuBLASLt (the preferred backend,
`kernel_backend: nvmath`) has no post-activation addend. `act` is `gelu_tanh` unless the user's
code calls the erf form; the SPEC names which. Spend the rows on the embed dim of `X` and on
the `(N, K)` shapes a transformer uses; strides as above apply to `X`. Set SPEC `row_inputs:
[x]` (the input carrying `M`): `W` shares `X`'s leading dim whenever `N == M`, and the
`zero_rows` contract probe must not empty it. `edge_huge` for this family is an `(M, N)` output
plane just above 2^31 elements with the shortest `K` the kernel accepts (`K = 256`): `M =
131073, N = 16384` gives `2^31 + 16384` elements; the output is the only 4 GiB tensor, `X` and
`W` are 64 MB and 8 MB, so the row runs. The harness compares in chunks, so its memory-fit skip is judged
on the inputs as for every other row. A `K = 4096` version of the same plane is 4.3 GiB of
`X` + `W` on top and a 90 GiB budget, which prevents the row from running on an 80 GiB card.

| row | why | required |
|---|---|---|
| `user_*` (the user's `M, N, K`, at least one) | the training shapes | yes |
| `user_up` (`K = D`, `N = 4D`) | up-projection: wide output tile, `N` dominates | yes when the user's layer is an MLP; else `rep_` |
| `user_down` (`K = 4D`, `N = D`, small `M`) | down-projection with a long reduction and few output tiles: exercises split-K | yes when the user's layer is an MLP; else `rep_` |
| `rep_square` (`M = N = K = D`) | the attention-projection shape, the baseline tile | no |
| `user_strided` (`strides: {x: [K + pad, 1]}`) | `X` sliced from a wider activation | yes |
| `rep_m_small` / `rep_m_large` (`M = 64` and `M = 65536` with the user's `N, K`) | token dims below one tile row and far above the SM count | no |
| `edge_huge` (`M * N > 2^31`, smallest: `M = 131073, N = 16384, K = 256`) | int64 indexing of the output plane; runs on an idle card | no |

`2.66D` (the SwiGLU hidden size, e.g. `D = 4096 -> 10944`) is the concatenated `[W1; W3]` GEMM
shape (`N = 2 * 2.66D`); list it under `rep_` only when the user's model is SwiGLU. "The user's layer is an
MLP" means the fused region covers that projection: a fuse of `fc1` only makes `user_up`
required and the down-projection shape a `rep_down` row (the other projection is not this
kernel's), even though the model has an `fc2`.

```yaml
workloads:
  - {name: user_up, source: user, required: true, dtype: bf16,
     shapes: {x: [8192, 4096], weight: [16384, 4096], bias: [16384]}, params: {act: gelu_tanh}}
  - {name: user_down, source: user, required: true, dtype: bf16,
     shapes: {x: [512, 16384], weight: [4096, 16384], bias: [4096]}, params: {act: none}}
  - {name: user_strided, source: user, required: true, dtype: bf16,
     shapes: {x: [8192, 4096], weight: [16384, 4096], bias: [16384]},
     strides: {x: [4224, 1]}, params: {act: gelu_tanh}}
  - {name: rep_square, source: representative, required: false, dtype: bf16,
     shapes: {x: [8192, 4096], weight: [4096, 4096], bias: [4096]}, params: {act: none}}
  - {name: rep_m_small, source: representative, required: false, dtype: bf16,
     shapes: {x: [64, 4096], weight: [16384, 4096], bias: [16384]}, params: {act: gelu_tanh}}
  - {name: edge_huge, source: edge, required: false, dtype: bf16,
     shapes: {x: [131073, 256], weight: [16384, 256], bias: [16384]}, params: {act: gelu_tanh}}
```

## Attention: `q (B, H, Sq, Dh)`, `k, v (B, Hkv, Skv, Dh)`

Contiguous inputs only (the kernel permutes nothing; the permute is its own token-wise op).
`heads > 1` in every row; `head_dim` from `{64, 96, 128, 192, 256, 288}`: multiples of 16,
mostly of 32, one `3 x 32` (96) or `9 x 32` (288) to exercise a tile that is not a power of two.

| row | why | required |
|---|---|---|
| `user_self_*` (two: `Sq = Skv` both small, e.g. 512, and both large, e.g. 8192) | training self-attention at the user's `H`, `Hkv`, `Dh` | yes |
| `rep_cross_qsmall` (`Sq = 128`, `Skv = 8192`) and `rep_cross_qlarge` (`Sq = 8192`, `Skv = 128`) | decode-like and encoder-like imbalance: the KV loop vs the Q grid | no |
| `rep_dh_<n>` (the user's `Dh` replaced by a neighbour from the list above) | tile shapes for another head size | no |
| `edge_huge` (`B * H * Sq * Skv > 2^31` in the score matrix, smallest: `B = 1, H = 8, S = 16384 + 128, Dh = 64`) | int64 indexing of the logical scores and of the `lse`/`m` rows; the inputs are small (`q`, `k`, `v` 16 MB each), so it always runs. Realistic, not only an edge: 56 heads x 6200 tokens already crosses | no |

```yaml
workloads:
  - {name: user_self_small, source: user, required: true, dtype: bf16,
     shapes: {q: [8, 32, 512, 128], k: [8, 8, 512, 128], v: [8, 8, 512, 128]}, params: {causal: true}}
  - {name: user_self_large, source: user, required: true, dtype: bf16,
     shapes: {q: [1, 32, 8192, 128], k: [1, 8, 8192, 128], v: [1, 8, 8192, 128]}, params: {causal: true}}
  - {name: rep_cross_qsmall, source: representative, required: false, dtype: bf16,
     shapes: {q: [4, 32, 128, 128], k: [4, 8, 8192, 128], v: [4, 8, 8192, 128]}, params: {causal: false}}
  - {name: rep_cross_qlarge, source: representative, required: false, dtype: bf16,
     shapes: {q: [4, 32, 8192, 128], k: [4, 8, 128, 128], v: [4, 8, 128, 128]}, params: {causal: false}}
  - {name: rep_dh_96, source: representative, required: false, dtype: bf16,
     shapes: {q: [8, 32, 512, 96], k: [8, 8, 512, 96], v: [8, 8, 512, 96]}, params: {causal: true}}
  - {name: edge_huge, source: edge, required: false, dtype: bf16,
     shapes: {q: [1, 8, 16512, 64], k: [1, 8, 16512, 64], v: [1, 8, 16512, 64]}, params: {causal: false}}
```

Non-causal attention (diffusion transformers, encoders, MiniMax-H3's packed multimodal
sequence) needs `causal: false` on the user rows and one long row: `S` is the packed token
count (tens of thousands for video), so `user_self_large` is the user's real `S`, not 8192,
and its score plane crosses 2^31 at ordinary head counts (`H = 56`: `S > 6200`).

**Block-causal attention** (video generation: the sequence is a run of frame chunks of
`block_size` tokens; a token attends to every token of its own chunk and of all earlier
chunks) carries `block_size` in `params` and needs three things the causal rows do not:

- rows where `S` is a multiple of `block_size` (the model's case) *and* one where it is not
  (a partial last chunk, e.g. `S = 5 * block_size + 37`), since the mask is `q_idx // block_size
  >= k_idx // block_size` and the last chunk is where an off-by-one hides;
- a `block_size` that is not a multiple of the kernel's K,V tile (e.g. 1560 = 30 x 52 latent
  tokens per frame) next to one that is (e.g. 1024): the first makes every chunk boundary
  straddle a tile and exercises the per-element select, the second lets the kernel skip it;
- `block_size >= S` in one row (one chunk: dense attention) and `block_size = 1` in another
  (plain causal), the two degenerate ends the same kernel must serve.

The roofline counts the attended area, `S * (S + block_size) / 2` pairs per head, not the
full plane; the eager reference builds the `(S, S)` boolean mask once per shape from indices
(`(arange(S) // block_size)[:, None] >= (arange(S) // block_size)[None, :]`) and the scaffold's
`make_inputs` must not pass that mask as an input: it is a function of `S` and `block_size`.

```yaml
# block-causal video attention (H = 56, Dh = 128, chunks of 1560 tokens = one 30 x 52 latent frame)
workloads:
  - {name: user_video_short, source: user, required: true, dtype: bf16,
     shapes: {q: [1, 56, 6240, 128], k: [1, 56, 6240, 128], v: [1, 56, 6240, 128]}, params: {block_size: 1560}}   # 4 frames
  - {name: user_video_long, source: user, required: true, dtype: bf16,
     shapes: {q: [1, 56, 21840, 128], k: [1, 56, 21840, 128], v: [1, 56, 21840, 128]}, params: {block_size: 1560}}  # 14 frames; score plane 2.7e10 > 2^31
  - {name: rep_partial_chunk, source: representative, required: false, dtype: bf16,
     shapes: {q: [2, 8, 7837, 128], k: [2, 8, 7837, 128], v: [2, 8, 7837, 128]}, params: {block_size: 1560}}       # 5 chunks + 37 tokens
  - {name: rep_tile_aligned, source: representative, required: false, dtype: bf16,
     shapes: {q: [2, 8, 8192, 128], k: [2, 8, 8192, 128], v: [2, 8, 8192, 128]}, params: {block_size: 1024}}
  - {name: edge_one_chunk, source: edge, required: false, dtype: bf16,
     shapes: {q: [2, 8, 2048, 128], k: [2, 8, 2048, 128], v: [2, 8, 2048, 128]}, params: {block_size: 4096}}       # dense
  - {name: edge_causal, source: edge, required: false, dtype: bf16,
     shapes: {q: [2, 8, 2048, 128], k: [2, 8, 2048, 128], v: [2, 8, 2048, 128]}, params: {block_size: 1}}          # plain causal
```

The eager reference at `S = 21840, H = 56` materialises a 107 GB fp32 score plane: `_eager.py`
must loop over heads (`B*S*S*4 = 1.9 GB` per head) or over query chunks, and the verifier
should expect the eager side of `user_video_long` to take seconds. `F.scaled_dot_product_attention`
with the boolean mask is the honest baseline for the speedup column (it dispatches to the
memory-efficient kernel, not flash, when a mask tensor is given: expect ~2x over it, not 20x).

## Modes (every class)

Each workload is run in every mode the op supports; no extra rows are needed.

| mode | what runs | SPEC | gate |
|---|---|---|---|
| training | forward with aux written (`fwd`), backward reading it (`bwd`) | `backward: fused` | always |
| inference | forward under `no_grad`, nothing saved (`infer`) | implicit | always |
| recompute | forward without aux, backward recomputing it (`bwd_recompute`) | `recompute: {available: true, default: <user's choice>, why: ...}` | numerics always; performance when `default: true` |

Recompute is offered at the M1 gate when the aux bytes are at least 10% of the activation
bytes on the user workload; `why` records the ratio and the overhead:

```yaml
# rmsnorm: rstd is 1/2048 of the activation bytes -> below 10%, not offered
recompute: {available: false, default: false, why: null}
# gemm + gelu: the saved pre-activation Z is as large as the output -> offered, user decides default
recompute:
  available: true
  default: false
  why: "Z is 2*M*N bytes = 50% of the activations; recomputing it re-reads X and W (+bytes of one GEMM operand pass, +2MNK FLOPs) in the backward"
```

"Activations" in that ratio are the storage-dtype tensors autograd keeps alive for this op's
backward when the aux is saved: the inputs it pins (`x`, `residual`), the user-visible outputs
it needs (`y`), and the aux itself; parameters (`weight`, `bias`) are excluded because they
exist regardless. Say the definition in the `why` when the set is not obvious
(`Z / (x + residual + y + Z)`).
