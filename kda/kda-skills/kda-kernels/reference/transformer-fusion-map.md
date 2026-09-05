# Where the fusions are in a transformer, and how to plan them for a whole model

Read this after discovery identifies transformer-like blocks in a whole-model request: the user hands you a repository
and does not know which operations belong together. This document gives the map (what a
transformer block is made of, which neighbours fuse, which never do, what the libraries
already do at full speed), then the procedure that turns a repository into a ranked plan of
kernels, each of which then goes through the ordinary M0-M6 workflow of `SKILL.md`. Every
fusion named here has a pattern row in
`.agents/skills/kda-kernel-implement/reference/common/compute-patterns.md`, and an example in
the KDA repository's `examples/<name>/` (a `TASK.md` written as a user would, a small
`user_repo/`, and for the marked ones a golden kernel with measured A800 numbers); the example
names below are those directories, not files in the user's repo.

## 1. The principle

A transformer step spends its time in three places:

1. **GEMMs** (the projections, the MLP, the output head): tensor-core bound. cuBLAS runs them
   at 70-85% of the datasheet peak and nothing you write in Triton or TileLang beats it on
   sm80 (`findings.md`). They are never fused *into*; they take an **epilogue** (bias,
   activation, a saved pre-activation) and that epilogue is fused by the library too
   (nvmath-python / cuBLASLt) or by one streaming kernel after `torch.matmul`.
2. **Attention**: tensor-core bound, and `F.scaled_dot_product_attention` (SDPA) is already
   FlashAttention-2 for the dense, causal and GQA cases. A kernel is written only when the
   *mask has structure SDPA cannot express* (block-causal frames, tile windows, key padding
   per batch) and forces torch onto the masked memory-efficient kernel, which visits the
   whole plane: then a block-sparse FA2 wins 5-20x (the three attention examples).
3. **Everything between them**: norms, residual adds, RoPE, activations, gates, casts,
   permutes, modulation. Each is memory-bound, each eager op reads and writes the whole
   activation once more, and PyTorch runs a residual add + RMSNorm as six to eight kernels
   moving ~45 bytes per element where one kernel moves 12. **This is where fusion lives**:
   between two GEMM boundaries, everything that touches the same activation is one kernel.

So the map is short: *the GEMMs and the attention are the skeleton; every stretch of
element-wise and row-wise work between two of them is one fused kernel; the residual add
belongs to the norm that follows it; the activation belongs to the GEMM that precedes it.*

Two consequences:

- **The residual is not a GEMM epilogue.** `act(X W^T + b) + R` looks fusable and is never
  what a transformer computes: the branch output is added to the residual *stream*, which is
  then normalised for the next branch. That add is the first line of the next norm kernel
  (`residual_then_norm`), where it comes for free (the norm has to read the sum anyway) and
  where the fp32 stream can be kept in fp32. A `GEMM + R` kernel would read `R` once more and
  round the stream to bf16.
- **Prefer fusing a permute or cast with its producer/consumer when the supported boundary permits it.** They are the *store side*
  (or load side) of the neighbouring kernel: the q/k norm + RoPE kernel writes `(B, H, S, D)`
  directly from `(B, S, H, D)`; the norm kernel reads the fp32 stream and writes bf16.

If `torch.compile` is already on in the repository, be honest at the plan gate: it fuses most
of the *forward* memory-bound chains nearly as well as a hand kernel (residual + RMSNorm:
compile 0.034 ms, golden 0.033 ms). What it does not do, and where a kernel still pays, is
(a) structured attention, (b) fusions whose adjoint it splits or whose permute it
materialises (q/k norm + RoPE + permute: compile 0.668 ms, golden 0.218 ms; backward 1.23 vs
0.36), (c) backward passes generally (residual + RMSNorm bwd 0.074 vs 0.056), (d) the saved-
tensor footprint (a fused norm saves `rstd`, the eager chain saves fp32 intermediates), which
is the binding constraint on long video sequences.

## 2. The block, operation by operation

A pre-norm decoder block as the examples' `minilm`, `minidit`, `minih3`, `minista` and
`minivggt` write it. `x` is the residual stream (fp32 in `minilm`/`minivggt`, bf16 in some
repos); everything else is bf16 under autocast. `K` marks a fused kernel KDA writes, `L` a
library call that stays, `-` an op that disappears into a neighbour.

| # | operation in the user's code | class | decision | fused with | pattern | example |
|---|---|---|---|---|---|---|
| 1 | `x = x + branch_out` (from the previous block or the embedding) | residual add | `-` | folded into #2 | | |
| 2 | `h = norm(x)` (RMSNorm / LayerNorm, affine or not) | row-wise norm | **K1** `residual_then_norm`: reads `x, branch_out` once, writes the new fp32 stream and the bf16 `h` | #1, the cast, DiT modulation (#2a), the gate of the previous branch (#2b) | `rowwise_onepass` | `fused_residual_rmsnorm` (golden), `fused_residual_layernorm`, `vggt_layerscale_residual_layernorm` |
| 2a | `h = h * (1 + scale[b]) + shift[b]` (DiT adaLN modulation, per-sample `(B, D)`) | modulation | `-` | inside K1: the normalised row is scaled and shifted before the store; `dscale`, `dshift` are reductions over the sample's tokens in the backward | `rowwise_onepass` | `f32_adaln` |
| 2b | `x = x + gate[b] * y` (DiT gating) or `x + gamma * y` (LayerScale) | gated residual | `-` | inside K1 as a third input: `x_new = x + g * y; h = norm(x_new)` | `rowwise_onepass` | `vggt_layerscale_residual_layernorm`, `f32_adaln` (gate row) |
| 3 | `qkv = h @ Wqkv^T (+ b)` | GEMM | **L** `torch.matmul` / `F.linear` | nothing: a bias is a cuBLASLt epilogue if you want one, and it is usually absent | `gemm_tensorcore` | |
| 4 | `q, k, v = split(qkv)`; `q = q_norm(q)`, `k = k_norm(k)` (per-head QK-norm, RMS or LN, shared or per-head weight) | per-head norm | **K2** `qk_norm_rope_permute`: one kernel over the `(B, S, 3, H, d)` GEMM output | #5, #6, the split (a strided read) | `rowwise_onepass` (row = one head) + `rope_pairs` + `permute_copy` | `qk_multihead_rmsnorm_rope_permute` (golden), `h3_qk_norm_mmrope`, `vggt_qkv_layernorm_rope2d` |
| 5 | `q, k = rope(q, k, cos, sin)` (half or interleaved pairing, partial rotation, 2-D / 3-D tables, in-kernel `cos/sin` from integer positions) | rotation | `-` | inside K2 | `rope_pairs` | same |
| 6 | `q.transpose(1, 2)` (to `(B, H, S, d)`), `.contiguous()` / `.reshape` | permute | `-` | the *store* of K2 when a copy really happens; `v` is re-laid out by the same kernel so attention gets three tensors in one layout. Check first whether it is a copy at all: `transpose` is a view, and SDPA / the FA2 references accept any strides with a unit head-dim stride, so the copy is often the `.contiguous()` or `.reshape` the user added "to be safe" and disappears by deleting it | `permute_copy` | same |
| 7 | `o = attention(q, k, v, mask)`: causal, dense, GQA (`H_kv < H`) | attention | **L** SDPA (`is_causal=True`, `enable_gqa=True`); no kernel | | `flash_attention_2` | reference `fa2_causal`, `fa2_gqa` (what to expect from a kernel: 0.85-0.94x of SDPA) |
| 7' | the same with a *structured* mask: block-causal by frame, sliding tile window, per-batch key padding, prefix-LM, document masks in a packed batch | attention | **K3** block-sparse FA2: mask lives in the loop bounds / a host-built block list, never in an `(S, S)` tensor | the output permute back to `(B, S, H, d)` is its store side | `flash_attention_2` | `h3_block_causal_attention`, `sliding_tile_attention` (golden), `vggt_padded_attention` |
| 8 | `o.transpose(1, 2).reshape(B, S, H*d)` | permute | `-` | a view when K3 stores `(B, S, H, d)`; else the store side of K3 / of SDPA's output (SDPA returns `(B, H, S, d)`; the `.transpose().reshape()` is one copy kernel you can only remove by owning the attention store) | `permute_copy` | |
| 9 | `y = o @ Wo^T` | GEMM | **L** | | `gemm_tensorcore` | |
| 10 | `x = x + y`; `h = norm2(x)` | residual + norm | **K1** again (second instance per block) | | `rowwise_onepass` | as #2 |
| 11 | `u = h @ W1^T (+ b1)`; `a = act(u)` | GEMM + activation | **L** with epilogue. cuBLASLt fuses exactly bias, ReLU and **tanh-GELU** (`approximate="tanh"`, with the saved pre-activation as `_AUX`): nvmath-python `GELU_AUX_BIAS` / `RELU_BIAS`, no kernel. Any other activation (SiLU, erf-GELU, GELU-tanh when nvmath is absent) is `torch.matmul` + one streaming activation kernel with its adjoint; never a hand GEMM | | `gemm_tensorcore` | `fused_gemm_epilogue` (golden = nvmath), `vggt_mlp_fc1_gelu` (erf GELU: `matmul` + kernel) |
| 11' | `a = silu(h @ W1^T) * (h @ W3^T)` (SwiGLU, Llama-style) | dual GEMM + gate | **L + K4**: one concatenated `[W1; W3]` GEMM (`h` read once), then one streaming gate kernel `silu(g) * u` with its adjoint | | `swiglu_dual_gemm` (which is *not* a dual-GEMM kernel) | `compute-patterns.md` row |
| 12 | `y = a @ W2^T` | GEMM | **L** | | `gemm_tensorcore` | |
| 13 | dropout on a branch output (`p > 0`) | mask + scale | `-` | inside the K1 that consumes the branch (`x + dropout(y)`): generate the mask from a seed in-kernel or read a saved bit-mask; never a standalone dropout kernel | `elementwise_stream` inside `rowwise_onepass` | (none yet; state it in the SPEC Math) |
| 14 | final `norm(x)`; `logits = h @ W_vocab^T`; `loss = cross_entropy(logits, targets)` | norm, GEMM, softmax-CE | K1 for the norm (with the last block's residual add as its first line; the stream output is not needed, only `h`); **L** for the GEMM; the CE is a `rowwise_onepass` over the vocab (`D = V` up to 128k: two-pass or chunked). Fusing GEMM + CE (never materialising `(B*S, V)` logits) is a *memory* win worth a plan line when `V >= 64k`; it is a chunked library GEMM + one CE kernel, not a GEMM kernel | | `rowwise_onepass` | (none yet) |
| 15 | token / patch embedding gather, positional-embedding add, the optimizer step, the data pipeline | | **out of scope for a kernel**: the gather is one library op, the optimizer wants a fused implementation from torch (`fused=True`) or apex, not a KDA kernel; mention them in the plan as "not a fusion target" | | | |

Reading the table: per block, a well-fused transformer runs **two K1 launches, one K2, one
attention (SDPA or K3), one MLP activation (an nvmath epilogue or K4), and four to six library
GEMMs** (fused or separate q/k/v, concatenated or separate SwiGLU projections).
Everything else in the eager trace (the `.float()`, the `pow`/`mean`/`rsqrt` of a norm, the
`transpose().contiguous()`, the `masked_fill`, the `softmax` over a materialised score plane)
disappears into one of those.

## 3. Variants and how they change the map

- **Decoder LM (Llama / Qwen / Mistral style; `minilm`)**: RMSNorm, GQA, RoPE (half or
  interleaved pairing), SwiGLU or a plain SiLU/GELU MLP, fp32 residual stream. Map: K1 x2, K2
  (per-head RMSNorm on q and k, RoPE, permute; `v` permuted by the same kernel), SDPA causal
  GQA, `[W1; W3]` GEMM + K4 (SwiGLU) or `matmul` + activation kernel / nvmath epilogue, final
  norm + head + CE. Nothing to write for attention.
- **DiT / video generation (`minidit`, `minih3`, `minista`; PixArt, HunyuanVideo, MiniMax-H3)**:
  adaLN instead of a plain norm (`(1 + scale) * LN(x) + shift`, `(B, D)` conditioning in fp32,
  computed by an MLP outside autocast), gating on every branch (`x + gate * y`), QK-norm with
  multi-axis RoPE (3-D MM-RoPE over `(t, h, w)`, partial rotation of 96/128 channels), attention
  over tens of thousands of tokens with a **structured** mask (block-causal by frame, sliding
  tile window with text tokens, or dense non-causal with text appended), GELU-tanh MLP. Map:
  K1 = adaLN with the gated residual of the previous branch as its inputs (`x, gate, y, scale,
  shift`; backward has `dscale`, `dshift`, `dgate` reductions over the sample's tokens), K2 =
  qk norm + MM-RoPE + permute, K3 = block-sparse FA2 when the mask is structured (dense
  non-causal is SDPA), MLP = nvmath `GELU_AUX_BIAS`. The attention kernel is where most of the
  step goes at these lengths; the saved-tensor memory of the eager attention is what makes
  training impossible before it.
- **ViT / multi-view (`minivggt`; DINOv2, VGGT)**: LayerNorm with weight and bias, LayerScale
  (`x + gamma * y`, `gamma` a learned `(D,)`), per-batch padding of tokens (a `(B, N)` validity
  mask that zeroes padded rows and masks their keys), 2-D RoPE from integer positions with
  cos/sin computed in-kernel, exact erf GELU. Map: K1 = LayerScale + residual + LayerNorm with
  the row mask, K2 = qkv LayerNorm + 2-D RoPE + relayout (one `dqkv` gradient), K3 = padded
  attention (SDPA with a boolean mask is the baseline; a kernel that skips fully-padded key
  blocks wins when padding is substantial, otherwise SDPA is enough: measure), MLP = `matmul` +
  erf-GELU kernel (erf is not a cuBLASLt epilogue). Two attention shapes (frame-local and
  global) share one kernel.
- **Encoder / bidirectional**: dense non-causal attention is SDPA; nothing changes elsewhere.
- **Cross-attention**: K2 splits into a q-side kernel per layer and a k/v-side kernel run once
  per encoder output; attention is SDPA (`S_q != S_kv` is fine) unless masked by structure.
- **Parallel blocks** (attention and MLP from the same norm, GPT-J / PaLM style): one K1 feeds
  both branches, and the residual add has two branch inputs (`x + y_attn + y_mlp`): still one K1.
- **Post-norm** (`x = norm(x + y)`): K1 with the norm output *as* the new stream (one output,
  bf16 or fp32 as the model keeps it); no separate `h`.
- **Sandwich / sub-LN** (`x = x + norm(y)`): the norm runs on the branch output and the add
  follows: still one kernel (`norm_then_residual`), two outputs if the model keeps `norm(y)`.
- **MoE MLP**: the router softmax/top-k is tiny, the expert GEMMs are grouped GEMMs (a
  library problem: `torch._grouped_mm`, cuBLASLt grouped, or a Triton grouped GEMM which is
  the one place a hand GEMM is justified because cuBLAS has no ragged batching on sm80). Out of
  the Level-1 map; say so.
- **Inference (KV cache, decode)**: a different map (paged attention, small-`M` GEMMs,
  quantisation). Not this document.

### The mask decides whether attention is a kernel

| mask in the user's code | torch path | KDA decision |
|---|---|---|
| none, `is_causal=True`, GQA via `enable_gqa` | SDPA flash kernel | **no kernel**; the eager `softmax(q k^T) v` in the repo is replaced by SDPA at integration if it is not already (say so in the plan; it is a one-line edit, not a kernel) |
| a HF-style 4-D additive mask `(B, 1, S, S)` of `0 / -inf` that is *plain causal* (no padding), or K/V heads copied `n_rep` times by a `repeat_kv` | SDPA memory-efficient kernel reading a mask, plus a K/V copy | **no kernel**: `is_causal=True` and `enable_gqa=True` (or the FA2 GQA reference's head mapping) remove both; this is the most common "attention is slow" finding in HF-derived code and costs one line |
| a boolean `(S, S)` / `(B, 1, S, S)` tensor with **structure** (frames, tiles, windows, padding, documents) | SDPA falls back to the memory-efficient kernel: reads the mask, visits the whole plane | **K3** block-sparse FA2 (compute-patterns.md "Block-sparse attention"); baseline = SDPA with the mask, and `torch.nn.attention.flex_attention` with a `create_block_mask` when the torch version has it (measure it: if flex is within 10% of the roof there is no kernel to write) |
| an additive float bias (ALiBi, relative position) | SDPA memory-efficient with the bias | a kernel only when the bias is *computable* in-kernel (ALiBi slopes, a `(2S-1,)` table); a dense learned `(H, S, S)` bias stays SDPA |
| softmax in fp32 with bf16 inputs, `.float()` on the scores | SDPA does this internally | not a reason for a kernel |

## 4. What not to fuse

- **GEMM + residual** (`act(XW^T + b) + R`): not a transformer pattern.
  The residual goes to the next norm.
- **A hand-written GEMM** (Triton or TileLang) where `torch.matmul` / nvmath does the job: 0.8x
  of cuBLAS in reference measurements (`findings.md`). The only justified hand GEMMs
  are grouped/ragged ones.
- **A FlashAttention kernel for a plain causal or dense mask**: SDPA is already one and is
  faster than any reference here by 6-18%.
- **Materialising the token mask** for a structured attention (`repeat_interleave` of a tile
  mask to `(S, S)`, `masked_fill` on scores): the mask belongs in the loop bounds or a block
  list (compute-patterns.md, "Block-sparse attention"); the token mask at HunyuanVideo size is
  320 GB.
- **One megakernel across a GEMM boundary** (norm + GEMM, GEMM + norm): the GEMM's tensor-core
  tiling and the norm's row reduction want different decompositions; the library GEMM plus
  two streaming kernels is faster and simpler.
- **Splitting a norm into two launches for few rows** ("only 32 rows, the SMs are idle"): one
  program per row is at the copy roof to `D = 32768`; the split costs a launch and a read
  (compute-patterns.md, "Row width rule").
- **Fusing the optimizer or the embedding gather** into a block kernel: different tensors,
  different lifetimes; use torch's fused optimizer.
- **A kernel for something under the latency floor**: an op whose roof is under 10 us gains
  nothing measurable; the verdict does not gate `sol_eff` there and neither should the plan.

## 5. The procedure for "optimize this model"

Do this inline before M0 for a matching model. It produces the internal PLAN checkpoint;
follow the orchestrator's authorization policy rather than adding an approval pause.

### 5.1 Find the block

Read the model file(s) the training script instantiates (`train*.py` -> the `nn.Module`), and
write down the forward of one block as the op list of §2, in the user's names: file:function
for each region, the tensors with their dtypes and shapes **at the training config** (batch,
sequence, hidden, heads, head dim, layers, the autocast policy, where `.float()` casts sit).
Note which norm (RMS/LN, affine or not), which RoPE pairing, which activation (`approximate=`),
whether the residual stream is fp32, what the attention mask is and how it is built, whether
`torch.compile` is already applied. The intake probe of `kda-kernel-scaffold` step 1 does this
for one region; here it is done once for the block, and the notes feed every later SPEC.

Estimate at the configuration the user **trains** with, not the smoke configuration if they
differ (a 2-layer smoke model has the same map and very different percentages); say which you
used in the plan. Signatures that locate the regions in code you have not seen before:

| region | grep for |
|---|---|
| K1 residual + norm | `rsqrt(`, `.pow(2).mean(`, `F.rms_norm`, `F.layer_norm`, `nn.LayerNorm`, `nn.RMSNorm`, `x.float()` next to a norm, `residual +`, `hidden_states = residual + ` |
| 2a/2b modulation, gating | `chunk(6`, `(1 + scale`, `shift`, `gate_msa`, `gate * `, `gamma_1 * ` (LayerScale), `modulation(` |
| K2 q/k norm + RoPE + permute | `q_norm`, `k_norm`, `rotate_half`, `apply_rotary`, `cos * ` and `sin * ` on q/k, `view(B, S, H, d).transpose(1, 2)`, `.contiguous()` after a transpose, `repeat_kv` |
| attention (which mask?) | `scaled_dot_product_attention` (which arguments), `softmax(` on `q @ k.transpose`, `masked_fill`, `attention_mask`, `causal_mask`, `tril`, `block_mask`, `flex_attention`, `float("-inf")` |
| MLP activation | `F.gelu(` (with or without `approximate="tanh"`), `F.silu(`, `act_fn(`, `gate_proj` + `up_proj` (SwiGLU), `fc1`/`fc2` |
| loss | `F.cross_entropy(logits.view(-1, V)`, `lm_head(` |
| already compiled? | `torch.compile(`, `@torch.compile`, `torch._dynamo` |

### 5.2 Profile one step

Time the training step and attribute it, with the user's GPU interpreter:

```python
# kda_kernels/_scratch/profile_step.py -- run from the repo root; 3 warm-up steps, 5 profiled.
# Scratch lives inside the repo (the GPU python may run in a container that sees nothing
# else) and is deleted at hand-over; the numbers go into PLAN.md.
import torch
from torch.profiler import ProfilerActivity, profile
# build model, optimizer, one batch exactly as train_smoke.py does, then:
for _ in range(3): step()
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    for _ in range(5): step()
    torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=60))
print(f"peak memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
```

Bucket the kernel rows by name: **GEMM** (`sm80_xmma`, `cutlass`, `gemm`, `nvjet`, `cublas`),
**attention** (`fmha`, `flash`, `attention`, `mem_efficient`), **compile** (`triton_`), and the
**rest** (`elementwise_kernel`, `vectorized_elementwise`, `reduce_kernel`, `softmax`,
`index`, `copy_`, `cat`, `masked_fill`, `native_layer_norm`, `rms_norm`, ...). Divide each
bucket's total by the number of profiled steps. Then say the Amdahl sentence in the plan:
*"GEMMs are A%, attention B%, the rest C%; fusion can remove most of C and, if the attention
mask is structured, most of B; A stays."* When `C + (structured ? B : 0)` is under 15% of the
step, the honest plan is short (one or two kernels, or "turn on `torch.compile` and stop").
Record `median_step_ms` and the peak memory as the baseline for every later `E2E.md`.

### 5.3 Map the rest onto kernels and estimate each

For each region of §2 that exists in this model, one candidate row. Estimate the gain from
bytes, not from hope: eager bytes = the sum over its eager ops of (inputs read + outputs
written), fused bytes = each input read once + each output written once, both times the
number of calls per step (forward + backward: the backward of a norm chain moves 2-3x the
forward); divide by the measured copy bandwidth (`_common/bench.py`'s copy roof, ~1.5 TB/s
effective on A800) for the fused time and take the eager time from the profile rows you can
attribute (or from the golden anchors in §6, scaled by size). For a structured attention, eager time is the masked-SDPA rows of the
profile; the kernel time is the attended-area FLOPs at ~170 TFLOP/s forward and ~130 backward
(the FA2 references' rates). Also write the saved-tensor bytes each fusion removes (a fused
norm saves `rstd` instead of fp32 intermediates; a fused attention saves `lse` instead of the
score plane): for long-sequence models this is the win that lets the batch grow.

Effort ranks by how far the candidate is from a reference: `residual_then_norm`,
`qk_norm_rope_permute`, the MLP epilogue and the SwiGLU gate are a tested reference each;
adaLN with gate is a reference plus two reductions; a block-sparse attention is the FA2
reference plus host block lists.

### 5.4 Record the internal plan

```
# KDA plan: <model> (<repo>)
- config: <file>: B, S, D, H (H_kv), d, layers, dtype policy, compile on/off
- step (<date>, <gpu>): <ms> median, <GiB> peak; GEMM <a>%  attention <b>% (<mask kind>)  rest <c>%  other <d>%
- ceiling: fusion can remove <c + b'>% at most; the rest is cuBLAS.

| # | region (file:function) | ops fused | pattern | eager ms/step | est. fused ms/step | est. step gain | saved-tensor bytes | reference / example | status |
|---|---|---|---|---|---|---|---|---|---|
| 1 | model.py:Block.forward, `x + y; rmsnorm(x)` (x2 per block) | residual add, cast, RMSNorm | rowwise_onepass | 2.9 | 0.7 | 4.1% | -1.2 GiB | fused_residual_rmsnorm | pending |
| 2 | ... | | | | | | | | |

Order: 1, 2, 3 (by gain; ties by effort). Not fusion targets: <embedding, optimizer, ...>.
Not a kernel: <attention is causal -> SDPA one-liner at integration>, <compile already covers ...>.
```

Record the plan and rationale in PLAN, with authorization source/scope and any requested
review checkpoints. Present a short model-level recommendation. Continue within the user's
optimization request; ask only about unresolved model-level intent or an explicitly requested
plan review. Integration still needs its concrete review unless already authorized.

### 5.5 Run the kernels, one at a time, re-measuring between them

For each row in order: the full M0-M6 of `SKILL.md`, one `kda_kernels/<op>/` per row, the
`STATUS.md` of that op as usual. After its M6 smoke, re-profile (§5.2), update the row
(`status: done, <fused ms>, step <before> -> <after>`), and re-estimate the remaining rows
against the new step (a fusion changes what is left; the attention row may grow to 60% of the
step once the norms are gone). Stop when the next row's estimated gain is under 2% of the step,
or the user's budget is spent, and write the closing line: step before, step after, peak
memory before and after, the rows shipped, the rows declined and why. A resumed session reads
`PLAN.md` first and then the `STATUS.md` of the first row that is not `done`.

When both K2 and K3 are on the plan, decide the attention layout once, before the first SPEC:
the FA2 references read `(B, S, H, d)` through strides, so K2 need not permute at all and K3
can store `(B, S, H, d)` directly for the output projection; two permutes vanish without a
kernel writing them. With SDPA the same holds for the input side (a `transpose` view is enough;
SDPA's output is `(B, H, S, d)` and the copy back is the one that remains).

Sequencing rules of thumb: K1 first (largest byte traffic, two per block, the safest
reference); then whichever of the attention or K2 the profile says; the MLP epilogue last
when nvmath is available (an afternoon) and never when the activation is already fused by
`torch.compile` and the GEMM dominates. If the model cannot train at its real length because
the eager attention does not fit, the attention row goes first regardless of the profile.

## 6. Worked maps from the examples

| model (`user_repo/`) | K1 | K2 | attention | MLP | examples |
|---|---|---|---|---|---|
| `minilm` (Llama-style LM: RMSNorm, GQA 32/8 heads, interleaved RoPE, SiLU or GELU-tanh MLP, fp32 stream) | residual + RMSNorm (x2) | per-head RMSNorm + RoPE + permute, q and k | causal GQA -> SDPA, no kernel | GELU-tanh: nvmath epilogue; SiLU: `matmul` + kernel; a SwiGLU variant: concat GEMM + gate kernel | `fused_residual_rmsnorm`, `qk_multihead_rmsnorm_rope_permute`, `fused_gemm_epilogue` |
| `minidit` (DiT: adaLN, gating, dense attention over 4096 tokens) | adaLN with the gated residual as input (x2), final adaLN + SiLU | (no QK-norm in this model) | dense non-causal -> SDPA | GELU-tanh: nvmath | `f32_adaln` |
| `minih3` (MiniMax-H3 video: 56 heads x 128, MM-RoPE, block-causal frames) | residual + norm | per-head RMSNorm (shared weight) + partial 3-D MM-RoPE + permute | block-causal by frame, `S` to 60k -> **K3** | GELU-tanh: nvmath | `h3_qk_norm_mmrope`, `h3_block_causal_attention` |
| `minista` (HunyuanVideo STA: tiles + text, per-head windows) | residual + norm | qk norm + RoPE + permute | per-head tile mask, 5-31% density -> **K3** block-sparse (golden 0.85 ms vs 15.5 ms masked SDPA) | GELU-tanh: nvmath | `sliding_tile_attention` |
| `minivggt` (VGGT: LayerNorm + LayerScale, padded multi-view tokens, 2-D RoPE, erf GELU) | LayerScale + residual + LayerNorm with a row mask | qkv LayerNorm + 2-D RoPE + relayout, one `dqkv` | dense with per-batch key padding -> SDPA with mask as baseline, **K3** if padding is substantial | erf GELU: `matmul` + kernel (not a cuBLASLt epilogue) | `vggt_layerscale_residual_layernorm`, `vggt_qkv_layernorm_rope2d`, `vggt_padded_attention`, `vggt_mlp_fc1_gelu` |

What the goldens measured on A800 (device time, user workloads), so the plan's estimates have
anchors: residual + RMSNorm `(4096 x 1024)` fwd 0.033 ms vs eager 0.154 / compile 0.034, bwd
0.056 vs 0.348 / 0.074; per-head RMSNorm + RoPE + permute `(4 x 4096, 32/8 heads x 128)` fwd
0.218 vs 6.26 / 0.668, bwd 0.355 vs 10.2 / 1.23; GEMM + GELU-tanh epilogue `8192 x 8192 x 2048`
nvmath 1.042 vs eager 1.212 / compile 1.198 (the GEMM alone is 1.017: an epilogue is worth 15%
of the GEMM at most, which is why it is last in the order); sliding tile attention `8 heads x
10.5k tokens, 31%` fwd 0.848 vs masked SDPA 15.5 / compile 10.3, bwd 2.70 vs 20.1 / 11.4.

## 7. Beyond kernels (say it in the plan, do not do it as a kernel)

- `torch.compile` on the whole model for what is left after the kernels (the custom ops are
  registered with fakes and compose with it; the integrate skill checks this).
- Activation checkpointing or the `recompute` switch of the fused kernels when memory, not
  time, is the constraint.
- Fused optimizer (`torch.optim.AdamW(fused=True)`), CUDA graphs for small batches, the data
  loader: standard engineering, not Level-1 kernels.
- Attention with a *standard* mask that the repo still computes by hand: replace with SDPA at
  integration time, note it in `E2E.md`; it is often the largest single win and costs one line.
- Work repeated for identical inputs: RoPE `cos/sin` tables or the causal mask rebuilt in every
  layer or every step, `repeat_kv` copies, `.contiguous()` on tensors that are already
  contiguous. Hoist or delete them at integration time and list them in the plan; they are
  Python edits, not kernels.
