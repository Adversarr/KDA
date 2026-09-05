# Golden: Sliding Tile Attention (`sliding_tile_attention`)

Hand-written target for `sliding_tile_attention` in `user_repo/minista/model.py`: attention over
a video sequence in 384-token tile order plus text, masked by a per-head `(H, NT, NT)` tile mask
(window centred on the query tile, text keys visible to all, text queries see all), softmax in
fp32. Recorded A800 reference measurements are below.

## Design

`kernel.py` is the FA2 reference (`kda-kernel-implement/reference/triton/attention/fa2_causal/`:
log2-domain online softmax, deterministic two-kernel backward with `P` recomputed from `lse`,
int64 `(b, h)` base + int32 in-loop offsets) with the causal loop bounds replaced by **block
lists built on the host from the tile mask**, once per (mask, shape, block geometry):

```
any[h, i, j]   some (query in Q block i, key in K block j) pair is allowed
full[h, i, j]  every pair is allowed             -> no element mask in the kernel
idx[h, i, :]   allowed j's, full ones first;  n_full[h, i], n_any[h, i]
perm           (head, block) rows sorted by n_any descending -> program order
```

The kernel runs an unmasked loop over `idx[:n_full]` and a masked loop over `idx[n_full:n_any]`;
the masked loop rebuilds the token mask from the tile mask with one 128 x 128 gather per block
(`tile_mask[h, m // ts, n // ts] | m >= V | n >= V`, and `n < S`). With `tile_size = 384 = 3 x 128`
every 128-row video block lies inside one tile, so the only partial blocks are a K block that
crosses `S` (when `text_len % 128 != 0`); the masked loop is empty or one block long and its
cost is invisible. Partial *video* blocks (tile size not a multiple of the block, such as
200 tokens) go through the same path and stay exact. The dK/dV kernel uses the transposed lists
(per K block, the Q blocks that attend to it) and the transposed tile mask for its gather.

Three things the causal reference did not need:

- **Heavy-first program order.** The windows differ per head: in the smoke config head 0 attends
  to all 27 tiles and head 7 to one, so a natural `(Q block, head)` grid leaves the last wave to
  the dense heads. Launching the longest lists first (the `perm` table, grid `(H * NQB, B)`)
  took the forward from 1.22 to 0.85 ms and the backward from 3.53 to 2.70 ms. This is the
  single largest lever on any block-sparse attention with heterogeneous heads.
- **Fully-denied rows.** A partial block can deny a whole row (the causal diagonal never does);
  before any block has contributed, `m_new = -inf` and `exp2(m - m_new)` is NaN. The masked path
  substitutes 0 for `-inf` in the subtraction.
- **Shared memory.** The 128 x 128 x 128 forward tile pipelines 3 stages in the causal kernel;
  with indirect K,V addresses Triton keeps one more tile in flight and 3 stages need 192 KB
  (> 167 KB on A800). 2 stages; `BLOCK_N = 64` with 3 stages is 10% slower.

The mask is never materialised at token granularity. The largest mask-derived tensor is `idx`,
`H x NQB x max_nnz` int32; `max_nnz` is the full row count because text queries attend to every
key, so it is 24 x 902 x 902 = 75 MB at HunyuanVideo size (a token mask there would be 24 x
115456^2 = 320 GB). The lists are cached by mask pointer and geometry, so a training loop
builds them once. Work is proportional to the
attended area: 31% of the plane in the smoke config, 43% in the large workload below, 5-15% with the
paper's windows on real videos.

Contract: `q, k, v` are `(B, H, S, D)` with unit stride in `D`; a `(B, S, H, D)` projection viewed
as `(B, H, S, D)` (the model's fused QKV) runs without a copy. `S` need not be a block multiple;
`D <= 256`. The backward is bitwise deterministic (no atomics).
`dq, dk, dv` come back in the storage dtype;
the mask has no gradient.

## A800, user workload B1 H8 S10496 D128 bf16, 31.3% attended

Device time from `torch.profiler`, min of three interleaved rounds; the
GPU is unlocked, compare within the table. FLOP rate counts the attended pairs only (`4 D` per
pair forward, `10 D` backward). `eager` is the user's function (materialised fp32 scores), `sdpa`
is `F.scaled_dot_product_attention` with the boolean token mask (the efficient-attention kernel;
FA2 refuses a mask), `compile` is `torch.compile` of the eager function. 2026-09-04.

| | fwd ms | TFLOP/s | bwd ms | TFLOP/s |
|---|---|---|---|---|
| **golden** (block-sparse FA2, heavy-first) | **0.848** | 167 | **2.700** | 131 |
| golden, natural program order | 1.222 | 116 | 3.527 | 100 |
| sdpa + token mask | 15.55 | 9 | 20.13 | 18 |
| torch.compile (eager fn) | 10.33 | 14 | 11.42 | 31 |
| eager (materialised scores) | 35.31 | 4 | 37.50 | 9 |

Speed-up over the strongest torch baseline (sdpa with the mask): 18.3x forward, 7.5x backward;
over the user's eager code: 42x / 14x. Against the achievable roof for this row (cuBLAS on the
same-FLOP cube, 0.62 ms fwd / 1.24 ms bwd) the golden is `sol_eff` 0.73 forward and 0.46
backward, the same fractions the dense FA2 reference reaches at this head dim (the deterministic
backward recomputes `QK^T` twice, 5 GEMMs for the 2.5x FLOP count).

HunyuanVideo size (`B1 H24 S115456 D128`, tile grid 5 x 6 x 10, windows `(3,3,3)`,
`(3,5,9)`, `(5,5,9)`, 43% attended, no eager path can allocate the scores): forward 396 ms =
179 TFLOP/s (57% of peak); the lists for that shape are 75 MB and take ~170 ms to build, once.
