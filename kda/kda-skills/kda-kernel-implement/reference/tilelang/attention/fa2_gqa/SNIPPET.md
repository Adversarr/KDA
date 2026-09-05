# FlashAttention-2, full attention, grouped-query (`flash_attention_2`, TileLang)

Files: `kernel.py`, `reference.py`: same contract and workloads as the Triton
twin [`triton/attention/fa2_gqa`](../../../triton/attention/fa2_gqa/SNIPPET.md) (math, group
semantics, the decode/window/bias fusion notes). The TileLang mechanics, and every pitfall met
on the way, are on the causal page [`fa2_causal`](../fa2_causal/SNIPPET.md); this page lists
what the GQA/non-causal variant changes and its numbers.

## What differs from the causal TileLang kernel

- `q, o: (B, H, S_q, D)`, `k, v: (B, H_kv, S_kv, D)`; two `T.const` groups
  (`"B, H, S_q, D"` and `"H_kv, S_kv"`) so `S_q != S_kv` compiles (cross-attention, decode-like
  shapes run; `D` a multiple of 16).
- `GROUP = H // H_kv` is a Python int argument (compile-time; one variant per ratio). The
  forward and dQ kernels index `k[bz, by // GROUP, ...]`.
- **No diagonal, one masked block.** The K,V loop covers `ceildiv(S_kv, block_N)` blocks; the
  accumulator fill is `T.clear` except on the tail block, where the zero-filled K rows must be
  *absent*, not score 0: `acc_s[i, j] = T.if_then_else(n0 + j < S_kv, 0, -inf)`. (The causal
  kernel got this for free from `j > i`.) The dQ kernel selects the tail keys out of `p`.
- **Group loop inside the dK/dV program**: grid `(ceildiv(S_kv, block_N), H_kv, B)`, then
  `for g in T.serial(GROUP): h = by * GROUP + g; for mb in T.Pipelined(ceildiv(S_q, block_M),
  num_stages=1): ...` accumulating into the same `dk_f`, `dv_f` fragments. Deterministic, K,V
  tile loaded once per group; `GROUP`x fewer, longer programs than the dQ kernel (fine at
  `B * H_kv * S_kv / 64 >= 2 * SMs`; below that split the group and reduce in a second kernel,
  never `T.atomic_add`).
- Same `select_config` as the causal kernel: forward `128 x 128`, 256 threads, one stage;
  backward `64 x 64`, 128 threads, one stage (two stages race on tail tiles and are slower).

## Measured (A800, bf16, `torch.profiler` device time; full rectangle FLOPs)

| shape | fwd sdpa | fwd tilelang | bwd sdpa | bwd tilelang | Triton twin fwd / bwd |
|---|---|---|---|---|---|
| B8 H32/8 S512 D128 | 0.204 ms | 0.270 (0.76x) | 0.931 | 0.779 (**1.19x**) | 0.323 / 0.901 |
| B1 H32/8 S8192 D128 | 5.24 | 5.59 (**0.94x**), 197 TFLOP/s | 14.71 | 19.35 (0.76x), 142 TFLOP/s | 6.00 / 21.28 |
| B1 H56/56 S4096 D128 (H3 dense) | 2.33 | 2.56 (0.91x), 188 TFLOP/s | 6.84 | 8.52 (0.80x) | 2.73 / 9.33 |
| B4 H32/8 Sq128 Skv8192 D128 (decode-like) | 0.555 | 0.591 (0.94x) | 1.88 | 1.70 (**1.11x**) | 0.635 / 1.92 |

The fastest of the four attention references: 7% ahead of the Triton GQA kernel on the long
forward and 9% on its backward, 0.94x of the FA2 kernel torch ships at 63% of the datasheet
peak. For `flash_attention_2` without a causal mask this is the backend to start from; with
the mask the two backends tie (causal page).
