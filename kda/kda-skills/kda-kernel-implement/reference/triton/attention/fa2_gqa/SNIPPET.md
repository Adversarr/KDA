# FlashAttention-2, full attention, grouped-query (`flash_attention_2`, Triton)

Files: `kernel.py`, `reference.py` (eager with `repeat_interleave`d K,V; `sdpa(enable_gqa=True)`
as the baseline). The same kernel structure as [`fa2_causal`](../fa2_causal/SNIPPET.md)
(read that page first: log2-domain softmax, operand layouts, tile sweep, int64 rule all
carry over); this page covers what changes when the mask goes and the heads are grouped.
TileLang twin: [`tilelang/attention/fa2_gqa`](../../../tilelang/attention/fa2_gqa/SNIPPET.md).

## Math

```
q, o: (B, H, S_q, D)      k, v: (B, H_kv, S_kv, D)      H % H_kv == 0,  GROUP = H / H_kv
query head h reads K,V head h // GROUP                    (repeat_interleave order, as in Llama / Qwen)
o = softmax(q k^T * scale) v   over all S_kv keys; S_q and S_kv independent (cross-attention shapes run)
lse: (B, H, S_q) fp32, log2 domain
```

`H_kv == H` is plain multi-head attention (the H3 dense non-causal case). The reference expands
K,V with `repeat_interleave(GROUP, dim=1)`; the kernels never do, they index the K,V head.

## What differs from the causal kernel

- **No diagonal; the only masked block is the `S_kv` tail.** The K,V loop is
  `[0, S_kv // BLOCK_N * BLOCK_N)` unmasked plus one masked block. FLOPs are the full
  `B*H*S_q*S_kv` rectangle: `attention_flops` takes both lengths.
- **dK/dV over the group, in one program.** The dK/dV program owns one `(b, h_kv, K,V block)`
  and loops `for g in range(GROUP)` over the query heads that read it, and inside over all
  `S_q` blocks: the `GROUP`-way reduction happens in the fp32 accumulator of one program, so
  it is deterministic and the K,V tile is loaded once for the group. The grid is
  `(cdiv(S_kv, BLOCK_N1), B * H_kv)`: `GROUP`x fewer programs than the dQ kernel, each `GROUP`x
  longer, which is fine at `B * H_kv * S_kv / 64 >= 2 * SMs` (`B = 1, H_kv = 8, S_kv = 8192`
  gives 1024 programs on 108 SMs). Below that, e.g. `H_kv = 1` at short `S_kv`, split the
  group across programs and reduce the partials in a second kernel; do not `tl.atomic_add`.
- `GROUP` is a runtime integer (a `range` bound), so there is one compiled variant per
  `(dtype, D, tile)` however the heads are grouped; `lse`/`delta` rows are indexed with
  `(b * H + h) * S_q` from inside the loop.
- Scale `dk`, `dv` tolerance by `sqrt(GROUP)` (they sum `GROUP` per-head
  contributions the eager chain rounds separately), matching the verifier's `reduced_over`
  rule for reductions.

## Measured (A800, bf16, `torch.profiler` device time; full rectangle FLOPs)

| shape | fwd sdpa | fwd triton | bwd sdpa | bwd triton | fwd eager | bwd eager |
|---|---|---|---|---|---|---|
| B8 H32/8 S512 D128 | 0.246 ms | 0.323 (0.76x) | 0.940 | 0.901 (**1.04x**) | 1.77 | 2.57 |
| B1 H32/8 S8192 D128 | 5.25 | 6.00 (0.87x), 183 TFLOP/s | 14.85 | 21.28 (0.70x), 129 TFLOP/s | 49.9 | 70.0 |
| B1 H56/56 S4096 D128 (H3 dense) | 2.33 | 2.73 (0.85x), 176 TFLOP/s | 6.84 | 9.33 (0.73x) | 21.0 | 31.3 |
| B4 H32/8 Sq128 Skv8192 D128 (decode-like) | 0.555 | 0.635 (0.87x) | 1.88 | 1.92 (0.98x) | 4.32 | 6.17 |

Same `select_config` table as the causal kernel; the forward is 0.85-0.87x of torch's FA2
and the backward 0.70-1.04x, 8-10x over eager. 183 TFLOP/s forward is 59% of the datasheet
peak, ~0.85 of cuBLAS on same-size GEMMs.

## Fusing into a user op

- **KV cache / decode** (`S_q` small, `S_kv` long): the grid has `cdiv(S_q, 128) * B * H`
  programs, too few at `S_q <= 128, B * H < 2 * SMs`; the standard fix is split-KV
  (flash-decoding): cut `S_kv` across a third grid axis, store per-split `(o, m, l)` and
  merge in a second kernel. Not in this snippet.
- **Sliding window / block-sparse**: bound the K,V loop per Q tile (`lo = max(0, start_m *
  BLOCK_M - window)`, `hi = ...`) and mask only the straddling blocks, exactly as the causal
  kernel does for its diagonal; the block-causal recipe is on the causal page.
- **Additive bias / ALiBi**: add the bias tile to `s` before the max (`s += bias`; in the
  log2 domain multiply the bias by `log2 e`) and to the recomputed `sT`/`s` in both backward
  kernels; a learned bias gets `d bias = ds` reduced over the batch, which is a
  separate reduction kernel.
- **QK-norm / RoPE** are token-wise and stay a separate kernel in front of this one
  (`fusion-exemplar/rmsnorm_rope_permute`).
