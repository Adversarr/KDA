# Golden: per-head RMSNorm + interleaved RoPE + permute (`qk_prep`)

Hand-written target for `qk_prep` in `user_repo/minilm/model.py` (Qwen3-style q/k
preparation in a grouped-query attention block), with recorded A800 reference measurements.

The kernel is the `fusion-exemplar/rmsnorm_rope_permute` Triton reference
(`kda-kernel-implement/reference/triton/fusion-exemplar/`) with three changes the user's code
forces, and one launch each for q (`Hq = 32` heads) and k (`Hk = 8`):

- **Per-head weight.** `w` is `(H, D)`; the forward loads a `[BLOCK_H, D]` weight tile and the
  backward accumulates `dw` as a `[BLOCK_H, D]` register tile, so it must hold *every head of a
  token* in one program (`BLOCK_H = next_pow2(H)`, `_geometry` refuses `H * D / 2 > 4096`).
- **Interleaved RoPE.** Pairs are `(x[2i], x[2i+1])`, not `(x[i], x[i + D/2])`: the rotation
  is `tl.split` of a `[BLOCK_H, D/2, 2]` reshape, rotate, `tl.join` back (`_rotate`); the
  backward rotates `dy` by `-angle` the same way.
- **Views of one `qkv` buffer.** q and k are slices of the fused projection with row stride
  `(Hq + 2 Hk) * D`; the kernels take `x` strides and `dy` strides, so no `.contiguous()` copy
  of either input.

The forward is one program per `(b, s, block of heads)` (`BLOCK_H` capped so a tile is about
2K elements: 8 heads of 128 for the user's shape), `rstd (B, S, H)` saved in fp32 only when a
backward can follow. The backward is a fixed grid of `2 x SMs` programs striding over tokens,
`dx` written per token and one fp32 `dw` partial per program summed on the host.

Contract (v1.1 lint rules): `x`, `dy` strided through; no `.contiguous()`, no `.reshape`;
int64 offsets on token and head indices; fp32 compute, storage dtype at the boundaries; `w` is
read in fp32 and `dw` returned in `w.dtype`.

## A800, user workload B 4 x S 4096 x (Hq 32 + Hk 8) x D 128, bf16

Device time from `torch.profiler`, min of three interleaved rounds; the
GPU is unlocked, so compare within the table. `eager` is the user's function (the
`stack(...).flatten` interleaving plus two `.contiguous()` permute copies: about 20 kernels);
`compile` is `torch.compile` of it. Bytes counted: forward reads x and writes y (`2 * es` per
element) plus `rstd`; backward reads dy and x, writes dx (`3 * es`) plus `rstd`. Copy roof is
the measured `y.copy_(x)` bandwidth.

| | time | GB/s | of copy | vs eager | vs compile |
|---|---|---|---|---|---|
| forward eager (user code) | 6.257 ms | 54 | 3% | | |
| forward `torch.compile` | 0.668 ms | 506 | 29% | 9.4x | |
| **forward golden** | **0.218 ms** | 1554 | 90% | 28.7x | 3.07x |
| backward eager | 10.200 ms | 50 | 3% | | |
| backward `torch.compile` | 1.234 ms | 410 | 24% | 8.3x | |
| **backward golden** | **0.355 ms** | 1425 | 82% | 28.7x | 3.47x |

`torch.compile` is unusually weak here: Inductor keeps the interleaved `stack/flatten` and the
permuted `.contiguous()` as separate passes over the 168 MB tensors instead of fusing them into
the norm, so both directions run at a quarter of the copy roof. The fused kernel is a single
read-once/write-once pass, which is why the gap is 3x rather than the 1.0-1.3x typical of
row-wise fusions (compare `fused_residual_rmsnorm/golden/GOLDEN.md`).

What to expect from a candidate kernel: a forward that computes the norm and RoPE in one kernel
but stores `(B, S, H, D)` and leaves the transpose to `.contiguous()` pays the two permute
copies on top, measured at 0.462 ms alone on this shape (so about 0.68 ms total, `compile`
parity); a backward that reduces `dw` with `tl.atomic_add` per token, or writes a `(B, S, H, D)`
partial and sums it in torch, adds at least one more full pass over `x`-sized data (0.17 ms of
traffic at the copy roof, more with atomics contention). A kernel that applies the half-split
rotation (`x[i], x[i + D/2]`) does not match the user's interleaved layout.

The two launches (q, then k) are the natural split: they have different `H`, so one grid would
need two geometries. Fusing v's transpose into the same pass would save its `.contiguous()` in
the attention block (another 42 MB read+write) but the user's request stops at q and k.
