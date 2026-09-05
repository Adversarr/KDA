# FlashAttention-2, causal, multi-head (`flash_attention_2`, TileLang)

Files: `kernel.py` (forward, preprocess, dK/dV and dQ kernels, autograd, `select_config`),
`reference.py`: the same contract and workloads as the Triton twin
[`triton/attention/fa2_causal`](../../../triton/attention/fa2_causal/SNIPPET.md), which holds
the algorithm (log2-domain online softmax, causal block skip, deterministic two-kernel
backward, int64 rule). This page is the TileLang-specific part. GQA / non-causal variant:
[`fa2_gqa`](../fa2_gqa/SNIPPET.md).

## Contract differences from the Triton twin

- `D` must be a multiple of 16 (the tile's last extent is `D` itself, no power-of-two
  padding): 64, 96, 128, 160, 192, 256 all compile. Triton pads `D = 96` to 128 in registers.
- Strides are compile-time ints (`T.StridedTensor((B, H, S, D), (sb, sh, sm, 1))`), one variant
  per distinct stride set; a `(B, S, H, D)` view of the projection output compiles once and
  passes without a copy (tested, `bshd` case).
- `o` is a fresh contiguous `(B, H, S, D)` (`T.empty`), not `empty_like(q)`.

## TileLang correctness constraints

- **`policy=T.GemmWarpPolicy.FullRow` on every GEMM.** The `Q K^T` accumulator fragment is
  the *A* operand of `P V` after a cast; with the default warp policy their layouts differ
  and lowering fails with `Layout infer conflict between acc_s and acc_s_cast`. FullRow gives
  each warp whole rows of both, so the layouts agree. The backward's `pT -> dv`, `dsT -> dk`
  and `ds -> dq` chains need it for the same reason.
- **The mask is the accumulator's initial value.** `acc_s` is filled with `0 / -inf` per
  element (`T.if_then_else(m0 + i >= n0 + j, 0, -inf)`) before `T.gemm` accumulates into it;
  off-diagonal blocks take `T.clear`. A select after the GEMM would need a second pass over
  the fragment.
- **`num_stages=1` in the backward, and it is faster.** With `T.Pipelined(lo, hi,
  num_stages=2)` the K block whose Q loop has a single iteration ending at the `S` tail
  produced nondeterministic `dk`/`dv` (a few rows differ run to run, sometimes `inf`); with one
  stage the backward is bitwise reproducible *and* measured 26% faster
  (11.4 vs 15.3 ms at 8192 tokens): five GEMM operands per iteration leave no shared memory for
  a second stage anyway. The forward's single-iteration tiles never showed the race, but its
  best config is also `num_stages=1` (table below), so nothing here pipelines.
- **No `if` around shared-memory reads inside a pipelined loop.** An `if m0 >= n0 + block_N:`
  choosing between the masked and unmasked `pT` expressions (both read `lse_s`, copied in the
  same loop iteration) can race; the dK/dV kernel masks
  every block instead (`T.if_then_else` per element, cheap next to five GEMMs). The forward's
  `if` only chooses between two *fills* and is fine.
- **1-D `T.copy` does not zero-fill the tail.** `T.copy(lse[bz, by, m0:m0 + block_M], lse_s)`
  past `S` leaves the shared tail *uninitialised* (2-D copies do zero-fill: probed with a
  poisoned buffer). The padded query columns must be *selected* out (`T.if_then_else(m0 + j <
  S and ..., exp2(...), 0)`, and the same guard on `dsT`), never multiplied by zero:
  `0 * NaN` from a stale shared word poisons the whole `dk`/`dv` row.
- **Two staging tiles for the two stores.** Reusing one shared tile for `dk` then `dv`
  (`T.copy(dk_f, s); T.copy(s, dk); T.copy(dv_f, s); ...`) raced; TileLang does not insert a
  barrier between a shared->global copy and the next fragment->shared copy into the same
  buffer. (The GEMM page's `Z`/`Y` epilogue got away with it; do not rely on that.)
- **Free-mode layout inference needs power-of-two `D` for fragment reductions.** The
  preprocess `delta = rowsum(o * do)` as `T.copy` into a `(block_M, D)` fp32 fragment +
  `T.reduce_sum` fails with `no available layout found` at `D = 96, 160, 192, 288` (fine at 64,
  128, 256). It is written as a shared tile + one serial row loop per thread instead; the
  attention kernels' own `(block, D)` fragments are GEMM outputs and infer fine at `D = 96`.
- **Tile shapes are constrained by inference too**: in the backward, 256 threads with
  `64 x 64` tiles and 32-row tiles both fail with a layout conflict (the sweep is only over
  what compiles).

## Launch geometry (`select_config`, A800, bf16, measured)

Forward, `B = 1, H = 32, S = 8192, D = 128` (sdpa flash 2.80 ms):

| `block_M x block_N` | stages | threads | ms |
|---|---|---|---|
| 128 x 128 | 1 | 256 | **3.11** |
| 128 x 128 | 2 | 256 | 3.14 |
| 64 x 64 | 1 | 128 | 3.20 |
| 64 x 64 | 2 | 128 | 3.29 |
| 128 x 64 | 2 | 256 | 3.53 |
| 128 x 64 | 3 | 256 | 3.61 |
| 64 x 128 | 2 | 128 | 3.58 |
| 128 x 64 | 2 | 128 | 4.30 |
| 128 x 32 | 2 | 256 | 4.48 |

Backward, same shape (sdpa flash 8.13 ms): `64 x 64` tiles, 128 threads, `num_stages=1`
11.39 ms; `num_stages=2` 15.31 ms and racy (above).

## Measured (A800, bf16, `torch.profiler` device time; FLOPs count the attended triangle only)

| shape | fwd sdpa | fwd tilelang | bwd sdpa | bwd tilelang | Triton twin fwd / bwd |
|---|---|---|---|---|---|
| B8 H32 S512 D128 | 0.169 ms | 0.224 (0.75x) | 0.694 | 0.618 (**1.12x**) | 0.226 / 0.570 |
| B1 H32 S8192 D128 | 2.80 | 3.11 (0.90x), 177 TFLOP/s | 8.13 | 11.39 (0.71x), 121 TFLOP/s | 3.11 / 10.70 |
| B1 H56 S4096 D128 (H3 heads) | 1.29 | 1.46 (0.88x) | 3.97 | 5.23 (0.76x) | 1.46 / 4.86 |
| B4 H32 S2048 D64 | 0.458 | 0.584 (0.78x) | 1.54 | 1.87 (0.82x) | 0.470 / 1.65 |

Parity with the Triton twin on the forward (unlike the GEMM, where TileLang trails by 20%:
the attention mainloop is two `T.gemm`s on shared tiles with a fragment in between, exactly
what TileLang lowers well); 6% behind it on the backward. Both are 0.7-0.9x of the FA2 kernel
torch ships, 14-22x over eager. The `sol_eff` bar for this pattern is discussed on the Triton page.

## Fusing into a user op

The block-causal, boolean-mask, bias and window recipes on the Triton page apply verbatim:
the condition in the accumulator fill (`acc_s[i, j] = T.if_then_else(cond, 0, -inf)`) and the
loop bounds (`n_blocks`, `lo`) are the only places the mask lives, in all three kernels.
Block-causal for video: `cond = (m0 + i) // block >= (n0 + j) // block`, `n_blocks =
ceildiv(min(S, ((m0 + block_M - 1) // block + 1) * block), block_N)`, and in the dK/dV kernel
`lo = (n0 // block * block) // block_M`.
