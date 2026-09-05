# FlashAttention-2, causal, multi-head (`flash_attention_2`, Triton)

Files: `kernel.py` (forward, three backward kernels, autograd, `select_config`), `reference.py`
(eager with materialised scores, and `F.scaled_dot_product_attention` as the baseline). Twin of
[`tilelang/attention/fa2_causal`](../../../tilelang/attention/fa2_causal/SNIPPET.md); the
non-causal GQA variant is [`fa2_gqa`](../fa2_gqa/SNIPPET.md).

## Math

```
q, k, v: (B, H, S, D)   storage dtype bf16/fp16, unit stride in D, any other strides (a (B, S, H, D)
                        view of the projection output is fine: no permute copy)
s = q k^T * scale        scale defaults to D^-1/2;  s[i, j] = -inf for j > i (causal, top-left aligned)
o = softmax(s) v         softmax in fp32; o in the storage dtype
lse = m + log2(l)        (B, H, S) fp32, log2 domain: saved for the backward
```

The kernels work in the log2 domain throughout: `s * scale * log2(e)` and `exp2`, so the
per-element scale folds into one FMA and the transcendental is the fast one. `lse` is stored
in that domain; the backward recomputes `p = exp2(s * scale_log2 - lse)`.

Backward, deterministic (no atomics), five GEMMs per K,V-block pair against two in the forward:

```
delta = rowsum(o * do)                              preprocess kernel, (B, H, S) fp32
dK/dV kernel  (one program per K,V block, loop over Q blocks at or below the diagonal):
    pT = exp2(k q^T * scale_log2 - lse)   dV += pT do    dpT = v do^T    dsT = pT (dpT - delta)    dK += dsT q
dQ kernel     (one program per Q block, loop over K,V blocks up to the diagonal):
    p  = exp2(q k^T * scale_log2 - lse)   dp = do v^T    ds = p (dp - delta)                       dQ += ds k
dK *= scale, dQ *= scale at the store
```

`q k^T` is thus computed three times (forward, dK/dV pass, dQ pass). A single-pass backward
with `tl.atomic_add` on `dQ` saves one GEMM and is not bitwise reproducible; the KDA verifier's
determinism probe rejects it, so the reference keeps two passes.

## Tensor-core and layout rules

- `tl.dot` operands come straight from `tl.load` in the storage dtype; `p` is cast to it
  before `p @ v` (`p.to(v.dtype)`), the accumulators (`acc`, `dk`, `dv`, `dq`) are fp32.
- The forward loads `k` **transposed** (`(BLOCK_D, BLOCK_N)`, offsets `offs_d[:, None] +
  offs_n[None, :] * stride`) so `tl.dot(q, k)` needs no `tl.trans`. In the backward every
  tile is loaded in its natural layout and transposed only as the **B** operand
  (`tl.dot(k, tl.trans(q))`, `tl.dot(v, tl.trans(do))`), which Triton folds into the MMA
  operand layout. The other two arrangements were measured and lose: loading `q` transposed and
  `tl.trans`-ing it back for `dsT @ q` costs 1.3x (18.1 vs 13.3 ms); loading each tile twice in
  both layouts overflows shared memory at 3 stages and is slower at 2.
- **Causal skip and the mask split.** A Q tile's K,V loop runs `[0, diag)` without any mask and
  `[diag, diag + BLOCK_M)` with the `offs_m >= offs_n` compare (the diagonal blocks, which also
  cover the `S` tail); the dK/dV kernel mirrors this with `[lo, mid)` masked and `[mid, S)`
  free. Masking every block cost 9% on the 8192-token forward (4.00 -> 3.11 ms with the
  split and the transposed K load together).
- **Tails.** `S` need not divide any tile: loads are masked to zero, masked scores set to
  `-inf`, `lse` of a padded row is `+inf` so its `p` is `0` in the backward. `D` is padded to
  `next_power_of_2(D)` in registers with a `mask_d` (`D = 96` runs as 128 with 25% of the MMA
  wasted; a 3 x 32 tiling would need a different `tl.dot` decomposition).
- **int64, but not in the loop.** `bh = tl.program_id(1).to(tl.int64)` before any multiply
  (the `(b, h)` base and the `lse`/`delta` row offset `bh * S` overflow int32 at
  `B*H*S*D >= 2^31`), and the program's *own* tile offsets (`offs_m64`, `offs_n64`: one load
  and one store outside the loop) are int64 too. The offsets *inside* the K,V / Q loops stay
  int32: promoting the row program id wholesale (`start_m.to(tl.int64)`) makes the loop
  bounds, `offs_n` and every in-loop address 64-bit and cost 11% on the 8192-token forward
  (3.11 -> 3.47 ms). The host asserts what that relies on, `S * stride_row < 2^31` per tensor
  (one `(b, h)` plane addressable in int32); the lint's `int64` rule is satisfied by the
  `.to(tl.int64)` on the same id. The *logical* score plane `B*H*S*S` crosses 2^31 at
  `56 heads x 6200 tokens`, which is why it is never an index here.

## Saved for the backward

`q, k, v, o` (autograd keeps them anyway) and `lse` (`B*H*S*4` bytes). The score plane is never
materialised; the eager reference's `B*H*S*S*4` bytes (4 GiB at `H = 32, S = 8192`) is the
memory the kernel saves besides the time. There is no recompute option: the backward already
recomputes `p` from `lse`.

## Launch geometry (`select_config`, A800, bf16, measured)

Forward, `B = 1, H = 32, S = 8192, D = 128` (sdpa flash 2.79 ms):

| `BLOCK_M x BLOCK_N` | warps | stages | ms |
|---|---|---|---|
| 128 x 128 | 8 | 3 | **3.11** |
| 128 x 32 | 4 | 2 | 3.18 |
| 128 x 32 | 4 | 3 | 3.22 |
| 128 x 64 | 8 | 3 | 3.38 |
| 128 x 64 | 8 | 4 | 3.51 |
| 128 x 64 | 4 | 3 | 3.79 |
| 64 x 128 | 4 | 3 | 5.12 |
| 64 x 64 | 4 | 3 | 5.96 |
| 128 x 128 | 4 | 3 | 162 (register spill) |

Backward, same shape (sdpa flash 8.13 ms); `(M1, N1)` = Q sub-block x K,V block of the dK/dV
kernel, `(M2, N2)` = Q block x K,V sub-block of the dQ kernel:

| `M1 N1 / M2 N2` | warps | stages | ms |
|---|---|---|---|
| 64 64 / 64 64 | 4 | 2 | **10.70** |
| 64 128 / 128 64 | 8 | 2 | 11.32 |
| 64 128 / 128 64 | 8 | 3 | 11.56 |
| 32 128 / 128 32 | 8 | 2 | 12.74 |
| 64 64 / 64 64 | 4 | 1 | 12.96 |
| 64 64 / 64 64 | 4 | 3 | 13.26 |
| 64 64 / 64 64 | 8 | 2 | 16.70 |
| 32 128 / 128 32 | 4 | 3 | 306 (spill) |
| 128 64 / 64 128 | 8 | 3 | out of shared memory |

Two stages beat three on the backward: five GEMM operands per iteration make the 3-stage
pipeline's shared memory the limit (`64 x 64` tiles at `D = 128` already need 57 KB per stage).

## Measured (A800, bf16, `torch.profiler` device time; FLOPs count the attended triangle only)

| shape | fwd sdpa | fwd triton | bwd sdpa | bwd triton | fwd eager | bwd eager |
|---|---|---|---|---|---|---|
| B8 H32 S512 D128 | 0.170 ms | 0.226 (0.75x) | 0.678 | 0.570 (**1.19x**) | 2.27 | 3.05 |
| B1 H32 S8192 D128 | 2.78 | 3.11 (0.89x), 177 TFLOP/s | 8.13 | 10.70 (0.76x), 129 TFLOP/s | - | - |
| B1 H56 S4096 D128 (H3 heads) | 1.28 | 1.46 (0.87x) | 3.97 | 4.86 (0.82x) | 31.7 | 41.0 |
| B4 H32 S2048 D64 | 0.459 | 0.470 (0.98x) | 1.54 | 1.65 (0.93x) | 16.8 | 22.7 |

Against the roofline: 177 TFLOP/s forward is 57% of the datasheet peak and ~0.8 of cuBLAS on
GEMMs of these shapes (the achievable compute roof, `bench.matmul_ms`); the FA2 kernel torch
ships reaches 63%. This is where a Triton FA2 lands on sm80 without warp specialisation; the
remaining gap to FA2 is its hand-scheduled softmax/GEMM overlap, not a missing tile config.
The kernel is 14-22x faster than the eager chain and the eager chain does not fit at
production lengths, so `sol_eff >= 0.5` against the FLOP roof (compute-patterns.md) is the
realistic bar for this pattern and the SPEC should say so.

## Fusing into a user op

- **Block-causal (video: frame chunks attend to all earlier chunks and their own).** Replace
  the compare `offs_m >= offs_n` with `offs_m // block >= offs_n // block` and the loop bounds
  `diag` with `(start_m * BLOCK_M) // block * block` (first key of the query tile's chunk) and
  `hi` with `min(S, ((start_m * BLOCK_M + BLOCK_M - 1) // block + 1) * block)`; the mask is
  needed only where the K,V tile straddles a chunk boundary of a query in the tile. The
  attended area is `S * (S + block) / 2` pairs, the FLOP roof scales with it. When `block` is a
  multiple of `BLOCK_N` no tile straddles and the masked loop is empty.
- **Boolean mask `(S, S)` or `(B, S, S)` from the model**: load the mask tile per block (`int8`
  or `bool` in memory, `tl.load(..., other=0)`), `tl.where(mask, s, -inf)`; every block is then a
  masked block and the split above degenerates. Prefer deriving the mask from indices when the
  model's mask has structure (causal, block, sliding window): a `(S, S)` byte mask is
  `S*S` bytes of traffic per head.
- **Non-causal / GQA / `S_q != S_kv`**: the [`fa2_gqa`](../fa2_gqa/SNIPPET.md) variant.
- **Norm/RoPE before attention** is a separate token-wise kernel
  (`fusion-exemplar/rmsnorm_rope_permute`): its output layout `(B, H, S, D)` is what this kernel
  reads, and fusing it into the attention loop would recompute the norm per K,V block.
