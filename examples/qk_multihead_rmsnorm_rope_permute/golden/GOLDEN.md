# Golden: per-head RMSNorm + interleaved RoPE + permute (`qk_prep`)

Current status: **pass**. The 2026-09-08 full A800 run, two independent small-row repeats,
and small/strided adjoints pass. H20 small-row adjoints also pass. The short-sequence backward
now assigns a complete head to each program and writes its final weight gradient directly,
removing partial-buffer allocation and separate reduction launches for up to 32 tokens.

Hand-written target for `qk_prep` in `user_repo/minilm/attention.py` (Qwen3-style q/k
preparation in a grouped-query attention block), with recorded A800 reference measurements.

The kernel is the `fusion-exemplar/rmsnorm_rope_permute` Triton reference
(`kda-kernel-implement/reference/triton/fusion-exemplar/`) with three changes the user's code
forces, and one launch each for q (`Hq = 32` heads) and k (`Hk = 8`):

- **Per-head weight.** `w` is `(H, D)`; the forward loads a `[BLOCK_H, D]` weight tile and the
backward accumulates `dw` as a `[BLOCK_H, D]` register tile, so it must hold *every head of a
token* in one program (`BLOCK_H = next_pow2(H)`, `_geometry` refuses `H * D / 2 > 4096`).
- **Interleaved RoPE.** Pairs are `(x[2i], x[2i+1])`, not `(x[i], x[i + D/2])`: the rotation
is `tl.split` of a `[BLOCK_H, D/2, 2]` reshape, rotate, `tl.join` back (`_rotate`); the backward
rotates `dy` by `-angle` the same way.
- **Views of one `qkv` buffer.** q and k are slices of the fused projection with row stride
`(Hq + 2 Hk) * D`; the kernels take `x` strides and `dy` strides, so no `.contiguous()` copy of
either input.

The forward is one program per `(b, s, block of heads)` (`BLOCK_H` capped so a tile is about 2K
elements: 16 heads of 128 for the user's shape), `rstd (B, S, H)` saved in fp32 only when a
backward can follow. For longer sequences, the backward is a fixed grid of `2 x SMs` programs striding over tokens,
`dx` written per token and one fp32 `dw` partial per program reduced by PyTorch on the GPU.

Contract (v1.1 lint rules): `x`, `dy` strided through; no `.contiguous()`, no `.reshape`; int64
offsets on token and head indices; fp32 compute, storage dtype at the boundaries; `w` is read in
fp32 and `dw` returned in `w.dtype`.

## A800 measurements

B 4 x S 4096 x (Hq 32 + Hk 8) x D 128, bf16.

Device time from `torch.profiler`, min of three interleaved rounds; the GPU is unlocked, so
compare within the table. `eager` is the user's function (the `stack(...).flatten` interleaving
plus two `.contiguous()` permute copies: about 20 kernels); `compile` is `torch.compile` of it.
Activation traffic includes x/y and saved `rstd` forward, then dy/x/dx and `rstd` backward. The
benchmark script also counts weights, rotary tables, partial reductions and final weight
gradients. Copy roof is the measured `y.copy_(x)` bandwidth.

| Phase | Golden | Eager | Compiled | SoL |
|---|---:|---:|---:|---:|
| forward | 0.2182 ms | 6.2524 ms | 0.6683 ms | 92.0% |
| inference | 0.2130 ms | 6.2511 ms | 0.9240 ms | 93.5% |
| backward | 0.3531 ms | 10.1797 ms | 1.2162 ms | 84.1% |

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

## Reproduce the checks

From the checkout root, using the GPU Python interpreter:

```bash
python examples/qk_multihead_rmsnorm_rope_permute/benchmark.py --verify --json tmp/golden-benchmarks/qk_multihead_rmsnorm_rope_permute/verify.json
python examples/qk_multihead_rmsnorm_rope_permute/benchmark.py --bench --json tmp/golden-benchmarks/qk_multihead_rmsnorm_rope_permute/report.json
python examples/qk_multihead_rmsnorm_rope_permute/benchmark.py --adjoint --json tmp/golden-benchmarks/qk_multihead_rmsnorm_rope_permute/adjoint.json
```

[Workloads and SoL accounting](../benchmark_cases.py) define the inputs, gradient reductions,
and per-phase bytes/FLOPs. The [shared runner](../../_benchmark.py) uses KDA's product runtime
directly. Reports are generated outside `golden/`; new numerical and timing reports require an
independent acceptance audit. Use a distinct output filename for scoped `--workload` or
`--fwd-only` runs.

## Current-source repeat audit

The repaired full run measures small-row backward at 0.008960 ms versus 0.014672 ms
compiled (1.638x). Both independent repeats pass at about 1.63x compiled. All four full-run
numerical workloads and eligible baselines pass, as do independent small and strided adjoints.
The short path counts final gradients but no partial-buffer traffic in its roof. The original
conflicting captures remain under `historical_capture` in `golden.json`; the new top-level
record, repeats and adjoints identify the repaired source by hash.
