# Fusion walkthrough: rmsnorm -> rope -> permute

Files: `kernel.py` (fused fwd + bwd), `reference.py` (the eager three-op chain).
Primitives: [`fp32_norms/rmsnorm`](../../fp32_norms/rmsnorm/SNIPPET.md),
[`rope/rope`](../../rope/rope/SNIPPET.md), [`permute/bnhd_to_bhnd`](../../permute/bnhd_to_bhnd/SNIPPET.md).

This is the QK-norm pattern (`q_norm(q)` -> `apply_rope` -> `transpose(1, 2)`) that the
`qk_rmsnorm_rope_permute` example fuses for a real user repo. Read it as a worked example of
how the implementer turns three correct primitives into one kernel *in context*: there is no
helper library to call, only the primitives' structure to carry over.

## 1. Draw the data flow, count the passes

```
x (B,S,H,D) --rmsnorm(D)--> n --rope(s)--> r --permute--> y (B,H,S,D)
   eager:      read x, write n | read n, write r | read r, write y     = 3 reads + 3 writes
   fused:      read x ........................................ write y = 1 read  + 1 write
```

Every fusion decision follows from this picture: whatever the middle ops need must be
available in registers between the load of `x` and the store of `y`.

## 2. Pick the program shape from the *reduction*

RMSNorm reduces over `D` per `(b, s, h)`; RoPE pairs element `d` with `d + D/2` of the same
row; the permute changes only where the row is stored. So the natural unit of work is one
row of `D` and the natural tile is `[BLOCK_H heads, D]` for one token `(b, s)`: the `cos`/`sin`
row for position `s` is loaded once per program and broadcast over the heads.

The row is loaded as **two half tiles** `x1 = x[:, :D/2]`, `x2 = x[:, D/2:]` because RoPE
needs the halves as separate operands and Triton tiles cannot be sliced. The RMS reduction
does not care: `sum(x1*x1, 1) + sum(x2*x2, 1)`.

## 3. Forward: primitive bodies, one after another, no stores in between

```
x1, x2 = load halves (fp32)                 # rmsnorm: load once
rstd = rsqrt((sum x1^2 + sum x2^2)/D + eps)
n1, n2 = x1*rstd*w1, x2*rstd*w2
y1 = n1*cos - n2*sin; y2 = n2*cos + n1*sin  # rope: registers only
store y1, y2 at y[b, h, s, :]               # permute: only the address changed
if SAVE_RSTD: store rstd[b, s, h]           # saved for the backward, only when one can follow
```

`SAVE_RSTD` is the rmsnorm primitive's flag carried through unchanged: the wrapper decides
it from grad mode and `requires_grad`, and an inference call allocates and writes no `rstd`.
Every saved-for-backward tensor of a fusion is gated the same way (`save_aux` in a kernel package).

Compare with `rmsnorm/kernel.py::_rmsnorm_fwd_kernel` and `rope/kernel.py::_rope_kernel`:
the fused body is their bodies concatenated, minus one store and one load, plus the permuted
store address from `permute/kernel.py`.

## 4. Backward: the adjoint of each step, in reverse order

```
dy1, dy2 = gather from y-layout dy[b, h, s, :]      # adjoint of the permuted store = permuted load
dn1 = dy1*cos + dy2*sin; dn2 = dy2*cos - dy1*sin    # adjoint of rope = rotate by -angle
x1, x2 = load x; rstd = load saved                  # rmsnorm backward, exactly as in the primitive
xhat = x*rstd; dxhat = dn*w
c = mean over D of dxhat*xhat
dx = rstd*(dxhat - xhat*c)  -> store in x layout
dw += sum over heads of dn*xhat                     # register accumulator, one partial per program
```

The backward grid is the rmsnorm primitive's grid (a fixed number of programs striding over
tokens) because `dw` must be accumulated across every row without per-row partials or
atomics. Each step handles all `H` heads of one token so `dw` sees every head; that bounds
`H * D/2 <= 8192` (a head loop would lift it).

## 5. What to save

| candidate | cost to save | cost to recompute | decision |
|---|---|---|---|
| `rstd (B,S,H)` fp32 | 4 bytes/row (~1.5% of `x` for D=128 bf16) | a full read of `x` plus a row reduction | save |
| `n` (norm output) | 100% of `x` | 3 FMAs per element from `x` and `rstd` | recompute |
| `cos`, `sin` | already exist | - | keep pointers |

## 6. Precision: the fusion is *more* accurate than the eager chain

Eager materialises `n` in `x.dtype` (bf16) before RoPE; the fused kernel never leaves fp32.
`y` and `dx` agree to bf16 rounding, but `dw`, a sum over `B*S*H` rows, accumulates the
reference's per-row bf16 rounding of `dn`: absolute differences of `~sqrt(rows) * 2^-8 * |term|`
that no elementwise tolerance at the *output* dtype can express. KDA's verifier therefore
(a) uses the tolerance of the lowest precision in the chain (the workload dtype) and (b)
scales `atol` by `sqrt(rows reduced)` for gradients smaller than the output. Expect this on
every fusion with a weight gradient; it is not a bug, and matching the eager rounding on
purpose would only make the kernel worse.

## 7. Measured on A800 (8x2048x32x128 bf16)

| | time | of copy roof |
|---|---|---|
| fused forward | 0.167 ms | 95% |
| eager chain forward | 4.56 ms | 3% |
| `torch.compile` chain forward | 0.28 ms | 56% |
| fused backward (incl. `dw` reduce) | 0.34 ms | 68% |
| eager chain backward | 6.87 ms | 3% |

Device time from `torch.profiler` (what `_run_dev.py` reports); CUDA-event wall time would add
the CPU launch overhead of the eager chain and of `autograd.Function`.

The forward is at the memory roof. The backward reads `x` and `dy` and writes `dx` (three
passes) plus the partial reduce; ~2/3 of copy speed is typical for norm backwards whose
per-program `dw` accumulation caps the grid at a couple of programs per SM. `dy` is read
through its own strides (`sdy_b, sdy_h, sdy_s`; only `stride(3) == 1` is required): attention
backwards commonly hand back a permuted view, and a `dy.contiguous()` in the launcher would be
a hidden fourth pass the roofline does not count (the lint's `copy` rule).
