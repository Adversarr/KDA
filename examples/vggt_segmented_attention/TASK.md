# Segmented multi-view attention (VGGT-style extension)

This is a synthetic variable-length extension, **not an official VGGT workload or an
equivalent replacement for the published VGGT model**. Official provenance is in `user_repo/README.md`.

Optimize `segmented/attention.py::segmented_attention` through implementation, verification,
benchmarking and integration behind a switch with an eager fallback. Inputs Q/K/V are bf16
CUDA tensors `(B,H,V*T,D)`; the representative geometry is B=1, V=4, T=1374, H=16, D=64.
Frame attention folds views into the batch, with one segment per sequence. Global attention
allows every scene's query to see keys from all of that scene's valid views.

`lengths` is an immutable tuple of CPU integer tuples, shape `(B,V)`, supplied by collation.
`tokens_per_view=T` is a positive integer. View i contributes keys `[i*T, i*T+lengths[b][i])`.
Lengths include special tokens; an absent view has length zero. They are not inferred from
pixel values or recovered from a CUDA mask. The layout must not silently interpret arbitrary
masks or rectangular image crops as prefixes. Scene boundaries must remain isolated.

Keep Q and the output at their original full length. Pack/skip invalid K/V; gradients for
excluded K/V must be zero. When every view of a scene is empty, attend to original key zero
as a dummy, matching the existing padded fixture. Invalid query outputs are still computed;
any loss exclusion or projection masking belongs to the caller. Output shape and dtype equal Q.

Math uses FP32 score accumulation and softmax. Probabilities round to the input bf16
before PV, which accumulates in FP32 and returns bf16, following FlashAttention's mixed-
precision convention. Low-precision tensor-core operands are also permitted in backward;
verify outputs and gradients with the unchanged numerical tolerances. Probability-cancellation
cases document this rounding boundary. No dropout or causal mask. This convention was adopted
on 2026-09-08; older FP32-probability measurements describe a stricter contract.

Measure the complete operation, including per-call KV concatenation/compaction and backward
scatter, not just its inner attention kernel. Compare with the key-masked mixed-precision
operation on the **same prefix-valid inputs**, plus the all-valid control. Do not compare a
random mask with a prefix mask. Report cold compilation separately, and keep the existing
numerical tolerances and useful-work SoL rules. Real 24-view sizes are in the extension's
intended workload scope, but any unavailable measurement must remain explicit.

Run the self-contained training fixture from user_repo:

```bash
python -m segmented.train --smoke --steps 3 --seed 0
python -m segmented.train --steps 50 --seed 0
```

The eager fixture spells out the Flash-style adjoint with PyTorch matrix operations:
`dP` and the row reduction `delta = sum(output * dOutput)` are fp32; `P` rounds
before dV and `dS = P * (dP - delta)` rounds before dQ/dK products. Gradient
products accumulate in fp32. This is the permitted mixed-precision adjoint,
not the additional gradient cast inserted by naive autograd through a bf16
probability tensor. Online softmax may round unnormalized block weights before
PV and normalize the fp32 output accumulator afterward, as FlashAttention does.
