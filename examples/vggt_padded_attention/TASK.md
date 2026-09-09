I am training a VGGT-style multi-view transformer (`multiview/attention.py`) and the attention
is the bottleneck of the global block: `padded_attention` materialises the fp32 `(B, H, N, N)`
logits. I want it as one fused flash-style kernel, forward and backward.

```python
def padded_attention(q, k, v, key_valid):
    # q, k, v: (B, H, N, d) bf16 with H = 16, d = 64;  key_valid: (B, N) bool, True = this key may be attended
    logits = torch.matmul(q.float(), k.float().transpose(-2, -1)) * (d ** -0.5)
    logits = logits.masked_fill(~key_valid[:, None, None, :], float("-inf"))
    prob = torch.softmax(logits, dim=-1)
    prob = torch.nan_to_num(prob, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.matmul(prob.to(v.dtype).float(), v.float()).to(v.dtype)
```

Dense and non-causal; the only mask is *key validity per batch entry*: padded patches and padded
views are keys nobody may attend to, and every query row of a batch entry shares the same
allowed key set, so the mask is the `(B, N)` vector, never an `(N, N)` plane. `key_valid` is
already sanitised before the call (`key_valid_from_token_valid`: a view with no valid token gets
key 0 as a dummy), so every row has at least one allowed key and the `nan_to_num` never fires in
practice; keep it out of the kernel. Padded *queries* are computed like any other row and zeroed
later, after the output projection. The softmax is fp32 (the function turns autocast off inside;
q, k, v arrive as bf16). Before PV, probabilities may round to the input dtype,
as in FlashAttention; PV accumulates in fp32 and the output returns to bf16. Backward
may similarly use low-precision tensor-core operands with fp32 accumulation. Numerical
verification tolerances are unchanged. This precision convention was explicitly adopted on
2026-09-08; earlier FP32-probability performance records describe a stricter contract.

The kernel is called twice per block pair with very different shapes: the frame block folds the
views into the batch (`B * S` sequences of `T` tokens), the global block sees every token of
every view (`B` sequences of `S * T`). Training config is `multiview/config.py` (`TrainConfig`,
`ModelConfig`): bf16 autocast, batch 1, 4 views of a `37 x 37` patch grid, `T` = 1374 tokens per
view (1 camera + 4 register + 1369 patches), so the global block runs at `N` = 5496, `dim` 1024
= 16 heads x 64. Real runs use 2 scenes of up to 24 views (global `N` above 30k) and mixed image
sizes, so `T` is not a multiple of anything convenient and the padding fraction varies; the
kernel has to be correct there too. The training run is `python -m multiview.train --steps 50`;
it prints `median_step_ms` and `final_loss`. `--attention sdpa` runs the library path for
comparison.

Please take it all the way: spec, kernel with the backward (`dq`, `dk`, `dv`), verification and
benchmark against speed of light and against `torch.compile` and
`F.scaled_dot_product_attention` with the boolean mask (`sdpa_attention` in the same file), and
integrate it into `multiview/attention.py` behind a flag so I can switch back to the eager code.
The q/k/v preparation in front of it (`qkv_prep`: per-head LayerNorm and 2-D RoPE) stays as it
is; do not fuse it into this kernel.

For a bounded correctness smoke, run `python -m multiview.train --smoke --steps 3 --seed 0` from
`user_repo/`. This keeps target channel/head dimensions but reduces batch/token counts; use the
unmodified representative config for performance measurements.

The eager fixture spells out the Flash-style adjoint with PyTorch matrix operations:
`dP` and the row reduction `delta = sum(output * dOutput)` are fp32; `P` rounds
before dV and `dS = P * (dP - delta)` rounds before dQ/dK products. Gradient
products accumulate in fp32. This is the permitted mixed-precision adjoint,
not the additional gradient cast inserted by naive autograd through a bf16
probability tensor. Online softmax may round unnormalized block weights before
PV and normalize the fp32 output accumulator afterward, as FlashAttention does.
