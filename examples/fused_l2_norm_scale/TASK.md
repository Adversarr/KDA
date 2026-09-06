L2-normalise over the last dim and multiply by a learned per-channel scale (nGPT-style
normalised hidden state):

```python
def l2_norm_scale(x, scale, eps):
    # x: (B, S, D) bf16;  scale: (D,) fp32
    h = x.float()
    y = h * torch.rsqrt(h.pow(2).sum(-1, keepdim=True) + eps) * scale.float()
    return y.to(x.dtype)
```

Shapes: B 8, S 4096, D 2048, bf16. Forward and fused backward (`dx`, `dscale`). The same
function is also called on `(B, S, H, D)` tensors with D 128 (per-head), so the kernel must
handle both a wide and a narrow last dim well.

The ordinary eager repository is in `user_repo/`; the definition to optimize is
`normalized/ops.py` and configuration is `normalized/__main__.py`. Run `python -m normalized
--steps 50 --seed 0` for representative training or `python -m normalized --smoke --steps 3
--seed 0` for a bounded correctness smoke. The latter preserves the operation's channel widths
and head dimensions while reducing batch/token counts. It prints `median_step_ms` and
`final_loss`.

Please implement and verify forward and backward, benchmark the representative shapes, and
integrate the result at that definition site with an eager fallback.
