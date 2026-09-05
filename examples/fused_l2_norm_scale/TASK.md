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

Status: TASK only.
