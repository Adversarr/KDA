Same pre-norm structure as `fused_residual_rmsnorm`, but the normalisation is LayerNorm with
weight and bias, all math in fp32, and the residual stream stays fp32:

```python
def add_layernorm(x, residual, weight, bias, eps):
    h = residual + x.float()
    mean = h.mean(-1, keepdim=True)
    var = (h - mean).pow(2).mean(-1, keepdim=True)
    y = (h - mean) * torch.rsqrt(var + eps) * weight.float() + bias.float()
    return h, y.to(x.dtype)
```

Shapes from the training config: batch 4, sequence 2048, `d_model` 2048, bf16 autocast;
`weight` and `bias` are fp32 parameters. I need forward and fused backward (`dx`,
`dresidual`, `dweight`, `dbias`); save `mean` and `rstd`, not the normalised tensor.

Status: TASK only (no `user_repo/` yet); arrives with the Dion optimizer analogue.
