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

Shapes from the training config: batch 4, sequence 2048, `d_model` 2048, bf16 autocast; `weight`
and `bias` are fp32 parameters. I need forward and fused backward (`dx`, `dresidual`, `dweight`,
`dbias`); save `mean` and `rstd`, not the normalised tensor.

The ordinary eager repository is in `user_repo/`; the definition to optimize is `model.py` and
configuration is `settings.json`. Run `python train.py --steps 50 --seed 0` for representative
training or `python train.py --smoke --steps 3 --seed 0` for a bounded correctness smoke. The
latter preserves the operation's channel widths and head dimensions while reducing batch/token
counts. It prints `median_step_ms` and `final_loss`.

Please implement and verify forward and backward, benchmark the representative shapes, and
integrate the result at that definition site with an eager fallback.
