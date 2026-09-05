Adaptive LayerNorm from my DiT blocks: LayerNorm without affine parameters, then a per-sample
scale and shift that come from the conditioning MLP. Everything in fp32 until the final cast.
The function is `adaln` in `minidit/model.py`; every block calls it twice (before the
attention and before the MLP branch) and the final layer once more.

```python
def adaln(x, scale, shift, eps):
    # x: (B, N, D) bf16;  scale, shift: (B, D) fp32 (already computed by the conditioning MLP)
    h = x.float()
    mean = h.mean(-1, keepdim=True)
    var = (h - mean).pow(2).mean(-1, keepdim=True)
    n = (h - mean) * torch.rsqrt(var + eps)
    return (n * (1 + scale[:, None, :]) + shift[:, None, :]).to(x.dtype)
```

Training config is `minidit/config.py` (`TrainConfig`, `ModelConfig`): bf16 autocast, batch 16,
N 4096 tokens, D 1152, 16 heads, 2 layers; `scale` and `shift` are fp32 because the
conditioning path runs outside autocast. The smoke run is `python train_smoke.py --steps 50`; it prints `median_step_ms` and `final_loss`.

I want the forward and a fused backward: `dx`, and `dscale` and `dshift` reduced over the N
tokens of each sample (they are `(B, D)`, like the inputs). The final layer follows `adaln`
with a SiLU (`FinalLayer.forward`); if it is cheap, make the activation an optional fused
epilogue, otherwise leave the SiLU where it is and say so.

Please take it all the way: spec, kernel with the backward, verification and benchmark against
speed of light and against `torch.compile`, and integrate it into `minidit/model.py` behind a
flag so I can switch back to the eager code. I care most about the shape the config actually
uses.

Status: TASK, user_repo.
