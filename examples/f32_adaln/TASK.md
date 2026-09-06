Adaptive LayerNorm from my DiT blocks: LayerNorm without affine parameters, then a per-sample
scale and shift that come from the conditioning MLP. Everything in fp32 until the final cast.
The function is `adaln` in `diffusion/layers.py`; every block calls it twice (before the
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

Training config is `diffusion/settings.py` (`TrainConfig`, `ModelConfig`): bf16 autocast, batch
16, N 4096 tokens, D 1152, 16 heads, 2 layers; `scale` and `shift` are fp32 because the
conditioning path runs outside autocast. The training run is `python -m experiments.train_noise
--steps 50`; it prints `median_step_ms` and `final_loss`.

I want the forward and a fused backward: `dx`, and `dscale` and `dshift` reduced over the N
tokens of each sample (they are `(B, D)`, like the inputs). The final layer follows `adaln` with
a SiLU (`FinalLayer.forward`); if it is cheap, make the activation an optional fused epilogue,
otherwise leave the SiLU where it is and say so.

Please take it all the way: spec, kernel with the backward, verification and benchmark against
speed of light and against `torch.compile`, and integrate it into `diffusion/layers.py` behind a
flag so I can switch back to the eager code. I care most about the shape the config actually
uses.

For a bounded correctness smoke, run `python -m experiments.train_noise --smoke --steps 3 --seed
0` from `user_repo/`. This keeps target channel/head dimensions but reduces batch/token counts;
use the unmodified representative config for performance measurements.
