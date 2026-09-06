I am training a VGGT-style multi-view transformer (`vision/blocks.py`). The blocks are written
residual-first, so every residual add is immediately followed by the LayerNorm that feeds the
next branch, and that pair is `layerscale_residual_layernorm`, called four times per block pair
(after the attention and after the MLP of the frame block and of the global block). I want it as
one fused kernel, forward and backward.

```python
def layerscale_residual_layernorm(x, branch, gamma, norm_w, norm_b, eps, valid, out_dtype):
    # x: (B, N, D) fp32 residual stream;  branch: (B, N, D) bf16 (attention or MLP output)
    # gamma: (D,) fp32 LayerScale;  norm_w, norm_b: (D,) fp32;  valid: (B, N) bool;  D = 1024
    x_new = x + branch.float() * gamma
    x_new = x_new.masked_fill(~valid.unsqueeze(-1), 0.0)
    y = F.layer_norm(x_new, (D,), norm_w, norm_b, eps).to(out_dtype)
    return x_new, y
```

Two outputs: the new fp32 residual `x_new` (the stream stays fp32 across the whole model; the
branch is bf16 from autocast) and its affine LayerNorm cast to bf16, the next branch's input.
LayerScale is the per-channel `gamma` (initialised at 0.01) on the branch, and the validity mask
zeroes the padded rows of the stream (padded patches, padded views) so that biases never revive
them — those rows must be exactly zero in `x_new`, and their `y` is then just the LayerNorm of a
zero row (`norm_b`). The backward needs `dx`, `dbranch` (bf16), `dgamma`, `dnorm_w` and
`dnorm_b`.

Training config is `vision/config.py` (`TrainConfig`, `ModelConfig`): bf16 autocast, batch 1, 4
views of a `37 x 37` patch grid, 1374 tokens per view, `dim` 1024; the function sees 5496 rows
of 1024 whether it runs in the frame block (`(B * S, T, D)`) or the global block (`(B, S * T,
D)`). Real runs use 2 scenes of up to 24 views (66k rows). The training run is `python -m
engine.trainer --steps 50`; it prints `median_step_ms` and `final_loss`.

Please take it all the way: spec, kernel with the backward, verification and benchmark against
speed of light and against `torch.compile`, and integrate it into `vision/blocks.py` behind a
flag so I can switch back to the eager code. The attention and the MLP around it are separate
requests; do not fuse them.

For a bounded correctness smoke, run `python -m engine.trainer --smoke --steps 3 --seed 0` from
`user_repo/`. This keeps target channel/head dimensions but reduces batch/token counts; use the
unmodified representative config for performance measurements.
