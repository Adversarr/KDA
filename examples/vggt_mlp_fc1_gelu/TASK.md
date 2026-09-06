I am training a VGGT-style multi-view transformer (`models/mlp.py`) and want the first half of
the MLP as one fused kernel: the `1024 -> 4096` GEMM with its bias and the exact (erf) GELU in
the epilogue. The function is `mlp_fc1_gelu`; every MLP calls it once.

```python
def mlp_fc1_gelu(x, w1, b1):
    # x: (B, N, 1024) bf16;  w1: (4096, 1024) and b1: (4096,) are fp32 parameters (bf16 under autocast)
    return F.gelu(F.linear(x, w1, b1))     # (B, N, 4096) bf16, exact erf GELU
```

Today this is cuBLAS plus a separate GELU pass over a `(rows, 4096)` bf16 tensor, and the
backward runs the GELU derivative as another elementwise pass before the two GEMMs. I want the
epilogue fused into the GEMM and a sensible backward: `dx`, `dw1`, `db1`, and your call on
whether to keep the pre-activation for the GELU derivative (that is 4x the input bytes) or to
recompute it — please state the trade-off in the spec rather than pick silently.

Training config is `models/config.py` (`TrainConfig`, `ModelConfig`): bf16 autocast, batch 1, 4
views of a `37 x 37` patch grid, 1374 tokens per view, `dim` 1024, MLP hidden 4096; the function
sees 5496 rows in both the frame block and the global block. Real runs use 2 scenes of up to 24
views (66k rows). The training run is `python train.py --steps 50`; it prints `median_step_ms`
and `final_loss`.

Please take it all the way: spec, kernel with the backward, verification and benchmark against
speed of light and against `torch.compile` (which fuses the same epilogue), and integrate it
into `models/mlp.py` behind a flag so I can switch back to the eager code. The second GEMM
(`fc2`) and the residual after it are separate; do not fuse them.

For a bounded correctness smoke, run `python train.py --smoke --steps 3 --seed 0` from
`user_repo/`. This keeps target channel/head dimensions but reduces batch/token counts; use the
unmodified representative config for performance measurements.
