I want the first projection of my MLP to be one fused operation: the GEMM with its epilogue (bias
add, tanh-GELU), output in bf16. The function is `fc1_gelu` in `minilm/model.py`; every block's
MLP calls it once and it produces the widest activation in the model.

```python
def fc1_gelu(x, weight, bias):
    # x: (..., K) bf16 under autocast, weight: (N, K) fp32 parameter, bias: (N,) fp32
    z = F.linear(x, weight.to(x.dtype), bias.to(x.dtype))
    return F.gelu(z, approximate="tanh").to(x.dtype)
```

Training config is `minilm/config.py` (`TrainConfig`, `ModelConfig`): bf16 autocast, batch 8,
sequence 1024, `d_model` 2048, MLP hidden 8192, 4 layers, so the GEMM is M 8192 x N 8192 x
K 2048. The smoke run is `python train_smoke.py --steps 50`; it prints `median_step_ms` and `final_loss`.

Please take it all the way: spec, the fused forward with its backward (`dx`, `dweight`,
`dbias`), fp32 accumulation on tensor cores, verification and benchmark against speed of light
and against `torch.compile`, and integrate it into `minilm/model.py` behind a flag so I can
switch back to the eager code. I do not care whether you write a kernel or drive a library, as
long as it is the fastest correct thing on this GPU. I care most about the shape the config
actually uses; the activation memory of this layer matters to me too, so tell me if there is a
cheaper way to save less for the backward.
