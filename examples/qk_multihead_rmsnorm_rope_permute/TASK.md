I want the q/k preparation in my attention block to be one fused kernel: per-head RMSNorm,
rotary embedding, and the transpose to `(B, H, S, D)` that `scaled_dot_product_attention` wants.
The function is `qk_prep` in `minilm/attention.py`; every block calls it once on q and k. Two
things differ from the usual RMSNorm+RoPE: the norm weights are per head (`q_norm_w` is `(Hq,
D)`, `k_norm_w` is `(Hk, D)`, each head scaled by its own row), and k has fewer heads than q
(grouped-query attention). The rotation is interleaved: pairs are `(x[2i], x[2i+1])`.

```python
def qk_prep(q, k, q_norm_w, k_norm_w, cos, sin, eps):
    # q: (B, S, Hq, D), k: (B, S, Hk, D) bf16 views of one fused qkv projection;
    # q_norm_w: (Hq, D), k_norm_w: (Hk, D) fp32; cos, sin: (S, D/2) fp32
    q = rmsnorm_per_head(q, q_norm_w, eps)
    k = rmsnorm_per_head(k, k_norm_w, eps)
    q = apply_rope(q, cos, sin, interleaved=True)
    k = apply_rope(k, cos, sin, interleaved=True)
    return q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous()
```

Note that q and k are slices of the `qkv` buffer (`Attention.forward`), so their row stride is
`(Hq + 2 Hk) * D`, not `H * D`; I would rather not pay a copy to make them contiguous first.

Training config is `minilm/config.py` (`TrainConfig`, `ModelConfig`): bf16 autocast, batch 4,
sequence 4096, `d_model` 4096, 32 query heads, 8 key/value heads, head dim 128, 2 layers. The
training run is `python -m minilm.training --steps 50`; it prints `median_step_ms` and
`final_loss`.

Please take it all the way: spec, kernel with the backward (`dq`, `dk` and the `(H, D)` weight
gradients), verification and benchmark against speed of light and against `torch.compile`, and
integrate it into `minilm/attention.py` behind a flag so I can switch back to the eager code. I
care most about the shape the config actually uses.

For a bounded correctness smoke, run `python -m minilm.training --smoke --steps 3 --seed 0` from
`user_repo/`. This keeps target channel/head dimensions but reduces batch/token counts; use the
unmodified representative config for performance measurements.
