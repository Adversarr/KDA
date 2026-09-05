I am reimplementing the MiniMax-H3 attention block (`minih3/model.py`) and want the q/k
preparation as one fused kernel: per-head RMSNorm, the 3D MM-RoPE, and the transpose to
`(B, H, S, D)`. The function is `qk_prep`; every block calls it once on q and k.

```python
def qk_prep(q, k, q_norm_w, k_norm_w, cos, sin, eps):
    # q, k: (B, S, H, D) bf16 with H = 56, D = 128;  q_norm_w, k_norm_w: (D,) fp32, shared by all heads
    # cos, sin: (S, 96) fp32: 3 axes (t, h, w) x 16 frequencies, duplicated for rotate-half
    q = apply_h3_rope(rmsnorm_heads(q, q_norm_w, eps), cos, sin)
    k = apply_h3_rope(rmsnorm_heads(k, k_norm_w, eps), cos, sin)
    return q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous()
```

Two things differ from the usual RMSNorm+RoPE. The rotation is *partial*: only the first 96 of
the 128 channels of each head are rotated (rotate-half convention, `x1, x2 = x[..., :48],
x[..., 48:96]`), the last 32 pass through untouched. And the positions are 3-D: each token has a
`(t, h, w)` coordinate and the tables are already built per token (`mm_rope_tables`), so the
kernel only has to index `cos`/`sin` by the token's position in the packed sequence. The norm
weight is one `(128,)` vector for all heads, not one row per head.

Training config is `minih3/config.py` (`TrainConfig`, `ModelConfig`): bf16 autocast, batch 1,
a `(4, 16, 32)` grid = 2048 packed tokens, `hidden_size` 5376, 56 heads of dim 128 (q/k/v are
7168 wide, wider than the residual), 1 layer. Real runs pack text, video and audio tokens into
one sequence of tens of thousands of tokens, so please make sure the kernel is correct there
too. The smoke run is `python train_smoke.py --steps 50`; it prints `median_step_ms` and `final_loss`.

Please take it all the way: spec, kernel with the backward (`dq`, `dk` and the two `(D,)`
weight gradients), verification and benchmark against speed of light and against
`torch.compile`, and integrate it into `minih3/model.py` behind a flag so I can switch back to
the eager code. The attention itself (dense, non-causal, fp32 softmax, `H3Attention.forward`)
is a separate request for later; do not fuse it into this kernel.

Status: TASK, user_repo.
