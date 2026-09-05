I am training a VGGT-style multi-view transformer (`minivggt/model.py`) and want the q/k/v
preparation after the QKV projection as one fused kernel: the per-head LayerNorm on q and k, the
2-D RoPE on q and k, and the re-layout of all three to head-major `(B, H, N, d)`. The function
is `qkv_prep`; every attention calls it once.

```python
def qkv_prep(qkv, q_norm_w, q_norm_b, k_norm_w, k_norm_b, positions, inv_freq, eps):
    # qkv: (B, N, 3, H, d) bf16, the (B, N, 3 * dim) projection output viewed, H = 16, d = 64
    # q_norm_w, q_norm_b, k_norm_w, k_norm_b: (d,) fp32, one LayerNorm weight/bias each, shared by all heads
    # positions: (B, N, 2) int64 (y, x) per token;  inv_freq: (d / 4,) fp32 = 1 / 100 ** (arange(0, 32, 2) / 32)
    q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
    q = apply_rope2d(layernorm_heads(q, q_norm_w, q_norm_b, eps), positions, inv_freq)
    k = apply_rope2d(layernorm_heads(k, k_norm_w, k_norm_b, eps), positions, inv_freq)
    return q.contiguous(), k.contiguous(), v.contiguous()
```

Three things differ from the usual RMSNorm+RoPE. The norm is an affine *LayerNorm* over the 64
channels of a head (mean and variance, weight and bias), not RMSNorm. The RoPE is *2-D*: the
first 32 channels of a head are rotated by the token's y coordinate and the last 32 by its x
coordinate, rotate-half inside each 32-channel half (`x1, x2 = half.chunk(2)`), with the angles
`pos * inv_freq` (duplicated for the two quarters) and their cos/sin computed in fp32 from the
integer positions — there is no precomputed table, the kernel gets `positions` and `inv_freq`.
Special tokens sit at `(0, 0)` and come out unrotated. And the input is the *interleaved* qkv
tensor: the rows are `(B, N)`, and q, k, v are three slices of one 3072-wide row, so the kernel
should read that layout directly rather than have me split it first. The backward likewise
should produce one `(B, N, 3, H, d)` gradient for the projection (plus the four `(64,)` weight
gradients), not three.

Training config is `minivggt/config.py` (`TrainConfig`, `ModelConfig`): bf16 autocast, batch 1,
4 views of a `37 x 37` patch grid, 1374 tokens per view (1 camera + 4 register + 1369 patches),
`dim` 1024 = 16 heads x 64. The function runs twice per block pair, on `(B * S, T)` rows in the
frame block and `(B, S * T)` rows in the global block (same number of rows, 5496 here). Real
runs use 2 scenes of up to 24 views (66k rows) and larger grids; patch coordinates stay small
non-negative integers. The smoke run is `python train_smoke.py --steps 50`; it prints `median_step_ms` and `final_loss`.

Please take it all the way: spec, kernel with the backward (`dqkv` and the four weight
gradients), verification and benchmark against speed of light and against `torch.compile`, and
integrate it into `minivggt/model.py` behind a flag so I can switch back to the eager code. The
attention after it (`padded_attention` / `sdpa_attention`) is a separate request; do not fuse it
into this kernel.

Status: TASK, user_repo.
