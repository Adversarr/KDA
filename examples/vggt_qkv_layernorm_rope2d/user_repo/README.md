# minivggt

A VGGT-style multi-view transformer core (the released aggregator's structure: camera + register
tokens per view, alternating *frame* blocks that attend inside each view and *global* blocks that
attend over every token of every view) trained on random DINOv2-sized patch tokens. It exists to
have a realistic training loop around the fusable pieces of `minivggt/model.py`, each a plain
function:

| function | what it does | called per block pair |
|---|---|---|
| `qkv_prep` | `(B, N, 3, H, d)` projection output -> q, k with per-head affine LayerNorm + 2-D RoPE, v re-laid out; all `(B, H, N, d)` | 2 |
| `padded_attention` | dense non-causal attention, per-batch key-validity mask, fp32 logits/softmax (`attention_impl = "eager"`; `"sdpa"` is the library path) | 2 |
| `layerscale_residual_layernorm` | `x + gamma * branch` (fp32 residual stream, padded rows zeroed) and the LayerNorm feeding the next branch | 4 |
| `mlp_fc1_gelu` | first MLP GEMM + bias + exact GELU | 2 |

```bash
python train_smoke.py --steps 50                # prints per-step loss/time, then median_step_ms and final_loss
python train_smoke.py --steps 50 --attention sdpa
```

Configuration is `minivggt/config.py` (`ModelConfig`, `TrainConfig`): bf16 autocast, batch 1,
4 views of a `37 x 37` patch grid (518 x 518 at patch 14) = 1369 patches + 1 camera + 4 register
tokens = 1374 tokens per view, 5496 tokens in the global block; `dim` 1024, 16 heads of dim 64,
MLP hidden 4096, one (frame, global) block pair. The smoke run pads the last 37 patches of view
1 and the whole of the last view, so the validity masks are live.
