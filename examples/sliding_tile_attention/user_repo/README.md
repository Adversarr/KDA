# minista (video DiT, Sliding Tile Attention)

One HunyuanVideo-style DiT block trained on random video latents plus text tokens; exists to
have a realistic training loop around `sliding_tile_attention` in `minista/model.py`. The
video tokens are kept in **tile order** (tiles of `(6, 8, 8)` = 384 contiguous tokens,
enumerated t-major over the tile grid) with the text tokens appended. Every head has its own
window in tiles: a query tile attends to the `kt x kh x kw` tiles centred on it (centre clamped
into the grid) plus every text token; text queries attend to everything. The mask is
block-sparse at tile granularity (`sta_tile_mask`, `(H, NT, NT)` bool, built once) and dense
inside a tile pair; the eager attention expands it to a token mask and materialises the fp32
score plane, softmax in fp32.

```bash
python train_smoke.py --steps 50      # prints per-step loss/time, then median_step_ms and final_loss
```

Configuration is `minista/config.py` (`ModelConfig`, `TrainConfig`): bf16 autocast, batch 1,
canvas `(18, 24, 24)` = 10368 video tokens in 27 tiles + 128 text tokens, `hidden_size` 1024,
8 heads of dim 128 with windows from `(3, 3, 3)` (dense on this grid) down to `(1, 1, 1)`,
1 layer.
