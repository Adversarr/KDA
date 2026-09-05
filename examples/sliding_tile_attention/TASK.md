I am training a HunyuanVideo-style DiT (`minista/model.py`) with Sliding Tile Attention and
the attention is the bottleneck: it expands the tile mask to a token mask and materialises the
full fp32 score plane, and at real video lengths it does not fit. I want
`sliding_tile_attention` as one fused flash-style kernel, forward and backward.

```python
def sliding_tile_attention(q, k, v, tile_mask, tile_size, text_len):
    # q, k, v: (B, H, S, D) bf16, D = 128, S = NT * tile_size + text_len, video tokens in tile order then text
    # tile_mask: (H, NT, NT) bool, built once per model (sta_tile_mask); tile_size = 384
    scores = (q @ k.transpose(-2, -1)).float() * (D ** -0.5)
    scores = scores.masked_fill(~sta_token_mask(tile_mask, tile_size, text_len)[None], float("-inf"))
    probs = torch.softmax(scores, dim=-1).to(v.dtype)
    return probs @ v
```

The structure is block-sparse at tile granularity: video query tile `qt` of head `h` attends
to the video key tiles with `tile_mask[h, qt, kt]` true (its window, a fixed number of tiles
per head because the window centre is clamped into the grid) and to all `text_len` text keys;
text queries attend to every key. Inside an allowed tile pair the attention is dense. So the
kernel should only visit the allowed `(q tile, k tile)` pairs plus the text block: with the
smoke windows that is 31% of the score plane, and on real HunyuanVideo shapes 5-15%. The
softmax is in fp32 (q, k, v are bf16 under autocast). `tile_size = 384 = 3 x 128`, so 128-row
tiles align with it; `text_len` is not a multiple of anything (128 here, 256 in the real
model, and it can be anything, including 0). Please do not assume `S` is a multiple of 128.

Training config is `minista/config.py` (`TrainConfig`, `ModelConfig`): bf16 autocast, batch 1,
canvas `(18, 24, 24)` = 27 tiles of 384 = 10368 video tokens + 128 text tokens, `hidden_size`
1024, 8 heads of dim 128 with per-head windows from `(3, 3, 3)` (dense on this 3 x 3 x 3 tile
grid) down to `(1, 1, 1)`, 1 layer. Real runs are HunyuanVideo: canvas `(30, 48, 80)` = 300
tiles = 115200 video tokens + 256 text, 24 heads of dim 128, windows such as `(3, 3, 3)`,
`(3, 6, 10)` and `(5, 6, 10)`; the eager code cannot allocate the scores there, so the kernel
has to be correct at that length too (24 heads x 115456 tokens is far past 2^31 score
elements; the tile mask there is `(24, 300, 300)`). The smoke run is
`python train_smoke.py --steps 50`; it prints
`median_step_ms` and `final_loss`.

Please take it all the way: spec, kernel with the backward (`dq`, `dk`, `dv`; the mask has no
gradient), verification and benchmark against speed of light and against `torch.compile` and
`F.scaled_dot_product_attention` with the boolean token mask, and integrate it into
`minista/model.py` behind a flag so I can switch back to the eager code. Keep the tile order
and the mask construction as they are; the kernel receives the tile mask, it does not
recompute windows.
