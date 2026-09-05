# minih3 (video, block-causal)

One MiniMax-H3-style transformer block trained on random video latents packed as a `(t, h, w)`
token grid, frame by frame; exists to have a realistic training loop around
`block_causal_attention` in `minih3/model.py`: scaled dot-product attention over 56 heads of
dim 128 whose mask is **causal at frame granularity and dense inside a frame**
(`frame(q) >= frame(k)`, `frame(i) = i // block_size`, `block_size = h * w` tokens per frame),
softmax in fp32. The q/k preparation in front of it (per-head RMSNorm + 3D MM-RoPE, `qk_prep`)
is eager and stays so.

```bash
python train_smoke.py --steps 50      # prints per-step loss/time, then median_step_ms and final_loss
```

Configuration is `minih3/config.py` (`ModelConfig`, `TrainConfig`): bf16 autocast, batch 1,
grid `(6, 16, 32)` = 6 frames of 512 tokens = 3072 tokens, `hidden_size` 5376, 56 heads of
dim 128 (q/k/v are 7168 wide), 1 layer.
