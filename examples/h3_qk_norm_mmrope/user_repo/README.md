# minih3

One MiniMax-H3-style transformer block trained on random latents packed as a `(t, h, w)` token
grid; exists to have a realistic training loop around `qk_prep` in `minih3/model.py`: per-head
RMSNorm of q and k (one `(128,)` weight shared by the 56 heads), 3D MM-RoPE that rotates the
first 96 of the 128 head channels (rotate-half, `cos`/`sin` tables `(S, 96)` already summed over
the three axes), and the transpose to `(B, H, S, D)`. The attention that follows is dense and
non-causal with an fp32 softmax, as in the public H3 code.

```bash
python train_smoke.py --steps 50      # prints per-step loss/time, then median_step_ms and final_loss
```

Configuration is `minih3/config.py` (`ModelConfig`, `TrainConfig`): bf16 autocast, batch 1,
grid `(4, 16, 32)` = 2048 tokens, `hidden_size` 5376, 56 heads of dim 128 (q/k/v are 7168
wide), 1 layer.
