# minilm

A small pre-norm transformer trained on random tokens; exists to have a realistic training
loop around `qk_prep` in `minilm/model.py`: per-head RMSNorm of q and k (one weight row per
head), interleaved RoPE, and the transpose to `(B, H, S, D)` for the attention kernel, on views
of a fused `qkv` projection.

```bash
python train_smoke.py --steps 50      # prints per-step loss/time, then median_step_ms and final_loss
```

Configuration is `minilm/config.py` (`ModelConfig`, `TrainConfig`): bf16 autocast, batch 4,
sequence 4096, `d_model` 4096, 32 query heads / 8 kv heads of dim 128, 2 layers.
