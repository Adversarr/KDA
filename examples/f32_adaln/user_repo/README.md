# minidit

A small DiT (diffusion transformer) with adaLN-Zero blocks trained on random latents; exists
to have a realistic training loop around `adaln` in `minidit/model.py`: LayerNorm without
affine parameters followed by the per-sample `scale` and `shift` the conditioning MLP produces,
computed in fp32 and cast back to the activation dtype. Every block calls it twice (attention
and MLP branches); the final layer calls it once more followed by a SiLU.

```bash
python train_smoke.py --steps 50      # prints per-step loss/time, then median_step_ms and final_loss
```

Configuration is `minidit/config.py` (`ModelConfig`, `TrainConfig`): bf16 autocast, batch 16,
4096 tokens, `d_model` 1152, 16 heads, 2 layers; the conditioning path stays fp32.
