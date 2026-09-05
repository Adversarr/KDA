# minilm

A small pre-norm transformer trained on random tokens; exists to have a realistic training
loop around the `add_rmsnorm` function in `minilm/model.py`.

```bash
python train_smoke.py --steps 50      # prints per-step loss/time, then median_step_ms and final_loss
```

Configuration is `minilm/config.py` (`ModelConfig`, `TrainConfig`): bf16 autocast, batch 8,
sequence 512, `d_model` 1024, 4 layers.
