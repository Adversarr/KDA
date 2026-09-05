# minilm

A small pre-norm transformer trained on random tokens; exists to have a realistic training
loop around `fc1_gelu` in `minilm/model.py`, the MLP's first projection with its epilogue
(bias, tanh-GELU).

```bash
python train_smoke.py --steps 50      # prints per-step loss/time, then median_step_ms and final_loss
```

Configuration is `minilm/config.py` (`ModelConfig`, `TrainConfig`): bf16 autocast, batch 8,
sequence 1024, `d_model` 2048, MLP hidden 8192, 4 layers.
