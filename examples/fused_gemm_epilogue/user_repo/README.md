# GEMM epilogue training repository

Decoder language modeling with a separately owned tanh-GELU feed-forward projection.

This is an ordinary eager PyTorch training repository. It runs independently with Python and
PyTorch on a CUDA GPU, without downloaded data or pretrained weights. Synthetic data is
generated from a fixed seed. No kernel package is required.

## Run

From this directory, run a bounded training check:

```bash
python train_lm.py --smoke --steps 3 --seed 0
```

For the representative workload described in the task:

```bash
python train_lm.py --steps 50 --seed 0
```

`--smoke` reduces batch size, token count and/or depth while retaining the target's channel
widths and head dimensions. It is a correctness smoke, not the representative benchmark. The
final two lines are `median_step_ms` and `final_loss`. Timing includes backward and the
optimizer update, uses CUDA synchronization, and discards the first ten steps when available.
`--steps` must be positive. The training loop rejects non-finite losses.

Use `--config experiment.json` to select a JSON experiment; omitted model fields use the Python
defaults.

## Code

The selected operation is defined in `language/feedforward.py`. Configuration lives in
`experiment.json`, with defaults in `language/config.py`. The surrounding model calls the
operation in its real forward path; the loss, backward and optimizer exercise its trainable
parameters.

- `language/config.py`
- `language/feedforward.py`
- `language/transformer.py`
- `train_lm.py`
