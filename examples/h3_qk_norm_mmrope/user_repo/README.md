# H3 QK normalization and MM-RoPE training repository

Packed video denoising with partial 3D rotation and wider attention projections.

This is an ordinary eager PyTorch training repository. It runs independently with Python and
PyTorch on a CUDA GPU, without downloaded data or pretrained weights. Synthetic data is
generated from a fixed seed. No kernel package is required.

## Run

From this directory, run a bounded training check:

```bash
python -m h3.train --smoke --steps 3 --seed 0
```

For the representative workload described in the task:

```bash
python -m h3.train --steps 50 --seed 0
```

`--smoke` reduces batch size, token count and/or depth while retaining the target's channel
widths and head dimensions. It is a correctness smoke, not the representative benchmark. The
final two lines are `median_step_ms` and `final_loss`. Timing includes backward and the
optimizer update, uses CUDA synchronization, and discards the first ten steps when available.
`--steps` must be positive. The training loop rejects non-finite losses.

## Code

The selected operation is defined in `h3/attention.py`. Configuration lives in `h3/config.py`.
The surrounding model calls the operation in its real forward path; the loss, backward and
optimizer exercise its trainable parameters.

- `h3/attention.py`
- `h3/config.py`
- `h3/model.py`
- `h3/positions.py`
- `h3/train.py`
