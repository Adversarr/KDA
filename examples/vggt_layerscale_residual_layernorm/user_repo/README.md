# VGGT LayerScale boundary training repository

Masked dense prediction through LayerScale residual boundaries.

This is an ordinary eager PyTorch training repository. It runs independently with Python and
PyTorch on a CUDA GPU, without downloaded data or pretrained weights. Synthetic data is
generated from a fixed seed. No kernel package is required.

## Run

From this directory, run a bounded training check:

```bash
python -m engine.trainer --smoke --steps 3 --seed 0
```

For the representative workload described in the task:

```bash
python -m engine.trainer --steps 50 --seed 0
```

`--smoke` reduces batch size, token count and/or depth while retaining the target's channel
widths and head dimensions. It is a correctness smoke, not the representative benchmark. The
final two lines are `median_step_ms` and `final_loss`. Timing includes backward and the
optimizer update, uses CUDA synchronization, and discards the first ten steps when available.
`--steps` must be positive. The training loop rejects non-finite losses.

Frame attention folds views into the batch; global attention flattens them into the token axis.
`--attention sdpa` selects the library comparison path.

## Code

The selected operation is defined in `vision/blocks.py`. Configuration lives in
`vision/config.py`. The surrounding model calls the operation in its real forward path; the
loss, backward and optimizer exercise its trainable parameters.

- `engine/trainer.py`
- `vision/attention.py`
- `vision/blocks.py`
- `vision/config.py`
- `vision/embedding.py`
- `vision/network.py`
- `vision/qkv.py`
