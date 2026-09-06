# VGGT QKV preparation training repository

Position-aware multi-view feature regression with special tokens.

This is an ordinary eager PyTorch training repository. It runs independently with Python and
PyTorch on a CUDA GPU, without downloaded data or pretrained weights. Synthetic data is
generated from a fixed seed. No kernel package is required.

## Run

From this directory, run a bounded training check:

```bash
python fit.py --smoke --steps 3 --seed 0
```

For the representative workload described in the task:

```bash
python fit.py --steps 50 --seed 0
```

`--smoke` reduces batch size, token count and/or depth while retaining the target's channel
widths and head dimensions. It is a correctness smoke, not the representative benchmark. The
final two lines are `median_step_ms` and `final_loss`. Timing includes backward and the
optimizer update, uses CUDA synchronization, and discards the first ten steps when available.
`--steps` must be positive. The training loop rejects non-finite losses.

Frame attention folds views into the batch; global attention flattens them into the token axis.
`--attention sdpa` selects the library comparison path.

## Code

The selected operation is defined in `geometry/qkv.py`. Configuration lives in
`geometry/config.py`. The surrounding model calls the operation in its real forward path; the
loss, backward and optimizer exercise its trainable parameters.

- `fit.py`
- `geometry/attention.py`
- `geometry/backbone.py`
- `geometry/blocks.py`
- `geometry/config.py`
- `geometry/embedding.py`
- `geometry/mlp.py`
- `geometry/qkv.py`
