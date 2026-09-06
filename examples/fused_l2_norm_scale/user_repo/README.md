# L2 normalization and scale training repository

Continuous representation regression with learned L2 scales at both hidden width 2048 and head
width 128.

This is an ordinary eager PyTorch training repository. It runs independently with Python and
PyTorch on a CUDA GPU, without downloaded data or pretrained weights. Synthetic data is
generated from a fixed seed. No kernel package is required.

## Run

From this directory, run a bounded training check:

```bash
python -m normalized --smoke --steps 3 --seed 0
```

For the representative workload described in the task:

```bash
python -m normalized --steps 50 --seed 0
```

`--smoke` reduces batch size, token count and/or depth while retaining the target's channel
widths and head dimensions. It is a correctness smoke, not the representative benchmark. The
final two lines are `median_step_ms` and `final_loss`. Timing includes backward and the
optimizer update, uses CUDA synchronization, and discards the first ten steps when available.
`--steps` must be positive. The training loop rejects non-finite losses.

Both 3D hidden-state tensors (D=2048) and 4D head tensors (D=128) call the same normalization
during every training step.

## Code

The selected operation is defined in `normalized/ops.py`. Configuration lives in
`normalized/__main__.py`. The surrounding model calls the operation in its real forward path;
the loss, backward and optimizer exercise its trainable parameters.

- `normalized/__main__.py`
- `normalized/data.py`
- `normalized/model.py`
- `normalized/ops.py`
