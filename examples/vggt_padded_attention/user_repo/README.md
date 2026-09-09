# VGGT padded attention training repository

Variable-validity multi-view feature reconstruction.

This is an ordinary eager PyTorch training repository. It runs independently with Python and
PyTorch on a CUDA GPU, without downloaded data or pretrained weights. Synthetic data is
generated from a fixed seed. No kernel package is required.

## Run

From this directory, run a bounded training check:

```bash
python -m multiview.train --smoke --steps 3 --seed 0
```

For the representative workload described in the task:

```bash
python -m multiview.train --steps 50 --seed 0
```

`--smoke` reduces batch size, token count and/or depth while retaining the target's channel
widths and head dimensions. It is a correctness smoke, not the representative benchmark. The
final two lines are `median_step_ms` and `final_loss`. Timing includes backward and the
optimizer update, uses CUDA synchronization, and discards the first ten steps when available.
`--steps` must be positive. The training loop rejects non-finite losses.

Frame attention folds views into the batch; global attention flattens them into the token axis.
`--attention sdpa` selects the library comparison path.

## Code

The selected operation is defined in `multiview/attention.py`. Configuration lives in
`multiview/config.py`. The surrounding model calls the operation in its real forward path; the
loss, backward and optimizer exercise its trainable parameters.

- `multiview/attention.py`
- `multiview/blocks.py`
- `multiview/collate.py`
- `multiview/config.py`
- `multiview/embedding.py`
- `multiview/encoder.py`
- `multiview/mlp.py`
- `multiview/train.py`

The mixed-precision eager path is validated with the repository CUDA/PyTorch 2.11
image. It uses `torch.bmm(..., out_dtype=torch.float32)` for low-precision
operands with FP32 product outputs, and an explicit Flash-style adjoint.
