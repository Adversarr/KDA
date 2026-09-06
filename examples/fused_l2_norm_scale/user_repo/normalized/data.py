"""Seeded synthetic regression data; no dataset download is needed."""
import torch


def make_batch(batch_size, seq_len, device, generator):
    """Local and neighboring-token features define a fixed regression problem."""
    x = torch.randn(batch_size, seq_len, 64, device=device, generator=generator)
    target = torch.tanh(x[..., :32] + 0.25 * x.roll(1, dims=1)[..., 32:])
    return x, target
