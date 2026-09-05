"""Eager reference: the permute plus the copy PyTorch would do lazily."""

import torch


def bnhd_to_bhnd_ref(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 2, 1, 3).contiguous()


__all__ = ["bnhd_to_bhnd_ref"]
