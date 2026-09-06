"""Normalization shared by the hidden stream and attention heads."""
import torch


def l2_norm_scale(x: torch.Tensor, scale: torch.Tensor, eps: float) -> torch.Tensor:
    """L2-normalize the last dimension, then apply a learned channel scale."""
    h = x.float()
    return (h * torch.rsqrt(h.square().sum(-1, keepdim=True) + eps) * scale.float()).to(x.dtype)
