"""Independent eager implementation of the example contract."""
import torch

def l2_norm_scale(x, scale, eps=1e-6):
    h = x.float()
    return (h * torch.rsqrt(h.square().sum(-1, keepdim=True) + eps) * scale.float()).to(x.dtype)
