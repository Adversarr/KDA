"""Independent eager implementation of the example contract."""
import torch

def adaln(x, scale, shift, eps=1e-6, silu=False):
    h = x.float()
    mean = h.mean(-1, keepdim=True)
    var = (h - mean).square().mean(-1, keepdim=True)
    y = ((h - mean) * torch.rsqrt(var + eps) * (1 + scale[:, None, :]) + shift[:, None, :]).to(x.dtype)
    return torch.nn.functional.silu(y) if silu else y
