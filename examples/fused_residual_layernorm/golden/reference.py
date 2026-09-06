"""Independent eager implementation of the example contract."""
import torch

def add_layernorm(x, residual, weight, bias, eps=1e-5):
    h = residual + x.float()
    mean = h.mean(-1, keepdim=True)
    var = (h - mean).square().mean(-1, keepdim=True)
    y = (h - mean) * torch.rsqrt(var + eps) * weight.float() + bias.float()
    return h, y.to(x.dtype)
