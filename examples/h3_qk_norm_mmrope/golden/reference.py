"""Independent eager shared RMSNorm and partial MM-RoPE."""
import torch

def qk_prep(q, k, q_norm_w, k_norm_w, cos, sin, eps=1e-6):
    def one(x, w):
        h = x.float()
        n = (h * torch.rsqrt(h.square().mean(-1, keepdim=True) + eps) * w).to(x.dtype).float()
        a, b = n[..., :48], n[..., 48:96]
        c, s = cos[None, :, None, :48], sin[None, :, None, :48]
        y = torch.cat((a*c-b*s, b*c+a*s, n[..., 96:]), -1).to(x.dtype)
        return y.transpose(1, 2).contiguous()
    return one(q, q_norm_w), one(k, k_norm_w)
