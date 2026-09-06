"""Independent eager affine LayerNorm and coordinate-driven 2-D RoPE."""
import torch

def qkv_prep(qkv, q_norm_w, q_norm_b, k_norm_w, k_norm_b, positions, inv_freq, eps=1e-6):
    q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
    def one(x, w, b):
        h = x.float()
        mean = h.mean(-1, keepdim=True)
        n = ((h - mean) * torch.rsqrt(h.var(-1, unbiased=False, keepdim=True) + eps) * w + b).to(x.dtype).float()
        result = []
        for axis, part in enumerate(n.chunk(2, -1)):
            angle = positions[..., axis].float()[:, None, :, None] * inv_freq
            c, s = angle.cos(), angle.sin()
            a, b = part.chunk(2, -1)
            result.append(torch.cat((a*c-b*s, b*c+a*s), -1))
        return torch.cat(result, -1).to(x.dtype).contiguous()
    return one(q, q_norm_w, q_norm_b), one(k, k_norm_w, k_norm_b), v.contiguous()
