"""Independent eager implementation of the example contract."""
import torch

def layerscale_residual_layernorm(x, branch, gamma, norm_w, norm_b, eps, valid, out_dtype=torch.bfloat16):
    h = (x + branch.float() * gamma).masked_fill(~valid.unsqueeze(-1), 0.)
    return h, torch.nn.functional.layer_norm(h, (h.shape[-1],), norm_w, norm_b, eps).to(out_dtype)
