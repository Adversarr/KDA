"""Independent eager reference preserving autocast parameter and output rounding."""
import torch
import torch.nn.functional as F


def mlp_fc1_gelu(x, w1, b1):
    with torch.autocast("cuda", dtype=x.dtype):
        return F.gelu(F.linear(x, w1, b1), approximate="none")
