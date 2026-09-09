"""Self-contained synthetic training with alternating frame/global attention.

Run from user_repo: python -m segmented.train --smoke --steps 3 --seed 0
Prints per-step loss/time, then median_step_ms and final_loss summary lines.
"""

import argparse
import statistics
import time
import torch
from torch import nn
from .attention import segmented_attention


class Model(nn.Module):
    def __init__(self, heads=16, width=64):
        super().__init__()
        self.heads, self.width = heads, width
        dim = heads * width
        self.qkv = nn.Linear(dim, 3 * dim)
        self.qnorm, self.knorm = nn.LayerNorm(width), nn.LayerNorm(width)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, lengths):
        b, s, t, d = x.shape
        for global_mode in (False, True):
            tokens = x.reshape(b, s * t, d) if global_mode else x.reshape(b * s, t, d)
            metadata = (
                lengths if global_mode else tuple((n,) for row in lengths for n in row)
            )
            q, k, v = (
                self.qkv(tokens)
                .reshape(*tokens.shape[:2], 3, self.heads, self.width)
                .permute(2, 0, 3, 1, 4)
                .unbind(0)
            )
            q, k = self.qnorm(q).to(v.dtype), self.knorm(k).to(v.dtype)
            out = segmented_attention(q, k, v, metadata, t)
            x = x + self.proj(out.transpose(1, 2).flatten(2)).reshape_as(x).float()
        return x


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    o = p.parse_args()
    if o.steps <= 0:
        p.error("--steps must be positive")
    torch.manual_seed(o.seed)
    t = 17 if o.smoke else 1374
    lengths = (
        (t, max(0, t - (3 if o.smoke else 37)), max(0, t - (6 if o.smoke else 74)), 0),
    )
    model = Model().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randn(1, 4, t, 1024, device="cuda")
    target = torch.randn_like(x)
    times = []
    for step in range(o.steps):
        torch.cuda.synchronize()
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x, lengths)
        # Ignore padding in the loss without constructing a token mask.
        errors = [
            (out[b, i, :n] - target[b, i, :n]).square().reshape(-1)
            for b, row in enumerate(lengths)
            for i, n in enumerate(row)
            if n
        ]
        loss = torch.cat(errors).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite loss")
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
        print(
            f"step {step:3d}  loss {loss.item():.4f}  step_ms {times[-1]:.2f}",
            flush=True,
        )
    print(f"median_step_ms {statistics.median(times[10:] or times):.2f}")
    print(f"final_loss {loss.item():.4f}")


if __name__ == "__main__":
    main()
