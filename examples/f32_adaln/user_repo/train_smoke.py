"""Deterministic training smoke run on random latents.

    python train_smoke.py [--steps 50] [--seed 0]

Prints one line per step (loss, step time) and two summary lines, ``median_step_ms`` (steps 10
onward) and ``final_loss``, so two runs can be compared by their last two lines.
"""

import argparse
import statistics
import time

import torch

from minidit.config import TrainConfig
from minidit.model import MiniDiT

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    cfg = TrainConfig()
    if args.steps is not None:
        cfg.steps = args.steps
    if args.seed is not None:
        cfg.seed = args.seed

    torch.manual_seed(cfg.seed)
    device = torch.device("cuda")
    dtype = DTYPES[cfg.autocast_dtype]
    model = MiniDiT(cfg.model).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.999), weight_decay=cfg.weight_decay)
    data_gen = torch.Generator(device=device).manual_seed(cfg.seed)
    shape = (cfg.batch_size, cfg.model.n_tokens, cfg.model.in_channels)

    times = []
    for step in range(cfg.steps):
        latents = torch.randn(shape, device=device, generator=data_gen)
        target = torch.randn(shape, device=device, generator=data_gen)
        t = torch.randint(0, 1000, (cfg.batch_size,), device=device, generator=data_gen)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
            loss = model(latents, t, target, compute_dtype=dtype)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
        print(f"step {step:3d}  loss {loss.item():.4f}  step_ms {times[-1]:.2f}", flush=True)

    steady = times[10:] if len(times) > 10 else times
    print(f"median_step_ms {statistics.median(steady):.2f}")
    print(f"final_loss {loss.item():.4f}")


if __name__ == "__main__":
    main()
