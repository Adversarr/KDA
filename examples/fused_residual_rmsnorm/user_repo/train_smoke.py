"""Deterministic training smoke run on random tokens.

    python train_smoke.py [--steps 50] [--seed 0]

Prints one line per step (loss, step time) and two summary lines, ``median_step_ms`` (steps 10
onward) and ``final_loss``, so two runs can be compared by their last two lines.
"""

import argparse
import statistics
import time

import torch

from minilm.config import TrainConfig
from minilm.model import MiniLM

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--smoke", action="store_true", help="Short sequence/batch; keep production channel widths")
    args = p.parse_args()

    cfg = TrainConfig()
    if args.smoke:
        cfg.batch_size = 1
        cfg.model.seq_len = 31
        cfg.model.n_layers = 1

    if args.steps is not None:
        cfg.steps = args.steps
    if args.seed is not None:
        cfg.seed = args.seed

    if cfg.steps < 1:
        p.error("--steps must be positive")

    torch.manual_seed(cfg.seed)
    device = torch.device("cuda")
    dtype = DTYPES[cfg.autocast_dtype]
    model = MiniLM(cfg.model).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.95), weight_decay=cfg.weight_decay)
    data_gen = torch.Generator(device=device).manual_seed(cfg.seed)

    times = []
    for step in range(cfg.steps):
        tokens = torch.randint(
            0, cfg.model.vocab_size, (cfg.batch_size, cfg.model.seq_len + 1), device=device, generator=data_gen
        )
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
            loss = model(tokens[:, :-1], tokens[:, 1:], compute_dtype=dtype)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite training loss")
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
