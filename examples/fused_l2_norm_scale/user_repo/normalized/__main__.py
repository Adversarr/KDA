"""Train with CLI configuration: python -m normalized --smoke --steps 3."""
import argparse
import statistics
import time
import torch
from .data import make_batch
from .model import NormalizedEncoder


def main():
    """Run reproducible full optimizer steps and report synchronized step time."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.batch_size, args.seq_len = 1, 37
    if min(args.steps, args.batch_size, args.seq_len) < 1:
        parser.error("steps, batch size and sequence length must be positive")
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    model = NormalizedEncoder().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    times = []
    for step in range(args.steps):
        features, target = make_batch(args.batch_size, args.seq_len, device, generator)
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(features, target)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1e3)
        print(f"step {step:3d}  loss {loss.item():.4f}  step_ms {times[-1]:.2f}", flush=True)
    steady = times[10:] if len(times) > 10 else times
    print(f"median_step_ms {statistics.median(steady):.2f}")
    print(f"final_loss {loss.item():.4f}")


if __name__ == "__main__":
    main()
