"""Deterministic training smoke run on random patch tokens.

    python train.py [--steps 50] [--seed 0] [--attention eager|sdpa]

Prints one line per step (loss, step time) and two summary lines, ``median_step_ms`` (steps 10
onward) and ``final_loss``, so two runs can be compared by their last two lines.
"""

import argparse
import statistics
import time

import torch

from models.config import TrainConfig
from models.aggregator import MiniVGGT

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def make_masks(cfg: TrainConfig, device: torch.device):
    m = cfg.model
    b, s, p = cfg.batch_size, m.num_frames, m.num_patches
    patch_valid = torch.ones(b, s, p, dtype=torch.bool, device=device)
    if s > 1 and cfg.invalid_patch_tail > 0:
        patch_valid[:, 1, p - cfg.invalid_patch_tail :] = False
    frame_valid = torch.ones(b, s, dtype=torch.bool, device=device)
    if s >= 3 and cfg.invalid_last_frame:
        frame_valid[:, -1] = False
    return patch_valid, frame_valid


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--attention", choices=("eager", "sdpa"), default=None)
    p.add_argument("--smoke", action="store_true", help="Short sequence/batch; keep production channel widths")
    args = p.parse_args()

    cfg = TrainConfig()
    if args.smoke:
        cfg.batch_size = 1
        cfg.model.grid = (5, 5)
        cfg.model.num_frames = 3
        cfg.invalid_patch_tail = 3

    if args.steps is not None:
        cfg.steps = args.steps
    if args.seed is not None:
        cfg.seed = args.seed
    if args.attention is not None:
        cfg.model.attention_impl = args.attention

    if cfg.steps < 1:
        p.error("--steps must be positive")

    torch.manual_seed(cfg.seed)
    device = torch.device("cuda")
    dtype = DTYPES[cfg.autocast_dtype]
    model = MiniVGGT(cfg.model).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.999), weight_decay=cfg.weight_decay)
    data_gen = torch.Generator(device=device).manual_seed(cfg.seed)
    m = cfg.model
    tok_shape = (cfg.batch_size, m.num_frames, m.num_patches, m.dim)
    tgt_shape = (cfg.batch_size, m.num_frames, m.num_patches, model.out.out_features)
    patch_valid, frame_valid = make_masks(cfg, device)

    times = []
    for step in range(cfg.steps):
        patch_tokens = torch.randn(tok_shape, device=device, generator=data_gen)
        # Reconstruct a locally smoothed feature field from the patch inputs.
        features = patch_tokens[..., :tgt_shape[-1]]
        target = (features + features.roll(1, dims=2) + features.roll(-1, dims=2)) / 3
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
            loss = model(patch_tokens, target, patch_valid, frame_valid, compute_dtype=dtype)
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
