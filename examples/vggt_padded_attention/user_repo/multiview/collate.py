"""Scene padding and validity masks."""
import torch
from .config import TrainConfig


def make_masks(cfg: TrainConfig, device: torch.device):
    """Collate views of different synthetic lengths into one padded scene batch."""
    m = cfg.model
    b, s, p = cfg.batch_size, m.num_frames, m.num_patches
    rows = torch.arange(p, device=device)[None, None, :]
    view = torch.arange(s, device=device)[None, :]
    scene = torch.arange(b, device=device)[:, None]
    lengths = p - ((view + scene) * cfg.invalid_patch_tail).remainder(p)
    patch_valid = rows < lengths[..., None]
    frame_valid = torch.ones(b, s, dtype=torch.bool, device=device)
    if s >= 3 and cfg.invalid_last_frame:
        frame_valid[:, -1] = False
    return patch_valid, frame_valid
