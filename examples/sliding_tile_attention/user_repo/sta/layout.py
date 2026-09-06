"""layout components for the local training model."""

from typing import Tuple

import torch



def tile_order(canvas: Tuple[int, int, int], tile: Tuple[int, int, int], device: torch.device) -> torch.Tensor:
    """``(video_len,)`` long: ``perm[i]`` is the raster index of the ``i``-th token in tile order."""
    T, H, W = canvas
    tt, th, tw = tile
    idx = torch.arange(T * H * W, device=device).view(T, H, W)
    idx = idx.view(T // tt, tt, H // th, th, W // tw, tw).permute(0, 2, 4, 1, 3, 5)  # (nT, nH, nW, tt, th, tw)
    return idx.reshape(-1)
