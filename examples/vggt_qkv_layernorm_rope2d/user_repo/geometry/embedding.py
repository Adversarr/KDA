"""embedding components for the local training model."""

from typing import Tuple

import torch



def make_positions(batch_size: int, num_frames: int, grid: Tuple[int, int], num_special: int, device: torch.device) -> torch.Tensor:
    """``(B, S, T, 2)`` int64: special tokens at ``(0, 0)``, patch ``(i, j)`` at ``(i + 1, j + 1)``."""
    gh, gw = grid
    yy, xx = torch.meshgrid(torch.arange(gh, device=device), torch.arange(gw, device=device), indexing="ij")
    patch_pos = torch.stack((yy.flatten(), xx.flatten()), dim=-1) + 1
    pos = torch.cat((torch.zeros(num_special, 2, dtype=torch.long, device=device), patch_pos), dim=0)
    return pos.view(1, 1, -1, 2).expand(batch_size, num_frames, -1, -1).contiguous()


def expand_first_vs_other(token: torch.Tensor, batch_size: int, num_frames: int) -> torch.Tensor:
    """``(1, 2, K, D)``: slot 0 for the first view, slot 1 for every other view -> ``(B, S, K, D)``."""
    first = token[:, 0:1].expand(batch_size, 1, *token.shape[2:])
    other = token[:, 1:2].expand(batch_size, max(num_frames - 1, 0), *token.shape[2:])
    return torch.cat((first, other), dim=1)
