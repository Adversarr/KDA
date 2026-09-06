"""positions components for the local training model."""

from typing import Tuple

import torch



def mm_rope_tables(grid: Tuple[int, int, int], rope_freq_dim: int, theta: float, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """``cos, sin: (S, 3 * rope_freq_dim * 2)`` for the packed ``(t, h, w)`` grid, rotate-half layout (angles duplicated)."""
    t, h, w = grid
    tt, hh, ww = torch.meshgrid(torch.arange(t), torch.arange(h), torch.arange(w), indexing="ij")
    position_ids = torch.stack([tt.flatten(), hh.flatten(), ww.flatten()], dim=-1).to(device)  # (S, 3)
    inv_freq = 1.0 / (theta ** (torch.arange(0, 2 * rope_freq_dim, 2, dtype=torch.float32, device=device) / (2 * rope_freq_dim)))
    freqs = position_ids.float().unsqueeze(-1) * inv_freq[None, None, :]  # (S, 3, 16)
    freqs = torch.cat(freqs.unbind(dim=1), dim=-1)  # (S, 48)
    freqs = torch.cat([freqs, freqs], dim=-1)  # (S, 96)
    return freqs.cos(), freqs.sin()
