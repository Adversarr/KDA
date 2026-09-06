"""masks components for the local training model."""

from typing import List, Tuple

import torch



def sta_tile_mask(
    tiles: Tuple[int, int, int], windows: List[Tuple[int, int, int]], device: torch.device
) -> torch.Tensor:
    """``(H, NT, NT)`` bool: ``mask[h, qt, kt]`` is True when key tile ``kt`` is inside head ``h``'s window around query tile ``qt``.

    Per axis with ``n`` tiles and window ``k`` (odd, ``k <= n``): ``centre = clamp(q, k // 2, n - 1 - k // 2)``
    and the key tile is allowed when ``|kt - centre| <= k // 2``. Tiles are enumerated t-major.
    """
    coords = torch.stack(torch.meshgrid(*[torch.arange(n, device=device) for n in tiles], indexing="ij"), dim=-1).view(-1, 3)
    masks = []
    for k in windows:
        allowed = torch.ones(coords.shape[0], coords.shape[0], dtype=torch.bool, device=device)
        for axis, (n, kk) in enumerate(zip(tiles, k)):
            assert kk % 2 == 1 and kk <= n, f"window {k} does not fit the tile grid {tiles}"
            q = coords[:, axis]
            centre = q.clamp(kk // 2, n - 1 - kk // 2)
            allowed &= (coords[None, :, axis] - centre[:, None]).abs() <= kk // 2
        masks.append(allowed)
    return torch.stack(masks)


def sta_token_mask(tile_mask: torch.Tensor, tile_size: int, text_len: int) -> torch.Tensor:
    """``(H, S, S)`` bool from the ``(H, NT, NT)`` tile mask: dense inside a tile pair, text keys for every query, text queries see all."""
    H, NT, _ = tile_mask.shape
    video_len = NT * tile_size
    S = video_len + text_len
    m = torch.ones(H, S, S, dtype=torch.bool, device=tile_mask.device)
    vid = tile_mask.repeat_interleave(tile_size, dim=1).repeat_interleave(tile_size, dim=2)
    m[:, :video_len, :video_len] = vid
    return m
