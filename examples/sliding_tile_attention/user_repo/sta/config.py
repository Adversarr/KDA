"""Training and model configuration (plain dataclasses; edit the defaults or override in code)."""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class ModelConfig:
    in_channels: int = 16  # latent channels per video token
    text_channels: int = 64  # text-encoder width per text token
    hidden_size: int = 1024
    num_heads: int = 8
    head_dim: int = 128
    canvas: Tuple[int, int, int] = (18, 24, 24)  # (t, h, w) latent grid -> 10368 video tokens
    tile: Tuple[int, int, int] = (6, 8, 8)  # STA tile: 384 tokens; canvas / tile = (3, 3, 3) = 27 tiles
    # Per-head window in tiles (t, h, w), odd sizes; (3, 3, 3) over a (3, 3, 3) tile grid is dense.
    windows: List[Tuple[int, int, int]] = field(
        default_factory=lambda: [(3, 3, 3), (1, 3, 3), (3, 1, 3), (3, 3, 1), (1, 1, 3), (1, 3, 1), (3, 1, 1), (1, 1, 1)]
    )
    text_len: int = 128  # text tokens appended after the video tokens
    n_layers: int = 1
    mlp_ratio: int = 4
    norm_eps: float = 1e-6

    @property
    def tile_size(self) -> int:
        t, h, w = self.tile
        return t * h * w

    @property
    def tiles(self) -> Tuple[int, int, int]:
        return tuple(c // s for c, s in zip(self.canvas, self.tile))

    @property
    def num_tiles(self) -> int:
        a, b, c = self.tiles
        return a * b * c

    @property
    def video_len(self) -> int:
        t, h, w = self.canvas
        return t * h * w

    @property
    def seq_len(self) -> int:
        return self.video_len + self.text_len


@dataclass
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    batch_size: int = 1
    steps: int = 50
    lr: float = 1e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    seed: int = 0
    autocast_dtype: str = "bf16"  # bf16 | fp16 | fp32
