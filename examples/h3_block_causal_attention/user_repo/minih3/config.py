"""Training and model configuration (plain dataclasses; edit the defaults or override in code)."""

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class ModelConfig:
    in_channels: int = 16  # latent channels per token
    hidden_size: int = 5376  # residual width (MiniMax-H3)
    num_heads: int = 56  # 56 x 128 = 7168-wide q/k/v, wider than the residual stream
    head_dim: int = 128
    rope_freq_dim: int = 16  # per axis; 3 axes x 16 x 2 = 96 rotated channels of 128
    rope_theta: float = 10000.0
    grid: Tuple[int, int, int] = (6, 16, 32)  # (t, h, w) latent patches -> 6 frames of 512 tokens = 3072 packed tokens
    n_layers: int = 1
    mlp_ratio: int = 2
    norm_eps: float = 1e-5
    # Backend override for fused kernels (None = the kernel package's default).
    kda_backend: Optional[str] = None

    @property
    def seq_len(self) -> int:
        t, h, w = self.grid
        return t * h * w

    @property
    def block_size(self) -> int:
        """Tokens per frame: the chunk of the block-causal attention mask."""
        _, h, w = self.grid
        return h * w


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
