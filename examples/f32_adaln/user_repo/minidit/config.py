"""Training and model configuration (plain dataclasses; edit the defaults or override in code)."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    in_channels: int = 16  # latent channels per patch token
    d_model: int = 1152
    d_cond: int = 256  # conditioning vector width (timestep + class embedding)
    n_layers: int = 2
    n_heads: int = 16
    n_tokens: int = 4096  # 64 x 64 latent patches
    mlp_ratio: int = 4
    norm_eps: float = 1e-6
    # Backend override for fused kernels (None = the kernel package's default).
    kda_backend: Optional[str] = None


@dataclass
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    batch_size: int = 16
    steps: int = 50
    lr: float = 1e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    seed: int = 0
    autocast_dtype: str = "bf16"  # bf16 | fp16 | fp32
