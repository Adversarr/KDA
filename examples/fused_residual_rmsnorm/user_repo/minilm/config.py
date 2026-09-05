"""Training and model configuration (plain dataclasses; edit the defaults or override in code)."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    vocab_size: int = 8192
    d_model: int = 1024
    n_layers: int = 4
    n_heads: int = 8
    seq_len: int = 512
    mlp_ratio: int = 4
    norm_eps: float = 1e-6
    # Backend override for fused kernels (None = the kernel package's default).
    kda_backend: Optional[str] = None


@dataclass
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    batch_size: int = 8
    steps: int = 50
    lr: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    seed: int = 0
    autocast_dtype: str = "bf16"  # bf16 | fp16 | fp32
