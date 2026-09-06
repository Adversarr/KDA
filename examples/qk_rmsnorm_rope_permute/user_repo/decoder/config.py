"""Training and model configuration (plain dataclasses; edit the defaults or override in code)."""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    vocab_size: int = 8192
    d_model: int = 4096
    n_layers: int = 2
    n_heads: int = 32  # query heads
    n_kv_heads: int = 32  # same number of key/value and query heads
    head_dim: int = 128
    seq_len: int = 2048
    mlp_ratio: int = 2
    norm_eps: float = 1e-6
    rope_base: float = 10000.0


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
