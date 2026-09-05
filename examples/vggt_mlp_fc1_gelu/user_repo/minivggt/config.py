"""Training and model configuration (plain dataclasses; edit the defaults or override in code)."""

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class ModelConfig:
    dim: int = 1024  # token width (VGGT aggregator: 1024)
    num_heads: int = 16  # 16 x 64
    depth: int = 1  # alternating frame/global pairs (VGGT: 24)
    mlp_ratio: float = 4.0  # MLP hidden 4096
    num_register_tokens: int = 4  # per view, after the camera token
    norm_eps: float = 1e-5
    rope_base: float = 100.0  # 2-D RoPE frequency base (VGGT uses 100, not 10000)
    layer_scale_init: float = 1e-2
    grid: Tuple[int, int] = (37, 37)  # patch grid of a 518 x 518 image at patch 14 -> 1369 patch tokens
    num_frames: int = 4  # views per scene
    # "eager": the explicit fp32-softmax attention (the model's reference math);
    # "sdpa":  torch.nn.functional.scaled_dot_product_attention with the boolean mask.
    attention_impl: str = "sdpa"
    # Backend override for fused kernels (None = the kernel package's default).
    kda_backend: Optional[str] = None

    @property
    def head_dim(self) -> int:
        return self.dim // self.num_heads

    @property
    def num_patches(self) -> int:
        gh, gw = self.grid
        return gh * gw

    @property
    def patch_start_index(self) -> int:
        """1 camera token + R register tokens precede the patches of every view."""
        return 1 + self.num_register_tokens

    @property
    def tokens_per_frame(self) -> int:
        return self.patch_start_index + self.num_patches

    @property
    def global_tokens(self) -> int:
        """Sequence length of the global block: every token of every view."""
        return self.num_frames * self.tokens_per_frame


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
    # Padding pattern of the smoke run, so the validity masks are exercised: the last
    # `invalid_patch_tail` patches of view 1 are padding, and the last view is entirely
    # padding when there are at least three views.
    invalid_patch_tail: int = 37
    invalid_last_frame: bool = True
