"""A VGGT-style alternating frame/global transformer on already-tokenised multi-view images.

The structure follows the released VGGT aggregator (DINOv2 patch tokens -> one camera token and
`R` register tokens prepended to every view -> `depth` pairs of a *frame* block, which attends
inside each view `(B * S, T, D)`, and a *global* block, which attends over every token of every
view `(B, S * T, D)`), with the same building blocks: pre-norm blocks with LayerScale, per-head
affine LayerNorm on q and k, 2-D RoPE (y half / x half of every head, base 100), exact-erf GELU
MLP, and validity masks (padded patches, padded views) that mask keys in attention and zero the
padded rows of the residual stream.

Every fusable piece is a plain function so a fused kernel can replace it one at a time:

- `qkv_prep`: the `(B, N, 3, H, d)` projection output -> q, k (per-head LayerNorm + 2-D RoPE) and
  v, all head-major `(B, H, N, d)`.
- `padded_attention`: dense non-causal attention with a per-batch key-validity mask, fp32
  logits and softmax (`ModelConfig.attention_impl = "eager"`; `"sdpa"` is the library path).
- `layerscale_residual_layernorm`: `x + gamma * branch`, padded rows zeroed, and the LayerNorm
  that feeds the next branch; the residual stream stays fp32.
- `mlp_fc1_gelu`: the first MLP GEMM with bias and exact GELU.

The blocks are written residual-first: a block receives the residual `x` and its own normalized
input `y = LN(x)` and returns the updated `x` together with the *next* block's normalized input,
so each residual add is immediately followed by the LayerNorm that consumes it.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig

# =============================================================================
# 1. q/k/v preparation: per-head LayerNorm + 2-D RoPE + head-major layout
# =============================================================================


def layernorm_heads(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
    """``x: (B, H, N, d)``, ``weight, bias: (d,)``: affine LayerNorm over the head dim of every head, in fp32."""
    xf = x.float()
    mean = xf.mean(dim=-1, keepdim=True)
    var = xf.var(dim=-1, unbiased=False, keepdim=True)
    y = (xf - mean) * torch.rsqrt(var + eps)
    return (y * weight.float() + bias.float()).to(x.dtype)


def rope2d_inv_freq(head_dim: int, base: float) -> torch.Tensor:
    """``(d / 4,)`` fp32: one frequency per rotated pair of one axis half (``axis_dim = d / 2`` channels)."""
    axis_dim = head_dim // 2
    exponent = torch.arange(0, axis_dim, 2, dtype=torch.float32) / axis_dim
    return 1.0 / (base**exponent)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope2d(x: torch.Tensor, positions: torch.Tensor, inv_freq: torch.Tensor) -> torch.Tensor:
    """2-D rotary embedding on ``x: (B, H, N, d)`` from integer ``positions: (B, N, 2)`` = (y, x).

    Channels ``[0, d/2)`` of every head are rotated by the token's y coordinate and channels
    ``[d/2, d)`` by its x coordinate, rotate-half inside each half (``x1, x2 = half.chunk(2)``).
    Angles, cos and sin are computed in fp32 (``angle = pos * inv_freq``, duplicated for the two
    rotate-half quarters); special tokens sit at position ``(0, 0)`` and are left unrotated.
    """
    xf = x.float()
    parts = []
    for part, pos in zip(xf.chunk(2, dim=-1), (positions[..., 0], positions[..., 1])):
        angle = pos.float()[:, None, :, None] * inv_freq  # (B, 1, N, d/4)
        angle = torch.cat((angle, angle), dim=-1)  # (B, 1, N, d/2)
        parts.append(part * angle.cos() + rotate_half(part) * angle.sin())
    return torch.cat(parts, dim=-1).to(x.dtype)


def qkv_prep(
    qkv: torch.Tensor,
    q_norm_w: torch.Tensor,
    q_norm_b: torch.Tensor,
    k_norm_w: torch.Tensor,
    k_norm_b: torch.Tensor,
    positions: torch.Tensor,
    inv_freq: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``qkv: (B, N, 3, H, d)`` (the projection output viewed) -> ``q, k, v`` each ``(B, H, N, d)``.

    q and k get the per-head affine LayerNorm (one ``(d,)`` weight and bias each, shared by all
    heads) followed by the 2-D RoPE; v is only re-laid out. All three come out contiguous and
    head-major, the layout the attention consumes.
    """
    q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
    q = apply_rope2d(layernorm_heads(q, q_norm_w, q_norm_b, eps), positions, inv_freq)
    k = apply_rope2d(layernorm_heads(k, k_norm_w, k_norm_b, eps), positions, inv_freq)
    return q.contiguous(), k.contiguous(), v.contiguous()


# =============================================================================
# 2. Attention with a key-validity mask
# =============================================================================


def key_valid_from_token_valid(valid: torch.Tensor) -> torch.Tensor:
    """``valid: (B, N)`` bool -> the keys attention may use; a view with no valid token gets key 0 as a harmless dummy."""
    key_valid = valid.clone()
    has_no_key = ~key_valid.any(dim=-1)
    if has_no_key.any():
        key_valid[has_no_key, 0] = True
    return key_valid


def padded_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, key_valid: torch.Tensor) -> torch.Tensor:
    """``q, k, v: (B, H, N, d)``, ``key_valid: (B, N)`` bool (True = this key may be attended) -> ``(B, H, N, d)``.

    Dense, non-causal. Logits, softmax and the PV accumulation are explicit fp32 (autocast is
    disabled inside, exactly as the model's reference path does it); the result is cast back to
    v's dtype. Every query row attends to the same key set of its batch entry, so the mask is a
    function of ``key_valid`` alone and is never stored as an ``(N, N)`` plane. The ``(B, H, N, N)``
    fp32 logits are what make this the memory and time bottleneck of the global block.
    """
    d = q.shape[-1]
    with torch.autocast(device_type=q.device.type, enabled=False):
        logits = torch.matmul(q.float(), k.float().transpose(-2, -1)) * (d**-0.5)
        logits = logits.masked_fill(~key_valid[:, None, None, :], float("-inf"))
        prob = torch.softmax(logits, dim=-1)
        prob = torch.nan_to_num(prob, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.matmul(prob, v.float()).to(v.dtype)


def sdpa_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, key_valid: torch.Tensor) -> torch.Tensor:
    """The library path: SDPA with the boolean key mask broadcast to ``(B, 1, 1, N)``."""
    return F.scaled_dot_product_attention(q, k, v, attn_mask=key_valid[:, None, None, :], is_causal=False)


# =============================================================================
# 3. Residual + LayerScale + LayerNorm (residual-first block boundary)
# =============================================================================


def layerscale_residual_layernorm(
    x: torch.Tensor,
    branch: torch.Tensor,
    gamma: torch.Tensor,
    norm_w: torch.Tensor,
    norm_b: torch.Tensor,
    eps: float,
    valid: torch.Tensor,
    out_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``x: (B, N, D)`` fp32 residual stream, ``branch: (B, N, D)`` (attention or MLP output, the
    autocast dtype), ``gamma: (D,)`` fp32 LayerScale, ``norm_w, norm_b: (D,)``, ``valid: (B, N)`` bool.

    Returns the new residual ``x + gamma * branch`` (fp32, padded rows set to zero so biases
    never revive them) and its affine LayerNorm in ``out_dtype``, the next branch's input.
    """
    x_new = x + branch.float() * gamma
    x_new = x_new.masked_fill(~valid.unsqueeze(-1), 0.0)
    y = F.layer_norm(x_new, (x_new.shape[-1],), norm_w, norm_b, eps).to(out_dtype)
    return x_new, y


# =============================================================================
# 4. MLP
# =============================================================================


def mlp_fc1_gelu(x: torch.Tensor, w1: torch.Tensor, b1: torch.Tensor) -> torch.Tensor:
    """``x: (B, N, D)`` -> ``(B, N, M)``: the first MLP GEMM with bias and the exact (erf) GELU."""
    return F.gelu(F.linear(x, w1, b1))


# =============================================================================
# 5. Modules
# =============================================================================


class SelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.h, self.d = cfg.num_heads, cfg.head_dim
        self.eps = cfg.norm_eps
        self.impl = cfg.attention_impl
        self.qkv = nn.Linear(cfg.dim, 3 * cfg.dim, bias=True)
        self.q_norm_w = nn.Parameter(torch.ones(self.d))
        self.q_norm_b = nn.Parameter(torch.zeros(self.d))
        self.k_norm_w = nn.Parameter(torch.ones(self.d))
        self.k_norm_b = nn.Parameter(torch.zeros(self.d))
        self.register_buffer("inv_freq", rope2d_inv_freq(self.d, cfg.rope_base), persistent=False)
        self.proj = nn.Linear(cfg.dim, cfg.dim, bias=True)

    def forward(self, y: torch.Tensor, positions: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """``y: (B, N, D)`` normalized tokens, ``positions: (B, N, 2)`` int64, ``valid: (B, N)`` bool -> ``(B, N, D)``."""
        b, n, _ = y.shape
        qkv = self.qkv(y).view(b, n, 3, self.h, self.d)
        q, k, v = qkv_prep(qkv, self.q_norm_w, self.q_norm_b, self.k_norm_w, self.k_norm_b, positions, self.inv_freq, self.eps)
        key_valid = key_valid_from_token_valid(valid)
        if self.impl == "eager":
            o = padded_attention(q, k, v, key_valid)
        elif self.impl == "sdpa":
            o = sdpa_attention(q, k, v, key_valid)
        else:
            raise ValueError(f"unknown attention_impl {self.impl!r}")
        out = self.proj(o.transpose(1, 2).reshape(b, n, self.h * self.d))
        # Padded queries produce finite garbage; zero them after the projection.
        return out.masked_fill(~valid.unsqueeze(-1), 0)


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden = int(cfg.dim * cfg.mlp_ratio)
        self.fc1 = nn.Linear(cfg.dim, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, cfg.dim, bias=True)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.fc2(mlp_fc1_gelu(y, self.fc1.weight, self.fc1.bias))


class Block(nn.Module):
    """One pre-norm block, residual-first: ``(x, y) -> (x', y')``.

        x  <- zero_padded(x + ls1 * Attention(y));   y  <- LN2(x)
        x' <- zero_padded(x + ls2 * MLP(y));         y' <- LN_out(x')     (the next block's LN1)
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.eps = cfg.norm_eps
        self.attn = SelfAttention(cfg)
        self.mlp = MLP(cfg)
        self.ls1 = nn.Parameter(torch.full((cfg.dim,), cfg.layer_scale_init))
        self.ls2 = nn.Parameter(torch.full((cfg.dim,), cfg.layer_scale_init))
        self.norm2_w = nn.Parameter(torch.ones(cfg.dim))
        self.norm2_b = nn.Parameter(torch.zeros(cfg.dim))
        self.norm_out_w = nn.Parameter(torch.ones(cfg.dim))
        self.norm_out_b = nn.Parameter(torch.zeros(cfg.dim))

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, positions: torch.Tensor, valid: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        a = self.attn(y, positions, valid)
        x, y = layerscale_residual_layernorm(x, a, self.ls1, self.norm2_w, self.norm2_b, self.eps, valid, y.dtype)
        m = self.mlp(y)
        x, y = layerscale_residual_layernorm(x, m, self.ls2, self.norm_out_w, self.norm_out_b, self.eps, valid, y.dtype)
        return x, y


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


class MiniVGGT(nn.Module):
    """Camera + register tokens, ``depth`` (frame, global) block pairs, a linear head on the patch tokens."""

    def __init__(self, cfg: ModelConfig, out_channels: int = 64):
        super().__init__()
        self.cfg = cfg
        self.camera_token = nn.Parameter(torch.empty(1, 2, 1, cfg.dim))
        self.register_token = nn.Parameter(torch.empty(1, 2, cfg.num_register_tokens, cfg.dim))
        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)
        self.norm_in_w = nn.Parameter(torch.ones(cfg.dim))
        self.norm_in_b = nn.Parameter(torch.zeros(cfg.dim))
        self.frame_blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.depth))
        self.global_blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.depth))
        self.out = nn.Linear(cfg.dim, out_channels)
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(
        self,
        patch_tokens: torch.Tensor,
        target: torch.Tensor,
        patch_valid: torch.Tensor,
        frame_valid: torch.Tensor,
        compute_dtype: torch.dtype,
    ) -> torch.Tensor:
        """``patch_tokens: (B, S, P, D)`` fp32, ``target: (B, S, P, C)``, ``patch_valid: (B, S, P)``, ``frame_valid: (B, S)`` -> masked MSE."""
        cfg = self.cfg
        b, s, p, d = patch_tokens.shape
        p0 = cfg.patch_start_index
        camera = expand_first_vs_other(self.camera_token, b, s)
        registers = expand_first_vs_other(self.register_token, b, s)
        x = torch.cat((camera, registers, patch_tokens), dim=2)  # (B, S, T, D) fp32 residual stream
        t = x.shape[2]

        positions = make_positions(b, s, cfg.grid, p0, x.device)
        special_valid = frame_valid[:, :, None].expand(b, s, p0)
        patch_valid = patch_valid & frame_valid[:, :, None]
        valid = torch.cat((special_valid, patch_valid), dim=2)  # (B, S, T)

        x = x.masked_fill(~valid.unsqueeze(-1), 0.0)
        y = F.layer_norm(x, (d,), self.norm_in_w, self.norm_in_b, cfg.norm_eps).to(compute_dtype)
        for frame_block, global_block in zip(self.frame_blocks, self.global_blocks):
            # Frame block: views fold into the batch, every view attends inside itself.
            x, y = frame_block(x.view(b * s, t, d), y.view(b * s, t, d), positions.view(b * s, t, 2), valid.view(b * s, t))
            # Global block: every token of every view in one sequence.
            x, y = global_block(x.view(b, s * t, d), y.view(b, s * t, d), positions.view(b, s * t, 2), valid.view(b, s * t))
        y = y.view(b, s, t, d)

        pred = self.out(y[:, :, p0:]).float()
        w = patch_valid.unsqueeze(-1).float()
        return ((pred - target.float()) ** 2 * w).sum() / (w.sum() * pred.shape[-1])
