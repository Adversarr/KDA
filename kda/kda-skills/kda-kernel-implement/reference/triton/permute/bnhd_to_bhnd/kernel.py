"""Head permute ``(B, N, H, D) -> (B, H, N, D)`` in Triton, as a contiguous copy.

Attention wants heads outermost; projections produce tokens outermost. ``.permute(0, 2, 1, 3)``
is free in PyTorch but the ``.contiguous()`` (or the copy hidden inside the next op) is a full
pass over the tensor. This snippet is that pass written explicitly so it can be fused into the
store of whatever produces the tensor (norm, RoPE, gating): the store address is all that
changes.

Both directions are the same kernel: swapping dims 1 and 2 of the input equals swapping dims 1
and 2 of the output, so the backward (``(B, H, N, D) -> (B, N, H, D)``) calls it with the roles
of ``N`` and ``H`` exchanged.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _permute_0213_kernel(
    x_ptr, y_ptr,
    D1, D2, D3,
    sx0, sx1, sx2,
    BLOCK_2: tl.constexpr, BLOCK_3: tl.constexpr,
):
    # y[a, c, b, :] = x[a, b, c, :]. One program per (a, b) and a tile over (c, d): loads are
    # D3-contiguous rows of x; stores are D3-contiguous rows of y at stride D1*D3.
    pid = tl.program_id(0).to(tl.int64)
    a = pid // D1
    b = pid % D1
    offs_c = (tl.program_id(1) * BLOCK_2 + tl.arange(0, BLOCK_2)).to(tl.int64)
    offs_d = tl.arange(0, BLOCK_3)
    mask = (offs_c[:, None] < D2) & (offs_d[None, :] < D3)
    tile = tl.load(x_ptr + a * sx0 + b * sx1 + offs_c[:, None] * sx2 + offs_d[None, :], mask=mask)
    y_off = a * (D2 * D1 * D3) + offs_c[:, None] * (D1 * D3) + b * D3 + offs_d[None, :]
    tl.store(y_ptr + y_off, tile, mask=mask)


def permute_0213(x: torch.Tensor) -> torch.Tensor:
    """Contiguous ``x.permute(0, 2, 1, 3)`` for any-strided 4-D ``x`` with a unit last stride."""
    D0, D1, D2, D3 = x.shape
    assert x.stride(3) == 1
    y = torch.empty(D0, D2, D1, D3, dtype=x.dtype, device=x.device)
    block_3 = triton.next_power_of_2(D3)
    block_2 = max(1, min(triton.next_power_of_2(D2), 4096 // block_3))  # ~4K-element tiles
    grid = (D0 * D1, triton.cdiv(D2, block_2))
    _permute_0213_kernel[grid](
        x, y, D1, D2, D3, x.stride(0), x.stride(1), x.stride(2),
        BLOCK_2=block_2, BLOCK_3=block_3, num_warps=4,
    )
    return y


class _Permute(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return permute_0213(x)

    @staticmethod
    def backward(ctx, dy):
        return permute_0213(dy)


def bnhd_to_bhnd(x: torch.Tensor) -> torch.Tensor:
    """``(B, N, H, D) -> (B, H, N, D)`` contiguous; differentiable (the adjoint is the inverse permute)."""
    return _Permute.apply(x)


bhnd_to_bnhd = bnhd_to_bhnd  # identical op with the middle dims read the other way round

__all__ = ["bnhd_to_bhnd", "bhnd_to_bnhd", "permute_0213"]
