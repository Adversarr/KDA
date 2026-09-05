# Head permute `(B, N, H, D) -> (B, H, N, D)`

Files: `kernel.py`, `reference.py`.

## What it is

`x.permute(0, 2, 1, 3).contiguous()` written as one explicit copy: `y[b, h, n, :] = x[b, n, h, :]`.
The permute itself is free in PyTorch; the copy it forces (either `.contiguous()` or the
hidden one inside the next kernel that needs a dense layout) is a full read and write of the
tensor. On A800 `permute+contiguous` reaches 45% of copy speed; this kernel reaches 97%.

Backward: the inverse permute is the same kernel with the middle dims read the other way
round (`permute_0213` applied to `dy` of shape `(B, H, N, D)` yields `(B, N, H, D)`).

## Layout rules

- 4-D input of any strides with a unit last stride; a head slice of a fused QKV projection
  (`qkv[..., H*D:2*H*D].view(B, N, H, D)`) goes straight in. Output is contiguous.
- Loads are `D`-element contiguous rows of `x`; stores are `D`-element rows of `y` at stride
  `N*D`. Both are fully coalesced when `D * itemsize >= 128 bytes` (`D >= 64` for bf16). For
  tiny `D` (< 32 elements) a shared-memory transpose would be needed; not covered.
- One program per `(b, n)` and a `BLOCK_2` block of heads; `BLOCK_3 = next_pow2(D)` with a
  mask (`D = 96` tested). Tiles are kept near 4K elements.

## Fusing with neighbours

A permute should never be its own launch inside a fused block: whoever produces the tensor
writes it in the permuted layout by changing the store address (`b*H*N*D + h*N*D + n*D + d`
instead of `b*N*H*D + n*H*D + h*D + d`). Its backward is then a *gather* on the permuted
gradient at the top of the fused backward kernel. `fusion-exemplar/rmsnorm_rope_permute`
shows both directions.

`NCHW <-> NHWC` for vision models is the same idea with three inner dims; the address
arithmetic generalises directly.
