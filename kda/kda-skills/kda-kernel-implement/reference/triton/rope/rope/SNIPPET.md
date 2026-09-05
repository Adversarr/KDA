# RoPE (half and interleaved pairings, fused backward)

Files: `kernel.py`, `reference.py`.

## Math

For each (batch `b`, position `s`, head `h`) row of `D` elements and angle table row `pos`:

```
y1 = x1 * cos[pos] - x2 * sin[pos]
y2 = x2 * cos[pos] + x1 * sin[pos]
```

- `half` (LLaMA, GPT-NeoX, Qwen): `x1 = x[:D/2]`, `x2 = x[D/2:]`.
- `interleaved` (GPT-J, some ViTs): `x1 = x[0::2]`, `x2 = x[1::2]`.

Backward: rotation by `-angle`, i.e. the same kernel with `sin -> -sin` (`BACKWARD` constexpr).
No tensors are saved besides the tables.

## Dtype and layout rules

- `x (B, S, H, D)`, any strides with a unit last stride: a head slice of a fused QKV
  projection (`qkv[..., :H*D].view(B, S, H, D)`) goes straight in. `y` is contiguous.
- `cos`, `sin`: fp32 `(S_max, D/2)`, contiguous. HF passes `(S, D)` tables built as
  `cat(freqs, freqs)`; use `[:, :D/2]`. Keep the tables fp32 even for bf16 activations:
  bf16 angles lose ~3 significant digits at long positions.
- `positions (B, S)` int, optional: packed sequences and KV-cache offsets index the tables
  through it instead of the implicit `0..S-1`.
- Math in fp32; cast at load and store only. The kernel matches the fp32 eager reference to
  bf16/fp16 rounding.

## Launch geometry (A800, 8x2048x32x128 bf16)

- One program per `(b, s)` and a `BLOCK_H` block of heads; the `cos`/`sin` row is loaded once
  per program and broadcast over the head tile. `BLOCK_H = min(next_pow2(H), 2048 / BLOCK_DH)`
  keeps tiles near 2K elements. Runs at 96% of copy for both pairings.
- Interleaved pairs must be separated in registers: load the contiguous row tile and
  `tl.split(tl.reshape(x, [BLOCK_H, BLOCK_DH, 2]))`, then `tl.join` + `tl.reshape` back before
  the store. Addressing memory with `2 * offs_d` compiles to scalar accesses and runs 13x slower.
- `D/2` need not be a power of two (`D = 96` is tested): `BLOCK_DH = next_pow2(D/2)` with a mask.

## Fusing with neighbours

- **After a norm**: the normalised row is already in registers as fp32; rotate before the
  store. For per-head QK-norm the norm reduction is over the same `D` as the rotation, so
  one `[BLOCK_H, D]` tile serves both (see `fusion-exemplar/rmsnorm_rope_permute`).
- **Into a permute**: only the store address changes; write `y[b, h, s, :]` instead of
  `y[b, s, h, :]` (see `permute/bnhd_to_bhnd`).
- **Backward through a fusion**: apply the inverse rotation to `dy` first, then the norm's
  backward, in the same program.
