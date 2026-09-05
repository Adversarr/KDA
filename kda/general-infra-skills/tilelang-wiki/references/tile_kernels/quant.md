# Quantization recipes

Production casts live in [`src/quant/`](src/quant/).
The SF ABI and helpers are centralized in
[`src/quant/common.py`](src/quant/common.py).
`QuantTensor` is just `tuple[torch.Tensor, torch.Tensor]` (data, SF)
([`src/quant/types.py`](src/quant/types.py)).

Tutorials (simpler, not the production ABI):

- [`../../examples/cast/example_per_token_cast_to_fp8.py`](../../examples/cast/example_per_token_cast_to_fp8.py)
  — row-major FP8 + `T.reduce_absmax`, eager `T.empty` outputs
- [`../../examples/deepseek_v4/act_quant.py`](../../examples/deepseek_v4/act_quant.py)
  — block-wise FP8/FP4 with power-of-two scale rounding

Public entry points: `per_token_cast`, `per_block_cast`, `per_channel_cast`
(+ `_fused` / `_and_transpose`), `swiglu_forward_and_per_*`,
`swiglu_backward_and_per_token_cast`, `cast_back`. Formats: `'e4m3'`, `'e2m1'`
(packed FP4 in `torch.int8`), `'e5m6'` (packed in `torch.uint32`).

## Scale-factor ABI

`BaseCastConfig` ([`src/quant/common.py`](src/quant/common.py)) has three
layout knobs:

| Mode | Config | Logical SF shape | Host dtype |
| --- | --- | --- | --- |
| Row-major | default | `(num_block_m, num_block_k)` | `float32` |
| TMA col-major | `use_tma_aligned_col_major_sf=True` | swapped, then `.T` in epilogue | `float32` |
| Packed UE8M0 | `use_packed_ue8m0=True` **and** TMA col-major | 4 scales packed on the token dim | `int32` view of `uint8` |

Packed UE8M0 is not an independent knob. Detection only happens on the TMA
path (`stride(0) == 1`, then `int32` → `uint8`).
`expand_to_fused_with_sf` asserts TMA when SF is `int32`.
`use_packed_ue8m0=True` without TMA col-major is the wrong contract.

`sf_block` is `(num_per_tokens, num_per_channels)`:

- per-token: `(1, K)` with `K` in `{16, 32, 64, 128}` (or full `hidden`)
- per-block: `(M, K)` with `M, K` in `{32, 128}`
- per-channel: `(M, 1)` with `M` typically `128`

`get_sf_shape` also expands UE8M0 (`num_block_m * 4`, `ceil_div(num_block_k, 4)`).
`alloc_scaling_factors` then TMA-aligns the last physical dim to 16 elements
(UE8M0) or 4 (`float32`) and returns a sliced view.

Detect an incoming `QuantTensor` with `get_cast_input_and_config`:
`x_sf.stride(0) == 1` means TMA col-major (then `.T`); packed UE8M0 is
recognized only after that, when `x_sf.dtype == int32` (then `.view(uint8)`).
Row-major SF must be `float32` with `stride(1) == 1` — it cannot be packed
UE8M0.

In-kernel access is the `load_sf` / `store_sf` / `transform_sf` macros — do
not index SF as a plain `[m, k]` matrix if either TMA or UE8M0 flag is set.

```python
@T.macro
def store_sf(tensor, sf, m_idx, k_idx, config):
    if config.use_packed_ue8m0:
        tensor[k_idx // 4, m_idx * 4 + k_idx % 4] = sf
    elif config.use_tma_aligned_col_major_sf:
        tensor[k_idx, m_idx] = sf
    else:
        tensor[m_idx, k_idx] = sf
```

`get_sf_and_inv(amax, out_config)` clamps `amax`, then `sf = amax / max_value`.
`round_sf=True` snaps to a power of two via the IEEE exponent
(`((bits - 1) >> 23) + 1 - 127`). Packed UE8M0 stores `uint8(exp + 127)` and
rebuilds the float with `transform_sf` (`uint32(sf) << 23`). Default clamps:
`1e-4` for E4M3; `max_value * 2**-126` for E2M1.

`cast_epilogue` is mandatory after launch: view UE8M0 as `int32` (empty-batch
allocates a fresh `int32` tensor), transpose TMA-col SF, then slice to
`ceil_div(num_tokens, sf_block[0])`.

FP4 packing: `torch.int8` stores two E2M1 values per byte.
`get_logical_hidden` / `get_physical_hidden` double/halve `hidden`. Debug
decode is `unpack_from_e2m1fn_x2`.

## Per-token cast

[`src/quant/per_token_cast_kernel.py`](src/quant/per_token_cast_kernel.py)
— factory `get_per_token_cast_kernel(hidden, token_stride, in_config, out_config, ...)`.

- Input may be BF16/FP32 **or** an already-quantized `QuantTensor` (requant).
- `T.StridedTensor` on `x` and both SF buffers; `out` is a compact `T.Tensor`.
- Grid: `(ceildiv(num_tokens, block_m), ceildiv(hidden, block_k))`, 128 threads.
- Amax via `T.reduce_absmax` on a reshaped fragment, then `get_sf_and_inv`,
  then scale + `T.copy(..., disable_tma=True)`.
- Variants: `per_token_cast_with_sf_only`, `per_token_cast_with_precomputed_sf`.
- `'e5m6'` redirects to `per_token_cast_to_e5m6` (warp-specialization **left on**)
  in [`src/quant/per_token_cast_to_e5m6_kernel.py`](src/quant/per_token_cast_to_e5m6_kernel.py).

Host: allocate `out` + `alloc_scaling_factors`, `if num_tokens > 0: kernel(...)`,
then `cast_epilogue`.

## Per-block cast

[`src/quant/per_block_cast_kernel.py`](src/quant/per_block_cast_kernel.py)
— 2D SF tiles. `block_size=(num_per_tokens, num_per_channels)` with both in
`{32, 128}`. 256 threads, 8192 elems/block. Same host contract as per-token
(`sf_only` / precomputed SF / `cast_epilogue`). Also
[`src/quant/per_block_cast_lossless_kernel.py`](src/quant/per_block_cast_lossless_kernel.py)
for SF-preserving requant.

Closest tutorial: [`../../examples/deepseek_v4/act_quant.py`](../../examples/deepseek_v4/act_quant.py)
(`fp8_quant_kernel` is per-row groups of 128, not the full TMA/UE8M0 ABI).

## Per-channel cast

Column-wise SF (`sf_block=(num_per_tokens, 1)`), E4M3 only.

- [`src/quant/per_channel_cast_kernel.py`](src/quant/per_channel_cast_kernel.py)
  — thin wrapper; requires `num_tokens % 128 == 0`, `hidden % 64 == 0`,
  `num_per_tokens == 128`, then calls the fused kernel.
- [`src/quant/per_channel_cast_fused_kernel.py`](src/quant/per_channel_cast_fused_kernel.py)
  — 128×128 tiles, 256 threads. Optional `pos_to_token` gather (`with_expand`)
  so MoE-expanded rows can be cast in the same launch. Expanded `num_tokens_out`
  must be `% 16 == 0`; non-expand must be `% 128 == 0`.
- [`src/quant/per_channel_cast_and_transpose_kernel.py`](src/quant/per_channel_cast_and_transpose_kernel.py)
  — writes `(hidden, num_tokens)` plus SF `(num_tokens // num_per_tokens, hidden)`.

## Fused SwiGLU + cast

Input is the packed gate/up concat `(N, 2H)` → SwiGLU → quantize `(N, H)`.

[`src/quant/swiglu_forward_and_per_token_cast_kernel.py`](src/quant/swiglu_forward_and_per_token_cast_kernel.py):

- E4M3, `num_per_channels` in `{128, hidden}`, `hidden % 128 == 0`.
- Optional `pos_to_token_topk` + `topk_weights` (scale by routing weight) and
  `pos_to_expert` (skip padded fused slots where expert is negative).
- Optional `swiglu_clamp_value`. If `clamped_count` is passed, the grid becomes
  persistent (`num_sms * 4` blocks via `get_num_sms()`) and three
  `T.alloc_reducer` counters accumulate clamp stats.

[`src/quant/swiglu_forward_and_per_channel_cast_and_transpose_kernel.py`](src/quant/swiglu_forward_and_per_channel_cast_and_transpose_kernel.py):

- BF16 in, E4M3 out. `num_tokens % 128 == 0`, `hidden % 128 == 0` (after the
  `// 2`). `num_per_tokens` in `{32, 128}`.
- `without_transpose=False` writes `(H, N)`; SF stays
  `(N // num_per_tokens, H)`.

Backward:
[`src/quant/swiglu_backward_and_per_token_cast_kernel.py`](src/quant/swiglu_backward_and_per_token_cast_kernel.py)
recomputes SwiGLU, writes BF16 `x_grad` plus a requantized FP8 copy, and uses
`T.alloc_reducer` to form `weight_grad`.

`cast_back` lives in
[`src/quant/cast_back_kernel.py`](src/quant/cast_back_kernel.py)
(E5M6 path:
[`src/quant/cast_back_e5m6_kernel.py`](src/quant/cast_back_e5m6_kernel.py)).

## Vectorization

`get_best_vectorize_size(dtype)` is `16 // dtype.bytes` on SM < 10 and
`32 // dtype.bytes` on SM ≥ 10. Per-token requant asserts
`num_per_channels >= num_vectorize` so a 16-wide SF block is illegal on
Blackwell FP8 (`num_vectorize=32`).
