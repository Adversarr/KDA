# TileKernels production recipes

DeepSeek [TileKernels](https://github.com/deepseek-ai/TileKernels) is a
production kernel library on TileLang. These pages extract the house
style and a few recipes that official `examples/` do not cover.

Requires TileLang ≥ 0.1.9, SM90 or SM100, CUDA 13.1+. Copied kernel
sources live in [`src/`](src/) and are the source of truth, not this
prose.

**MHC is Manifold HyperConnection**, not MLA and not a codebook. Kernels
live in [`src/mhc/`](src/mhc/). Do not route MHC questions to
`examples/deepseek_mla/`.

## Recipes

- [quant.md](quant.md) — scale-factor ABI; per-token / per-block / per-channel
  cast; fused SwiGLU+cast
- [moe.md](moe.md) — stable top-k; `get_fused_mapping`; expand/reduce fused
- [patterns.md](patterns.md) — `T.StridedTensor`, `T.alloc_reducer`,
  `T.Persistent`, `T.async_copy` + `T.ptx_wait_group`, `T.make_tensor` pointer
  tables, `get_num_sms()`

## Tutorial counterparts (skill `examples/`)

Use these for a first kernel. Use TileKernels when you need the production ABI,
empty-batch host contract, or fused routing/quant.

| Family | Tutorial | Production |
| --- | --- | --- |
| FP8/FP4 cast | [`../../examples/cast/`](../../examples/cast/) | [`src/quant/`](src/quant/) |
| V4 act quant / SF rounding | [`../../examples/deepseek_v4/act_quant.py`](../../examples/deepseek_v4/act_quant.py) | [`src/quant/common.py`](src/quant/common.py) |
| Top-k | [`../../examples/topk/`](../../examples/topk/) | [`src/moe/topk_gate_kernel.py`](src/moe/topk_gate_kernel.py) |
| MHC | [`../../examples/deepseek_mhc/`](../../examples/deepseek_mhc/) | [`src/mhc/`](src/mhc/) (split ops; see note) |

`src/mhc/` is the split production ops: mix, pre_apply, pre_split, post,
norm, and multilayer recompute. The tutorial's fused pre
(`mhc_pre_big_fuse_tilelang` in `example_mhc_pre.py`) has no copied
production twin — `pre_big_fuse` / `sinkhorn` / `expand` were not vendored.

## House style

Almost every public op is a **factory + host launcher**, not an eager
`T.const` / `T.empty` kernel.

1. **Factory** decorated with `@tilelang.jit(...)` returns a nested
   `@T.prim_func`. Compile-time knobs are ordinary Python args (`hidden`,
   `num_topk`, `CastOutputConfig`, `num_sms`). Batch-like sizes stay
   `T.dynamic`.
2. **No autotune.** There is no `tilelang.autotune` in this tree. Specialize
   with factory args; do not add a config scan.
3. **Disable warp specialization** unless a kernel explicitly opts out:

   ```python
   @tilelang.jit(
       pass_configs={
           tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
       },
   )
   def get_topk_gate_kernel(num_experts: int, num_topk: int):
       num_tokens = T.dynamic('num_tokens')
       ...
   ```

   From [`src/moe/topk_gate_kernel.py`](src/moe/topk_gate_kernel.py).
   The known exception is `get_per_token_cast_to_e5m6_kernel`
   (`TL_DISABLE_WARP_SPECIALIZED: False`) in
   [`src/quant/per_token_cast_to_e5m6_kernel.py`](src/quant/per_token_cast_to_e5m6_kernel.py).
   MHC multilayer recompute also sets `TL_PTXAS_REGISTER_USAGE_LEVEL` /
   `TL_DISABLE_VECTORIZE_256`
   ([`src/mhc/multilayer_recompute_kernel.py`](src/mhc/multilayer_recompute_kernel.py)).
4. **Host allocates.** Launchers `torch.empty` outputs, then call the JIT
   kernel with explicit in/out buffers. Do not return `T.empty` from the
   prim_func.
5. **Empty-batch skip.** Allocate the output shape, then skip the launch:

   ```python
   topk_idx = torch.empty((num_tokens, num_topk), dtype=torch.int64, device=scores.device)
   if num_tokens == 0:
       return topk_idx
   kernel(scores, topk_idx)
   ```

   Same idea as `if num_tokens > 0: kernel(...)` in the quant launchers.
   Packed-UE8M0 SF also special-cases `num_tokens == 0` in `cast_epilogue`.
6. **Print generated CUDA** with `TK_PRINT_KERNEL_SOURCE=1`:

   ```python
   if int(os.getenv('TK_PRINT_KERNEL_SOURCE', 0)):
       print(kernel.get_kernel_source())
   ```

Occupancy for persistent / grid-sync kernels comes from
`get_num_sms()` in [`src/config.py`](src/config.py) (overridable via
`set_num_sms`). See [patterns.md](patterns.md).

## Typical factory

Compile-time: `hidden`, dtypes, SF config, thread/tile math.
Runtime-dynamic: `num_tokens` and often SF strides (TMA padding changes the
physical stride). Activation `token_stride` is usually a factory arg, not
`T.dynamic`.

```python
@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def get_per_token_cast_kernel(hidden, token_stride, in_config, out_config, ...):
    num_tokens = T.dynamic('num_tokens')
    out_sf_stride = T.dynamic('out_sf_stride')
    # ...
    @T.prim_func
    def per_token_cast_kernel(
        x: T.StridedTensor[(num_tokens, hidden), (token_stride, 1), in_config.dtype],
        ...
        out_sf: T.StridedTensor[sf_shape, (out_sf_stride, 1), out_config.sf_dtype],
    ):
        ...
    return per_token_cast_kernel
```

From [`src/quant/per_token_cast_kernel.py`](src/quant/per_token_cast_kernel.py).

## Copied sources

```
src/
├── quant/      # FP8/FP4/E5M6 cast + fused SwiGLU
├── moe/        # top-k, fused mapping, expand/reduce
├── mhc/        # Manifold HyperConnection kernels
├── engram/     # Engram gate (async pipeline)
├── transpose/  # batched transpose
├── config.py   # get_num_sms / set_num_sms
└── utils.py    # align / ceil_div / is_power_of_two
```
