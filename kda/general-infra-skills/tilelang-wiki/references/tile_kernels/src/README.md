# DeepSeek TileKernels excerpts

Production kernels copied from
[TileKernels](https://github.com/deepseek-ai/TileKernels) for local
reference. MIT license is in [LICENSE](LICENSE).

These files keep their original `tile_kernels.*` imports. They are the
full source for the recipes in the parent folder — not a runnable
install of the package.

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
