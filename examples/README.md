# Examples

Examples for the Level-1 kernel workflow. Each directory is one kernel a user might
ask for; `TASK.md` is the request as they would type it. A complete example also carries
`golden/` (a hand-written kernel with measured A800 numbers; `golden.json` is the
machine-readable part) and `user_repo/` (a small training codebase for integration and
an end-to-end smoke run). TASK-only entries are requests, not complete runnable examples.
KDA-authored development tests are not distributed. Golden measurements are recorded
reference results, not promises of performance in a new environment.

| example | fuses | status |
|---|---|---|
| `fused_residual_rmsnorm` | residual add + RMSNorm, fp32 stream | TASK, golden, user_repo |
| `fused_residual_layernorm` | residual add + LayerNorm | TASK only |
| `qk_rmsnorm_rope_permute` | RMSNorm + RoPE + permute (q and k) | TASK only |
| `qk_multihead_rmsnorm_rope_permute` | per-head RMSNorm (GQA) + interleaved RoPE + permute | TASK, golden, user_repo |
| `f32_adaln` | LayerNorm (no affine) + per-sample adaptive scale/shift (`rowwise_onepass`, `(B, D)` modulation, `dscale`/`dshift` over tokens), optional SiLU epilogue | TASK, user_repo (`minidit`) |
| `h3_qk_norm_mmrope` | MiniMax-H3 q/k prep: per-head RMSNorm (shared weight, 56 heads) + partial 3D MM-RoPE (96 of 128 channels, rotate-half) + permute; dense non-causal attention is the follow-up | TASK, user_repo (`minih3`) |
| `h3_block_causal_attention` | MiniMax-H3 video attention: 56 heads x 128, block-causal mask at frame granularity (`frame(q) >= frame(k)`, `block_size = h * w` tokens, dense inside a frame), fp32 softmax; `flash_attention_2` with a structured mask, partial last frame, `S` up to 60k (int64 scores); `sdpa` with the boolean mask is the baseline | TASK, user_repo (`minih3`, video variant) |
| `sliding_tile_attention` | HunyuanVideo Sliding Tile Attention (FastVideo): tokens in 384-token tile order + text, per-head `(H, NT, NT)` bool tile mask (window centred on the query tile, clamped), text keys for all, text queries see all; block-sparse `flash_attention_2` that visits only allowed tile pairs (31% of the plane in the smoke, 5-15% at 115k tokens), fp32 softmax, non-128-multiple `S`; `sdpa` with the token mask is the baseline. The golden is the FA2 reference driven by host-built per-(head, Q block) lists of allowed K blocks in heavy-first order: 0.85 ms fwd / 2.70 ms bwd on the smoke row, 18x / 7.5x over masked `sdpa` | TASK, golden, user_repo (`minista`) |
| `vggt_padded_attention` | VGGT multi-view attention: 16 heads x 64, dense non-causal with a per-batch key-validity mask `(B, N)` (padded patches/views), fp32 softmax; frame block `(B*S, T=1374)` and global block `(B, S*T=5496)` shapes; `sdpa` with the boolean mask is the baseline | TASK, user_repo (`minivggt`) |
| `vggt_qkv_layernorm_rope2d` | VGGT q/k/v prep: interleaved `(B, N, 3, H, d)` input, per-head affine LayerNorm (weight + bias) on q/k, 2-D RoPE (y half / x half, base 100) from integer positions with in-kernel cos/sin, v re-laid out; one `dqkv` gradient | TASK, user_repo (`minivggt`) |
| `vggt_layerscale_residual_layernorm` | VGGT block boundary: `x + gamma * branch` (fp32 stream, bf16 branch, LayerScale gamma), padded rows zeroed by a `(B, N)` mask, affine LayerNorm to bf16; two outputs, five gradients | TASK, user_repo (`minivggt`) |
| `vggt_mlp_fc1_gelu` | VGGT MLP fc1: `1024 -> 4096` GEMM + bias + exact erf GELU (`gemm_tensorcore`), save-vs-recompute decision for the GELU derivative | TASK, user_repo (`minivggt`) |
| `fused_l2_norm_scale` | L2 norm + learned scale, wide and narrow D | TASK only |
| `fused_gemm_epilogue` | GEMM + bias + tanh-GELU (`gemm_tensorcore`); the golden is nvmath-python's cuBLASLt epilogue when importable, else the Triton twin. No residual: `act(XW^T+b)+R` is not a transformer pattern | TASK, golden, user_repo |

## Running an example

With the [Linux prerequisites](../README.md#install) installed and a GPU available,
copy a sample repository and install KDA into it. From the parent of your KDA checkout:

```bash
cp -R KDA/examples/fused_residual_rmsnorm/user_repo ./rmsnorm-example
bash KDA/kda/install.sh ./rmsnorm-example
cd rmsnorm-example
python train_smoke.py --steps 50
```

Open your coding agent in `rmsnorm-example`. Start with “Can you optimize the model here?” For a selected region, ask
“Can you fuse the residual addition and normalization in `minilm/model.py`, with inputs x,
residual and weight and outputs the residual stream and normalized activation?”
The longer [example request](fused_residual_rmsnorm/TASK.md) is an optional detailed specification. The workflow creates
`kda_kernels/<op>/`, verifies the kernel, and shows measured results and a proposed model edit for integration review, unless already authorized.
Use the Python interpreter with the GPU visible. The training command prints
`median_step_ms` and `final_loss`; compare on your own hardware, and keep the eager
backend available with `KDA_BACKEND=eager`.

## Optional container environment

`Dockerfile.cu128` supplies CUDA 12.8, torch, Triton, TileLang, and nvmath-python.
From the KDA checkout, with Docker and the NVIDIA Container Toolkit available:

```bash
docker build -f examples/Dockerfile.cu128 -t kda:dev .
docker run --rm -it --gpus all -v "$PWD":/workspace/KDA kda:dev bash
```

The image does not install a coding-agent CLI. Its package indexes can be overridden
with `--build-arg PIP_INDEX_URL=...` and `--build-arg TORCH_INDEX_URL=...`; for example,
use `https://pypi.org/simple` and `https://download.pytorch.org/whl/cu128` instead of
the supplied mirrors. `PIP_EXTRA_INDEX_URL` is an optional additional index.

## Files

- `Dockerfile.cu128`: CUDA 12.8 + torch 2.11 + triton + tilelang dev image.
- `<op>/TASK.md`: the request to give your agent.
- `<op>/user_repo/`: sample model, configuration, and training entrypoint where provided.
- `<op>/golden/`: standalone reference kernels and recorded measurements where provided.
