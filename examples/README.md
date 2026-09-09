# Examples

Evaluation cases for the Level-1 kernel workflow. Each directory tests how a coding agent and
model perform on a particular kernel task when KDA is supplied in context.

- `user_repo/` is the input fixture: ordinary model code, configuration and a runnable workload.
- `TASK.md` describes the request and mathematical contract.
- `golden/` is the evaluator's reference solution, with numerical and performance evidence.
Keep it outside the repository and context given to the agent under evaluation.

Install KDA into an isolated copy of `user_repo/` and run the selected harness there. Generated
kernels, trajectories and other run artifacts belong to that run workspace, not to `golden/`.
The implementation and validation rules in `kda/` are the single source of truth; examples do
not carry their own copy of the KDA runtime or another evaluation framework.

For automated isolated runs with Cursor, Pi, Claude Code or Codex, see the
[Python blind-evaluation runner](../scripts/BLIND_EVAL.md). It provides explicit GPU
assignment, per-example containers, live logs, stopping and preserved run artifacts.
Its report summary is agent-reported evidence, not independent golden acceptance.

A golden is accepted only when both numerical correctness and performance correctness pass.
Performance uses KDA's audited achievable Speed-of-Light roof and baseline gates; recorded
timings alone do not establish acceptance. Pending, failing and tuning results must remain
explicit in the measurement record.

| Example | Status | Outstanding gate |
|---|---|---|
| [Residual RMSNorm](fused_residual_rmsnorm/TASK.md) | [`pass`](fused_residual_rmsnorm/golden/GOLDEN.md) | None. |
| [Residual LayerNorm](fused_residual_layernorm/TASK.md) | [`pass`](fused_residual_layernorm/golden/GOLDEN.md) | None. |
| [QK RMSNorm and RoPE](qk_rmsnorm_rope_permute/TASK.md) | [`pass`](qk_rmsnorm_rope_permute/golden/GOLDEN.md) | None. |
| [GQA RMSNorm and RoPE](qk_multihead_rmsnorm_rope_permute/TASK.md) | [`pass`](qk_multihead_rmsnorm_rope_permute/golden/GOLDEN.md) | None. |
| [Adaptive LayerNorm](f32_adaln/TASK.md) | [`pass`](f32_adaln/golden/GOLDEN.md) | None. |
| [H3 QK normalization and MM-RoPE](h3_qk_norm_mmrope/TASK.md) | [`pass`](h3_qk_norm_mmrope/golden/GOLDEN.md) | None. |
| [H3 block-causal attention](h3_block_causal_attention/TASK.md) | [`pass`](h3_block_causal_attention/golden/GOLDEN.md) | None. |
| [Sliding-tile attention](sliding_tile_attention/TASK.md) | [`incomplete`](sliding_tile_attention/golden/GOLDEN.md) | Real-model profiling reached backward round 3 but timed out after 90 minutes; complete report missing. |
| [VGGT padded attention](vggt_padded_attention/TASK.md) | [`pass`](vggt_padded_attention/golden/GOLDEN.md) | None. |
| [Segmented multi-view attention (VGGT-style extension)](vggt_segmented_attention/TASK.md) | [`pass`](vggt_segmented_attention/golden/GOLDEN.md) | Two explicit H20 backward acceptance exceptions; synthetic extension, not official VGGT. |
| [VGGT QKV preparation](vggt_qkv_layernorm_rope2d/TASK.md) | [`pass`](vggt_qkv_layernorm_rope2d/golden/GOLDEN.md) | None. |
| [VGGT LayerScale boundary](vggt_layerscale_residual_layernorm/TASK.md) | [`pass`](vggt_layerscale_residual_layernorm/golden/GOLDEN.md) | None. |
| [VGGT exact-GELU MLP](vggt_mlp_fc1_gelu/TASK.md) | [`pass`](vggt_mlp_fc1_gelu/golden/GOLDEN.md) | Compiled parity policy; SoL diagnostic. |
| [L2 normalization and scale](fused_l2_norm_scale/TASK.md) | [`pass`](fused_l2_norm_scale/golden/GOLDEN.md) | None. |
| [GEMM epilogue](fused_gemm_epilogue/TASK.md) | [`pass`](fused_gemm_epilogue/golden/GOLDEN.md) | None. |

## Running an example

With the [Linux prerequisites](../README.md#install) installed and a GPU available, copy a
sample repository and install KDA into it. From the parent of your KDA checkout:

```bash
cp -R KDA/examples/fused_residual_rmsnorm/user_repo ./rmsnorm-example
bash KDA/kda/install.sh ./rmsnorm-example
cd rmsnorm-example
python train_smoke.py --steps 50
```

Open your coding agent in `rmsnorm-example`. Start with “Can you optimize the model here?” For a
selected region, ask “Can you fuse the residual addition and normalization in `minilm/model.py`,
with inputs x, residual and weight and outputs the residual stream and normalized activation?”
The longer [example request](fused_residual_rmsnorm/TASK.md) is an optional detailed
specification. The workflow creates `kda_kernels/<op>/`, verifies the kernel, and shows measured
results and a proposed model edit for integration review, unless already authorized. Use the
Python interpreter with the GPU visible. The training command prints `median_step_ms` and
`final_loss`. User fixtures start as eager code; the agent adds an eager fallback when it
integrates its generated kernel.

## Optional container environment

`Dockerfile.cu128` supplies CUDA 12.8, torch, Triton, TileLang, and nvmath-python. From the KDA
checkout, with Docker and the NVIDIA Container Toolkit available:

```bash
docker build -f examples/Dockerfile.cu128 -t kda:dev .
docker run --rm -it --gpus all -v "$PWD":/workspace/KDA kda:dev bash
```

The image does not install a coding-agent CLI. Its package indexes can be overridden with
`--build-arg PIP_INDEX_URL=...` and `--build-arg TORCH_INDEX_URL=...`; for example, use
`https://pypi.org/simple` and `https://download.pytorch.org/whl/cu128` instead of the supplied
mirrors. `PIP_EXTRA_INDEX_URL` is an optional additional index.

## Files

- `Dockerfile.cu128`: CUDA 12.8 + torch 2.11 + triton + tilelang dev image.
- `<op>/TASK.md`: the request to give your agent.
- `<op>/user_repo/`: standalone eager model, configuration and training entrypoint.
- `<op>/golden/`: withheld kernel, independent eager reference, summary and recorded evidence.
- `<op>/benchmark.py`: runnable numerical, timing, and adjoint checks of the golden.
- `<op>/benchmark_cases.py`: operation-specific workloads, gradient reductions, and SoL bytes/FLOPs.
- `_benchmark.py` and `_check_adjoint.py`: shared orchestration that imports the product runtime directly; the runtime is not copied into examples.

New examples follow this same five-entry layout (`TASK.md`, `user_repo/`, `golden/`,
`benchmark.py`, `benchmark_cases.py`). Put fixture setup and provenance in
`user_repo/README.md`, and evaluator commands and results in `golden/GOLDEN.md`.
Reuse the common `benchmark.py` wrapper; expose `kernel_fn`, `reference_fn`, `ROOF`,
`cases()` and `roof()` from `benchmark_cases.py`, plus any operation-specific hooks.
Training entries support `--smoke`, `--steps` and `--seed` and retain the
`median_step_ms` and `final_loss` summary lines.

Keep the standard `golden.json` headline fields (`status`, `workload`, `gpu`,
`measurement`, `verification`, `kernel_ms`, `baseline_ms`, `roof_ms`, `sol_ms`,
`derived`, `coverage`, `performance_verdict`, `workloads` and `samples`). A headline
identifies one GPU/workload; additional-device evidence is supplementary. Unmeasured
values remain null and acceptance stays incomplete until its gates have been measured.

Run `python examples/<op>/benchmark.py --help` for options. Each golden document lists
reproducible commands. Generated reports belong under `tmp/golden-benchmarks/` or another
directory outside `golden/`. Full benchmarks preserve all three timing rounds and use the
minimum round median, matching the recorded measurement protocol. Scoped checks require an
explicit output path. The input fixture for an evaluation is `user_repo/`; keep the golden
solution and its validation scripts withheld from the agent being evaluated.

[Benchmarking](BENCHMARKING.md) documents the shared measurement protocol, SoL accounting and
workload scope. Odd-row and deliberately abnormal stress cases remain visible as optional
diagnostics; normal model workloads retain all acceptance gates.
