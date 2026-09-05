# CUDA Kernel Wiki

Structured knowledge base for CUDA kernel optimization, implementation lookup, and architecture-aware tuning. The current corpus is deepest on NVIDIA Blackwell and Hopper work, but the navigation is organized around reusable kernel concerns rather than a single event or ingestion project.

For command-line usage and trigger guidance, start with [SKILL.md](SKILL.md). For worked query flows, see [references/examples.md](references/examples.md).

## Recommended Query Tools

```bash
python3 scripts/query.py "<natural language>" [--tag <t>] [--type <kernel|technique|pr|...>]
python3 scripts/get_page.py <page-id-or-path> [--follow-sources]
python3 scripts/grep_wiki.py "<regex>" [--only wiki|sources]
```

## Quick Navigation

| I want to... | Go to |
|---|---|
| Diagnose a performance problem | [queries/by-problem.md](queries/by-problem.md) |
| Learn a specific optimization technique | [queries/by-technique.md](queries/by-technique.md) |
| Study a hardware feature | [queries/by-hardware-feature.md](queries/by-hardware-feature.md) |
| Browse PR-backed implementation history | [queries/by-repo.md](queries/by-repo.md) |
| Find a kernel family | [queries/by-kernel-type.md](queries/by-kernel-type.md) |
| Compare languages and DSLs | [queries/by-language.md](queries/by-language.md) |

## Hardware Features

- [hw-tcgen05-mma](wiki/hardware/tcgen05-mma.md) — tensor-core MMA on Blackwell
- [hw-tmem](wiki/hardware/tmem.md) — dedicated accumulator storage
- [hw-clc](wiki/hardware/clc.md) — cluster launch control for persistent scheduling
- [hw-tma](wiki/hardware/tma.md) — bulk tensor movement and multicast
- [hw-2sm-cooperative](wiki/hardware/2sm-cooperative.md) — two-SM cooperative MMA
- [hw-nvfp4](wiki/hardware/nvfp4.md) — NVFP4 and block-scaled narrow precision
- [hw-pdl-gdc](wiki/hardware/pdl-gdc.md) — dependent launch and launch coordination

## Optimization Techniques

- [technique-warp-specialization](wiki/techniques/warp-specialization.md) — assign distinct warp roles
- [technique-persistent-kernels](wiki/techniques/persistent-kernels.md) — persistent scheduling and tail-effect reduction
- [technique-swizzling](wiki/techniques/swizzling.md) — shared-memory bank-conflict control
- [technique-pipeline-stages](wiki/techniques/pipeline-stages.md) — multi-stage load/compute overlap
- [technique-epilogue-fusion](wiki/techniques/epilogue-fusion.md) — fuse post-processing into the main kernel
- [technique-tile-scheduling](wiki/techniques/tile-scheduling.md) — improve locality and work balance
- [technique-double-buffering](wiki/techniques/double-buffering.md) — overlap memory movement with compute
- [technique-software-exp](wiki/techniques/software-exp.md) — software softmax exponentials
- [technique-fine-grained-quantization](wiki/techniques/fine-grained-quantization.md) — FP8/FP4 scaling strategies
- [technique-vectorized-loads](wiki/techniques/vectorized-loads.md) — saturate bandwidth with wide loads

## Kernel Case Studies

- [kernel-flash-attention-4](wiki/kernels/flash-attention-4.md) — Blackwell-native attention design
- [kernel-deepgemm](wiki/kernels/deepgemm.md) — FP8 GEMM with block scaling
- [kernel-flashmla](wiki/kernels/flashmla.md) — dense and sparse MLA kernels
- [kernel-nsa](wiki/kernels/nsa.md) — native sparse attention
- [kernel-gated-delta-net](wiki/kernels/gated-delta-net.md) — chunk-parallel linear attention
- [kernel-grouped-gemm](wiki/kernels/grouped-gemm.md) — grouped GEMM for MoE workloads
- [kernel-fused-moe](wiki/kernels/fused-moe.md) — fused expert kernels

## Problem Patterns

- [pattern-low-sm-utilization](wiki/patterns/low-sm-utilization.md) — under-filled SMs and poor tile balance
- [pattern-memory-bound](wiki/patterns/memory-bound.md) — bandwidth-limited kernels
- [pattern-register-pressure](wiki/patterns/register-pressure.md) — low occupancy from register demand
- [pattern-compute-bound](wiki/patterns/compute-bound.md) — kernels missing peak throughput
- [pattern-tail-effect](wiki/patterns/tail-effect.md) — late-wave underutilization

## Languages And DSLs

- [lang-cute-dsl](wiki/languages/cute-dsl.md) — CuTe DSL
- [lang-cuda-cpp](wiki/languages/cuda-cpp.md) — CUDA C++ with inline PTX where needed
- [lang-ptx](wiki/languages/ptx-sm100.md) — low-level PTX reference
- [lang-triton](wiki/languages/triton-blackwell.md) — Triton lowering and practical limits

## Migration Guides

- [migration-wgmma-to-tcgen05](wiki/migration/wgmma-to-tcgen05.md) — Hopper to Blackwell tensor-core migration
- [migration-register-to-tmem](wiki/migration/register-to-tmem.md) — moving accumulators out of registers

## Source Coverage

| Repository | Focus |
|---|---|
| [NVIDIA/cutlass](queries/by-repo.md#nvidiacutlass) | tensor-core kernels, templates, and schedules |
| [sgl-project/sglang](queries/by-repo.md#sgl-projectsglang) | serving-oriented kernel integrations |
| [vllm-project/vllm](queries/by-repo.md#vllm-projectvllm) | attention, MoE, and decode kernels |
| [flashinfer-ai/flashinfer](queries/by-repo.md#flashinfer-aiflashinfer) | optimized inference kernels |
| [pytorch/pytorch](queries/by-repo.md#pytorchpytorch) | compiler and kernel backend support |

## Notes

- Archived source evidence under `sources/`, `candidates/`, and `artifacts/` remains available for traceability.
- User-facing docs focus on reusable CUDA-kernel navigation, even when a page's evidence includes event or benchmark material.
