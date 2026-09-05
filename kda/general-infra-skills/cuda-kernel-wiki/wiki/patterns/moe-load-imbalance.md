---
id: pattern-moe-load-imbalance
title: "MoE Expert Load Imbalance"
type: pattern
tags: [moe, grouped-gemm, tile-scheduling, clc]
symptoms: [load-imbalance, tail-effect, low-sm-utilization]
candidate_techniques: [technique-tile-scheduling, technique-persistent-kernels, technique-kernel-fusion]
related: [kernel-grouped-gemm, kernel-fused-moe, pattern-tail-effect]
sources: [contest-gpumode-p4, contest-flashinfer-track-a, blog-deepgemm]
---

# MoE Expert Load Imbalance

## Symptom

MoE grouped GEMM shows uneven per-expert compute time. Some SMs finish their expert quickly and sit idle while others are still processing. Overall latency is dominated by the slowest expert.

## Likely Causes

1. **Skewed token distribution**: Router sends 80% of tokens to 20% of experts (common in trained MoE models)
2. **Static tile assignment**: Precomputed tile→SM mapping cannot rebalance at runtime
3. **Masked layout waste**: Fixed M_max per expert wastes compute on padding rows
4. **Small-M per expert**: When M < BLOCK_M, thin-GEMM underutilizes tensor cores

## Candidate Techniques

| Technique | Effect |
|---|---|
| [CLC (Cluster Launch Control)](../hardware/clc.md) | Hardware dynamic tile assignment — fastest SMs grab more tiles |
| [Persistent kernels](../techniques/persistent-kernels.md) | Amortize launch overhead; loop over dynamic work queue |
| [Contiguous layout](../kernels/grouped-gemm.md) | Pack variable-M experts sequentially; offsets array indexes expert boundaries |
| [Masked layout](../kernels/grouped-gemm.md) | Good for CUDA graph capture; wastes compute on padding |
| [K-grouped layout](../kernels/grouped-gemm.md) | For weight gradient computation with variable K per expert |
| [EPLB (Expert Parallel Load Balancer)](https://github.com/deepseek-ai/EPLB) | Replicate heavy experts across GPUs; 1.49x prefill speedup, 2.54x decode |

## Evaluation Caution

Archived grouped-GEMM benchmark history includes cases where harness behavior distorted the apparent benefit of a scheduling strategy. Use this as a reminder to isolate correctness, benchmarking, and process reuse when evaluating MoE kernels; otherwise measured gains can reflect the harness more than the kernel.

## Caveats

- CLC only available on SM100 datacenter (not SM120 consumer)
- Dynamic scheduling has small per-tile overhead vs static precomputed
- Small experts may not benefit — minimum viable tile size is a floor
- EPLB works at cluster scale, not single-device

## When NOT An Issue

- Uniform routing (rare in practice)
- Very large batch sizes (statistics average out)
- Training with auxiliary load balancing loss
