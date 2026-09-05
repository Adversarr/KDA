# CUDA Kernel Wiki

`cuda-kernel-wiki` is a self-contained knowledge skill for CUDA kernel optimization, implementation lookup, and architecture-aware tuning.

The current corpus is strongest on NVIDIA Blackwell and Hopper kernels, but the package is organized around reusable kernel concerns:

- hardware features and memory hierarchy
- optimization techniques and failure patterns
- kernel case studies such as GEMM, attention, MLA, and MoE
- language and DSL guides for CUDA C++, PTX, CuTe DSL, and Triton
- PR-backed source references from upstream kernel-heavy repositories

## What This Distillation Keeps

- queryable wiki pages under `wiki/`
- raw source summaries under `sources/`
- archival evidence under `artifacts/`
- candidate ledgers under `candidates/`
- local query and maintenance scripts needed for normal use

## What This Distillation Intentionally Drops

- ingestion and refresh helpers used to build the corpus from upstream
- phase-specific audits, fixture checks, and pre-commit workflow scripts
- user-facing wording that treated the package as a contest-oriented artifact

## Query Tools

Run from this directory:

```bash
python3 scripts/query.py "persistent kernel scheduling"
python3 scripts/get_page.py kernel-deepgemm --follow-sources
python3 scripts/grep_wiki.py "register pressure" --only wiki
```

## Core Docs

- `SKILL.md` — skill entry point and query contract
- `index.md` — top-level navigation
- `references/primer.md` — topic map
- `references/schema.md` — schema and vocabulary reference
- `references/examples.md` — worked query patterns

## Local Maintenance

The retained maintenance surface is intentionally small:

- `scripts/generate-indices.py` regenerates `queries/*.md`
- `scripts/validate.py` checks frontmatter, cross-links, and asset structure

## Scope Notes

- Best current coverage: Blackwell and Hopper CUDA kernels
- Out of scope: distributed-system runtime topics and non-CUDA accelerators
- Archived source material may still mention contests or benchmark events, but the visible skill interface is organized around durable CUDA-kernel learning
