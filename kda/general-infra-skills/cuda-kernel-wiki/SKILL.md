---
name: cuda-kernel-wiki
description: Use when the user asks about CUDA kernel optimization, GPU memory hierarchy, kernel scheduling, GEMM or attention implementations, PTX or CUDA DSL techniques, or wants concrete PR-backed kernel references. The deepest coverage is NVIDIA Blackwell and Hopper, but the navigation patterns are useful for broader CUDA-kernel research.
argument-hint: "[natural-language-question] | [--tag foo --type kernel] | [page-id]"
allowed-tools: "Bash Read Grep Glob"
---

# CUDA Kernel Wiki

> **Knowledge cutoff: 2026-04-27.** Tracked upstream PRs, blog summaries, and version-sensitive notes are anchored to repository state on or before that date.

Query a structured, cross-referenced knowledge base for CUDA kernel optimization. The corpus is strongest on NVIDIA Blackwell and Hopper kernel work, with reusable navigation around techniques, kernel case studies, hardware features, language/DSL guidance, and PR-backed implementation examples.

## When To Use This Skill

Trigger this skill when the user asks about:

- **CUDA kernel optimization patterns** — memory-bound kernels, register pressure, occupancy, tail effects, pipelining, fusion, vectorized loads
- **Kernel implementations** — GEMM, attention, MoE, grouped GEMM, MLA, sparse attention, fused kernels
- **GPU memory hierarchy and scheduling** — shared memory, TMA, TMEM, warp specialization, persistent kernels, tile scheduling
- **Low-level programming surfaces** — CUDA C++, PTX, CuTe DSL, Triton, and architecture-aware kernel tuning
- **Architecture-specific references** — especially NVIDIA Blackwell (`sm100`) and Hopper (`sm90`)
- **PR-backed examples** — “how did CUTLASS, SGLang, vLLM, FlashInfer, or PyTorch implement this kernel pattern?”

This skill is less useful for:

- Host-side framework integration, serving, routing, or scheduler policy
- Distributed systems topics such as DeepEP, EPLB, and DualPipe
- Purely non-CUDA accelerator questions

## How To Query

All commands below run from the skill directory. The scripts resolve the wiki root from their own location.

### Path 1: Unified Search

```bash
python3 scripts/query.py "how to reduce register pressure in a fused attention kernel"
python3 scripts/query.py --tag warp-specialization --type technique
python3 scripts/query.py --repo cutlass --limit 20
python3 scripts/query.py --symptom memory-bound --compact
```

Filters: `--type`, `--tag`, `--repo`, `--language`, `--architecture`,
`--symptom`, `--confidence`, `--limit`, `--compact`, `--paths-only`.
Alias-aware terms are expanded automatically, so inputs such as `UMMA`, `TMEM`,
`B200`, and `H100` map to their canonical terms.

### Path 2: Fetch A Specific Page

```bash
python3 scripts/get_page.py kernel-deepgemm
python3 scripts/get_page.py pr-cutlass-2472
python3 scripts/get_page.py kernel-flash-attention-4 --follow-sources
python3 scripts/get_page.py lang-cuda-cpp --body-only
```

### Path 3: Regex Search Across Bodies And Source Pages

```bash
python3 scripts/grep_wiki.py "tcgen05\\.mma"
python3 scripts/grep_wiki.py "warp specialization" --only wiki
python3 scripts/grep_wiki.py "persistent" "tail effect" --any
```

### Path 4: Pre-Built Indices

Auto-generated under `queries/`:

- `queries/by-problem.md` — symptom-driven navigation into relevant pattern and technique pages
- `queries/by-technique.md` — optimization techniques with architectures, confidence, and source counts
- `queries/by-hardware-feature.md` — hardware features to related wiki pages and source references
- `queries/by-kernel-type.md` — kernel families such as GEMM, attention, MoE, MLA, and grouped GEMM
- `queries/by-language.md` — CUDA C++, PTX, CuTe DSL, Triton, and related kernel pages
- `queries/by-repo.md` — PR coverage across tracked upstream repositories

### Path 5: Companion Docs

- `references/primer.md` — quick topic map and canonical page IDs
- `references/schema.md` — condensed schema and vocabulary guide
- `references/examples.md` — worked query patterns from question to answer shape

## Output Pattern

When answering from this knowledge base:

1. **Cite specific pages** with both path and page ID.
2. **Follow `sources:` links** back to PRs, blogs, or docs when making claims.
3. **Respect confidence levels** — `verified` > `source-reported` > `inferred` > `experimental`.
4. **Use code snippets** from wiki pages when they clarify the technique.
5. **Report performance claims completely** with `gpu`, `dtype`, `shape`, `metric`, `value`, and `source_id`.

## Knowledge Base Contents

- **2,000+ markdown pages** across wiki syntheses, upstream PR summaries, blogs, docs, and archived reference material
- **6 query indices** under `queries/`
- **Controlled vocabulary** in `data/tags.yaml` plus aliases in `data/aliases.yaml`
- **Version-sensitive notes** in `data/version-claims.yaml`
- **Archived source evidence** under `sources/`, `candidates/`, and `artifacts/`
- **Local maintenance tools** for querying, validation, and index regeneration under `scripts/`

## Quality Guarantees

- Every `verified` page carries both official-doc and upstream-code evidence
- Every technique, kernel, and language page includes at least snippet-level reproducibility
- PR pages remain traceable through source metadata and upstream links
- Architecture-specific claims stay attached to explicit `architectures` fields
