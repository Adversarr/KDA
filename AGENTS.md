# Kernel Design Agent (KDA)

KDA is an experimental workflow for agentic kernel development. The goal is
automatic discovery, implementation and verification for model developers. PLAN and SPEC
are internal checkpoints by default. Integration receives a concrete review unless prior
user authorization covers it; explicit review-only and per-stage approval requests prevail.

## Code overview

- `kda/kda-skills/`: workflow skills and kernel references. The scaffold's
  `template/` is the source of truth for generated packages and their `_common/` runtime.
- `kda/general-infra-skills/`: CUDA infrastructure references, including upstream
  code with separate licenses and notices.
- `kda/agents/`: implementer and verifier definitions; keep Markdown and TOML
  instruction bodies consistent.
- `examples/`: requests, sample training code, and golden kernels.

## Working conventions

- Keep edits small and readable. This is an experimental project, not a broad hardening effort.
- Generated kernels are standalone: they must not import KDA at runtime.
- Keep the compute-pattern list consistent with the scaffold runtime's supported patterns.
- Use the GPU interpreter for kernel checks. Report unavailable checks and observed
  limitations rather than implying validation succeeded.
- Preserve numerical contracts, verification, and eager fallback. Re-measure before
  changing documented performance numbers.
- Preserve upstream notices and test files. KDA development test suites are not part
  of this distribution.
