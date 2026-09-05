# Kernel Design Agent (KDA)

KDA is an experimental exploration of agentic, fully automatic kernel development,
aiming toward no-human-in-the-loop operation. It is not a production-hardened tool;
generated kernels need verification and performance depends on the workload and GPU.
The agent handles specification and kernel design internally. It presents measured results
and a concrete model edit before integration, unless you already authorized that step.

The project explores two levels of optimization:

- Level 1: Kernels (e.g., Fused-GEMM-Epilogue, Fused-Adaptive-LayerNorm, Fused-QK-Norm-RoPE, etc.)
- Level 2 (not implemented): Model Architecture. (e.g. Host-Device Synchronization Reduction, Zero-Copy Optimization, etc.)

**Why not a new kernel library or operation library, but a new agent harness?**

A new kernel library or operation library requires manual effort to implement and maintain, which is time-consuming and error-prone:

1. Different compute capability, such as nvidia's next generation GPU, requires manual effort to implement and maintain to achieve optimal performance.
2. Same compute capability, different GPUs can have different performance characteristics, such as different memory bandwidth, different compute units, etc. (e.g. H20 vs. H100)
3. Same compute capability, same GPU, but different input dtype/shape, compute dtype, output dtype, etc. A kernel library should support all these cases to be practical.
4. Everything is same, but the real application have additional operations, such as the Fused-Adaptive-LayerNorm-SiLU, Fused-QK-RMSNorm-RoPE-Permute kernel in your new experimental and cutting-edge models.

To conclude, it is quite difficult to maintain a kernel library or operation library for all these cases above.

An agentic workflow can automatically diagnose and accelerate common deep learning workflows, which is more efficient, scalable, and customizable for many users.

## Kernels (Level 1)

> Status: v1.1 workflow (SPEC v2). Triton backend for token-wise ops, GEMM epilogues and
> FlashAttention-2 (causal / block-causal / dense GQA); a TileLang backend is wired with tested
> twins and is **experimental** (slower GEMM, long per-shape JIT). Reference measurements
> are from A800; they are not performance guarantees for other environments.

KDA is not a kernel library. It is a set of skills your coding agent (Cursor, Claude Code,
Codex, pi) runs inside *your* repository to produce a fused kernel for a region of your training
code, verify it against your own eager code, benchmark it against speed of light, and wire it
in behind a flag. Generated kernels live in `<your repo>/kda_kernels/<op>/` and import nothing
from KDA at runtime.

### Install

Installation targets Linux with Bash 4+, GNU coreutils (`realpath`), Git, and Python 3.
Kernel execution requires a supported NVIDIA GPU and a CUDA-enabled Python environment.
Install the coding-agent CLI you intend to use separately.

```bash
git clone https://github.com/Adversarr/KDA.git
bash KDA/kda/install.sh /path/to/your/repo  # copies the skills into your repo (about 25 MB)
```

The copy lands in `.agents/skills/` and `.agents/agents/` (read by pi and Cursor), with
`.claude/skills/` and `.claude/agents/` symlinks for Claude Code and `.codex/agents/*.toml` for
Codex; `.kda/manifest.json` records what was installed so a re-run refreshes it. Commit the
directories or ignore them, as you prefer (the installer never edits `.gitignore`). `--link`
symlinks into the KDA checkout instead, for developing KDA itself.

After installing, ask your coding agent:

- **Basic:** “Can you optimize FooModel here?”
- **Advanced:** “Can you fuse the operations from this line to that line, with inputs x and
  weight and output y?” Select the code or name its location.

The agent finds the model, configuration, execution command and available GPU environment,
profiles the workload and chooses the implementation. You do not need to specify a kernel
backend, launch geometry, numerical test matrix or backward kernel. If several training or
inference runs are plausible, it asks which matters. Technical contracts and reports are
recorded under `kda_kernels/` and linked when useful.

A normal optimization request covers discovery through verification. Before integration the
agent shows the measured result, proposed code change and eager fallback. “Take this through
integration” can authorize that step in advance; “review only” and requested stage approvals
are respected. A result with unmet performance criteria remains explicit and needs acceptance.
The agent can also conclude that existing libraries already cover the useful opportunities.

The code targets torch >= 2.1 (>= 2.6 for `torch.compile`-safe custom ops), `pyyaml`,
and Triton >= 3.0 or TileLang >= 0.1.6 for the backend the SPEC names; these minimums
are not a validated compatibility matrix. The supplied [container recipe](examples/Dockerfile.cu128)
uses CUDA 12.8, torch 2.11.0, Triton >= 3.5.0, TileLang 0.1.14, and nvmath-python 1.0.0.
It is an environment recipe, not a guarantee that every GPU or dependency combination works.

Start with the [residual RMSNorm example](examples/README.md#running-an-example).
See [agent permissions](kda/agents/README.md) before running the workflow; the
implementer configuration requests broad filesystem access.

### Workflow (M0-M6)

| stage | what happens | who |
|---|---|---|
| M0 scaffold | `kda_kernels/<op>/` rendered from the template for one kernel backend; `_eager.py` is your code verbatim; `SPEC.md` gets your real shapes, strides and dtypes, the compute dtype with its provenance, the compute pattern, saved-for-backward and recompute decisions, roofline, hardware peaks | your agent, inline |
| M1 contract checkpoint | agent checks SPEC against observed code and records its rationale; asks about unresolved semantics | your agent |
| M2 implement | fused forward and backward from the reference snippets in the pattern's structure; `_run_dev.py --verify` and the mechanical lint pass | subagent |
| M3 verify + bench | independent audit, lint, contract probe, numerics at fixed tolerances over the training, inference and recompute phases, timings vs eager and `torch.compile`, SOL efficiency; `report.json.verification` decision `pass` / `tune` / `fail` / `incomplete` (`fail` returns to M2 at most twice) | subagent |
| M4 tune | at most 3 implement/verify iterations while the verdict is `tune` (SOL eff < 0.7 or slower than the baseline) | subagents |
| M5 freeze | autotuning removed, <= 4 frozen launch configs keyed by GPU, `TUNED.json`, re-verified | subagent |
| M6 integrate | definition-site edit behind `KDA_BACKEND` (and `KDA_RECOMPUTE` when the op offers it, plus config fields), before/after smoke run, `E2E.md` | your agent, inline, within recorded integration authorization |

Kill switch at any time: `KDA_BACKEND=eager` or `backend="eager"` per call.

Existing generated packages need their customized runner updated as well as the shared
runtime to use coverage, provenance and finalized verification. `--sync-common` alone does
not upgrade operation runners; preserve their input generators. Legacy reports remain readable
but are not current verification evidence.

### What is in the box

- `kda/kda-skills/`: the workflow skills (`kda-kernels` orchestrator, with the transformer
  fusion map for whole-model requests; `kda-kernel-scaffold`,
  `-implement`, `-verify-and-bench`, `-integrate` stages). The scaffold `template/` is the
  single source of truth for a kernel package and the vendored `_common/` runtime
  (registration for torch >= 2.6 via `torch.library.custom_op`, older via `autograd.Function`;
  profiler-based timing with an achievable copy-calibrated roof; tolerance rules; report/verdict).
- `kda/kda-skills/kda-kernel-implement/reference/`: complete, tested primitives the
  implementer fuses in context (Triton: fp32 norms with a split-D wide-row path,
  elementwise/residual, RoPE, permute, GEMM epilogue, FA2 causal and dense-GQA attention;
  nvmath-python: the GEMM epilogue through cuBLASLt, preferred over any hand-written GEMM;
  TileLang, experimental: the GEMM epilogue and attention twins), a fusion
  exemplar with a walkthrough, and `common/compute-patterns.md`, the closed list of kernel
  structures a SPEC can name.
- `kda/kda-skills/kda-kernel-verify-and-bench/scripts/lint_kernel.py`: the mechanical checks
  (hidden copies, int32 offsets, unguarded aux stores, fp32 tensor-core operands, leftover
  autotune) that fail an M3 verdict.
- `kda/general-infra-skills/`: generic CUDA skills the stages point at (`ncu-report`,
  `cuda-kernel-wiki`, `torch-profile-reading`, `diagnose-cuda-program-performance`, `tilelang-wiki`).
- `kda/agents/`: `kda-implementer` and `kda-verifier` agent types as Markdown (pi, Claude
  Code, Cursor) and TOML (Codex).
- `examples/`: example tasks, sample training repositories, and golden kernels
  ([guide](examples/README.md)).

Backends: one per package, named in SPEC `kernel_backend`. Honor a supported explicit backend choice; otherwise use applicable nvmath for GEMM
epilogues and Triton elsewhere. TileLang is experimental: its GEMM twin measured 0.8x of the Triton one on the large
forwards after the same effort, its attention twins tie or lead by 7-9%, and its per-shape JIT
(tens of seconds per kernel) can dominate multi-workload development; the
[TileLang reference](kda/kda-skills/kda-kernel-implement/reference/tilelang/README.md)
has the measurements. Agents take it only when the user names it. Gluon and CuTeDSL are planned;
hand-written CUDA is not.

## Model Architecture (Level 2)

> Status: not implemented.

Many users, even frontier LLM agents, are still writing slow code, such as:

1. frequent explicit/implicit synchronization between host and device, such as `.item()`, `.tolist()`, `.cpu()` etc.
2. slow code without zero-copy optimization, such as `torch.cat([x, y])` -> `torch.cat([x.contiguous(), y.contiguous()])` and so on.
3. duplicated computing for even same input, such as RoPE embed across different layers or different iterations within a larger training loop.
4. ...

Level 2 will name regions and hand them to the Level-1 entry (`kda-kernels`); there is no
separate kernel path.

## License

KDA-owned material is licensed under [MIT](LICENSE), copyright 2026 Adversarr.
Bundled third-party material retains its own terms; see [third-party notices](THIRD_PARTY.md).
