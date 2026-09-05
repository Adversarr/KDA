---
name: kda-kernel-implement
description: M2/M4/M5 of the KDA kernel workflow, run as a sequential worker or explicit serial role on one `kda_kernels/<op>/`. Implements the fused forward and backward (the SPEC `kernel_backend`) from SPEC.md and the reference snippets (M2), applies one tuning hypothesis from report.json (M4), or freezes launch configs and removes autotuning (M5). Use when dispatched by kda-kernels with an op directory and a mode; the implementer runs `_run_dev.py --verify` and the lint itself and never edits SPEC.md.
---

# kda-kernel-implement (M2, M4, M5)

You are the implementer for one kernel package `<op_dir>` in the user's repo. Use the dispatch and files as context; in a fresh worker they are your only task context; you finish by leaving everything the
next implementer or the verifier needs in the same files. You run no subagents. Follow host instructions, explicit user constraints and the
orchestrator's authorized stage scope.

Three modes, chosen by the dispatch prompt:

| mode | when | you produce |
|---|---|---|
| `implement` | M2, first iteration | `_<kb>/*` passing `_run_dev.py --verify` for both backends and the lint |
| `tune` | M4, remaining STATUS tune budget, finalized verdict was `tune` | one hypothesis applied and measured, logged |
| `freeze` | M5, passed or correct best-so-far candidate (also frozen repair) | `_configs.py` branches, `TUNED.json`, autotune removed, re-verified |

`<kb>` is SPEC `kernel_backend` (`triton`, `nvmath` or `tilelang`): the kernel directory is `_<kb>/`, the
backends are `<kb>` and `<kb>_fwd_only`. Kernel work happens in `python` of the user's
environment (GPU visible): every `python` in this file means the `python:` line of your
dispatch, verbatim (a repo-specific wrapper or an absolute path), never the bare word.
`_run_dev.py` writes `report.json` and `REPORT.md` into `<op_dir>`.

Every file you need is under `<op_dir>` or `.agents/skills/` in the repo root. The reference
directory is `.agents/skills/kda-kernel-implement/reference/` (beside this SKILL.md). There is
nothing to search the filesystem for. Scratch scripts go in `<op_dir>/_scratch/` (delete it
before you report) and run as modules from the repo root,
`python -m kda_kernels.<op>._scratch.<name>`, which is what puts `kda_kernels` on the path;
`python <op_dir>/_scratch/<name>.py` fails with `ModuleNotFoundError: kda_kernels`.

## 0. Read, in this order

1. `<op_dir>/STATUS.md`, then `SPEC.md` in full: front matter (workloads with their `strides`,
   `saved_for_backward`, `backward`, `compute_dtype`, `compute_pattern`, `recompute`,
   `tolerances`) and the Math section.
2. `<op_dir>/IMPL_NOTES.md` (previous iterations) and `<op_dir>/report.json` if present
   (`verdict`, `reasons`, per-workload `time_ms`, `roof_ms`, `derived.sol_eff`, `contract`).
   A SPEC disagreement recorded by an earlier iteration is re-checked against the current
   `SPEC.md` and `_speed_of_light.py` before you repeat it; the orchestrator may have fixed it.
3. `<op_dir>/_eager.py` and `interface.py`: the exact signature you must match.
4. `reference/common/compute-patterns.md`: the row named by SPEC `compute_pattern` is the
   structure your kernel must have (grid, loop, what stays in registers); its "grepable
   signature" is what the verifier looks for.
5. `reference/table-of-content.md`, then under `reference/<kb>/` the `SNIPPET.md` **and**
   `kernel.py` of every primitive the op contains, then
   `reference/triton/fusion-exemplar/rmsnorm_rope_permute/FUSION.md` (the fusion walk-through
   applies to both DSLs). The snippets are complete, tested kernels; the fused kernel is their
   bodies concatenated in registers, as FUSION.md walks through. There is no helper library to
   import. TileLang: also `reference/tilelang/README.md`. `gemm_tensorcore`: read
   `reference/nvmath/epilogue/gemm/SNIPPET.md` first; when nvmath is the selected backend, imports successfully and the
   epilogue is bias / ReLU / tanh-GELU (+ aux), the forward is a planned cuBLASLt `Matmul`
   and no GEMM kernel is written (the `_nvmath/` template); the Triton twin supplies the adjoint.
6. `reference/common/speed-of-light.md` when the mode is `tune`.
7. `reference/common/findings.md`: the measurements behind the rules above (row width, the
   cuBLAS roof, int32 at 2^31, numerics-first mainloop), from runs where an implementer
   guessed otherwise.

Completion: you can state, without reopening the files, the forward math, what is saved for
the backward, every workload shape and stride, the pattern's required structure, and which
snippet each fused step comes from.

## 1. Plan in `IMPL_NOTES.md` before code (`implement` mode)

Fill the Fusion plan section: pattern and how the kernel follows it, primitives used (paths),
memory-traffic plan (loaded once / written once / registers only), compute dtype and cast
points, strides passed, saved-vs-recomputed (must mirror SPEC, `RECOMPUTE` variant when
`recompute.available`), launch geometry per workload pattern including the widest row and the
small-rows workload, known risks. When SPEC and eager disagree on semantics, record the exact discrepancy and return it
to the orchestrator before dependent implementation. An optional optimization (such as
recomputing an aux) is a suggestion: record it and follow the current valid SPEC. Only the
orchestrator changes the contract.

Completion: no `TODO` remains in the Fusion plan.

## 2. Implement `_<kb>/` (`implement` mode)

Files and what each must satisfy (the template's `TODO(implementer)` comments mark the spots):

- `_impl_fwd.py`: the kernel(s) and a **fully type-annotated** launcher
  `<op>_fwd(..., save_aux: bool) -> Tuple[...]` returning the user-visible outputs followed by
  the aux tensors from SPEC `saved_for_backward`, plus `<op>_fwd_fake` producing the same
  shapes/dtypes with `torch.empty`. The op schema is inferred from the annotations;
  unannotated or `Optional` tensor arguments break registration.
  Aux tensors are optional work: every aux store sits behind `if SAVE_AUX:` (Triton: a
  `tl.constexpr` fed from `save_aux`; TileLang: the Python `bool` argument, specialised per
  value), and with `save_aux=False` the launcher neither allocates nor writes them and returns
  `torch.empty(0, dtype, device)` in their place (the fake mirrors this). `_common.compat`
  sets `save_aux` per call from grad mode, `requires_grad` and the recompute flag, so
  inference never pays for the backward. `_run_dev.py` checks that path as the `infer` rows.
  Zero rows return the empty outputs before any launch (a zero-size grid is a CUDA error; the
  `zero_rows` contract row checks it).
- `_impl_bwd.py`: the fused adjoint, one gradient per differentiable input; weight and bias
  gradients accumulated per program into fp32 partials of shape `(n_programs, D)` and reduced
  with host-launched `.sum(0)` on the GPU (see the rmsnorm snippet). Plus `<op>_bwd_fake`. The launcher
  takes `recompute: bool`; under `RECOMPUTE` the kernel recomputes the aux from the saved
  inputs (same code as the forward) instead of loading it. When SPEC `recompute.available` is
  false the flag stays in the signature and the `RECOMPUTE` branch is never taken.
- `_register.py`: `N_OUTPUTS` is the count of user-visible outputs (SPEC `outputs`); the
  template's `1` is a placeholder, and the launcher returns those outputs first and the aux
  after them (`make_differentiable` splits the tuple there). `_setup_context` saves the SPEC
  `saved_for_backward` list, read as: the *written* aux tensors are what SPEC lists; inputs
  and user-visible outputs needed by backward may also be retained, documented separately
  in the Fusion plan. Retaining them can extend their lifetime without a new allocation. `inputs` in `_setup_context(ctx, inputs,
  output)` is the launcher's full argument tuple in signature order, keyword-only arguments
  (`act`, `eps`, `save_aux`) appended after the positional ones (`compat.make_differentiable`
  re-attaches torch's `keyword_only_inputs`), so read a scalar option as
  `inputs[<its position>]`, never from a global. Under `ctx.recompute` save the inputs the
  aux is recomputed from, since the aux is the empty placeholder. `_backward` returns a tuple
  aligned with the forward's positional inputs
  (`None` for non-tensors, `save_aux` included, and inputs without gradients) and passing
  `ctx.recompute` to the backward launcher. Keep `save_aux_arg="save_aux"` and
  `recompute_arg="recompute"` on both `make_differentiable` and `with_eager_backward`.
- `_configs.py`: during M2 and M4, host-side heuristics or autotuning are both fine; keep
  `select_config(dims, gpu)` the single place launch parameters come from so M5 can freeze it.
  `dims` is the op's own dict (`{"n_rows", "d", "itemsize"}` for token-wise ops; `{"m", "n", "k"}`
  for a GEMM); the launcher builds it.

Kernel conventions, all of them checked by the verifier and by `lint_kernel.py`:

- **Compute dtype**: elementwise math and reductions in SPEC `compute_dtype` (Triton
  `COMPUTE: tl.constexpr`, TileLang `accum_dtype`); one cast to the storage dtype at the store.
  The eager reference's *own* rounding points are part of the numerical contract: where the
  user's code combines a storage-dtype tensor with a scalar or another storage-dtype tensor
  before its upcast (`1 + scale` with a bf16 `scale`, `x * gate` in bf16), eager rounds to
  that dtype there, and the fp32 kernel must reproduce the rounding (`(1 + s).to(bf16).to(f32)`)
  or fail near the rounding boundary (for example, at `scale ~ -1`). List
  these points from `_eager.py` before writing the kernel; they belong in SPEC Math and, when
  the SPEC misses one, in a SPEC disagreement.
- **Tensor cores**: operands enter `tl.dot` / `T.gemm` in their storage dtype (bf16/fp16), never
  upcast at load; the accumulator is `compute_dtype`; the single cast happens in the epilogue.
  An fp32 upcast before the MMA silently turns a tensor-core GEMM into a CUDA-core one.
- **Strides**: every non-last stride of every input and gradient is a launcher argument
  (`x.stride(0)`, `dy.stride(0)`), read with `_helpers.rows_and_stride`. The rendered checks in
  `interface.py` reject a non-unit last-dim stride, so `.contiguous()`, `.reshape(...)` and
  `.view(...)` on an input or a gradient are never needed: the first two are a hidden extra
  pass the roofline does not count (lint `copy`, hard), a `.view` raises on a layout it cannot
  express (lint `view`, soft), and `dy` from autograd arrives with whatever strides the graph
  gave it. Outputs you allocate yourself are contiguous and may be viewed freely.
- **Offsets**: int64 (`.to(tl.int64)` on the program id before multiplying by a stride);
  masks on every load and store that can touch a tail; `BLOCK_D = triton.next_power_of_2(D)`
  when `D` is not a power of two.
- **Row width, not row count**: the measured reference uses one program per row at the copy roof,
  including its `user_small_rows` workload (measured: rmsnorm SNIPPET table; the copy it is
  compared with is latency-bound at those sizes too), so begin with that geometry. A different geometry requires measured justification.
  What forces a second geometry is a row that does not fit registers: forward past
  `D = 32768`, fused backward past `D = 8192` (row width rule in `compute-patterns.md`); the
  rmsnorm reference's split-D pair is the model when a workload is that wide.
- **Aux**: materialized auxiliaries match SPEC; retained original inputs/outputs follow the
  documented backward data dependencies. Recompute cheap
  intermediates. Saved tensors that are not user-visible outputs are written only under
  `SAVE_AUX`; with `RECOMPUTE` they are rebuilt in the backward.

Then run, in this order, and fix until all pass:

```bash
python -m kda_kernels.<op>._run_dev --verify --backend <kb>_fwd_only --json <op_dir>/report.fwd_only.json --md <op_dir>/report.fwd_only.md   # forward kernel alone
python -m kda_kernels.<op>._run_dev --verify --backend <kb> --json <op_dir>/report.verify.json --md <op_dir>/report.verify.md   # fused backward too
python .agents/skills/kda-kernel-verify-and-bench/scripts/lint_kernel.py <op_dir>   # exit 0 = no hard finding
```

All `_run_dev.py` commands run from the user's repo root, the directory that contains
`kda_kernels/` (the module resolves through cwd). The `--json`/`--md` side files keep a plain
verify from overwriting `<op_dir>/report.json`, which only the M3 verifier writes as the complete timed record. M2/M5 report
performance pending M3; M4 uses diagnostic paths.

`<kb>_fwd_only` takes the backward through autograd on `_eager.py`; when it passes and `<kb>`
fails, inspect `_impl_bwd.py` and `_backward` first; classify other errors by their evidence. `REPORT.md` lists the failing output
names with `max_abs`, `max_rel` and the effective tolerance. For a failing `contract` row, locate the violated expectation and source owner:
interface checks belong to the orchestrator, launcher failures to you. The lint's hard findings are the conventions above; fix them, do not argue them.

The `compile` column is the `torch.compile(fullgraph=True)` probe on the first user workload
(`<kb>` backend only). It fails when a fake disagrees with its launcher in shape, dtype or
rank of any output, aux included; inductor asserts that when the saved tensor reaches the
backward op. First inspect `_impl_*.py`: print `[(t.shape, t.dtype) for t in ...]` for
the fake and the launcher on the failing workload and make mismatched outputs identical. If those agree, report the observed
compiler/harness failure and ownership evidence instead of assuming a fake bug. If you write your own compile probe in
`_scratch/`, call `kda_kernels._common.bench.configure_compile_for_dev()` before
`torch.compile`: it turns inductor's on-disk cache off, which is keyed on graph code and not on
the fake's outputs, so a probe without it reuses the artefact built against your *previous*
fake and reports shapes your code no longer produces (`_run_dev.py` does this for you).

Completion: both correctness reports have verdict `pass` on all required coverage and lint
has no hard finding. Exit 0 alone is insufficient: an incomplete result is not a pass.
Return `performance: pending M3`; no closing benchmark is required in implement mode.
Optional skipped rows remain notes. Required missing GPU/memory evidence is incomplete and
does not justify changing the workload or numerical requirements. Classify a traceback from
its source and ownership boundary; report needed harness/contract edits to the orchestrator.

## 3. Tune one hypothesis (`tune` mode)

Read the dispatch's candidate record `reasons` and the per-workload numbers. Pick the single worst
`(workload, phase)` by `sol_eff`, name one mechanism from the table in
`.agents/skills/kda-kernel-implement/reference/common/speed-of-light.md`, and change only
what that mechanism needs. The verifier's `Diagnosis:` line in your dispatch is the first
hypothesis to test, not an instruction: measure it (a register-count or occupancy claim is
checked with `ncu` or a `num_warps` sweep before rewriting the tile), and when the
measurement says otherwise (for example, fp32-issue bound rather than register-bound), write the falsification into `IMPL_NOTES.md` and tune the mechanism you measured.

| class | mechanisms |
|---|---|
| token-wise | uncoalesced access, spills from a row too wide for one program (split-D), a hidden pass (`.contiguous()`, a host reduce), partials too large, rows-per-program and `num_warps`; an SFU-bound activation epilogue (`sigmoid`/`exp`/`erf` per element) on a `BLOCK_D = next_pow2(D)` tile much wider than `D` (masked lanes still issue the SFU op: 2048 lanes for `D = 1152` is 78% wasted) - a fast sigmoid (`1 / (1 + exp2(-x * log2e))`) and chunking the row so the MUFU count equals `D` buy 5-10 points; the rest is structural (speed-of-light.md "SFU work under the copy roof") and is *not* a second tune round |
| GEMM | tile shape (`BLOCK_M/N/K` vs `M, N`), split-K for small `M` or large `K`, pipeline stages (`num_stages` / `T.Pipelined`), tensor-core underuse (an fp32 operand cast, a tile below the MMA shape; ncu `sm__pipe_tensor_cycles_active` when available), epilogue not fused |

When the mechanism is unclear after reading the kernel, profile the named kernel with the
`ncu-report` skill (`dram__throughput`, achieved occupancy, local-memory traffic) before
editing, if `ncu` works here (containers without `--cap-add=SYS_ADMIN` report "no kernels
were profiled"; then reason from the source and say so); when you need an architecture
pattern, read `cuda-kernel-wiki` (Triton) or `tilelang-wiki` (TileLang).

When the `compile` baseline in `report.json` is the one you cannot beat (Triton backend only):
inductor is itself a Triton code generator, so read what it generated for the user's eager
function and copy the tricks. `torch.compile(_eager.<fn>, mode="max-autotune-no-cudagraphs")`
on the workload prints an `AUTOTUNE` table with the winning template and its config
(`BLOCK_M/N/K`, `num_stages`, `num_warps`), and `TORCH_LOGS=output_code` prints the path of
every generated kernel under `/tmp/torchinductor_<user>/`; `grep -l triton_mm
/tmp/torchinductor_*/*/*.py` finds the GEMM template afterwards. Things worth lifting from
that source: `tl.multiple_of` / `tl.max_contiguous` on the load indices (16-byte vector
loads; the `% M` wrap of the tutorial matmul hides them and costs ~20%), `tl.assume` on the
program ids, `eviction_policy` on the epilogue loads, the tile configs it chose for this GPU,
and which pointwise ops it fused into the GEMM epilogue vs left in a separate kernel (the
GEMM+epilogue SNIPPET records why a transcendental epilogue can be *slower* fused). Log the
comparison in `IMPL_NOTES.md` as the hypothesis.

For a `gemm_tensorcore` op, time `torch.matmul` alone on the SPEC workloads before tuning the
mainloop: when the fused kernel's mainloop is further from cuBLAS than the epilogue's traffic
costs (`3 * M * N * es` bytes at HBM speed), no tile config will close it, and the right kernel
is `torch.matmul` plus one epilogue kernel over the product (the reference's
`mainloop="cublas"` path; on A800 that is every wide-output shape, `compute-patterns.md`
mainloop note). Record which shapes went which way in `IMPL_NOTES.md`.

All performance comparisons use device time: `--method profiler` is the measurement method, and the
roofs are only comparable with it. `--method events` adds host launch overhead that no kernel
edit can change; use it at most as a side measurement into `report.events.json`. `auto` can
fall back to `events` on a CUPTI hiccup: if `env.bench_method` in the record you just wrote says `events`, re-run
with `--method profiler`; an events record is not a performance hint, its `sol_eff` can read
0.00 or above 1 on rows whose kernels it missed.

Re-run `--verify` for both backends and lint, then measure the hypothesis in side files:

```bash
python -m kda_kernels.<op>._run_dev --bench --backend <kb> --method profiler --iteration <n> --json <op_dir>/report.tune.<n>.json --md <op_dir>/report.tune.<n>.md
```

Keep the change only with a measured improvement to the targeted mechanism and no new required
gate regression. Use [the evidence protocol](../kda-kernels/reference/evidence.md) for noise; conflicting gates are unresolved.
For a rejected hypothesis, report `revert requested` and the checkpoint path. The orchestrator
restores code and candidate reports mechanically; never overwrite the accepted canonical
record with diagnostic timings. Preserve the hypothesis and its measurements in IMPL_NOTES.

Completion: the Iteration log in `IMPL_NOTES.md` has an entry `Iteration <n>` with the
change, the before/after `sol_eff` of the targeted phase, and the next hypothesis.

## 4. Freeze (`freeze` mode)

1. For frozen-stage repairs, fix the named defect while retaining frozen-config/no-autotune
   requirements. Remove every `@triton.autotune` / `tilelang.autotune` and every dev-only sweep. Launch
   parameters come from `select_config(dims, gpu)` only.
2. Write the branches: one per configuration that measured best on some workload, at most 4,
   keyed on `dims` thresholds (`d`, `n_rows` for token-wise; `m`, `n`, `k` for GEMM) and on
   `torch.cuda.get_device_name()` substrings in `_VALIDATED`; the conservative `_DEFAULT`
   stays for unvalidated GPUs. Only GPUs you measured on go into `_VALIDATED`. When every
   workload measured best with the same geometry (M4 ran no rounds), `select_config` is that
   one geometry and `_DEFAULT`; do not write branches that return identical values. A
   geometry that is a formula of `dims` (`BLOCK_D = next_power_of_2(d)`) is one branch, and
   `_DEFAULT` may be the same formula: an unvalidated GPU must still get a legal config for
   every `d`, so `_DEFAULT` is a rule, not a fixed number that only fits the user's shape.
3. Write `<op_dir>/TUNED.json`: `{gpu, torch, <kb>, date, branches: [{condition, config, workload, sol_eff}]}`.
   It is the measurement record: one entry per branch of `select_config` naming the workload
   that chose it and its `sol_eff`. The 4-branch cap is on `select_config`, not on this file,
   but do not list nine workloads for one branch either; one entry per branch is the shape.
4. Run both backend verifies and lint. Populate TUNED from the candidate's measured
   configurations and evidence paths; a new configuration needs targeted diagnostic measurement.
   Frozen performance remains pending the subsequent M3 on the same run number.

Completion: no autotune decorators or dev sweeps remain, TUNED identifies the measured
candidate configurations, and both numerical verifies plus lint pass. Report `frozen performance:
pending M3`. A best-so-far candidate remains `tune`; freezing does not waive a performance gate.

## 5. Report back

Append to `IMPL_NOTES.md` (Iteration log), then answer the orchestrator in at most 10 lines:
mode, correctness verdicts, lint result, candidate/diagnostic evidence paths, measured
performance or pending M3, files changed, SPEC disagreements, keep/revert decision and next hypothesis. Paths, not code.

## Constraints

- Owned source: `<op_dir>/_<kb>/*`, `_helpers.py`, IMPL_NOTES and TUNED.
  Generated artifacts: create/remove `<op_dir>/_scratch/` and write stage-specific verification
  and diagnostic reports. The canonical M3 report and checkpoints belong to the orchestrator/verifier. `SPEC.md`, `_eager.py`, `interface.py`, `backends.py`, `_run_dev.py`,
  `_speed_of_light.py` and `kda_kernels/_common/` belong to other stages; report needed
  changes instead. Deleting `<op_dir>/_scratch/` is always permitted, whoever wrote it: the
  M0/M1 intake probes the orchestrator left there are scratch too.
- A required workload that cannot meet tolerance because the eager reference is itself noisy
  is a SPEC `tolerances` question for the orchestrator, not a reason to loosen `_common`.
- Stop after the completion criterion of your mode; the orchestrator dispatches the next stage.

## Dispatch

The orchestrator copies this block verbatim into the subagent prompt, filling the angle
brackets, and dispatches it as a fresh subagent (the `kda-implementer` agent type where the
harness has one).

```
Read `.agents/skills/kda-kernel-implement/SKILL.md` first and follow it exactly.
mode: <implement | tune | freeze>
repo_root: <absolute repository root>
op_dir: <absolute path to kda_kernels/<op>>
iteration: <run number from STATUS>
candidate_report: <candidate/checkpoint report path, or none>
checkpoint: <operation-relative path for tuning, or none>
python: <command that runs python with the GPU visible, default `python`>
Do not spawn subagents. Do not edit SPEC.md. Reply with the 10-line report the skill asks for.
```
