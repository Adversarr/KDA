---
name: kda-kernel-verify-and-bench
description: Audit and verify one generated KDA operation after implementation, tuning or freezing. Run numerical checks and benchmarks, preserve full evidence, and finalize pass/tune/fail/incomplete without changing kernel or contract source. Used by kda-kernels for M3 or revalidation on a new GPU.
---

# KDA verifier (M3)

Goal: establish whether the current package satisfies its SPEC and performance gates, with
reproducible evidence. With an independent worker, review code you did not implement. In a
serial fallback, perform the same checks and record `same_context`; do not claim independence.
Follow host instructions, explicit user constraints and the orchestrator's authorized scope.

Source is read-only: SPEC, interface, eager code, helpers, backend code, input generator,
roofline and shared runtime. You may write canonical reports, `audit.json`, diagnostic
reports and `<op_dir>/_scratch/`. Delete scratch after recording its useful evidence.
Do not spawn workers or change thresholds/workloads to get a pass.

The dispatch's `python:` is the GPU command, verbatim. Commands run from `repo_root`, which
contains `kda_kernels/`; scratch probes run as modules under that package. `<kb>` is SPEC
`kernel_backend` (`triton`, `nvmath`, `tilelang`). Read the
[authoritative evidence protocol](../kda-kernels/reference/evidence.md) before writing reports.

## 1. Audit the current package

Read STATUS, full SPEC, implementation notes, interface and eager reference, then the selected
backend and relevant compute-pattern/snippet references. Read
`../kda-kernel-implement/reference/common/findings.md` when a roof/numerics diagnosis needs
its measurement context. Answer every audit row in the audit's Markdown `notes`:

| Check | Evidence |
|---|---|
| SPEC and coverage | `validate_spec(load_spec(...))` yields no errors; observed user workloads and the class's required rows exist. A missing required row is incomplete, not an optional note. |
| Eager identity | Compare source region and `_eager.py`: same signature, outputs, eps, casts and rounding. After integration, use the preserved reference and recorded integration change set. |
| Interface | Read all input checks for device, dtype, shape, layout and op-specific limits. Contract probes must reject bad inputs before a launch and accept supported empty/no-grad paths. The harness probes one workload/first tensor; inspect constraints it cannot invent. |
| Compute semantics | Kernel arithmetic and documented rounding points match SPEC. Compare tensor-core operand/storage dtypes and accumulator dtype. |
| Pattern | Check the selected structure and its actual supported dimensions. Measured alternatives need recorded rationale and evidence. Hardware observations are not additional passing thresholds. |
| Autograd state | SPEC `saved_for_backward` lists materialized auxiliaries. Check those in saved-aux mode and the separately documented retained inputs/outputs needed for backward. Under recompute, retain reconstruction inputs instead of requiring absent aux; inference stores no aux. |
| Auxiliary work | Aux allocation/store guards follow `save_aux`; fake outputs match real outputs, including empty placeholders. Registration passes save_aux/recompute options correctly. |
| Phases | Every required forward/inference phase and applicable backward/recompute phase has results. Required measured phases have positive timings; optional skipped rows are recorded. Recompute performance gates only when default. |
| Index/layout safety | Inspect int64 base offsets, masks, leading-stride addressing and legal geometry at the widest required shape. Scope collapsible-row helpers honestly; no hidden input/gradient copy. |
| Roof | Re-derive bytes/FLOPs from the math independently. Count every output gradient, aux traffic and reduction partials. For a GEMM, forward/backward/recompute are 1/2/3 whole GEMMs. Attention uses attended-area math and its documented roof method. |
| Freeze | For mode freeze or a previously frozen package: measured TUNED provenance, legal default/config branches and no autotune or dev sweeps, even before M5 is ticked. |

Run the mechanical lint and paste its output into notes:

```bash
python .agents/skills/kda-kernel-verify-and-bench/scripts/lint_kernel.py <op_dir>
```

Every hard finding is an audit `hard` finding; soft findings are notes unless inspection
establishes a correctness/contract failure. Completion: every row answered with source or
measurement evidence, including any untested limitation.

## 2. Produce the complete record

```bash
python -m kda_kernels.<op>._run_dev --verify --bench --backend <kb> --method profiler --iteration <n>
python -m kda_kernels.<op>._run_dev --verify --backend <kb>_fwd_only --json <op_dir>/report.fwd_only.json --md <op_dir>/report.fwd_only.md --iteration <n>
```

The first command covers all SPEC rows and owns the canonical report. The second isolates
forward registration/numerics from a fused backward error. Preserve source between runs;
source edits require new evidence. Missing GPU, required OOM/skips, zero profiler samples or
missing required results are incomplete evidence. Record the blocker; never infer a timed
pass from successful numerics. Optional memory-heavy skipped diagnostics remain notes.

The runner computes `sol_eff = roof_ms / kernel_ms`, with `roof_ms` the measured achievable
roof and `sol_ms` the datasheet bound. They are different: exceeding one against a measured
roof does not alone violate the datasheet. Use the runtime's below-datasheet check and inspect
its counts before attributing impossible speed. Missing a GEMM can lower `roof_ms` and thus
lower its ratio to a fixed kernel time. Preserve all measured reference numbers.

Run two scoped repeats for tagged near-gate workloads as the evidence protocol specifies;
include all samples in `audit.reruns`. A gate crossing the observed spread is incomplete,
not an opportunity to select the better sample. After those repeats report unresolved
measurement conditions without cycling. Numerical/audit failure remains failure.

When a host-overhead question matters, run an event-timed side diagnostic with explicit
`--json` and `--md` paths, compare it with profiler time and name the limitations. Otherwise
record it as not measured. Never report an unmeasured difference as zero.

Completion: full current report plus current forward-only evidence, all repeat paths
preserved, and any unavailable evidence explicitly identified.

## 3. Probe and diagnose

- Read the existing `torch.compile(fullgraph=True)` probe results on supported torch versions.
  Compare fake/real outputs first on a mismatch. If they agree, locate the observed failure
  rather than assuming every compile error is in the fake. Scratch compile probes call
  `_common.bench.configure_compile_for_dev()` before compiling.
- Run the first user workload twice with identical inputs; compare outputs and gradients
  bit-for-bit. Differences are recorded and fail only when SPEC demands determinism.
- For a tuning result, name a testable mechanism for each failing required phase. When source
  and measurements leave it unclear and NCU works, use `ncu-report` with the KDA handoff:
  named verified workload/kernel, timing evidence, hypothesis and output root
  `<op_dir>/_scratch/profile/<unique-run>`. If counters are denied, record it and reason from
  available evidence; retry NCU at most once. Do not change host permissions.
- A traceback is evidence, not a diagnosis by itself. Identify the failing ownership boundary
  and report source/contract/harness changes to the orchestrator; preserve your read-only role.

## 4. Finalize and return

Write `audit.json` using the evidence protocol: audit completion and context, all checklist
notes/lint/probes, hard/unresolved/note findings, repeat paths and the diagnosis. Then:

```bash
python -m kda_kernels.<op>._run_dev --finalize-audit <op_dir>/audit.json
```

Read `report.json.verification` and its findings. The finalizer checks current provenance and
coverage and regenerates REPORT.md from the same decision; do not hand-edit a verdict or
append a contradictory Markdown decision. Missing finalization means M3 is unfinished.

Return at most 10 lines: finalized verdict, iteration, worst efficiency per applicable phase,
user-workload speed versus the available baseline, audit failures, skipped/missing evidence,
context (`independent`/`same_context`), diagnosis and report paths. Required correctness and
contract checks are never waived to reduce tool calls.

## Dispatch

```text
Read `.agents/skills/kda-kernel-verify-and-bench/SKILL.md` and perform its checks.
repo_root: <absolute repository root>
op_dir: <absolute operation directory>
mode: <verify | freeze>
iteration: <run number from STATUS>
python: <GPU interpreter command>
context: <independent | same_context>
candidate_report: <record/checkpoint path, or none>
Do not spawn subagents. Source files are read-only; return the stage report.
```
