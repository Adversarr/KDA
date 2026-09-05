---
name: kda-kernel-integrate
description: M6 of the KDA kernel workflow, run inline by the orchestrator after M5. Swaps the user's definition site to the fused kernel behind the `KDA_BACKEND` flag, runs the observed training or inference smoke command before and after with the same inputs, and records timing and behavior deltas in E2E.md. Use when STATUS.md shows M5 done and integration is authorized in the current or an earlier request, or when an integrated kernel must be re-validated after the user's code changed.
---

# kda-kernel-integrate (M6)

Runs inline in the orchestrator's context. Input: `<op_dir>` with `STATUS.md` M5 ticked and
`SPEC.md` `integration:` filled. Output: the user's definition site calls
`kda_kernels.<op>.interface.<op>`, the eager path is one environment variable away, and
`<op_dir>/E2E.md` holds the before/after evidence. Use the orchestrator's recorded authorization. Show the concrete change set and measured
result before requesting any still-needed integration approval. Unmet performance gates
remain explicit; review-only requests never authorize integration.

## 1. Baseline before touching anything

Run `integration.smoke_command` from SPEC as two repeats, unchanged, using the observed training or inference mode, fixed inputs/seed and the smallest representative configuration. For training, use about 50 steps and capture per-step time and final loss. For inference, capture request/batch latency and the existing output/quality metric; if none exists, compare same-input outputs at SPEC tolerances in a scratch driver. If measurements are missing, obtain them with a scratch driver before integration;
include any permanent measurement instrumentation in the authorized change set and E2E.md.

Completion: `E2E.md` has a `before` row with median execution time and the applicable behavior metric.

## 2. Edit the definition site

Default to replacing the owning definition or selected region. The concrete change set may
also include the necessary epilogue opt-in call site, backend/recompute configuration and
initialization, or missing measurement instrumentation. List all affected user files in the
integration review and E2E.md. Preserve unrelated call sites and behavior.

- Function: the body becomes one call; unpack as many outputs as the interface returns:

  ```python
  from kda_kernels.<op>.interface import <op>
  ...
  h, y = <op>(x, residual, weight, eps=eps)   # KDA_BACKEND=eager restores the original math
  ```

  Two cases the one-line swap does not cover by itself. **Dtype subset**: when the kernel's
  contract accepts fewer dtypes than the eager code (a bf16/fp16 tensor-core GEMM under a
  repo whose `autocast_dtype` can be set to fp32), keep the original math reachable for the
  rest: the stub routes inputs outside the contract to the package's `<op>_eager`
  (`from kda_kernels.<op>._eager import <op>_eager`; it is not exported by `interface.py`)
  before the interface's checks would raise, so no flag setting can break a configuration
  that worked before. **Optional epilogue** (SPEC `params: {act: ...}` the user asked to fuse):
  give the definition site a keyword with the *old* behaviour as default (`act="none"`) and
  let the one call site that composed the activation after the op opt in
  (`F.silu(adaln(...))` -> `adaln(..., act="silu")`); that call-site edit is part of the
  integration, is listed in `E2E.md`, and the `KDA_BACKEND=eager` row still has to match
  `before` (the eager backend applies the same epilogue).

  The interface already resolves `KDA_BACKEND` (env wins over everything). When the repo has
  a config system, expose the same choice there: a field `kda_backend: Optional[str] = None`
  (`None` = the package default, so the env var keeps working). A definition site that sees
  the config (a Module) passes `backend=cfg.kda_backend`; a free function that does not is
  covered by one call `kda_kernels.<op>.interface.set_default_backend(cfg.kda_backend)` in
  the model class's `__init__` (every construction site, including eval scripts, goes through
  it; the training script does not). Its position inside `__init__` does not matter (it sets
  a process-wide default read at call time, so anywhere before the first forward is fine);
  an unknown value raises there, at construction. It does mean constructing the model
  *resets* the default: a probe that calls `set_default_backend` and then builds the model
  runs the package default on both sides of its comparison; in probes set the
  backend after construction, or pass `backend=` per call. A field that already exists with either
  semantics is kept as is.
- Recompute, when SPEC `recompute.available: true`: the interface also resolves
  `KDA_RECOMPUTE=1|0` (env wins) and a `recompute=` kwarg. Expose it beside the backend the
  same way: a config field `kda_recompute: Optional[bool] = None` (`None` = SPEC
  `recompute.default`), passed as `recompute=cfg.kda_recompute` by a Module or set once through
  `kda_kernels.<op>.interface.set_default_recompute(cfg.kda_recompute)`. When SPEC says
  `available: false`, add nothing: the kwarg is accepted and ignored, and a flag that does
  nothing misleads.
- `nn.Module`: replace the body of `forward` the same way; parameters and their names stay
  where they were so checkpoints load unchanged.
- The eager code the kernel replaced is deleted only when the user asks; otherwise it lives
  on as the `eager` backend in `_eager.py`.

Completion: `KDA_BACKEND=eager python -c "<import the module>"` and the default both import
without error.

## 3. Smoke after

For inference-only workloads, use the three before/eager/kernel configurations below with inference latency and the output/quality metric in place of step time and loss. Preserve every measured repeat. Compare eager with original behavior and kernel outputs at SPEC tolerances on identical inputs. Omit recompute, backward, training-loss and 50-step checks; a flat latency result can trigger a forward/host-overhead diagnostic. The remaining training-specific instructions apply only when backward is part of the observed workload.

Run the smoke command under two backends: `KDA_BACKEND=eager` (proves the flag) and default
(the SPEC `kernel_backend`: `triton`, `nvmath` or `tilelang`); with `before`, that is three
configurations, each run as two repeats (below). For training with SPEC `recompute.available: true`, add a
fourth: the kernel backend with `KDA_RECOMPUTE=1` (or `=0` when `recompute.default: true`), so
the user sees the step-time and peak-memory trade in the same table (`torch.cuda.max_memory_allocated`
printed as one `peak_mem_gb <value>` line inserted *after the step loop and before* the
`median_step_ms` / `final_loss` prints, so those stay the last two lines). The `before` row
ran the unmodified script and carries `-` there; the `eager` flag row is the same code path
and stands in for its memory number. The `peak mem
GB` column is required only when that recompute row exists; without it write `-` and do not
add a memory print or a probe for it (the smoke script's last two lines stay as they are).
Fill the `after` rows. If the training loop uses `torch.compile`, run the kernel smoke with
`TORCHINDUCTOR_FORCE_DISABLE_CACHES=1` the first time (or after any `_impl_*.py` edit):
inductor's on-disk cache is keyed on graph code, not on a custom op's fake outputs, so a stale
artefact from an earlier fake fails with an `assert_size_stride` naming shapes the kernel no
longer produces. The user's later runs can use the cache normally.

```
# E2E: <op>
| run | backend | median step ms | final loss | peak mem GB | notes |
|---|---|---|---|---|---|
| before | original code | ... | ... | ... | commit <sha> |
| after  | eager (flag)   | ... | ... | ... | must match before within noise |
| after  | <kernel_backend> | ... | ... | ... | |
| after  | <kernel_backend>, KDA_RECOMPUTE=1 | ... | ... | ... | only when SPEC recompute.available |
```

Two repeats of each configuration; report the mean of their two median step times, both
medians and their spread in notes, and the final loss of
the **first** repeat (loss is a correctness signal, not a speed one, so it is never the better
of two), and the spread between the two repeats is the noise floor: a delta inside it is
noise, not a regression or a speedup. When the two repeats of one configuration differ by
more than ~10%, run a
third and report the median of the three medians plus their range in the notes column; a step-time delta smaller than that range is
"unresolved on this card", and the device-time table of `REPORT.md` remains the performance
evidence, while the step-time row is indicative. When `before` ran at a different time from
`eager` and the kernel runs (other GPU work in between, a shared card that quietened), the
step-time pair to report is `eager` (flag) vs the kernel, run back to back; `before` then
serves only the loss / bit-identity row.
Judge nothing beyond arithmetic: report the step-time delta in ms and percent, and the loss
delta, and whether the `eager` run matches `before` (same seed: it should be bit-identical or
within that noise floor). "Matches" is judged on what the smoke prints: the same `final_loss`
at its printed precision, or a difference no larger than the spread between the two `before`
repeats; do not add tensor or weight comparisons to the script for it. The kernel row is not generally bit-identical within the approved tolerance, although it
must preserve SPEC's required rounding points. Small differences can accumulate during training; the correctness gate was M3 (SPEC tolerances on the op),
report the delta with that context; a new non-finite result or a contract violation
requires investigation before calling the integration complete. Only the
`eager`-vs-`before` pair is the bitwise check. A smoke that regenerates its targets every
step (random latents, loss swinging by 10x step to step) amplifies a 1-ulp difference far
past that guide; when the delta exceeds ~1e-2, isolate
once instead of investigating the trajectory: one step on identical weights and data with
the region eager vs kernel (`backend=` per call), and report that step-0 loss difference
(1e-4 there) next to the 50-step number, in the kernel row's `notes` column of the `E2E.md`
table (`one-step isolation: eager 3.2278244 vs kernel 3.2278101`). Set the step-time expectation before reading
the number: `report.json` compares device time against the *fastest* of eager and
`torch.compile`, while the smoke usually trains the eager model, so the available win per
call is the eager-minus-kernel device time (the `base` and `kernel` columns of the eager
rows) times the call count per step, with host-dispatch costs deducted only when measured. Without that diagnostic, label this a device-time estimate; measure host overhead only if needed to explain the step result. A step-time delta in that
range is the expected middle case and needs no investigation. When the step time did not move although `report.json` promised a
speedup, capture one torch profiler trace of a few steps and read it with the
`torch-profile-reading` skill to see whether the op is on the critical path at all (the
kernels may sit below the top-k cut-off or be unnamed; a call counter at the definition
site settles "is it called" faster than the trace). When it **is** on the critical path and
the step is still flat, the per-call device cost differs between the isolation bench and
the model: time the model-context forward and backward *separately* for the kernel path and
the eager path (a few steps each, `torch.cuda.synchronize` around `loss.backward()`), and
expect one phase's win to be offset by the other. Known mechanisms: autograd's eager backward
reuses the autocast-cached bf16 weight while the fused path re-casts the fp32 master every
call; host dispatch and device work can overlap, while kernels on one stream execute in order; each custom-op
call adds its aux hand-off. A forward gain that the fused backward gives back in-model is a
finding to write into `E2E.md` with the two phase numbers, not a dead end and not a reason
to re-tune the kernel.

Completion: three rows filled (four with recompute), deltas stated, `STATUS.md` M6 ticked
with the `E2E.md` path.

## 4. Hand over

Show the user the `E2E.md` table and the complete integration change set, how to turn the kernel off
(`KDA_BACKEND=eager`, or the config field) and, when available, how to toggle recompute
(`KDA_RECOMPUTE=1|0`, or the config field). If the user runs `torch.compile`,
mention that torch >= 2.6 registers the kernel as a custom op (no graph break) and older
versions graph-break at the `autograd.Function`.
