---
name: kda-kernel-scaffold
description: M0 of the KDA kernel workflow. Creates the kernel package for a named region of user code (spec, eager reference, workloads, roofline, hardware peaks) and checks the M1 contract checkpoint. Use when a new fused kernel is requested, when `kda_kernels/<op>/` does not exist yet, or when SPEC.md must be re-derived from changed user code.
---

# kda-kernel-scaffold (M0)

Runs inline in the orchestrator's context (no subagent). Output: `kda_kernels/<op>/` in the
user's repo, rendered from `template/` next to this file, with `SPEC.md` filled well enough
that an implementer can work from it alone. Follow the orchestrator's authorization scope;
SPEC is internally checked by default and presented for approval only when requested.

The template is the single source of truth for the package layout; read
`template/op/SPEC.md` once before starting so you know every field you must fill, and
`../kda-kernel-implement/reference/common/findings.md` for the measurements behind the
workload, roofline and numerics rules below.

## 1. Intake

Gather, from the user's message, the code and the training config, before writing anything:

- **Region**: `path::symbol` of the function or `nn.Module` whose body becomes the kernel.
  The user names a region or the orchestrator selects it from model profiling. For a
  selected span, derive all boundary inputs/outputs and externally consumed intermediates;
  preserve aliases, side effects and parameter dependencies. Ask only about ambiguous semantics.
- **Real shapes, dtypes and layouts**: batch, sequence, hidden sizes and the autocast dtype the
  training run actually uses. Read the config files (Hydra yaml, dataclasses, argparse
  defaults); one real workload outranks any number of representative ones. Note how the
  inputs are produced: a `q, k, v = qkv.split(...)` or a `[:, :, :D]` slice means the kernel
  receives a strided row (SPEC `strides`). **Probe the dtypes, do not trust the prose**: run
  one representative forward step (including backward only for training) of the smoke command with the region wrapped to print
  `dtype`, `shape` and `stride()` of every input, and take SPEC `dtype`/`dtypes`/`strides`
  from that print: a producing `Linear` under autocast can emit bf16 conditioning tensors
  even when the task describes fp32. The probe is the script
  `<op_dir>/_scratch/probe_intake.py` (render the package first: step 2 is deterministic and
  content-free, so do it before this probe), run from the repo root as
  `python -m kda_kernels.<op>._scratch.probe_intake` (`python _scratch/x.py` does not put the
  repo root on the path); the `_scratch/` directory is deleted before M1:

  ```python
  import sys; sys.path.insert(0, ".")
  import minilm.model as m                      # the module that *defines* the region
  orig = m.fc1_gelu                             # keep the original: the wrapper must call it, not itself
  def probe(*a, **k):
      for i, t in [*enumerate(a), *k.items()]:
          if hasattr(t, "dtype"): print(i, t.dtype, tuple(t.shape), t.stride(), t.requires_grad)
      out = orig(*a, **k)
      print([(o.dtype, tuple(o.shape), o.grad_fn) for o in (out if isinstance(out, tuple) else (out,))])
      return out
  m.fc1_gelu = probe
  try:
      # Build model/data from the existing entrypoint; reproduce the full representative step.
      # Include backward only for a training workload.
      run_representative_step()  # replace with the observed entrypoint/step
  finally:
      m.fc1_gelu = orig
  ```

  Print **every** call of the step, not the first: the region is called from several sites
  and they differ (adaLN: the `Block` sites pass `chunk(6)` views of the modulation output,
  row stride `6D`; `FinalLayer` passes `chunk(2)` views, row stride `2D`). Each
  distinct layout is a `user_` row.

  Patch the name in the module the call sites resolve it from (a function called by module
  global is patched on that module; a method on the class), and restore it after the complete
  representative step in `finally`. The wrapper calls the saved original, not the patched name. `requires_grad` of each input decides SPEC `grad_inputs` (a `cos`/`sin`
  table built under `no_grad` gets no gradient; a norm weight does), and inspect the actual downstream call sites for cloning/views or other consumers.
  For an inference-only workload record `grad_inputs: []`; do not infer training from a
  Parameter having `requires_grad=True` when the entry point runs under `no_grad`.
- **Compute dtype** (`compute_dtype` + `compute_dtype_source`): the user named it -> `user`;
  the user's code casts (`x.float()`, `.to(torch.float32)`, an fp32 accumulator) -> `inferred`,
  with `path:line` for the gate; neither -> `default` fp32, and the gate says so in one
  sentence. Never silently assume.
- **Pattern and kernel backend**: match the region to one row of
  `.agents/skills/kda-kernel-implement/reference/common/compute-patterns.md` (`compute_pattern`);
  the row's default `kernel_backend` is the package's unless the user asks for the other.
- **Backward**: does the region run under autograd in training? Default `backward: fused`.
- **Recompute ratio**: aux bytes (the `saved_for_backward` tensors that are not user-visible
  outputs) over activation bytes (inputs + outputs) on the user workload. At >= 10% propose
  `recompute.available: true` at the gate with the backward overhead (extra bytes and FLOPs
  of recomputing the aux); below, `available: false`. `default` is false unless existing configuration or explicit user priorities justify it;
  record the decision internally and show a measured memory/time trade at integration.
- **Integration**: the owning definition or selected region and any necessary ancillary edits, whether the repo has
  a config system that can carry `kda_backend` and `kda_recompute` fields, and the smallest
  training command that can serve as the M6 smoke run.
- **Torch and kernel-DSL versions** in the user's environment (`python -c "import torch, triton"`;
  for a `gemm_tensorcore` region also check `import nvmath` and epilogue support. Honor an
  explicit supported backend choice; otherwise choose applicable nvmath, then Triton. Probe
  TileLang when explicitly selected).

Completion: every bullet is derived from evidence, with only unresolved model-level intent
asked of the user. Report unavailable execution evidence honestly.

## 2. Render

Op name: the fused ops in data-flow order, snake_case, e.g. `fused_residual_rmsnorm`,
`qk_rmsnorm_rope_permute`.

From the **user's** repo root (the directory that will contain `kda_kernels/`; every later
`python -m kda_kernels.<op>._run_dev` in this workflow runs from there too, since the module
resolves through cwd), with a relative destination (it resolves against the cwd wherever
python runs, including a container that mounts the repo elsewhere). This command needs no
GPU and no torch: any python 3.8+ runs it, the GPU interpreter is fine too:

```bash
python .agents/skills/kda-kernel-scaffold/scaffold.py --op <op> --dest kda_kernels --kernel-backend triton
```

Honor a supported explicit backend choice first. Otherwise use `--kernel-backend triton`
except for `gemm_tensorcore`, which takes `nvmath` when
`import nvmath` works in the user's environment and the epilogue is in cuBLASLt's set (bias,
ReLU, tanh-GELU, aux store; compute-patterns.md, mainloop note) and `triton` otherwise;
`tilelang` is experimental and chosen when explicitly requested and applicable (backend note). Only
the chosen backend's kernel directory (`_triton/`, `_nvmath/` or `_tilelang/`) is rendered.
`_common/` is installed once per repo; the script says when a newer template version exists
(`--sync-common` updates it). Completion: the command printed `rendered op`.

## 3. Eager reference (`_eager.py`)

Copy the user's code verbatim into `<op>_eager(...)`, wrapped in the functional signature:
tensors positional, scalars keyword-only, output always a tuple. Exception: a scalar the
user's call sites pass *positionally* (`adaln(x, scale, shift, eps)`) stays positional with
the same default in `_eager.py` and `interface.py`, so the M6 one-line swap leaves the call
sites untouched; new options the user did not have (`act`) are keyword-only. The harness
passes `params` as keywords either way. Keep every dtype cast
exactly as the user wrote it (an fp32 upcast in their LayerNorm is part of the contract). The
harness runs without autocast: when the region ran under autocast in the model (the probe
showed bf16 activations meeting fp32 `Parameter`s, `F.linear(bf16, fp32)` raises outside
autocast), add the cast autocast performed, `w.to(x.dtype)`, at the top of `<op>_eager` with a
comment naming it as the autocast cast, and keep the fp32 Parameter in SPEC `dtypes`; the
kernel reads the fp32 weight and casts once.
Autograd through this function is the reference backward. Record `source.path` and
`source.symbol` in the SPEC front matter; after M6 this file is the record of the original code.

Completion: `_eager.py` has the user's math and no `TODO`; any semantic contradiction is
resolved before dependent kernel implementation.

## 4. Interface (`interface.py`)

Same parameter names as `_eager.py` plus `backend=` and `recompute=`. The template renders
the generic checks (`check_cuda`, `check_dtype_in`, `check_last_dim_contiguous` from
`_helpers.py`) on `x`; extend `_check_contract` to every tensor input and add the op-specific
rows. The checklist per op:

| row | rule |
|---|---|
| device | every tensor input on the same CUDA device |
| dtype | each input's allowed set is what the user's code feeds it (a weight the model keeps fp32 is `{fp32}`; the activation `{bf16, fp16, fp32}`); the eager reference accepting more is not the contract |
| rank / shape | ranks fixed; inputs sharing a dimension checked with `check_same_shape` / `check_last_dim` |
| last stride | `check_last_dim_contiguous` on every tensor the kernel indexes; leading layouts match SPEC; `rows_and_stride` requires collapsible leading dimensions |
| kernel limits | the pattern's caps (`D <= 8192` for a fused row-wise backward, MMA-shape multiples for GEMM) |
| row count | `n_rows < 2**31` when program ids index rows (int64 offsets cover elements, not the grid) |

Every check raises `ValueError`/`TypeError` before any launch. `_run_dev.py --contract`
exercises the table mechanically on the first user workload, mutating the first tensor input:
`cpu_tensor`, `bad_last_stride` (last stride 2), `shape_mismatch` (last dim narrowed by one,
only when another input shares that dim), `bad_dtype` (int32) must raise; `zero_rows`
(leading dim 0 on the token/row inputs) and `no_grad` must return. Set SPEC `row_inputs` to
the inputs that carry the token dim (`[x, residual]` for a GEMM epilogue, `[x, residual]`
for a residual norm); left `null`, the probe empties every input sharing `x`'s leading dim,
which also catches a `(N, K)` weight whenever `N == M` (an 8192-token batch into an 8192-wide
projection) and makes the call ill-formed. The probe slices the *leading* dim, whatever it
is: for a batch-first op (`x (B, N, D)` with per-sample `scale`, `shift (B, D)`; heads
`q, k (B, S, H, D)`) the leading dim is the batch, so `zero_rows` means an empty batch, and
every input that shares `B` must be emptied together or eager raises a broadcast error
(`row_inputs: [x]` leaves `scale` nonempty against an empty `x`). Rule: list
every input whose leading dim is the one being emptied (`[x, scale, shift]`, `[q, k]`), or
leave `null` when *all* inputs with that leading size should follow; inputs without that dim
(`cos`/`sin (S, R)`, a `(D,)` weight) are left alone. An empty sequence with a full batch is
not probed; the kernel's `S = 0` path is covered by `user_small_rows`' contract only if you
add it. The verdict fails when a
bad input gets through or is rejected by a CUDA error instead of a Python one. Copy the math
and the shape/dtype tables from SPEC into the docstring. Add an `nn.Module` only when the
user's source op is a Module. Then fix the package docstring in `__init__.py`: its one-line
description and the usage block are rendered for a one-argument op (`(y,) = op(x)`); rewrite
the call with the real signature and output tuple.

Completion: `interface.py` checks every row of the table; no `TODO` left in `interface.py`
or `__init__.py`.

## 5. SPEC front matter and prose

Fill every field of `template/op/SPEC.md`:

- `kernel_backend`, `compute_pattern`, `compute_dtype`, `compute_dtype_source`, `recompute`:
  from the intake; `recompute.why` states the ratio and the overhead in one sentence.
- `backward`: `fused` unless the user asked otherwise; `eager` needs `backward_reason`.
- `saved_for_backward`: materialized auxiliary tensors, with name, shape, dtype, bytes and
  save-vs-recompute reasoning. Separately document retained original inputs/user outputs in
  the Math/autograd-state prose. Retention can extend lifetimes even without new allocation.
  `bytes` is free text that a reader can evaluate: a formula in the shape symbols
  (`rows * 4`, `B*S*D*2`) is the normal form, a number only when the shape is fixed.
  Materialized auxiliaries are written only when a backward can follow (`save_aux`); count
  their traffic in `fwd` and `bwd`, not `infer`. For `gemm_tensorcore`, preserve the eager
  mainloop's rounding point. Cancellation in `gelu(z) + R` can expose differences among a
  fused fp32 accumulator, bias-inclusive rounded `addmm`, and `matmul` followed by fp32 bias.
  Set `make_inputs` to reproduce the model's observed/init scale first, then run the step-8
  probe; use its zero-failure candidate to choose the mainloop and aux form. Unscaled random
  GEMM operands grow as `sqrt(K)` (45 at `K = 2048`), while the reference model's weight init
  yields `|z| ~ 1`; an observation in one regime is not a universal backend rule.
  Record the decision in SPEC Math, Contracts and the auxiliary table.
- `workloads`: the rows `reference/workloads.md` (beside this SKILL.md) prescribes for the
  op's class, in its order: the user's real shapes (`source: user`, `required: true`), the
  small-rows and strided rows (`required: true`), the `D` sweep, the ultra-large row
  (`source: edge`: the **smallest** shape with one tensor above 2^31 elements, 4 GiB in bf16,
  so that it runs; `validate_spec` rejects rows above 13 GiB of inputs and an `edge_huge` that
  crosses nothing. Shapes that fill the card are a mistake, not thoroughness). Use `strides` for
  the strided row (same rank as the shape, last entry 1), `dtypes` for inputs that stay fp32
  (weights, gates) **and for non-float inputs** (`key_valid: bool`, `positions: int64`;
  `bool`/`int32`/`int64` are accepted there, never as the workload `dtype`; they take no
  gradient and the harness fills a mask ~90% true and small random indices; a table with
  meaning, positions on the model's grid or an `inv_freq` table, is built in
  `_run_dev.make_inputs` with the model's own function, step 8), `grad_inputs` when some
  inputs never receive gradients, `params` for scalars such as `eps`. Write floats with a
  dot (`1.0e-6`): YAML reads `1e-6` as a string. YAML does no arithmetic: compute a strided row's numbers (`N * (K + pad)`) with python before you type them. A `torch.dtype`-valued argument (`out_dtype`)
  is a param spelled as its name (`out_dtype: bf16`); `interface.py` accepts the name and the
  `torch.dtype` (the call sites pass `y.dtype`) and maps it once.
- `tolerances`: leave empty. The defaults (fp32 1e-5, fp16 1e-3, bf16 1.6e-2) are applied at
  the workload's dtype and `atol` is scaled by `sqrt(rows)` for weight gradients automatically;
  an override is for references that are noisier still and needs a sentence of justification.
  The other case that is settled here, not at M2: a fixed `atol` on an output that the math
  can cancel to ~0 from large intermediates (see `saved_for_backward` above) either fixes the
  mainloop that reproduces eager's rounding or gets an override with the arithmetic that
  justifies it.
- `integration`: definition site, flag (`KDA_BACKEND` plus the config field if any),
  `recompute_flag` (`KDA_RECOMPUTE` plus its config field), smoke command.

Prose sections: description, math with every backward-relevant intermediate named,
signature tables, contracts (the compute-dtype provenance sentence, the pattern regime),
saved-for-backward table with the Recompute paragraph, roofline for `fwd`, `infer`, `bwd`
(and `bwd_recompute`), workload rationale grouped as `workloads.md` groups them.

Completion: `SPEC.md` contains no `TODO` and `python -c "from kda_kernels._common.spec import *; print(validate_spec(load_spec('kda_kernels/<op>/SPEC.md')))"` prints `[]`.
When `recompute.available` is true, set `RECOMPUTE_AVAILABLE`/`RECOMPUTE_DEFAULT` in
`backends.py` to match (`_run_dev.py` refuses to run when they disagree).

## 6. Roofline (`_speed_of_light.py`)

Implement `estimate(phase, **inputs)` for `fwd`, `infer`, `bwd` and (when available)
`bwd_recompute` from the Roofline prose: bytes = every input read once plus every output
written once; FLOPs from the math. `infer` is `fwd` without the aux writes; `bwd_recompute`
reads the inputs the aux depends on instead of the aux. Backward checklist: one incoming
gradient read **per forward output** (a two-output op reads two), every saved tensor read
once, one gradient written per differentiable input (weight gradients: the fp32 partials
count too). Set `ROOF` to the compute unit doing the arithmetic (`fp32` for norms, gating,
RoPE, permutes; `bf16` for tensor-core epilogues). Read
`.agents/skills/kda-kernel-implement/reference/common/speed-of-light.md` for the worked examples.

For a GEMM pattern count **whole GEMMs**, not factors: with a helper `gemm = 2*M*N*K` (one
GEMM), `fwd`/`infer` are `gemm`, `bwd` is `2*gemm` (`dX = dZ W`, `dW = dZ^T X`) and
`bwd_recompute` is `3*gemm` (plus the recomputed `Z`), i.e. `2`/`4`/`6 * M*N*K`. Writing
`4*gemm` doubles the backward roof and reports a `sol_eff` above 1 that the prose, copied
from the same reasoning, will not catch.

For a tensor-core `ROOF` also fill `gemm_shapes(phase, **inputs)`: the list of `(M, N, K)` the
phase runs (fwd `[(M, N, K)]`, bwd `[(M, K, N), (N, K, M)]`, bwd_recompute all three). The
harness times cuBLAS on them and that is the achievable compute roof the verdict uses (the
datasheet peak is only reported); the compile baseline becomes `max-autotune`. Say so in the
SPEC Roofline prose: "roof = max(measured cuBLAS time, measured copy time for the phase bytes)". **Not for
`flash_attention_2`**: leave `gemm_shapes` raising `NotImplementedError` so the harness times
cuBLAS on one same-FLOP cube (the achievable tensor-core rate); the per-head `(B*H*S, S, d)`
GEMMs an attention "executes" are skinny in `K = d` and cuBLAS runs them slower than the
fused kernel, which reports `sol_eff > 1`. The FLOPs are the attended
area (compute-patterns.md), and the SPEC prose names `F.scaled_dot_product_attention` as the
number to beat, timed by the implementer, since the harness baseline is the materialised path.

Completion: the prose byte formula and `estimate()` agree term by term for every phase, and
for a GEMM pattern the phase FLOPs are `1 : 2 : 3` GEMMs (`fwd : bwd : bwd_recompute`) and
`gemm_shapes` lists the same GEMMs; check the ratio with
`python -c "from kda_kernels.<op>._speed_of_light import estimate; ..."` on the user workload
before M1.

## 7. Hardware peaks

```bash
python -c "from kda_kernels._common.gpu_info import get_gpu_info; print(get_gpu_info().as_dict())"
```

Paste the dict as one `hardware:` entry, replacing the template's `hardware: []` with a
one-item YAML list (`- {name: ..., cc: "8.0", ...}`; quote `cc`, or YAML reads it as a
float). When `flops_source` is `cc-derived`, the SKU is not
in the built-in table: web-search the vendor datasheet for dense fp32 and bf16 TFLOP/s and
HBM bandwidth, overwrite the numbers, and cite the source in the Hardware prose section.
`unknown` means SOL will be reported as n/a; say so at the gate.

## 8. Harness sanity run

`make_inputs` in `_run_dev.py` must produce the model's scales, because numerics (tolerance
failures, the cancellation question above) and even the verdict depend on them: norm weights
around 1 (`1 + 0.1 * randn`), gates in (0, 1), positions as integers, **GEMM weights at the
model's init std** (`nn.init.normal_(std=0.02)` -> `0.02 * randn`; read the init), fp32
conditioning vectors at the scale the model's MLP emits, and **tables as the model builds
them**: `cos`/`sin` from the model's own table function (values in `[-1, 1]`, the duplicated
halves `c[R/2 + j] == c[j]` a kernel may legitimately exploit), position ids as integers in
range. A `randn` table passes M2 on the scaffold's inputs and fails a kernel that assumed
the duplication at M3, or hides that it did. Unscaled `randn` on every input is the template
default and is wrong for any op with a matmul or a table. Then:

```bash
python -m kda_kernels.<op>._run_dev --verify --backend eager --json kda_kernels/<op>/report.eager.json --md kda_kernels/<op>/report.eager.md
```

Eager against eager exercises the SPEC, the workloads (the strided view included), the input
generator, the gradient plumbing, the `infer` phase and the contract probe (every `contract`
row must pass: a failing row is a missing check in `interface.py`). The side files keep
`report.json` free for kernel results (a `report.json` with `backend: eager` would read as a
passing kernel to anyone who skips the `backend` field). An optional row that runs out of
memory on a shared card is recorded as skipped (`report.json` `skipped:`), not as a failure.

For a `gemm_tensorcore` op, run the cancellation probe now, on these inputs, for each
candidate rounding point of the mainloop output, and pick the design with **zero failures**
on the user row at the SPEC tolerance:

```python
import torch
from kda_kernels._common.spec import load_spec
from kda_kernels._common.verify import compare
from kda_kernels.<op> import _run_dev, _eager
spec = load_spec("kda_kernels/<op>/SPEC.md"); w = next(w for w in spec.workloads if w.name == "<user row>")
i = _run_dev.make_inputs(w, torch.device("cuda")); x, wt, b, r = (i[k] for k in ("x", "weight", "bias", "residual"))
# adapt to the op: no residual -> drop `r` from y_from; rank-3 `x` -> `x.flatten(0, -2)` (addmm wants a matrix)
ref = _eager.<op>_eager(**i, **w.params)[0]                                   # the contract
def y_from(z): return (torch.nn.functional.gelu(z) + r.float()).to(x.dtype)  # the epilogue in fp32
cands = {
  "fp32 Z + fp32 b (fused tl.dot, bias-free aux)":    y_from(x.float() @ wt.float().t() + b.float()),
  "bf16 Z, fp32 b (cuBLAS matmul, bias-free aux)":    y_from((x @ wt.t()).float() + b.float()),
  "bf16 (Z + b) (cuBLAS addmm, bias-inclusive aux)":  y_from(torch.addmm(b, x, wt.t()).float()),
}
atol, rtol = spec.tolerance(x.dtype)
for name, y in cands.items(): print(name, compare(y, ref, atol=atol, rtol=rtol).as_dict())
```

The three rows are the three mainloops the GEMM SNIPPET offers; the rule of thumb (cancellation
-> cuBLAS `addmm` with the bias-inclusive aux, else the fused mainloop) names the usual
winners, but the probe decides: cuBLAS `matmul` with an fp32 bias in the epilogue can
also be the only candidate that matches the eager rounding. Write the winning row, and the
failure counts of the others, into SPEC Math as the mainloop restriction.
Optionally time the baselines for the gate message
(`--bench --backend eager --workload <user_row> --json <op_dir>/report.baseline.json --md <op_dir>/report.baseline.md`; eager and compiled times appear in the diagnostic report), so the M1 gate can state what "win" is on the table.
Completion: exit code 0 and verdict `pass`.

## 9. Record the M1 contract checkpoint

Check SPEC, interface and eager reference agree, validate workloads and complete eager sanity.
Tick M0/M1 in STATUS with evidence, backend/pattern, recompute rationale and the authorization
scope. Present a short model-level update; link SPEC instead of asking the user to review YAML,
roofline formulas or launch choices. Ask only for unresolved semantics or a material change to
numerical requirements, or when the user explicitly requested stage-by-stage approval.
The orchestrator owns continuation and the remaining integration review.
