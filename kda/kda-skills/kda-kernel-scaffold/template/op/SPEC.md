---
kda_spec: 2
op: {{op}}
common_version: {{common_version}}
source:
  path: TODO/path/in/user/repo.py
  symbol: TODO.Class.method_or_function
kernel_backend: {{kernel_backend}}   # supported explicit user choice; otherwise applicable nvmath (GEMM with supported cuBLASLt epilogue); otherwise triton. TileLang is experimental.
compute_pattern: TODO                # a pattern name from compute-patterns.md (e.g. rowwise_onepass)
backward: fused            # fused | eager  (eager requires backward_reason)
backward_reason: null
compute_dtype: fp32                  # fp32 | bf16 | fp16: elementwise math and the MMA accumulator dtype
compute_dtype_source: default        # user | inferred | default (the M1 gate states which)
recompute:                           # second training path that recomputes aux instead of saving it
  available: false
  default: false
  why: null
target_archs: [sm80, sm90, sm100]
row_inputs: null                     # inputs whose leading dim is the token/row dim, e.g. [x, residual]; null = every input sharing x's leading dim (wrong for a GEMM weight (N, K) when N == M)
tolerances: {}             # per-dtype [atol, rtol] overrides; defaults: fp32 1e-5, fp16 1e-3, bf16 1.6e-2
saved_for_backward: []     # materialized auxiliaries [{name, shape, dtype, bytes, why}]; retained inputs/outputs documented separately
workloads:
  # The user's real shapes come first and are required; then representative and edge shapes.
  # See reference/workloads.md for the rows each op class needs (small-rows, strided, ultra-large).
  - name: user_TODO
    source: user
    required: true
    dtype: bf16
    shapes: {x: [TODO]}
    strides: {}            # optional per input, same rank as shapes, last stride 1; omitted = contiguous
    dtypes: {}
    grad_inputs: null
    params: {}             # scalar kwargs; write floats with a dot (1.0e-6, not 1e-6)
hardware: []               # paste get_gpu_info().as_dict() (scaffold SKILL step 7); one entry per validated GPU
integration:
  definition_site: TODO/path/in/user/repo.py::function_or_class
  flag: {env: KDA_BACKEND, config: null}
  recompute_flag: {env: KDA_RECOMPUTE, config: null}
  smoke_command: null
---

# SPEC for `{{op}}`

**Description**: TODO one paragraph: what the user's code does and which ops are fused into one kernel.

**Math**:

$$
\text{TODO: write the forward equations; name every intermediate that the backward needs.}
$$

## Signature, Shape, Dtype rules

```python
# interface.py
def {{op}}(x: torch.Tensor, *, backend: str | None = None, recompute: bool | None = None) -> tuple[torch.Tensor, ...]:
    ...
```

**Inputs**

| Variable | Meaning | Shape rule | Dtype rule |
|----------|---------|------------|------------|
| `x` | TODO | `(..., D)` | fp16, bf16, fp32 |

**Outputs** (always a tuple)

| Variable | Meaning | Shape rule | Dtype rule |
|----------|---------|------------|------------|
| `y` | TODO | same as `x` | same as `x` |

## Additional contracts

- **Compute dtype**: `compute_dtype` (front matter) is the dtype of every elementwise
  operation and reduction inside the kernel and of the tensor-core accumulator. TODO one
  sentence on where it came from, matching `compute_dtype_source`: "the user asked for X" /
  "inferred from `x.float()` at path:line" / "KDA default fp32, the user did not specify".
  Tensor-core operands (`tl.dot`, `T.gemm`) are fed in their storage dtype, never upcast at
  load; the single cast happens in the epilogue.
- **Compute pattern**: `compute_pattern` (front matter) names the row of
  `reference/common/compute-patterns.md` this kernel must follow; TODO one sentence on the
  regime (e.g. row-wise reduction over `D <= 8192`; measured launch geometry for small row counts).
- **Constraints**: TODO (e.g. last dim contiguous, `D <= 8192`, `D` power of two).
- **Failure mode**: violated constraints raise a Python error before any launch (the rendered
  checks in `interface.py`); kernels never index out of range (int64 indexing, masked
  loads/stores). `_run_dev.py --contract` exercises these rows.

## Saved for backward

Materialized auxiliaries listed in `saved_for_backward` are written only when a
backward can follow: the launcher's `save_aux` flag, set by `_common.compat` from grad mode and
`requires_grad`, gates their stores. Inference and `torch.no_grad()` skip auxiliary stores; retained input/output storage is accounted for separately below.

| Tensor | Shape | Dtype | Bytes | Save or recompute, and why |
|--------|-------|-------|-------|----------------------------|
| TODO | | | | |

**Recompute** (front matter `recompute`): TODO aux bytes on the user workload, activation
bytes (inputs + outputs), their ratio, and the overhead of recomputing the aux in the backward
(extra bytes and FLOPs). When the ratio is at least 10% the scaffold proposes
`available: true` at the internal M1 checkpoint. Keep `default: false` unless existing configuration or explicit model-level priorities justify enabling it; the runtime switch is
`KDA_RECOMPUTE=1` / `recompute=` / the config field in `integration.recompute_flag`.

## Roofline

Formulas are implemented in `_speed_of_light.py`; keep them in sync. Phases: `fwd` (training
forward, aux written), `bwd`, `infer` (forward under `no_grad`, no aux), and `bwd_recompute`
when `recompute.available`.

- **Forward bytes**: TODO (sum of every input read once and every output written once, aux included).
- **Forward FLOPs**: TODO.
- **Inference bytes / FLOPs**: TODO (forward without the aux writes).
- **Backward bytes / FLOPs**: TODO.
- **Roof**: TODO memory-bound / compute-bound / latency-bound, and which compute unit (`fp32` CUDA cores or `bf16` tensor cores).

## Workloads

Why each workload in the front matter exists and what launch behaviour it is meant to exercise,
grouped as `reference/workloads.md` describes for this op class: the user's shapes, the
small-rows case (below the SM count), the strided case, the ultra-large case, the `D` sweep.

## Hardware

Peaks recorded in the front matter come from `_common.gpu_info`. One sentence: which GPU,
and where the peaks came from (`flops_source`: `table` = the built-in SKU table, nothing to
confirm; `cc-derived` or `unknown` = confirm by web search and cite the source here).

## Integration

List the replacement definition and justified ancillary call-site, configuration and instrumentation edits in the integration change set. State how the kernel is
switched off (`KDA_BACKEND=eager` and the config field, if any), how recompute is switched
on (`KDA_RECOMPUTE=1` and its config field), and the smoke command M6 runs before and after.

## Retained autograd state

Document original inputs and user outputs retained for backward separately from the
`saved_for_backward` materialized-auxiliary list. State which survive in saved-aux,
recompute and inference modes; retaining storage can extend its lifetime.
