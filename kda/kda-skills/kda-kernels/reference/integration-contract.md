# Integration contract and early path experiment

Create one `INTEGRATION.md` per change set, beside PLAN or the selected operation.
One-operation requests use the same short record. Front matter schema:

```yaml
---
integration_schema: 1
operations: [kda_kernels/example/SPEC.md]
entrypoint: model.py::Model.forward
replacements:
  - symbol: model.py::region
    consumers: [attention]
    boundaries: [normalization stays after redistribution, preserve dtype casts]
command: python existing_request_driver.py
inputs: [artifacts/input.npz]
configuration: {mode: inference, sp: 2}
ranks: [0, 1]
branches: [cached]
variants:
  current: {root: /snapshot/current, files: [model.py], backend: eager}
  candidate: {root: /snapshot/candidate, files: [model.py], backend: triton}
  rollback: {root: /snapshot/candidate, files: [model.py], backend: eager}
lifecycle: {owner: worker request, cleanup: finally on success and exception}
fallback: unsupported inputs and explicit eager retain original behavior
quality: unchanged preprocessing and decoder; compare all saved outputs
primary_metric: request_wall_ms
matrix:
  - id: cached_sp2
    data_root: /snapshot/data
    input_hashes: {artifacts/input.npz: <sha256>}
    config_hashes: {resolved_config.json: <sha256>}
authorization: {source: user request, integration: false}
---
```

Replace the example with observed paths/configuration. Variant `backend` names the actual
executed selection expected in that path; for mixed backends use the same symbol-to-backend
mapping in variant.backend and worker witnesses. Auto is a selection policy, not proof of the
backend actually executed. Bind resolved input/configuration hashes per matrix cell, not only
file labels. The invocation command must equal its contract (optional per-variant/cell override). `files` covers runtime and driver dependencies;
optional `loaded_files` restricts which of those files must be observed as imported modules.
Keep all external consumers and movement constraints in the replacement rows. A lifecycle
without persistent resources records tested no-op cleanup explicitly. Existing mathematical
and quality requirements are authoritative; do not invent a model tolerance from op tolerance.

Before a second tuning round, run a minimal correct candidate in an isolated physical source
snapshot with the real process/rank topology. Record loaded files from each worker using
`integration.runtime_witness`, actual call counters, backend, and success/failure teardown
probes. Parent-only monkeypatching is insufficient for spawned workers. The initial experiment
checks actual reachability, numerical behavior, lifetime and direction of benefit; it does not
authorize production edits or replace final request validation.

Use the user's driver, not a new scheduler. `integration.invocation` binds source/input/config
hashes and witnesses. Each `experiment.run_ab` output inspector returns `invocation_id` linking
the saved output to that unique invocation. Create the invocation inside the timed callable,
using real started_at/finished_at boundaries, before returning to the output inspector.
Samples cannot reuse invocation ids; the assessor checks request time containment, contract
digest and declared data identities. All repeats and failures remain in the record.
The independent integration audit contains `original_contract`, `integration_diff`,
`input_preprocessing`, and `quality`, each `{passed: bool, evidence: path-or-source-reference}`.
Set audit.evidence_digest with `integration.audit_digest(contract, invocations, experiments)`
after reviewing those exact records. Old audits cannot validate a changed execution record.
Authorization and target are decision-only fields: changes require reassessment but do not
pretend the execution changed. Input/configuration files are rehashed from each data_root.
Read original callers and input sources before accepting the derived SPEC/eager reference.

Validate before GPU work:

```bash
python -m kda_kernels._common.integration INTEGRATION.md
```

Assess saved evidence (`invocations`, `experiments` keyed by matrix id, and `audit`):

```bash
python -m kda_kernels._common.integration INTEGRATION.md --evidence artifacts/integration.json --out artifacts/acceptance.json
```

Missing or contradictory path evidence prevents integration acceptance. This result does not
promote a kernel report into production acceptance. Final delivery additionally requires the
task performance assessment and recorded authorization. Preserve negative/noisy measurements.
