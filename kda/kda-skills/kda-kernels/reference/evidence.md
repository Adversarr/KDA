# Evidence protocol

The runner computes raw numerics/performance; the verifier finalizes the audit. STATUS owns
workflow progression. Existing thresholds in `_common/report.py` are the machine rules;
reference measurements and proposed optimizations do not change them.

## Reports and finalization

`report.json` is the complete M3 benchmark. `report.verify.json` and
`report.fwd_only.json` are correctness side reports. Scoped repeats, tuning experiments and
event diagnostics always use their own explicit `--json` and `--md` destinations. The runner
rejects canonical or already-existing destinations for `--workload` runs before GPU setup; use a new dispatch/sample name to retain previous evidence.

Raw `verdict` is `pass`, `tune`, `fail` or `incomplete`. Required missing/skipped evidence is
incomplete; optional skips are diagnostic notes. Backward applies to gradient-bearing
workloads, recompute numerics when available, recompute performance when default. Verify-only
runs do not require timings. Compiled baselines remain optional. Exit codes: pass/tune 0,
fail 1, invalid inputs 2, incomplete 3. Exit 0 alone does not mean performance passed.

After the complete record, forward-only verification and source/lint/probe audit, write
`<op_dir>/audit.json`. All audit rows in the verifier skill must be answered in `notes`;
include the lint table and source/measurement references. The audit format is:

```json
{
  "report_digest": "<digest of current raw report>",
  "audit_complete": true,
  "independence": "independent",
  "findings": [
    {"severity": "note", "reason": "Concrete observation", "evidence": "SPEC.md or workload/phase"}
  ],
  "reruns": {},
  "notes": "Audit checklist, lint table and additional probe evidence in Markdown",
  "diagnosis": "workload/phase: observed mechanism and evidence, or none: pass"
}
```

Compute the digest from the repository root:

```python
import json
from pathlib import Path
from kda_kernels._common.evidence import report_digest
print(report_digest(json.loads(Path("kda_kernels/<op>/report.json").read_text())))
```
It binds the audit to this raw measurement; an old audit cannot finalize a new run. Keep the
audit file inside the operation directory. An empty findings list is valid after a complete clean audit. Severity is `hard` (failed
correctness/contract/lint/source audit), `unresolved` (missing evidence or uncertainty), or
`note`. Use `same_context` when no independent worker was available. Then run:

```bash
python -m kda_kernels.<op>._run_dev --finalize-audit <op_dir>/audit.json
```

This performs no GPU measurements. It requires complete benchmark and forward-only evidence
bound to current SPEC, operation code and shared runtime hashes. The JSON `verification`
object is authoritative for M3/M5; absent finalization is unfinished. Its precedence is
`fail > incomplete > tune > pass`. Hard findings force failure, unresolved findings prevent
acceptance, and notes cannot relax raw gates. Markdown is regenerated from the same object
and ends with `Diagnosis:`. Preserve raw measurements; do not hand-edit a verdict.

## Noise and missing measurements

For each tagged near-gate workload, repeat twice into unique side files:

```bash
python -m kda_kernels.<op>._run_dev --bench --workload <row> --method profiler --json <op_dir>/report.rerun.<run>.<row>.1.json --md <op_dir>/report.rerun.<run>.<row>.1.md
python -m kda_kernels.<op>._run_dev --bench --workload <row> --method profiler --json <op_dir>/report.rerun.<run>.<row>.2.json --md <op_dir>/report.rerun.<run>.<row>.2.md
```

List both operation-relative JSON paths under `reruns: {"<row>": ["report.rerun.<run>.<row>.1.json",
"report.rerun.<run>.<row>.2.json"]}`. Include additional near-gate pass-side repeats when the
measurement spread needs checking. Report all samples. A gate that changes between record
and repeat is incomplete; do not choose the fastest sample or tune automatically. After
these repeats, report unresolved environment/measurement limitations rather than cycling.
A missing/zero profiler sample cannot prove performance from numerics alone. Event timing is
a separate optional host-overhead diagnostic, never the final M3 record.

## Checkpoints and retries

Before dispatching a tune hypothesis, the orchestrator runs:

```bash
python -m kda_kernels._common.evidence <op_dir> before-run-<n>
```

Record `_checkpoints/before-run-<n>` in STATUS and dispatch it with the candidate record.
The snapshot includes worker-owned code/configuration and current reports, with source and
artifact hashes. It requires current finalized pass/tune evidence and never reuses a name.
A rejected hypothesis is restored with:

```bash
python -m kda_kernels._common.evidence <op_dir> before-run-<n> --restore
```

Restoration checks protected contract/harness hashes and snapshot integrity, restores code
and reports, and checks resulting source hashes. It preserves IMPL_NOTES and STATUS counters.
If protected files changed, repair/reverify; do not resurrect a stale result. The prior
report keeps its iteration while STATUS records the rejected run as consumed.

## Existing generated packages

Runtime version 0.5.0 adds expected coverage, source hashes and finalized verification.
Legacy callers remain callable but old reports without this evidence cannot pass the new
M3 checkpoint. `scaffold.py --sync-common` updates only the shared runtime: merge the new
runner argument/coverage/finalization changes into a customized `_run_dev.py` and rerun
verification. Preserve its operation-specific input generator. Never use `--force` to
replace an implemented package as an upgrade shortcut.
