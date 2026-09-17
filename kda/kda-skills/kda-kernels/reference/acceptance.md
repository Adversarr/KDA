# Scoped acceptance and evidence dependencies

New operation reports use schema_version 2 and scope operation. `verification.verdict`
is the audited stage summary, never model acceptance. `outcomes` separates correctness,
integration (not_assessed at operation scope), performance measurement validity and delivery.
The task's `acceptance.json` is produced with:

```sh
python -m kda_kernels._common.acceptance INTEGRATION.md artifacts/request-evidence.json --out acceptance.json
```

The tool rereads op reports referenced by the contract, checks dependency fingerprints and
finalized report digests, reassesses invocation/rank/output evidence and recomputes A/B statistics.
Historical reports cannot obtain new acceptance. No supplied `pass` flag replaces missing samples.
Candidate files must cover the finalized operation's SPEC, executable source, runtime VERSION
and TUNED configuration identities. Use `variants.candidate.operations` to map an operation
SPEC to its candidate SPEC when the layout differs. This prevents validating one implementation
and delivering another. The current assessor supports `primary_metric: request_wall_ms`;
other task metrics need an explicit assessor and remain unverified here.

The command also writes `manifest.json` beside acceptance (override with `--manifest`). It
binds the acceptance digest, execution contract, current operation reports and candidate files,
with source revision when available, delivery limitations and unfinished items. Keep raw traces,
experiments and tuning history in artifacts; retain runtime, required regressions and concise
integration documentation in the delivered change.

`performance_policy: diagnostic` is the new default: SOL and best-isolated-baseline ratios
inform a decision but do not request extra tune rounds or block freezing. `strict_kernel`
explicitly restores existing gates, cold profiler measurements and bounded near-gate repeats.
Both policies retain runtime/numerical errors and incomplete evidence. Neither promotes
an operation pass to request correctness or request benefit.

The contract may declare `target.minimum_reduction` (fractional request wall-time reduction).
Only task-declared thresholds apply; absent a threshold, report whether positive request benefit
was observed. An explicitly accepted performance limitation is recorded as
`authorization.accept_performance_limitations: true`; negative samples remain negative. A positive median does not establish statistical significance. Shared load,
spread, failed requests and negative benefit remain visible. Missing correctness or integration
evidence cannot be waived by authorization. An unmet target remains incomplete; any revised
task goal must be explicit in a revised contract and re-assessed without rewriting samples.

## Dependency groups

`dependencies.py` records contract, arithmetic, adapter/state, inputs, measurement, analysis
and caller file groups. SPEC/INTEGRATION front matter is parsed before hashing; ordinary prose
changes are excluded. Unknown executable files conservatively belong to every group.
TUNED.json belongs to arithmetic dependencies; shared runtime VERSION belongs to adapter/state.
Each evidence item lists its dependencies, environment, command/rank and sample keys.

- Arithmetic changes invalidate dependent numerical and performance results.
- Registration, adapter and plan-state changes invalidate behavior and results through that path.
- Caller, input and decode changes invalidate integration/quality results.
- Analysis-only changes retain raw samples but require reanalysis and a new audit binding.
- Changed collection boundaries require measurement again. Unknown dependencies require rerun.

`invalidation(previous, current, depends_on)` returns reuse, reanalyze or rerun.
For an analysis-only change, run `python -m kda_kernels._common.evidence <op_dir>
--reanalyze <op_dir>/report.json` and issue a fresh audit. This removes old finalization,
retains raw samples and analysis history, and rejects changed execution dependencies. Reanalysis
must preserve raw samples and their original measurement/source identities; it must not replace
old execution fingerprints with current ones for changed execution groups.

Keep task-wide repair/tune/integration counters in STATUS in addition to existing per-op
budgets. `record_experiment` consumes declared limits and a finite matrix cell; each experiment
names the decision it can change. Exhaustion stops performance experiments, not disclosure of
correctness gaps. Diagnostic SOL alone is never a reason for another experiment.
