---
name: kda-kernel-integrate
description: M6 of the KDA kernel workflow, run inline by the orchestrator after M5. Swaps the user's definition site to the fused kernel behind the `KDA_BACKEND` flag, runs the observed training or inference smoke command before and after with the same inputs, and records timing and behavior deltas in E2E.md. Use when STATUS.md shows M5 done and integration is authorized in the current or an earlier request, or when an integrated kernel must be re-validated after the user's code changed.
---
# Integrate the verified change set

Run inline in the orchestrator. Use the recorded authorization; review-only and explicit
stage approval requests prevail. Prepare a concrete diff and measured result before asking
for any still-needed final integration permission.

1. Read [INTEGRATION](../kda-kernels/reference/integration-contract.md) and the original caller.
   The contract names entrypoint, replacement, consumers, collective/cache/dtype boundaries,
   real commands, input/configuration hashes, ranks, branches, lifecycle, quality, matrix and
   authorization. Single-op tasks use the same short record.
2. After first operation correctness, before a second tuning round, try the candidate in an
   isolated importable source snapshot. Verify calls, loaded source hashes and actual backends
   in every worker; check outputs and cleanup on success and exception. Parent monkeypatches
   do not establish spawned-worker coverage. Record early benefit direction and limitations.
3. Preserve the original API, mathematical domain, gradient semantics and state ownership.
   The public adapter handles valid unsupported inputs under auto. Explicit backend selection
   is strict; execution exceptions propagate. Use auto or per-op config for mixed backends.
   KDA_BACKEND=eager remains global rollback. Keep parameters and checkpoint names unchanged.
   Optional epilogues retain the old default and list each opt-in caller in the diff.
4. Independently review the complete integration diff, original preprocessing, decoded output
   comparison and lifecycle. Use verifier scope integration. Model quality uses the established
   request contract; op tolerance or a scalar loss print is not model quality evidence.
5. On frozen/trimmed source, warm up both variants then run five alternating A/B pairs via
   the synchronous request driver. Long-lived workers alternate invocation blocks. Check all
   saved outputs, including warmups; retain every failed request and raw sample. Add rollback
   coverage. Use [measurement semantics](../kda-kernel-scaffold/reference/measurement.md).
6. Write E2E.md from the saved results and produce [task acceptance](../kda-kernels/reference/acceptance.md).
   Separate deployment delta, eager ratio and best-available-implementation ratio. A relocated
   region requires an extra separate/relocated group only for attribution or diagnosis; without
   it claim the net change only. Nested phase medians and activity sums are not additive elapsed
   attribution. No fastest-repeat selection, sample-count significance claim or profiler/event
   subtraction. Missing request evidence cannot be replaced by an isolated op report.
7. Apply the authorized concrete change set when acceptance is ready. Link runtime source/commit,
   contract digest, finalized op reports, request evidence, acceptance and unresolved limitations
   in the final manifest. Retain runtime code, needed regression tests and concise docs; put
   traces, experiment drivers and tuning history in artifacts. Reassess after trimming changes
   dependencies. Mark M6 complete only for the final applied source.

Preserve existing repair/tune limits and task-wide experiment counters. Each extra experiment
must name its decision consequence. Diagnostic SOL misses never force tuning. Exhausted
performance experiments end with disclosed limitations; unresolved correctness and integration
remain blockers. Authorization cannot change a measured loss into a gain.

Return the complete change set, scoped conclusions, actual request results, fallback command,
evidence paths and unverified hardware. Do not alter historical benchmark numbers.
