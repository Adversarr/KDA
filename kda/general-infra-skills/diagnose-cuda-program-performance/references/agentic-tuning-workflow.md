# Agentic CUDA performance-tuning workflow

## Contents

After the gate, then:

1. Inventory schema
2. Isolate without destroying representativeness
3. Cheap falsification
4. Implement one attributable change
5. Validate behavior
6. Controlled measurement
7. Causal re-profile
8. Accept, revise, or revert
9. Durable handoff

## After the gate

[SKILL.md](../SKILL.md) section 9 sends you here. This page is the **change cycle** — isolate, falsify, implement one change, validate, measure, re-profile, decide, hand off. It starts only after [SKILL.md](../SKILL.md) section 7 can be answered. An unknown answer there is a measurement, not an implementation.

Treatments live in [issue-catalog.md](issue-catalog.md). Ask moments stay in [SKILL.md](../SKILL.md) section 4.

NVIDIA's APOD cycle — Assess, Parallelize, Optimize, Deploy — uses the same iterative principle: find real hotspots, make bounded changes, test correctness and performance, and repeat. See [APOD](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#assess-parallelize-optimize-deploy).

Satisfy every *Done when* before moving on.

*Done when:* every question in [SKILL.md](../SKILL.md) section 7 has an answer, and the task is to change something.

## 1. Inventory schema

The table lives in [report-and-receipts.md](report-and-receipts.md) section 3. Every open row this cycle will attack carries:

- **location** — source region, phase, or transaction
- **category** — from the vocabulary on that table
- **evidence** — digest heading, timeline feature, or counter that supports it
- **frequency** — calls, distribution, many-short versus few-long
- **ceiling** — conservative upper bound on accepted wall-time reduction
- **falsifier** — cheapest experiment that can disprove the mechanism
- **risk** — correctness, ordering, memory, determinism
- **confidence** — measured, inferred, or speculative
- when known: activity-union vs exclusive semantics, critical-path position, causal explanation (evidence is not the explanation)

Rank by expected accepted-wall reduction, then confidence, risk, and effort. Default attack order: host/device synchronization; launch and transaction fragmentation; redundant work and materialization; allocation/driver churn and transfers; work decomposition; a proven dominant kernel; concurrency only for proven independent critical-path work. Override when one kernel or CPU phase clearly owns the accepted metric.

Quote the *ceiling* from the accepted critical path. A speculative row gets that bound, not a precise speedup.

*Done when:* the chosen target has a filled field list, a *ceiling* taken from the accepted critical path, and a named *falsifier*.

## 2. Isolate without destroying representativeness

Construct the smallest workload that still produces the suspected mechanism:

- shapes, dtypes, device, strides, sparsity, ordering, and distributions
- call frequency and transaction sequence
- synchronization and host decisions
- upstream-produced values that affect branches
- downstream consumption that affects laziness or lifetime
- warm or cold state
- correctness and determinism checks

Keep the call frequency and transaction sequence that produced the signature. A one-call kernel benchmark cannot validate a million-call launch hypothesis; a kernel-only benchmark cannot validate host synchronization.

*Done when:* the isolated workload reproduces both the suspected dynamic counters and representative output.

## 3. Cheap falsification

Prefer the smallest reversible experiment that can *falsify* the hypothesis:

- defer or batch one scalar host read and observe the gap
- pack several publications into one device-produced certificate
- precompute a repeated invariant to estimate its *ceiling*
- batch a bounded number of independent transactions
- preallocate one workspace to test allocator churn
- replace a whole-state clear with a sparse-delta prototype
- disable optional logging or audit materialization while retaining core work
- fix a shape to separate compilation from execution
- prototype fusion in Triton
- compare a dependency-correct one-stream versus two-stream probe
- profile an isolated long kernel with a reduced NCU set

The experiment estimates a mechanism. It is not a production patch.

Several of these probes break the contract on purpose to measure a *ceiling*: deferring a scalar read changes when the loop terminates, disabling audit materialization removes an output, fixing a shape narrows the workload. That is legitimate for a throwaway measurement. Label the result as a *ceiling*, then implement a production surface only after section 4.

**Ask** before a probe that breaks a contract the human protected in [SKILL.md](../SKILL.md) section 4, and before one whose side effects escape the worktree — overwriting a reference, occupying a shared GPU for a long stretch, or leaving the repository in a state someone else builds from. A local worktree is not permission: if section 4 named audit output or determinism as protected, disabling it is the human's call even when nothing outside the checkout changes.

### Triton as a probe

Use Triton when a compact prototype can test fusion, launch reduction, locality, or scheduling for tractable shapes and dtypes. Require a strong oracle.

Prefer existing optimized libraries such as CUB, Thrust, cuBLAS, cuSPARSE, or framework primitives when the operation matches them. NVIDIA recommends parallel libraries as the straightforward first choice when they fit, and specifically identifies Thrust's scan, sort, and reduce primitives. See [Parallel Libraries](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#parallel-libraries).

Discard the Triton experiment when it cannot improve representative end-to-end time. If it proves the idea, choose the maintainable production surface — Triton, CUDA, or library composition — from measured overhead, validation, integration, and architecture coverage.

### CPU binding

Move or retain work on CPU only when profiling proves active CPU compute is the constraint and the transfer and synchronization cost is acceptable. Use a compiled CPU binding when the refactor is mostly mechanical and safe. A CUDA-wait boundary moved onto the host is still a wait, not CPU compute.

*Done when:* the experiment changes the predicted counter or timeline feature, or the hypothesis is rejected.

## 4. Implement one attributable change

Keep one conceptual change per experiment. Separate:

- instrumentation
- cleanup or refactoring
- transaction restructuring
- algorithmic change
- kernel implementation
- concurrency

For streams, document producer and consumer dependencies, buffers, allocator ownership, events, default-stream semantics, and the final join. For batching, document ordering, capacity, fallback, and deterministic publication. For certificate reuse, prove identity and producer authority. For allocation caching, define refresh barriers and external-memory assumptions.

After each source edit, follow the repository's synchronization, build, and test rules.

*Done when:* the patch is small enough to attribute, fully reversible, and its predicted dynamic effect is written down.

## 5. Validate behavior

Create immutable known-good outputs in new candidate directories; never overwrite references.

Run layered gates before trusting any timing:

1. operator or kernel unit tests
2. intermediate tensor or authority comparisons
3. ordering, counts, choices, and decisions
4. manifests, certificates, and audit receipts
5. API or CLI output
6. domain invariants
7. repeated determinism
8. empty, degenerate, adversarial, and representative large inputs

Classify every difference:

- real semantic
- order-only
- approved exception
- nondeterministic or flaky
- measurement or serialization artifact

NVIDIA recommends known-good reference comparison after each change and couples correctness with performance validation in the APOD loop. See [Verification](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#verification).

*Done when:* all required gates pass, or every difference is explicitly approved. Otherwise revert or seek approval before further performance work.

## 6. Controlled measurement

Run baseline and treatment:

- same physical GPU and UUID
- uncontended
- identical input and settings
- exact source states
- equivalent warmup and allocator or cache state
- explicit completion at the accepted boundaries
- alternating order such as ABBA when practical
- enough samples to separate improvement from noise

Report raw samples, median, spread, absolute reduction, relative speedup, affected phase, end-to-end metric, memory footprint, and tail or cold-start tradeoffs.

Accept only on the agreed wall-time boundary. A microbenchmark win that does not move that number is not an accepted program optimization.

*Done when:* baseline and treatment are comparable on the checklist above, raw samples are retained, and the accepted-wall delta is either outside noise or recorded as within it.

## 7. Causal re-profile

Capture a post-change broad timeline with [nsys-workflow.md](nsys-workflow.md). Confirm:

- the targeted gap, API wait, launch, copy, allocation, or kernel duration changed
- wall-time improvement agrees with the predicted mechanism
- work was removed or accelerated, not shifted outside the measured boundary
- no new synchronization, memory pressure, compilation, tail, or CPU bottleneck erased the gain
- the next bottleneck is identified

For a kernel treatment, profile the exact accepted candidate on the representative workload, and only for a named kernel plus a digest launch ordinal. If `ncu-report` is installed, start from its `analyze_reports.py` digest rather than raw metrics. That skill writes under `profile/<run_name>/` and is written against B200 / sm_100 (verified on sm_80). Other languages, layouts, or GPUs use [profiling-and-attribution.md](profiling-and-attribution.md) section 10. If `cuda-kernel-wiki` is installed, map the measured mechanism and architecture to implementation patterns; otherwise use the architecture's tuning guide.

*Done when:* clean wall time and dynamic evidence tell the same causal story.

## 8. Accept, revise, or revert

Revise and revert are the agent's calls; reverting a failed experiment needs no permission. Accept is a shared call: the evidence is yours to establish, but whether a trade-off is worth taking belongs to whoever owns the code. **Present the decision rather than announcing it** whenever acceptance spends something — peak memory, cold-start time, tail latency, determinism, a changed reduction order, or added complexity — or whenever the improvement lands near the threshold agreed in [SKILL.md](../SKILL.md) section 4.

### Accept when

- correctness contract passes
- clean accepted wall time improves materially
- baseline and treatment are comparable
- predicted causal counters or timeline change
- memory and tail tradeoffs are acceptable
- evidence is preserved

### Revise when

- the hypothesis remains plausible but measurement or implementation is inconclusive
- the experiment changed the right counter but another local cost consumed the win
- representativeness needs repair

### Revert when

- correctness changes without approval
- end-to-end performance regresses or remains within noise
- the microbenchmark win disappears in the routine
- complexity exceeds demonstrated value
- the expected mechanism does not appear
- concurrency increases contention or lifetime
- the candidate narrows the workload to favorable cases

Revert completely. Preserve a short failed-experiment receipt so later runs do not repeat it.

*Done when:* the worktree contains only accepted changes, the decision is supported by receipts, and every present-the-decision case has been shown to the owner.

## 9. Durable handoff

Use [report-and-receipts.md](report-and-receipts.md). Preserve:

- commands and environment
- source identities and patches
- raw timing samples
- traces, reports, and analysis
- correctness comparisons
- predicted versus observed causal changes
- accept / revise / revert decision
- remaining ranked bottlenecks
- next cheapest discriminating experiment

Then separate the two kinds of durable output. The receipt records *this* run and stays with the artifacts. The small subset that will be true next time — the verified environment, the boundary, the workload, the oracle, the approval list, and any question now settled — is offered for the repository's project memory block, so the next cycle can Recall it instead of asking again. See [report-and-receipts.md](report-and-receipts.md) section 8.

Bottlenecks migrate after every accepted change. Begin the next cycle from a fresh clean profile of the accepted source.

*Done when:* the receipt is filed with the artifacts, remaining bottlenecks are ranked with a next *falsifier*, and any fact that will still be true next time is offered for section 8.
