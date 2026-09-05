# Bottleneck reports and experiment receipts

## Contents

1. Reporting principles
2. Current bottleneck report template
3. Bottleneck inventory
4. Experiment plan
5. Experiment receipt
6. Acceptance language
7. Portable repository integration
8. Project memory

## 1. Reporting principles

A performance report explains why the program is slow. Profiler tables are evidence, not the report. Label every claim **measured** (timing, trace, counter, or source identity), **inferred** (causal reading consistent with measured evidence), or **speculative** (plausible, waiting on a discriminating experiment). Rank by recoverable accepted wall time. State uncertainty and a conservative **ceiling**. Keep raw artifacts and commands next to the summary.

Every diagnosis reports: (1) exact slow boundary and clean wall time; (2) phase decomposition; (3) CUDA-activity temporal union and no-CUDA gap; (4) main/control-thread classification during gaps; (5) launch, copy, sync, allocation, and query counts; (6) copy count and bytes by direction and size; (7) top low-utilization source owners; (8) top device kernels with `digest.txt` columns — name, calls, total s, mean us, max ms — plus duration-distribution bucket share from `## Kernel duration distribution`; (9) measured / inferred / speculative labels; (10) instrumentation overhead and known blind spots; (11) cheapest next discriminating experiment; (12) correctness and acceptance boundary. Flag an unowned interval and name the next capture.

Explain the CPU/GPU relationship in plain language for each high-value target:

```text
The CPU needs a device-produced count before it can choose the next launch.
The scalar copy is small, but it forces completion of the producer stream and
prevents launch-ahead. The predicted win is fewer dependency round trips, not
higher copy bandwidth.
```

Give every label a mechanism and the arithmetic. "GPU underutilized," "memory-bound," "Python overhead," and "too many copies" name a category. "180 µs of GPU idle per iteration × 40,000 iterations, each caused by the host waiting for the residual before it can launch the next step" names a mechanism the reader can act on. When the report goes to the person who wrote the code, [cpu-mental-model-antipatterns.md](cpu-mental-model-antipatterns.md) section 7 is how to pitch that mechanism.

*Done when every required claim is present, every number is labeled measured / inferred / speculative, and every high-value target has mechanism plus arithmetic.*

## 2. Current bottleneck report template

Quote `digest.txt` before filling kernel, gap, or copy tables. The kernel block copies `## Kernels by device time` and `## Kernel duration distribution` — see [nsys-workflow.md](nsys-workflow.md) section 6. Kernel columns are name, calls, total s, mean us, max ms, and duration-distribution bucket share. Bucket share is the digest bucket that contains this kernel's mean us (`<1us`, `<5us`, `<10us`, `<50us`, `<100us`, `<1ms`, `>=1ms`), that bucket's program-wide share (calls and total s), and the max-ms bucket when it is later.

```markdown
# Performance diagnosis: <routine/program>

## Outcome

<One paragraph: exact slow boundary, dominant causal categories, current
measured ceiling, and next cheapest experiment.>

## Scope and authority

- User-visible boundary:
- Internal boundaries:
- Workload/input identity:
- Cold or warm:
- Correctness authority:
- Source identity and patch:
- Physical GPU/UUID:
- Container/toolchain:
- CPU/thread settings:
- Contention state:

## Assumptions and open questions

| Item | Source | Effect if wrong |
|---|---|---|
| | asked / project memory / assumed | |

<Anything the human has not confirmed goes here, and is repeated beside every
number it affects.>

## Clean timing

| Boundary | Samples | Median | Spread | Synchronization semantics |
|---|---|---:|---:|---|
| End-to-end | | | | |
| Stage | | | | |
| Target phase | | | | |

## CPU/GPU critical path

| Phase | Wall | CUDA-active union | No-CUDA gap | Main CPU state | Confidence |
|---|---:|---:|---:|---|---|
| | | | | | |

## Low-utilization owners

| Rank | Owner | Calls | Exclusive wall | CUDA active | Gap | Leading evidence |
|---:|---|---:|---:|---:|---:|---|
| | | | | | | |

## Device-time kernels

Quote `## Kernels by device time` and `## Kernel duration distribution`.

| name | calls | total s | mean us | max ms | bucket share |
|---|---:|---:|---:|---:|---|
| | | | | | |

| bucket | calls | total s | share |
|---|---:|---:|---:|
| | | | |

## Transfers and synchronization

| Direction/API | Calls | Bytes | Engine time | Host API time | Size distribution | Owners |
|---|---:|---:|---:|---:|---|---|
| | | | | | | |

## Bottleneck ranking

<Inventory table in section 3.>

## Blind-spot audit

- Clean timing separate from ownership trace:
- API time compared with engine time:
- Copy count and bytes separated:
- Visible waits correlated with producers:
- Runtime/driver queries attributed:
- CPU active gaps sampled:
- Many-short vs few-long kernels separated (duration-distribution buckets quoted):
- Instrumentation overhead bounded:
- Unattributed critical path:

## Next experiment

Hypothesis / predicted counter or timeline change / minimal treatment / correctness risk / acceptance boundary / reversion condition:

## Artifacts

Commands / clean samples / Systems report / ownership report / CPU samples / NCU reports / correctness comparisons:
```

*Done when the Outcome paragraph names a ceiling, both kernel digest headings are quoted, the kernel table uses those columns, and the inventory has at least one row with a falsifier.*

## 3. Bottleneck inventory

One row per causal candidate. Required fields: **location** (source region, phase, or NVTX owner), **category**, **evidence** (critical-path fact from `digest.txt` or clean timing), **frequency** (calls, duration-distribution bucket share, or iteration count), **ceiling** (conservative accepted-wall bound against the first-pass clean number), **falsifier** (cheapest experiment that would drop the row), **risk** (correctness, ordering, determinism, audit, or peak memory), **confidence** (measured / inferred / speculative). Also record when known: activity-union vs exclusive semantics, critical-path position, and causal explanation (evidence is not the explanation).

| Rank | location | category | evidence | frequency | ceiling | falsifier | risk | confidence |
|---:|---|---|---|---|---:|---|---|---|
| 1 | | host/device synchronization | | | | | | measured |

Required category vocabulary: measurement/environment; host/device synchronization; launch/transaction fragmentation; transfer/materialization; allocation/driver/lifetime; CPU/framework/I/O; work decomposition/redundancy; avoidable custom kernel; compilation/initialization/cache; stream/concurrency; GPU memory bandwidth; GPU memory latency; GPU compute/instruction; occupancy/resource pressure; divergence/imbalance/tail; atomics/serialization; shared memory/barriers; unknown/mixed.

A speculative row gets a conservative **ceiling** from the accepted critical path, not a precise speedup. *Done when every proposed target has location, category, evidence, frequency, ceiling, falsifier, risk, and confidence.*

## 4. Experiment plan

Write this before coding, and only after [SKILL.md](../SKILL.md) section 7 can be answered. The change cycle is [agentic-tuning-workflow.md](agentic-tuning-workflow.md).

```text
Experiment ID:
Question:
Measured evidence:
Hypothesis:
Predicted wall-time effect:
Predicted dynamic counters/timeline:
Representative workload:
Isolation strategy:
Minimal implementation:
Protected semantics:
Expected internal differences:
Approval required: yes/no and why
Correctness gates:
Memory/lifetime risks:
Baseline procedure:
Candidate procedure:
Post-change profile:
Acceptance threshold:
Reversion condition:
Artifact directory:
```

A strong hypothesis predicts a structural observation:

```text
Packing four scalar publications should remove three D2H transfers and three
stream synchronizations per transaction while leaving producer kernel time
unchanged.
```

A weak hypothesis names a desired outcome without a test:

```text
Improve GPU utilization.
```

*Done when the plan names a predicted timeline or counter change that could falsify the row, plus an acceptance threshold and a reversion condition.*

## 5. Experiment receipt

Preserve accepted and rejected experiments:

```text
Experiment ID:
Date/time/timezone:
Decision: accept / revise / revert
Question:
Hypothesis:
Expected timeline or metric change:
Correctness risk:
Source revision:
Uncommitted patch identity:
Repository instructions used:
Container image/digest:
Complete container command:
GPU model/UUID/logical mapping:
Driver/CUDA/framework/compiler/profiler versions:
Native architecture targets:
CPU affinity and thread settings:
Relevant environment variables:
Contention/preflight:
Input fixtures/hashes:
Shapes/distributions/seeds:
Algorithm settings:
Warmup/cold-start procedure:
Output behavior:
Baseline command:
Candidate command:
Profiler commands:
Raw baseline samples:
Raw candidate samples:
Median/spread:
Stage and end-to-end deltas:
Memory/tail/cold-start deltas:
Correctness commands/results:
Reference location:
Candidate location:
Difference classification:
Predicted dynamic counters:
Observed dynamic counters:
Clean wall-time result:
Causal interpretation:
Instrumentation overhead:
Confounders:
Trace/report locations:
Logs/analysis locations:
Reason for decision:
Remaining bottlenecks:
Next cheapest discriminating experiment:
```

Performance reproducibility and output equivalence may retain different metadata. A product comparison may omit physical UUID; a performance receipt records it.

*Done when a later run can reconstruct the decision from this receipt alone — commands, GPU identity, raw samples, and the predicted-versus-observed counters.*

## 6. Acceptance language

### Accept
```text
Accepted: all required correctness gates pass; clean same-GPU median improved
from X to Y (Z% / S×); the predicted launch/copy/synchronization/kernel change
appears in the post-change trace; memory and tail tradeoffs are acceptable.
```

### Revise
```text
Inconclusive: the treatment changed the predicted local counter, but clean
end-to-end improvement is within noise / consumed by another measured cost.
The hypothesis remains plausible; the next discriminating experiment is ...
```

### Revert
```text
Rejected and reverted: correctness changed / clean wall time regressed / the
microbenchmark win did not survive integration / the predicted mechanism did
not appear. Preserve this receipt to prevent repetition.
```

Acceptance is that Accept sentence, supported by the receipt. These are not acceptance by themselves: higher utilization; higher occupancy; fewer launches; lower isolated kernel time; a single sample; a different GPU; a heavily instrumented trace; visually similar output; a narrowed favorable workload.

*Done when the spoken sentence is one of the three above, and the receipt records the supporting samples and the predicted-versus-observed change.*

## 7. Repository integration

At the start of each use:

1. **Recall** this repository's agent instructions and performance contracts. *Done when the contract is quoted or marked absent.*
2. Discover build, remote, container, synchronization, test, and artifact conventions. *Done when runners, comparators, and artifact paths are named.*
3. Identify protected semantics and approval boundaries. *Done when the approval list is written into the experiment plan.*
4. Reuse existing runners, comparators, trace analyzers, and output layouts. *Done when the receipt cites those commands rather than inventing new ones.*
5. Keep machine-specific commands and facts in the experiment receipt or project docs, not in this skill. *Done when receipts name the paths this run used.*

## 8. Project memory

Asking is the price of the *first* investigation, not of every one. Once the environment is verified and the contract agreed, write both into the repository's always-loaded agent instructions (`AGENTS.md`). **Recall** this block before asking anything; **Ask** only about what has changed.

### 8.1 What earns a place

The file is loaded on every turn, so the block earns its cost only by caching lookups that are expensive, hidden, or impossible to repeat. A line that a one-shot command answers belongs in the command, not in the memory.

| Keep | Leave out |
|---|---|
| The effective run and profile commands, including container flags — routinely buried in wrapper scripts | Driver, toolkit, and framework versions — volatile, and `nsys --version` answers them |
| Which GPU to measure on, its UUID, whether it is shared, and the agreed protocol for getting a quiet run | The device list — `nvidia-smi -L` answers it |
| Permission state: whether CPU sampling and hardware counters are available here, and who grants them | Timing numbers, profiler output, and bottleneck rankings — these belong in receipts and go stale immediately |
| The representative workload and how to invoke it | Anything already written in the repository's build or test docs |
| The accepted wall-time boundary and how it is measured | Credentials, tokens, or private host details |
| The correctness oracle command and what counts as equivalent | The reasoning behind a past decision — link the receipt instead |
| Protected semantics and what needs approval before it changes | Advice already in this skill |
| Environment gotchas that cost real time to rediscover — "profile the worker directly, the launcher forks and the child is not traced" | Anything that was guessed rather than verified or confirmed |
| Questions already settled, as one line plus a pointer to the receipt — "the custom top-k beats `cub::DeviceRadixSort` at our shapes, receipt 2026-05-02" | |

### 8.2 Template

Use one delimited block so a later run can replace it in place instead of appending a second copy:

```markdown
<cuda-perf-environment>
Verified: <date> by <agent/human>. Re-verify when any check below fails.

- Runs: <bare metal | container | scheduler>. Effective command: <exact command,
  including --cap-add=SYS_ADMIN and -e NVIDIA_DRIVER_CAPABILITIES=all if containerized>
- Measure on: <GPU model, UUID>. Shared: <yes/no>. Quiet-run protocol: <agreed steps>
- CPU sampling: <available | denied, ask <who>>. Hardware counters: <available |
  denied, ask <who>>
- Preflight: `nsys status --environment` plus <the project's CUDA smoke command>
- Representative workload: <command and fixture path>. Cold or warm: <which is the
  product metric>
- Accepted boundary: <what is timed, and how> — currently <median> on <GPU>
- Correctness oracle: <command>. Equivalent means: <exact | tolerance | rule>
- Needs approval before changing: <reduction order, determinism, audit output,
  peak memory, cold start, ...>
- Artifacts: <path convention>
- Gotchas: <the things that cost an hour to find>
- Settled: <question — answer — receipt pointer>
</cuda-perf-environment>
```

Trim every line that does not apply. Unfilled placeholders are worse than absent lines: a later run will trust them.

### 8.3 Rules

- **Ask once before writing it.** Show the block and get agreement. If the repository has no such file, ask before creating one.
- **One block, replaced in place.** Find the existing `<cuda-perf-environment>` block and rewrite it. Two blocks that disagree are worse than none.
- **Date it, and name the check that proves it.** A memory without a verification command cannot be distinguished from a guess later.
- **Treat it as stale** when the GPU model or UUID differs, the container image changed, the preflight command fails, the fixture or oracle command is missing, or the repository layout moved. Re-verify the failing line, update it, and move the date.
- **Record only what was verified or confirmed by the human.** An assumption promoted into always-loaded memory becomes indistinguishable from a fact.
- **Keep results out.** Memory holds how to measure and what counts; receipts hold what was measured.

*Done when one `<cuda-perf-environment>` block is agreed, dated, and written, or the human declined and that refusal is recorded in the receipt.*
