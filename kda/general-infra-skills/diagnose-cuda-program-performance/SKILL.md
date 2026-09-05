---
name: diagnose-cuda-program-performance
description: Diagnose why a CUDA program is slow across the whole CPU–GPU critical path and rank bottlenecks before changing code. Use for whole-program bottleneck reports; GPU idle or low utilization; Nsight Systems traces; host synchronization such as `.item()` or `cudaDeviceSynchronize`; launch storms; host/device copies; allocation churn; CPU-side phases; profiler and container setup; CUDA written with a CPU mental model; hand-written kernels that duplicate CUB, Thrust, or cuBLAS; and regressions where a fast kernel did not make the program fast. Not for tuning internals of a kernel already proven to dominate.
---

# Diagnose CUDA Program Performance

Explain what makes a CUDA program slow across the whole CPU–GPU critical path, and rank fixes by the wall time each can recover. A fast kernel does not make a program fast.

## 1. Scope

**In scope:** where wall time goes; why the GPU idles; synchronization, launches, transfers, allocation, CPU phases, initialization, and concurrency; whether a given kernel is worth tuning; the report that explains it.

**Out of scope:** tuning internals of a kernel this skill has already proven to dominate. After the [gate](#7-gate-before-implementation), hand off in [§11](#11-companion-skills).

**Dated material (2026-05):** profiler flags, permission settings, container arguments, and library API names. Linked NVIDIA guides are versionless. Verify a flag against `--help` before treating a mismatch as a skill bug.

Assumes an NVIDIA GPU and CUDA only. PyTorch names are examples with C++/CUDA equivalents. Companion skills are optional with fallbacks. Read this repository's agent instructions first — [report-and-receipts.md](references/report-and-receipts.md) section 7.

## 2. Bundled tooling

Default analysis path — run this before any query of your own. First-pass form has **no** `--kernel`. If the checkout-relative path is wrong, resolve `scripts/nsys_analyze.py` from this skill directory.

```bash
python3 .agents/skills/diagnose-cuda-program-performance/scripts/nsys_analyze.py REPORT --out ARTIFACTS/nsys-analysis
```

Quote `digest.txt` in the report before any custom SQL or a reimplementation of this analysis. Named SQL exceptions live in [nsys-workflow.md](references/nsys-workflow.md) section 10 and cross-check the digest.

The interval algebra is full of traps:

1. Summing concurrent kernel durations instead of merging them (reports more GPU-busy time than the window contains).
2. Adding CUDA API duration to the GPU work it launched (they overlap).
3. Adding inclusive NVTX durations across nested ranges (counts the same wall time once per nesting level).
4. Attributing a kernel to an NVTX range open on some other thread, rather than the thread that launched it.
5. Ranking gaps by length and concluding from the top of that list, when most idle sits in gaps too short to appear on it.

## 3. Recall, detect, ask, flag

Sort every input in this order:

- **Recall** — a `<cuda-perf-environment>` block in the repository's agent instructions may already hold the verified setup and the agreed contract. Treat it as stale: re-verify with the preflight it names, and Ask only what it does not cover.
- **Detect** — a command or a file answers it. Run the command; Detect commands and permission reasons are in [profiler-environment-setup.md](references/profiler-environment-setup.md).
- **Ask** — only the human knows it, or acting needs their permission. One batch per moment in [§4](#4-ask-moments). Wait.
- **Flag** — no answer arrived and the work must proceed. State the assumption beside every number it affects.

## 4. Ask moments

Ask in **one batch per moment**, with a recommended default for each question. State the answers back before a long run. Questions 1, 10, and 11 of the [§7](#7-gate-before-implementation) gate (boundary, representativeness, oracle) live here.

**Before measuring**

| Ask | Default |
|---|---|
| Which wall-time boundary counts as slow, and is it user-visible or internal? | The user-visible request or CLI wall time (widest path that still matches the complaint) |
| Which input is representative, and where do the fixtures live (shapes, distributions, edge cases)? | The project's existing benchmark or CI fixture |
| Cold start or steady state as the product metric? | Steady state, unless the complaint is first request |
| What must remain identical versus what may differ (oracle / equivalence: exact, tolerant, or product-semantic; outputs, determinism, ordering, internal tensors, decisions, certificates, manifests, audit receipts, memory policy, errors; explicitly allowed exceptions)? | Exact match; nothing protected may differ |
| Hardware or platform constraints (GPU model, MIG, memory cap)? | The GPU Detect named; no extra constraints |
| How much improvement is worth the change? | Any reduction that beats clean-timing noise on the accepted boundary |
| Report only, or implement? | Report only until the gate |
| Is the GPU shared, and may it be quieted? | Shared; do not quiet |
| Where does the workload actually run, and can its launch command be changed? | This shell unless Detect says otherwise; do not change launch flags |
| How long does one run take, and how many are acceptable? | Time one run first, then 3–5 clean samples |
| May host settings (`perf_event_paranoid`, driver counters) or profiler installs be changed? | Do not change; ask an administrator |

**Before changing:** anything that alters results, ordering, determinism, error timing, audit output, or peak memory (batching a reduction, deferring a convergence check, caching a capacity query, reusing a buffer). Default: wait for approval.

**Before touching the host:** `perf_event_paranoid`, driver counter permission, installing tooling, or anything else affecting other users of a shared machine. Default: report the current value and the evidence it costs, then wait.

If an answer does not arrive and the work proceeds, Flag the most conservative reading — the widest boundary and the strictest equivalence — beside every number it affects.

*Done when answers are stated back, and every missing one is Flagged beside the numbers it affects.*

## 5. Critical path and GPU debt

A CUDA application is a CPU program that submits asynchronous work to a GPU. User-visible latency is the full dependency chain: host control, dispatch, runtime/driver, stream ordering, kernels/copies/memsets, allocation, device-to-host decisions, and postprocessing.

```text
CPU: prepare ─ launch ─ prepare ─ read scalar ─ branch ─ launch ─ publish
GPU:           kernel ──────────┘ wait/idle ─── kernel ────────────┘
```

The CPU returns from a launch before the GPU finishes. Independent enqueue overlaps; a GPU-produced scalar drains the queue and idles the device while the host branches. Ask:

1. Who produces the next required value?
2. Where is it consumed?
3. Which dependency forces completion?
4. What can remain queued or device-resident?
5. Which work lies on the user-visible critical path?

NVIDIA describes CUDA as heterogeneous computing across concurrently operating host CPUs and GPU devices, with separate memories and different latency/throughput designs — one cohesive system, not isolated processors. See [Heterogeneous Computing](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#heterogeneous-computing).

```python
launch_large_cuda_work()
value = device_tensor.item()
```

The arithmetic of `.item()` is tiny; its dependency is not. A profiler can charge a long producer wait to that line. Call the accumulated wait **GPU debt**: the visible line is the collection point, not necessarily the producer to optimize. The same mechanism: `.cpu()`, `.tolist()`, `.numpy()`, `bool(tensor)`, tensor predicates, logging, asserts, exception formatting, explicit `synchronize()`, nominally async copies consumed immediately, allocator reuse that honors outstanding stream use.

Frequency turns crumbs into walls: `10 microseconds × 1,000,000 calls = 10 seconds`. The larger cost may be the GPU bubbles between calls.

**Bytes do not measure control latency.** A four-byte D2H copy can serialize milliseconds of producer work; a multi-gigabyte D2D copy may be bandwidth without host sync. Analyze count, bytes, and dependency position. NVIDIA recommends keeping intermediates on the device and batching small transfers. See [Data Transfer Between Host and Device](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#data-transfer-between-host-and-device).

## 6. First pass

Work in this order. Each step exists to stop a later one from producing a confident wrong answer.

1. **Fix the boundary and the contract.** What interval is slow, on which input, and what must not change. Mostly an Ask ([§4](#4-ask-moments)). *Done when "faster" and "still correct" are both unambiguous.*
2. **Time it cleanly.** Synchronized outer timer, several samples, median and spread, cold and warm separated, no profiler attached. *Done when a reproducible number exists — the denominator for every **ceiling** you will quote.*
3. **Read the source** with [cpu-mental-model-antipatterns.md](references/cpu-mental-model-antipatterns.md). Costs nothing and works when profiling is blocked. *Done when candidate mechanisms are listed as hypotheses, each with the profile signature that would confirm it.*
4. **Get a timeline.** If Systems is blocked, report the exact preflight failure and continue on [profiling-and-attribution.md](references/profiling-and-attribution.md) section 4 plus the step-3 hypotheses — do not invent kernel internals, and do not stop. When a report exists, capture and analyze with [nsys-workflow.md](references/nsys-workflow.md) sections 3–6 (read the digest in that page's order). Classify what the host was doing in the no-CUDA gaps. Recapture-with-annotation is [nvtx-bootstrap.md](references/nvtx-bootstrap.md) section 1, and only after `## No-CUDA time by host API in flight` still leaves material idle unclassified. *Done when either (a) `digest.txt` exists and these headings are quoted: `## CUDA activity versus wall` (union + no-CUDA gap), `## No-CUDA time by host API in flight`, and both `## Kernels by device time` and `## Kernel duration distribution`; or (b) Systems is blocked, the blocker is recorded, and the no-profiler path plus source hypotheses are in the report. Do not require the NVTX idle-owner table on an unannotated trace.*
5. **Rank an inventory.** Use the field list in [report-and-receipts.md](references/report-and-receipts.md) section 3. Every candidate carries a mechanism, its evidence, a conservative **ceiling** against the step-2 number, and the cheapest experiment that could **falsify** it. *Done when nothing is proposed that lacks a ceiling and a falsification.* Stop. Answer [§7](#7-gate-before-implementation) before applying a treatment.
6. **Record what the next run should not have to Ask again.** Offer to write the verified environment and the agreed contract as a `<cuda-perf-environment>` block — [report-and-receipts.md](references/report-and-receipts.md) section 8. *Done when steps 1 and 4 would start from Recall.*

## 7. Gate before implementation

Do not propose or implement an optimization until the investigation can answer:

1. What exact user-visible or internal wall-time boundary is slow?
2. How much of it contains CUDA kernels, copies, or memsets as a temporal union?
3. What is the CPU doing during the remaining time: compute, dispatch, wait, driver call, allocation, I/O, or unknown?
4. Which semantic phase and source transaction own every major interval? (Host-API idle owners and source hypotheses count; NVTX names are not required.)
5. Is the target many short repetitions, few long operations, or both?
6. Are copy count, bytes, direction, engine time, API time, and dependency position separated?
7. Are visible operators doing work, waiting for producers, or both?
8. Are runtime queries, allocations, system calls, compilation, and CPU-only work accounted for?
9. Is the proposed metric on the critical path and bounded by the accepted wall-time ceiling?
10. Is the workload representative and the environment uncontended?
11. Is there a correctness oracle and an exact equivalence contract?
12. Could instrumentation have created or exaggerated the apparent bottleneck?

Unknown answer → smallest measurement, not implementation. During first pass, read catalog **Symptom / Mechanism / Evidence / Misdiagnosis** only. Do not apply a **Treatment**, do not invoke `ncu-report`, do not edit code.

## 8. Blind-spot gates

| Blind spot | Failure | Why it survives | Gate |
|---|---|---|---|
| Kernel-only hotspot hunting | Rank long kernels before accounting for CPU-only phases, CUDA gaps, synchronization, allocation, and dispatch | Kernel tables contain concrete names and durations; empty timeline regions look like absence of evidence | Before selecting a kernel, report phase wall time, CUDA-activity union, no-CUDA gap, main-thread state during representative gaps, and launch/copy/sync counts |
| Coarse or wrapper-only NVTX | A phase range locates time but cannot assign repeated transactions, copies, and waits to source owners | The trace looks annotated while the actionable layer remains opaque | Maintain semantic phase ranges plus repeated transaction/function ownership. Add selected operator detail only to resolve remaining ambiguity. Quantify unattributed critical-path time |
| Blaming the visible operator | Treat long `to`, `_to_copy`, `copy_`, `.item()`, or allocation ranges as expensive local work | Synchronization charges earlier producer time to the consumer boundary; nested framework ranges repeat the same interval | Correlate every apparently expensive boundary with its producer, stream sync, copy-engine interval, and nested parent/child ranges. Label it as work, wait, or mixed |
| Treating API totals as wall or engine time | Interpret summed `cudaMemcpyAsync` duration as data-transfer cost or add API and kernel totals | API names suggest their nominal operation, while overlap and waiting are hidden in aggregates | For each important API family, report call count, host duration, actual engine duration, temporal overlap, direction/size distribution, and source owner |
| Ignoring tiny control traffic | Dismiss scalar copies because total bytes are negligible | Bandwidth intuition replaces dependency reasoning | Bucket D2H copies by `<=8 B`, `<=64 B`, `<=256 B`, and bulk sizes; map small copies and synchronization to transactions and host decisions |
| Ignoring driver and safety helpers | Skip `cudaMemGetInfo`, allocation, device queries, and correlated `ioctl` because they do not launch kernels | Source lines look like cheap bookkeeping and are outside framework operator lists | Count and attribute driver/runtime queries, allocation/free calls, and OS-runtime activity. Flag loop-invariant queries at fine granularity |
| Confusing many-short with few-long kernels | Send a 35-microsecond kernel launched hundreds of thousands of times to kernel analysis before considering batching | Aggregate kernel ranking hides the distribution | Report calls, total s, mean us, max ms, and the kernel duration-distribution buckets from `digest.txt`. Route high-count short work to transaction redesign; route consequential long work to kernel analysis |
| Accepting instrumented wall time | Use all-function, all-ATen, stack, shape, or memory tracing as the speedup authority | Ownership traces contain the richest evidence and therefore feel most authoritative | Separate clean timing, routine ownership, CPU sampling, and focused kernel diagnostics into different captures. Bound overhead before using any diagnostic capture for timing |
| Assuming streams create speedup | Add streams because the trace is serialized or utilization is low | Potential concurrency is confused with independent critical-path work and compatible resource use | Draw producer/consumer dependencies, prove buffer lifetime and stream semantics, identify an actual gap to fill, predict resource contention, then verify real overlap and clean wall time |
| Tuning a kernel that should not exist | Optimize a hand-written reduction, scan, sort, filter, histogram, or GEMM without benchmarking the CUB, Thrust, cuBLAS, CUTLASS, or framework equivalent | The kernel is present, correct, and named after the project's domain rather than the primitive it implements, so it reads as application code. Its metrics look healthy because it is a reasonable implementation — just not a tuned one, and not one that gains the next architecture's tuning | Before tuning any custom kernel, name the standard primitive it implements or state that there is none, and benchmark the library equivalent at the real call site as the baseline. See [cpu-mental-model-antipatterns.md](references/cpu-mental-model-antipatterns.md) section 5 |
| Mixing performance boundaries or environments | Compare algorithm time with wrapper time, CPU cleanup with CUDA work, different physical GPUs, contended runs, stale source, or different thread settings | All values are labeled "runtime" | Record explicit boundaries and identities: request, wrapper, algorithm, phase, transaction; source and patch; input; physical GPU/UUID; container; CPU threads; warmup; contention; output behavior |

## 9. After the gate

The change cycle is [agentic-tuning-workflow.md](references/agentic-tuning-workflow.md). [issue-catalog.md](references/issue-catalog.md) is the encyclopedia: first pass may open a landing for Evidence and Misdiagnoses; apply a Treatment only after this gate.

When a named kernel plus a digest launch ordinal already prove that kernel owns material wall time, read issue-catalog section 10 (GPU kernel efficiency categories) and hand off per [§11](#11-companion-skills).

## 10. Symptom router

Read only what the current branch needs, then return to the **first pass**. This table does not open the change cycle.

| Observed symptom | Read |
|---|---|
| A tiny line looks implausibly expensive, or the CPU/GPU dependency chain is unclear | [§5](#5-critical-path-and-gpu-debt) |
| No profile exists yet, or CUDA code needs review; `cudaDeviceSynchronize` after every launch, per-item launches, device values steering host control flow, locks or serial loops inside kernels | [cpu-mental-model-antipatterns.md](references/cpu-mental-model-antipatterns.md) |
| A custom kernel does a reduction, scan, sort, filter, histogram, or GEMM that a library already provides | [cpu-mental-model-antipatterns.md](references/cpu-mental-model-antipatterns.md) section 5 |
| Unsure where wall time goes or which profiler to use | [profiling-and-attribution.md](references/profiling-and-attribution.md) |
| An Nsight Systems report needs capturing, exporting, or analyzing; a `.nsys-rep` or `.sqlite` is in hand | [nsys-workflow.md](references/nsys-workflow.md), driving [§2](#2-bundled-tooling) |
| Idle time must be split into a few long stalls versus a flood of tiny gaps | [nsys-workflow.md](references/nsys-workflow.md) section 6 |
| GPU idle needs a cause and the program has no NVTX annotation | [nsys-workflow.md](references/nsys-workflow.md) section 6, idle by host API in flight |
| Host-API idle still leaves material idle unclassified, or ranges were added but do not appear, or a range must select the capture window | [nvtx-bootstrap.md](references/nvtx-bootstrap.md) section 1, helpers in [code_templates/](code_templates/) |
| No profiler is installed, permitted, or attachable | [profiling-and-attribution.md](references/profiling-and-attribution.md) section 4 |
| Low, sawtooth, or bursty GPU utilization; long gaps between CUDA work | [issue-catalog.md](references/issue-catalog.md): host synchronization, fragmentation, CPU/framework, allocation |
| `.item()`, `.cpu()`, `.numpy()`, `.tolist()`, a tensor predicate, a small D2H `cudaMemcpy`, logging, or an assert appears slow | [§5](#5-critical-path-and-gpu-debt), then [issue-catalog.md](references/issue-catalog.md): host/device synchronization |
| Thousands or millions of short kernels; one launch per item, tile, component, or graph node | [issue-catalog.md](references/issue-catalog.md): launch and transaction fragmentation |
| Large `aten::to`, `_to_copy`, `copy_`, `clone`, `cat`, or `contiguous` totals | [profiling-and-attribution.md](references/profiling-and-attribution.md) section 6, then [issue-catalog.md](references/issue-catalog.md): transfers |
| Frequent `cudaMemGetInfo`, allocation/free calls, driver `ioctl`, allocator waits, OOM retries, or memory growth | [issue-catalog.md](references/issue-catalog.md): allocation, lifetime, and memory pressure |
| GPU is idle while a CPU thread is busy, or a faster GPU barely helps | [issue-catalog.md](references/issue-catalog.md): CPU, framework, I/O, and environment |
| First iteration is much slower, or shapes trigger pauses | [issue-catalog.md](references/issue-catalog.md) section 9 |
| Profiled and clean timings disagree, or the timer boundary is unclear | [issue-catalog.md](references/issue-catalog.md) section 1 |
| A microbenchmark improved but the routine did not | After the [gate](#7-gate-before-implementation): [agentic-tuning-workflow.md](references/agentic-tuning-workflow.md) sections 2, 6, and 7 |
| Work looks redundant, padded, or repeated in full when a delta would do | [issue-catalog.md](references/issue-catalog.md) section 7 |
| More streams, async copies, or overlap were proposed or made the program slower | [issue-catalog.md](references/issue-catalog.md): streams and concurrency |
| `nsys` fails, misses activity, lacks counters, sees the wrong GPU, or the workload runs in a container | [profiler-environment-setup.md](references/profiler-environment-setup.md) |
| A bottleneck report, experiment plan, or durable handoff is needed | [report-and-receipts.md](references/report-and-receipts.md) |
| This repository was profiled before, or the setup questions are being asked a second time | [report-and-receipts.md](references/report-and-receipts.md) section 8 |

## 11. Companion skills

Optional. This skill never links into another skill's folder. Every hand-off has a fallback.

| Skill | Hand off when | If it is not installed |
|---|---|---|
| `ncu-report` | a named kernel plus a digest launch ordinal already prove that kernel owns material wall time, or the user already has an `.ncu-rep` for that kernel | Drive `ncu` directly using [profiling-and-attribution.md](references/profiling-and-attribution.md) section 10 (NCU hand-off contract). Occupancy, balance, stalls, memory, source counters, time-series. |
| `cuda-kernel-wiki` | the limiting mechanism and architecture are known and you need implementation patterns | The CUDA C++ Best Practices Guide and the tuning guide for the specific architecture |
| `torch-profile-reading` | a PyTorch Kineto or Chrome JSON trace needs analysis | Read the trace directly, keeping the questions in [profiling-and-attribution.md](references/profiling-and-attribution.md) section 9 |

`ncu-report` writes under `profile/<run_name>/` and is written against B200 / sm_100 (verified on sm_80), including Python-DSL kernels. Use it after the gate, and only for a named kernel plus a digest ordinal. Other languages, layouts, or GPUs use [profiling-and-attribution.md](references/profiling-and-attribution.md) section 10.

## 12. Common pitfalls

- Optimize accepted end-to-end wall time, not utilization, occupancy, kernel count, or novelty.
- Treat CUDA execution as asynchronous. A synchronizing line may be paying accumulated producer work rather than doing expensive local work.
- Inspect the whole CPU/GPU timeline before selecting a kernel for deep profiling.
- Check whether a library already implements a kernel before writing or tuning one. A team fluent in CUDA will hand-roll a reduction, scan, sort, or GEMM that CUB, Thrust, cuBLAS, CUTLASS, or the framework already ships. Benchmark the library version as the baseline; keep a custom kernel only for fusion the library cannot express, a dtype or layout it lacks, or measured overhead — and treat "the library was slower" as a claim to verify.
- Treat a source-level finding as a hypothesis until the timeline shows its predicted signature, and treat frequency as part of the finding — the same line is free once and fatal in a loop.
- Separate clean timing captures from high-overhead ownership captures.
- Compare CUDA API duration with actual kernel/copy-engine activity; never add nested or overlapping totals.
- Separate copy count from copy bytes. Tiny control transfers and bulk payload transfers are different bottlenecks.
- Classify high cumulative kernel time into many-short-calls versus few-long-calls before choosing fusion or kernel tuning.
- Prove stream independence and an overlap opportunity before introducing concurrency.
- Use realistic workloads, immutable correctness references, the same physical GPU, and the same source and environment for comparisons.
- Re-profile after a source change. Accept only when correctness, clean wall time, and the predicted causal counters all agree.
- Report the mechanism in plain language with the arithmetic attached, so the person who wrote the code can find the next instance themselves.
