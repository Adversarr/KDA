# Profiling and attribution

Which evidence answers which question, and how to read the numbers once you have them. Capture, export, analyzer flags, and the digest read order live in [nsys-workflow.md](nsys-workflow.md) sections 3–6. When to add NVTX is [nvtx-bootstrap.md](nvtx-bootstrap.md) section 1; how to wire it is sections 2–3. Report shape, inventory, and receipts live in [report-and-receipts.md](report-and-receipts.md).

## Contents

1. [Evidence ladder](#1-evidence-ladder)
2. [Measurement boundaries](#2-measurement-boundaries)
3. [What utilization and unions mean](#3-what-utilization-and-unions-mean)
4. [When no profiler is available](#4-when-no-profiler-is-available)
5. [CUDA API and engine interpretation](#5-cuda-api-and-engine-interpretation)
6. [Copy analysis](#6-copy-analysis)
7. [Kernel distribution analysis](#7-kernel-distribution-analysis)
8. [CPU and OS-runtime attribution](#8-cpu-and-os-runtime-attribution)
9. [Framework profiler use](#9-framework-profiler-use)
10. [Nsight Compute handoff](#10-nsight-compute-handoff)

## 1. Evidence ladder

Collect the least intrusive evidence that resolves the current uncertainty. Start at the first rung that answers the open question.

| Question | Primary evidence |
|---|---|
| How long does the user wait? | Clean synchronized wall-clock timing |
| Which stage owns wall time? | Low-overhead timers; coarse NVTX only after [nvtx-bootstrap.md](nvtx-bootstrap.md) section 1 |
| Which patterns are worth suspecting at all? | Source reconnaissance — [cpu-mental-model-antipatterns.md](cpu-mental-model-antipatterns.md) section 6 |
| When are CPU and GPU active or waiting? | Nsight Systems — [nsys-workflow.md](nsys-workflow.md) sections 3–6 |
| Which source transaction owns launches, copies, waits, and gaps? | First: host-API idle table. Then targeted NVTX plus Systems — [nvtx-bootstrap.md](nvtx-bootstrap.md) section 1 |
| Which framework operators and stacks create the transaction? | Framework profiler; a trace-reading skill such as `torch-profile-reading` if installed |
| What is active CPU code doing? | CPU sampling and main-thread stacks |
| Why is a thread blocked? | CUDA API backtraces, OS-runtime tracing, thread state |
| Why is a proven important kernel slow? | Focused Nsight Compute (section 10); a kernel-profiling skill such as `ncu-report` if installed |
| Which low-level implementation pattern fits the measured limit? | A kernel-optimization corpus such as `cuda-kernel-wiki` if installed, after architecture and mechanism are known |

Nsight Compute cannot explain CPU-only gaps, host synchronization chains, launch starvation, I/O, or framework dispatch. Reach it only after a named kernel plus a digest launch ordinal already prove that kernel owns material wall time.

Source reconnaissance is the one cheap rung available before any environment work, and the only one that costs nothing when a profiler is unavailable. It produces hypotheses, not findings: promote a source hit only when the timeline shows its predicted signature.

*Done when the current question is named and the next measurement matches that row.*

## 2. Measurement boundaries

For every reported duration, record:

- exact start and end;
- whether the host synchronized before and after;
- whether it is host wall time, CUDA event time, summed activity, or interval union;
- cold or warm state;
- included input, output, audit, and I/O work;
- source revision and working-tree patch;
- input identity and algorithm settings;
- physical GPU, UUID, and contention state;
- container, driver, runtime/toolkit, framework, compiler, profiler;
- CPU thread and library settings.

Use a synchronized outer wall timer for accepted user-visible latency. Use CUDA events for selected stream/device intervals. Synchronize only at intentional outer boundaries; inserting sync inside the workload to make timing easier can change the schedule.

NVIDIA requires synchronization around CPU timers used for asynchronous CUDA calls. See [CUDA C++ Best Practices Guide: Timing](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#timing).

*Done when every number you will quote has those fields written next to it.*

## 3. What utilization and unions mean

### Critical path beats summed activity

CPU work, CUDA API calls, kernels, and copies can overlap. Summing their durations can double-count the same wall interval. Compute temporal unions and exclusive ownership where possible.

A kernel owning 20% of device time is not automatically a 20% wall-time opportunity. It may overlap CPU work, run off the critical path, or be followed by a larger host bottleneck. Establish an upper bound — the ceiling — against the exact wall-time boundary before prioritizing it.

### Utilization is a symptom, not a score

High utilization can mean productive throughput, redundant scans, atomics, poor memory access, or a slow algorithm keeping the device busy. Low utilization can mean host starvation, synchronization, launch fragmentation, grid underfill, serial dependencies, allocation, I/O, or a naturally small workload.

Removing unnecessary GPU work can lower utilization while improving wall time. Raising utilization through competing streams can regress wall time.

### Timing must close the asynchronous boundary

A CPU timer that stops immediately after enqueue measures submission, not completion. Synchronize at intentional outer boundaries or use CUDA events for an appropriate stream interval. NVIDIA explicitly requires host/device synchronization around CPU timing of asynchronous CUDA calls. See [Timing](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#timing).

The Systems digest heading `## CUDA activity versus wall` already reports the CUDA-activity union, the no-CUDA gap, kernel span sum, copy engine sum, and activity islands. Quote that union before treating any kernel or copy total as wall time.

*Done when every quoted duration is labeled wall, event, sum, or union, and every kernel ceiling is against the accepted wall boundary.*

## 4. When no profiler is available

Nsight Systems may be absent, forbidden by policy, or impossible to attach to the process that matters. A useful diagnosis is still reachable; a kernel-internals diagnosis is not. Say which one you are delivering.

### 4.1 Evidence that needs no profiler

| Technique | What it establishes |
|---|---|
| Nested synchronized host timers around each phase | Where wall time goes at phase granularity; the denominator for every ceiling |
| CUDA events around a stream interval | Device-side duration of a region without a full trace |
| Events around a loop of *N* identical launches | Separates per-launch overhead from per-launch execution as *N* grows |
| `CUDA_LAUNCH_BLOCKING=1` A/B against a clean run | See 4.2 |
| Framework sync detection, e.g. PyTorch's `torch.cuda.set_sync_debug_mode("warn")` | Names the exact lines performing implicit host synchronization |
| Framework allocator counters, e.g. `torch.cuda.memory_stats()` | Allocation and free counts, retries, peak reserved versus allocated |
| `nvidia-smi --query-gpu=utilization.gpu --format=csv --loop-ms=100` | Coarse starvation signal: sustained low utilization during a phase that should be GPU-bound |
| Stubbing a phase with a precomputed result | Amdahl ceiling for removing that phase entirely |
| Source reconnaissance | Candidate mechanisms to test |

Verify `nvidia-smi` flags against `--help` before treating a mismatch as a skill bug (flags current as of 2026-05).

### 4.2 The launch-blocking discriminator

Run the workload with `CUDA_LAUNCH_BLOCKING=1` and compare against the clean run:

| Observation | Meaning |
|---|---|
| Wall time barely changes | The program was already serialized by its own host synchronization. Asynchrony was not being exploited, so host synchronization or launch fragmentation is the dominant mechanism. This is the strongest cheap evidence available without a profiler. |
| Wall time gets substantially worse | Real overlap existed and the async pipeline is working. Look elsewhere: CPU phases, transfer volume, or genuine kernel cost. |

Treat this strictly as a discriminator. It changes scheduling and error semantics, so it never produces an acceptance number.

### 4.3 What cannot be claimed

Without a timeline: exact gap ownership, per-transaction attribution, CPU stacks during gaps, and the temporal CUDA-activity union. Without hardware counters: occupancy, achieved bandwidth, stall reasons, cache behavior, and every roofline statement.

Report these as unavailable and name the capture that would resolve them. A phase-level attribution honestly labeled is worth more than a kernel-level story that the evidence cannot support.

*Done when the report names the phase-level owners it has, lists what is unavailable, and states whether this is a whole-program diagnosis or that a kernel-internals diagnosis is out of reach.*

## 5. CUDA API and engine interpretation

Host API duration is not device duration and is not automatically additive wall time. Quote `digest.txt` headings `## CUDA runtime API by family` (family, calls, host s) and `## CUDA runtime API by host time` (host s, calls, name) first; then compare those host totals to engine intervals from `## CUDA activity versus wall` and `## Copies by direction and size`.

For important API families such as `cudaMemcpyAsync`, stream/event synchronization, launch, allocation/free, and memory/device queries, report:

- count;
- summed and distributional host duration;
- actual device-engine duration where applicable;
- overlap with kernels/copies and other API calls;
- call-stack/NVTX owner;
- payload direction and size where applicable;
- whether the call waits for an earlier producer.

### API/engine mismatch

If copy APIs accumulate far more host duration than copy-engine intervals, possible causes include:

- producer readiness waits;
- pageable-memory staging;
- stream synchronization;
- allocation/lifetime dependencies;
- nested framework attribution;
- driver serialization.

Treat the mismatch as a clue, not as recoverable transfer time.

### Synchronization provenance

For every major synchronization family, attribute:

- explicit application sync — `cudaDeviceSynchronize`, `cudaStreamSynchronize`, `cudaEventSynchronize`, `torch.cuda.synchronize`;
- framework scalar extraction — `.item()`, `_local_scalar_dense`, tensor predicates;
- blocking transfer — the synchronous `cudaMemcpy` form, or a pageable-memory copy;
- stream/event dependency;
- allocator/lifetime safety;
- library boundary, including primitives that return a value to the host such as `thrust::reduce`;
- profiler or diagnostic behavior, including `CUDA_LAUNCH_BLOCKING`.

The line containing the sync is the consumption boundary. Also identify the producer it waits for.

*Done when each important API family has count, host duration, engine duration, overlap, owner, payload, and wait-versus-work.*

## 6. Copy analysis

Quote `## Copies by direction and size` from `digest.txt`: direction, calls, MiB, engine s, and the size buckets `<=8B`, `<=64B`, `<=256B`, `<=4KiB`, `<=1MiB`, `>1MiB`.

| Direction | Calls | MiB | Engine s | `<=8B` | `<=64B` | `<=256B` | `>1MiB` | Top owners |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| D2H | | | | | | | | |
| H2D | | | | | | | | |
| D2D | | | | | | | | |

Fill `<=4KiB` and `<=1MiB` from the same digest heading. Interpret separately:

- tiny D2H → control decisions, counters, validation, logging, dynamic sizes;
- tiny H2D → per-transaction commands or metadata;
- bulk H2D/D2H → input/output/checkpoint payload;
- bulk D2D → clone, materialization, layout/dtype conversion, state snapshot, full clear.

Ask whether each visible framework conversion performs a copy. Device/dtype normalization can be a no-op. `cat` and `contiguous` are candidates only when frequency, bytes, allocation, and critical-path evidence support them.

NVIDIA advises batching many small transfers, retaining intermediate data on device, and using pinned memory for asynchronous host transfers. See [Data Transfer Between Host and Device](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#data-transfer-between-host-and-device).

*Done when count, bytes, direction, engine time, and size buckets are filled, and each row has an owner and a control-versus-payload reading.*

## 7. Kernel distribution analysis

Rank from two digest headings together, not from total device time alone:

- `## Kernels by device time` — total s, calls, mean us, max ms, name;
- `## Kernel duration distribution` — bucket, calls, total s.

For each top kernel, record those four per-kernel columns, the duration-distribution buckets, owning phase and transaction (`## Kernel device time by launching NVTX range`), shape/work distribution, and surrounding host gaps or repeated sequence. When mean us and max ms diverge, quote the duration-distribution buckets to see whether the mass is many short calls or a few long ones.

`(unattributed)` on the launching-NVTX table is leftover launches — graph replay, a submitting thread with no NVTX, or a launch outside the window. It is not an idle owner. Recapture-with-annotation is [nvtx-bootstrap.md](nvtx-bootstrap.md) section 1, and only after `## No-CUDA time by host API in flight` still leaves material idle unclassified.

Route by pattern:

| Pattern | First response |
|---|---|
| Hundreds of thousands of tens-of-microsecond calls | Batch, fuse, capture, or redesign the transaction |
| A few long, material calls | Isolate representative inputs and profile the kernel |
| Healthy kernel metrics but excessive calls/bytes | Remove redundant total work |
| Long-tail calls or high per-SM variance | Analyze workload distribution and scheduling |
| A custom kernel implementing a standard primitive | Benchmark the library equivalent before tuning it — [cpu-mental-model-antipatterns.md](cpu-mental-model-antipatterns.md) section 5 |
| Low total device contribution | Deprioritize unless it blocks a larger dependency |

*Done when top kernels have calls, total s, mean us, max ms, and bucket shares quoted from `digest.txt`, and each is routed as many-short, few-long, or deprioritized.*

## 8. CPU and OS-runtime attribution

### CPU active during GPU gaps

Collect CPU samples and resolve native symbols. Check:

- Python/GIL and framework dispatch;
- native graph, geometry, sorting, serialization, compression;
- OpenMP/BLAS behavior;
- locks and work queues;
- memory allocation and page faults;
- I/O and filesystem operations.

Sampling time across all threads is not equal to critical-path wall time. Focus on the main/control thread and workers whose completion gates the next CUDA submission.

### CPU blocked during GPU gaps

Collect thread state, CUDA API backtraces, and selected OS-runtime calls:

- stream/device/event sync;
- `ioctl` and driver queries;
- futex/condition variables;
- poll/epoll;
- file and network I/O;
- child process joins.

Correlate background worker sleep with the critical path before treating those totals as actionable. Host-API idle already appears under `## No-CUDA time by host API in flight` in the Systems digest; sampling and OS-runtime rows explain the remainder.

*Done when representative no-CUDA gaps are classified as active compute, dispatch, blocked-on-producer, driver/allocation, I/O, or unattributed, on the thread that gates the next launch.*

## 9. Framework profiler use

Use a framework profiler to connect source stacks with operators, shapes, and memory behavior after Systems identifies a bounded region. This rung applies only to framework-based programs; a pure C++/CUDA application goes straight from Systems to kernel analysis.

Use expensive options selectively:

- stack capture;
- shape recording;
- memory profiling;
- all-operation NVTX;
- long Chrome traces.

Questions to answer:

- Which source loop produces the operator cardinality?
- Does the operator launch work, move data, allocate, or wait?
- Are nested operations the same interval represented repeatedly?
- Are shapes stable enough for batching, compilation, or graph capture?
- Are graph breaks or specializations multiplying dispatch/JIT work?

For PyTorch Kineto or Chrome traces, `torch-profile-reading` handles the analysis if it is installed; otherwise read the trace directly and keep the same questions. Either way, keep the Systems critical-path model as the timing authority; framework operator totals sit beside it.

*Done when each important operator in the bounded region is labeled work, wait, or mixed, with the source loop and shape stability named.*

## 10. Nsight Compute handoff

Use NCU only when all are true:

- the kernel passes correctness checks;
- a clean Systems capture proves material device and wall-time relevance;
- the kernel is few-long rather than many-short work;
- the kernel is not a reimplementation of a library primitive that has never been benchmarked against it;
- representative shapes/workloads are known;
- an isolated or tightly filtered driver reproduces the production mechanism;
- source mapping is available when line attribution is needed;
- profiler permissions and architecture are valid.

The Systems side of the handoff — named kernel plus digest launch ordinal — is [nsys-workflow.md](nsys-workflow.md) section 8.

Analyze occupancy, balance, stalls, memory, compute, source counters, and time-series/tail evidence. `ncu-report` covers this workflow if it is installed: start from its `analyze_reports.py` digest, which reads a `.ncu-rep` and states which metric name supplied each number. Without that skill, or for portable C++/CUDA and architectures that skill does not cover, drive `ncu` directly and keep the same questions. Once the limiting mechanism is known, a corpus such as `cuda-kernel-wiki` maps it to implementation patterns.

NCU replay can be extremely expensive and perturb caches, clocks, and execution. Treat it as diagnostic evidence, not application timing authority.

*Done when NCU is opened only after the eight conditions hold, with occupancy, balance, stalls, memory, source counters, and time-series addressed, or when the missing condition is named and the next Systems or library-benchmark step is chosen instead.*
