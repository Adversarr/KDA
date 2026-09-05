# CUDA program performance issue catalog

## Contents

1. Measurement boundary and profiler artifacts
2. Host/device synchronization
3. Launch and transaction fragmentation
4. Transfers, layout, and materialization
5. Allocation, lifetime, and memory pressure
6. CPU, framework, I/O, and environment
7. Work decomposition and redundant algorithms
8. Streams, concurrency, and overlap
9. Compilation, initialization, and caching
10. GPU kernel efficiency categories
11. Avoidable custom kernels
12. Common real-project patterns

This catalog starts from what a profile shows. When there is no profile yet, or the question is what the source itself gets wrong, start from [cpu-mental-model-antipatterns.md](cpu-mental-model-antipatterns.md) and return here to confirm each hypothesis.

Use the same reasoning frame for every category:

1. **Symptom:** what is observable?
2. **Mechanism:** what CPU/GPU dependency causes wall time?
3. **Evidence:** which measurement distinguishes it from alternatives?
4. **Misdiagnosis:** what attractive explanation is wrong?
5. **Treatment:** what bounded change tests the mechanism?
6. **Risk:** which correctness, ordering, lifetime, or memory property may change?

A category is confirmed when its Evidence distinguishes it from the Misdiagnoses. During first pass, read **Symptom / Mechanism / Evidence / Misdiagnosis** and return to [SKILL.md](../SKILL.md) section 6. Apply a **Treatment** only after section 7 can answer every question — that *Done when* is a clean A/B on the accepted wall-time boundary that moves the predicted digest counters, with Risk items checked against the equivalence contract. Default evidence is the first-pass `digest.txt` — quote the named heading. Further queries, if any, are the named exceptions in [nsys-workflow.md](nsys-workflow.md) section 10.

## 1. Measurement boundary and profiler artifacts

### Symptoms

- A fast kernel does not improve the application.
- A newer GPU barely changes end-to-end time.
- Profiled and unprofiled runs disagree materially.
- Loading, cleanup, validation, output copies, export, compilation, or warmup move in and out of the reported “stage.”
- A CPU timer reports implausibly small CUDA time.

### Mechanism

Applications have distinct performance boundaries:

- request/end-to-end wall;
- input I/O and CPU preprocessing;
- device publication;
- CUDA algorithm;
- phase and repeated transaction;
- output device-to-host materialization;
- CPU postprocessing and export;
- cold initialization, JIT, autotuning, and allocator warmup.

An optimization affects only boundaries containing its critical-path work. An asynchronous CPU timer can stop before the GPU finishes. Detailed tracing can perturb high-call-count paths more than coarse paths.

### Evidence

- State every timer's start, end, and synchronization semantics.
- Record nested clean phase timings and a low-overhead timeline.
- Separate cold and warm runs.
- Compare diagnostic capture wall time against uninstrumented wall time.
- Record exact input, source, physical GPU, container, thread settings, and output behavior.
- Estimate the ceiling using the fraction of accepted wall time owned by the target.

NVIDIA's APOD cycle starts by locating the portions responsible for the bulk of application execution time, estimating the possible speedup, making incremental changes, and verifying each result. It also requires realistic workloads. See [Application Profiling](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#application-profiling).

### Misdiagnoses

- “Kernel speedup equals application speedup.”
- “Profiled ownership time is the acceptance baseline.”
- “CUDA events and request wall time are interchangeable.”
- “The first run represents steady state.”
- “Every stronger GPU improves every stage.”

### Treatments and risks

- Define the performance contract before profiling.
- Keep clean timing, ownership tracing, CPU sampling, and NCU diagnostics separate.
- Synchronize only intentional outer timing boundaries.
- Preserve cold-start measurements when cold start is a product requirement.
- Re-baseline after relevant source or environment changes.

Risk: hiding required initialization or output work can create a benchmark-only win.

## 2. Host/device synchronization

The worked `.item()` / GPU debt example lives in [SKILL.md](../SKILL.md) section 5.

### Symptoms

- Bursty GPU timeline with repeated empty gaps.
- Large counts of small D2H copies and stream synchronizations.
- Framework surfaces: `.item()`, `.cpu()`, `.numpy()`, `.tolist()`, tensor predicates, logging, assertions, or host-visible counters inside loops.
- C++/CUDA surfaces: `cudaDeviceSynchronize` or `cudaStreamSynchronize` once per launch, the blocking `cudaMemcpy` form, an error-check macro that synchronizes, `CUDA_LAUNCH_BLOCKING` left enabled, or a library call that returns a value to the host such as `thrust::reduce`.
- Sync count per phase closely tracking launch count per phase.
- Iterative solver, BFS, graph, worklist, convergence, or transactional code progresses one decision at a time.
- A harmless-looking scalar line owns surprising duration.

### Mechanism

The host cannot consume a device-produced value until its producer dependencies finish. A small host read can therefore:

1. drain relevant queued GPU work;
2. transfer a tiny payload;
3. wake Python or native host control;
4. choose the next branch;
5. submit the next work only after the device is already idle.

The bandwidth cost of a scalar is negligible; the round trip and serialized dependency chain are not. “Async” describes API submission, not freedom from dependencies.

### Evidence

- Quote `## Copies by direction and size` (`<=8 B` / `<=64 B` D2H), `## CUDA runtime API by host time` (synchronize / blocking memcpy), `## No-CUDA time by host API in flight`, and `## Largest individual no-CUDA gaps`.
- Correlate small D2H copies, `cudaStreamSynchronize`, `_local_scalar_dense`, framework scalar operations, and source/NVTX owners.
- Inspect the producer immediately before the read and the GPU gap immediately after it.
- Count reads per phase, transaction, and useful work item. Distinguish one phase-final read from repeated inner-loop reads.
- Search source for device-derived Python control and logging.

### Misdiagnoses

- “It is only one integer.”
- “PCIe bandwidth makes scalar reads free.”
- “`non_blocking=True` prevents every host wait.”
- “The `.item()` implementation itself is computationally expensive.”
- “Changing syntax removes the dependency.”

### Treatments and risks

- Keep predicates, counters, convergence state, and error codes device-resident.
- Batch host reads at true phase boundaries.
- Pack related scalar publications into one certificate.
- Replace per-item host control with masks, scans, segmented compaction, or batched kernels.
- Move stable iterative control into a device transaction when semantics permit.
- Capture a stable launch sequence with CUDA Graphs only after eliminating dynamic host dependencies.

Risk: deferring reads can change exception timing, ordering, termination, deterministic decisions, audit publication, and buffer lifetime.

## 3. Launch and transaction fragmentation

### Symptoms

- Thousands or millions of kernels, mostly shorter than 10–50 microseconds.
- Significant CUDA launch API time.
- One repeated sequence of tiny selection, reduction, copy, validation, and publication operations.
- One launch or native/Python call per logical object, row, tile, component, token, or graph node.
- No individually dominant source line despite large total wall time.

### Mechanism

Each launch and framework call carries fixed host/runtime/driver/device scheduling cost. Fragmented transactions also repeat:

- Python and dispatcher work;
- tensor-object and metadata construction;
- allocator interaction;
- argument validation;
- intermediate global-memory round trips;
- scalar publication and synchronization;
- partial-grid tails.

Dependencies between tiny operations prevent launch-ahead, so GPU bubbles may exceed the launch API time itself.

### Evidence

- Quote `## Kernels by device time` (total s, calls, mean us, max ms, name), `## Kernel duration distribution` (bucket, calls, total s), `## CUDA runtime API by family` (launch), and `## CUDA activity versus wall` (union vs no-CUDA gap).
- Build kernel duration histograms and repeated-name sequence counts.
- Report launches, copies, memsets, syncs, and API calls per useful transaction.
- Compute temporal CUDA-active union and the gaps between transaction launches.
- Classify top kernels: high total + tiny per call → batching/fusion; high total + substantial per call → possible kernel target; low total → deprioritize.
- Attribute framework and native calls to semantic transactions with NVTX.

### Misdiagnoses

- “Launch overhead is only a few microseconds.”
- “A high cumulative kernel must be instruction-tuned.”
- “The GPU is busy because many kernel rows exist.”
- “One kernel per object is naturally parallel.”
- “A one-call microbenchmark represents the loop.”

### Treatments and risks

- Batch independent items in packed, CSR, or segmented representations.
- Fuse adjacent elementwise, reduction, selection, publication, and audit work.
- Reuse stable schedules, indices, metadata, and workspaces.
- Use optimized primitives such as CUB/Thrust for standard scan, reduction, selection, sort, and compaction when their stream and temporary-storage contracts fit.
- Use Triton as a cheap fusion/scheduling falsification prototype for tractable shapes; port or replace it only after end-to-end evidence.
- Use CUDA Graphs for stable repeated launch sequences, not for dynamic CPU-controlled paths.

Risk: batching and fusion can change ordering, reduction sequence, tie-breaking, error timing, memory ceiling, and deterministic transaction behavior.

## 4. Transfers, layout, and materialization

### Symptoms

- High `cudaMemcpyAsync` count or host API duration.
- Large framework totals for `to`, `_to_copy`, `copy_`, `clone`, `contiguous`, `cat`, gather/scatter, casts, or packing.
- Alternating H2D/D2H traffic in a loop.
- Copy-engine time is much smaller than copy API time.
- Large D2D volume or repeated whole-state clones/clears.

### Mechanism

Separate four classes:

1. **Tiny control copies:** few bytes, huge count; synchronization/fragmentation dominates.
2. **Bulk payload copies:** few calls, most bytes; bandwidth and overlap dominate.
3. **D2D materialization:** layout conversion, clone, gather, concatenate, snapshot, full clear; memory traffic, launches, allocation, and lifetime dominate.
4. **No-copy normalization:** dtype/device/layout request already satisfied; only framework/metadata overhead remains.

An async copy can still appear slow because the source is not ready, pageable host memory needs staging, a later consumer waits, allocator/lifetime rules add dependencies, or the API interval is charged producer completion.

### Evidence

- Quote `## Copies by direction and size` (direction, calls, MiB, engine s, size buckets). Use at least `<=8 B`, `<=64 B`, `<=256 B`, intermediate, and `>1 MiB` groups.
- Compare API intervals with actual copy-engine intervals; quote one leaf, not nested `to → _to_copy → copy_` sums.
- Inspect tensor strides and whether `contiguous()` actually copied.
- Identify whole-array traffic relative to the number of changed or consumed elements.

NVIDIA recommends minimizing host/device transfers, keeping intermediate structures on the device, and batching small transfers. Async host/device transfer requires pinned memory; copy/compute overlap also depends on streams, device capability, and dependencies. See [Data Transfer Between Host and Device](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#data-transfer-between-host-and-device) and [Asynchronous and Overlapping Transfers](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#asynchronous-and-overlapping-transfers-with-computation).

### Misdiagnoses

- “The largest byte total is always first.”
- “Small transfers cannot matter.”
- “`cudaMemcpyAsync` cannot block.”
- “Every `.to()` moves data.”
- “`cat` or `contiguous` must be expensive by name.”
- “D2D traffic is free because it stays on the GPU.”

### Treatments and risks

- Batch tiny control publications and keep intermediate control state on device.
- Pre-normalize stable dtype/device/layout outside hot loops.
- Reuse buffers and update sparse deltas instead of copying or clearing full state.
- Fuse producers and consumers to remove intermediates.
- Use pinned host buffers only where real asynchronous transfer/overlap is required; pinned memory is scarce and pinning is heavyweight.
- Measure actual overlap before double buffering.

Risk: eliminating copies may alias mutable storage, extend lifetime, change contiguity contracts, or expose races.

## 5. Allocation, lifetime, and memory pressure

### Symptoms

- Frequent `cudaMalloc`, `cudaFree`, `cudaMemGetInfo`, allocator events, or driver `ioctl`.
- Variable shapes cause pauses or history-dependent performance.
- Tensor construction dominates GPU-idle ranges.
- Capacity gates or OOM checks run inside fine-grained loops.
- Extra streams increase memory use or allocator waits.

### Mechanism

Allocation can include driver calls, synchronization, free-list management, block splitting, stream-lifetime tracking, event insertion, release/garbage collection, initialization, and framework metadata construction. A caching allocator reduces raw allocation frequency but does not make allocation, fragmentation, or stream safety free.

Safety bookkeeping also multiplies: querying physical memory before every small transaction crosses the runtime/driver boundary even when the result is effectively invariant over that transaction.

### Evidence

- Quote `## CUDA runtime API by family` and `## CUDA runtime API by host time` for allocation, free, and memory-info families. Count and attribute allocation/free, memory-info, device/property, and OS-runtime calls.
- Track active, reserved, and physical-free memory separately.
- Identify stable maximum shapes and repeated temporary shapes.
- Determine true allocation barriers where external pressure may change.
- Check whether extra streams prolong buffer lifetime or increase peak memory.
- Test bounded workspace reuse as a falsification experiment.

### Misdiagnoses

- “The caching allocator makes allocation free.”
- “A memory safety helper is too small to matter.”
- “Reserved memory equals active memory.”
- “Preallocation is always faster and safe.”
- “More streams do not affect lifetime.”

### Treatments and risks

- Hoist stable tensor construction and allocate bounded reusable workspaces.
- Reuse size classes and shorten temporary lifetimes.
- Cache conservative resource observations only across explicitly proven transactions.
- Refresh at real phase, slice, external-interaction, or large-allocation boundaries.
- Keep primitive temporary storage under the framework allocator and current-stream contract.
- Preserve fail-closed behavior and telemetry for refresh/allocation barriers.

Risk: stale headroom estimates can break coexistence with other processes; oversized workspaces can raise peak memory and force eviction.

## 6. CPU, framework, I/O, and environment

### Symptoms

- GPU is idle with no CUDA API activity.
- A faster GPU barely improves the stage.
- Python/ATen calls or native CPU functions dominate no-CUDA intervals.
- Host preprocessing, data loading, decompression, cleanup, validation, logging, serialization, or export dominates.
- CPU is unexpectedly single-threaded or oversubscribed.
- Runtime varies with filesystem, thread settings, or container configuration.

### Mechanism

GPU work cannot start until the CPU submits it and its inputs are ready. Common host limits include:

- Python loops, GIL, function/dispatcher/schema overhead;
- small tensor/list/dict/dataclass construction;
- graph breaks and specialization lookup;
- native CPU algorithms and locks;
- OpenMP/BLAS undersubscription or nested oversubscription;
- thread-pool startup and contention;
- NUMA placement and page faults;
- remote filesystem I/O;
- logging, audit, and serialization.

Summed sleeping time across background threads is not critical-path time. Identify the main/control thread and the exact GPU-starvation interval.

### Evidence

- Quote `## CUDA activity versus wall` (union, no-CUDA gap) and `## No-CUDA time by host API in flight` (idle-with-call-open vs idle-with-none-open).
- Measure CPU-only phases separately. Sample active CPU stacks during GPU gaps.
- Trace thread states and OS runtime for blocked gaps.
- Record CPU affinity, OpenMP/BLAS/data-loader thread counts, NUMA, and I/O path. Compare intended thread configuration against inherited container defaults.
- Count Python/framework calls and graph recompilations in the hot phase.
- Determine whether a host function performs compute or waits for CUDA.

### Misdiagnoses

- “The GPU is slow because it is idle.”
- “All poll/futex time is a bottleneck.”
- “Python always matters” or “Python never matters.”
- “More CPU threads are always faster.”
- “The container sees CPUs, so libraries use them correctly.”
- “Move CPU work to GPU” without transfer and granularity analysis.

### Treatments and risks

- Fix the environment before code: intended thread counts, affinity, libraries, filesystem, and architecture build.
- Batch at the semantic transaction level before micro-optimizing Python syntax.
- Move stable orchestration into one native/batched boundary and return packed results.
- Reuse metadata and hoist validation/normalization where the contract allows.
- Parallelize a measured CPU hotspot with a compiled binding only when CPU compute—not CUDA waiting—is proven dominant and the refactor is low-risk.
- Pipeline independent CPU preparation with GPU work when dependencies permit.

Risk: moving work across CPU/GPU can add transfers and synchronization; CPU parallelization can create oversubscription or nondeterministic ordering.

## 7. Work decomposition and redundant algorithms

### Symptoms

- Full arrays are scanned, copied, or cleared for sparse changes.
- Stable topology, schedule, offsets, or certificates are rebuilt.
- Duplicate audits rediscover an invariant already proven by a typed producer.
- Runtime scales with padded capacity rather than active work.
- Kernels look healthy but total device time remains large.
- Variable-size objects create severe imbalance.

### Mechanism

Hardware can execute unnecessary work efficiently. Kernel metrics may look excellent while the algorithm performs too many bytes, threads, passes, retries, sorts, scans, or validations. CPU-style per-object decomposition also prevents cross-object parallelism and creates partial waves.

### Evidence

- Count useful items versus issued threads, traversed elements, and bytes scanned.
- Measure launches/copies/syncs per changed element or accepted transaction.
- Record rebuild and audit counts for supposedly immutable authorities.
- Inspect batch-size and per-object work distributions.
- Compare dense capacity against active frontier/delta size.
- Establish which producer certificate already proves which consumer invariant.

### Misdiagnoses

- “Healthy NCU metrics mean little optimization remains.”
- “A fast per-item kernel makes the loop efficient.”
- “Batching only helps dense linear algebra.”
- “Bigger batches are always better.”
- “Reusing a proof automatically weakens correctness.”

### Treatments and risks

- Use sparse frontier/delta updates instead of dense full-state passes.
- Batch related items into segmented/CSR representations with bounded capacity.
- Reuse immutable authorities, schedules, topology, and producer certificates.
- Replace repeated global atomics with warp/block aggregation and one global publication.
- Split common/fast and exceptional/slow paths when distributions justify it.
- Prefer library primitives for standard data-parallel operations before hand-rolling them — section 11, then [cpu-mental-model-antipatterns.md](cpu-mental-model-antipatterns.md) section 5 for the library table.

Risk: algorithm-level changes can alter internal tensors, ordering, decisions, receipts, error handling, and determinism. Obtain approval where the equivalence contract requires it.

## 8. Streams, concurrency, and overlap

### Symptoms

- One stream serializes activity and a team proposes “add streams.”
- Multiple streams exist but kernels/copies do not overlap.
- Secondary-stream experiments increase utilization but regress wall time.
- Events, allocator waits, memory growth, races, or stale results appear.

### Mechanism

Streams permit concurrency; they do not create independent work. Overlap can fail because of:

- true producer/consumer dependencies;
- shared buffers or incomplete lifetime tracking;
- default-stream semantics;
- pageable host memory;
- contention for SMs, registers, shared memory, bandwidth, cache, or copy engines;
- conservative event waits and final joins;
- host submission starvation;
- increased peak memory and allocator events;
- worse locality or tail latency.

NVIDIA notes that operations in streams retain ordering, copy/compute overlap is device-dependent, host memory must be pinned for asynchronous host transfers, and appropriate non-default streams are needed for certain overlap. See [Asynchronous and Overlapping Transfers](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#asynchronous-and-overlapping-transfers-with-computation).

### Evidence

- Draw a dependency graph and name every producer, consumer, buffer, event, and join.
- Compute actual temporal overlap in Nsight Systems; stream count is not evidence.
- Check `asyncEngineCount`, pinned-memory use, and stream IDs for transfers.
- Inspect resource saturation of candidate concurrent kernels.
- Compare one-stream and multi-stream clean A/B wall time, peak memory, and allocator activity.

### Misdiagnoses

- “More streams increase utilization and therefore speed.”
- “Independent-looking functions are dependency-independent.”
- “Copy and compute overlap automatically.”
- “Events and final joins are free.”
- “A single stream is inherently a bottleneck.”

### Treatments and risks

- First remove host scalar dependencies and submission gaps (sections 2–3).
- Add concurrency only for proven independent critical-path work.
- Use narrow producer events and consumer waits rather than a global synchronize.
- Preserve framework allocator stream ownership.
- Chunk transfers/work only when the dependency structure and device engines support overlap.
- Revert if clean wall time regresses, even when overlap or utilization rises.

Risk: races, use-after-free, changed ordering, greater memory footprint, and contention.

## 9. Compilation, initialization, and caching

### Symptoms

- First call is much slower than later calls.
- New shapes or dtypes cause pauses.
- Timing changes after process restart.
- Profiling captures compilation rather than steady execution.
- Native extension loads or architecture errors appear.

### Mechanism

Cold work includes CUDA context creation, module/library loading, JIT compilation, kernel specialization, autotuning, allocator-pool growth, cache population, dynamic linking, and extension compilation. Dynamic shapes can trigger repeated specialization or graph breaks.

### Evidence

- Record first-call and steady-state timings separately.
- Trace compilation/module load and count specializations or graph breaks.
- Print the exact kernel names/shapes after warmup.
- Verify native code contains the target GPU architecture.
- Restart the process to reproduce cold behavior intentionally.

### Misdiagnoses

- “One warmup fixes every production scenario.”
- “JIT time is kernel execution time.”
- “An imported extension is built for the visible GPU.”
- “A microbenchmark's cached state represents a service cold start.”

### Treatments and risks

- Define whether cold start or steady state is the product metric.
- Warm or ahead-of-time compile known variants where appropriate.
- Bound shape specialization and preserve cache keys.
- Build native extensions for the actual architecture and test real kernel execution.

Risk: hiding a production cold-start requirement or precompiling an unbounded variant set.

## 10. GPU kernel efficiency categories

Use Nsight Systems first to prove that a named kernel is consequential on the accepted wall-time boundary. Then use section 11 to establish that the kernel should exist at all. Only then profile it on representative shapes — with `ncu-report` if that skill is installed, otherwise `ncu` directly using [profiling-and-attribution.md](profiling-and-attribution.md) section 10 — and map the measured limit to implementation patterns with `cuda-kernel-wiki` if it is installed, otherwise the architecture's tuning guide.

Quote `## Kernels by device time` (total s, calls, mean us, max ms) and `## Launch ordinals for kernels matching '…'` before collecting NCU.

### 10.1 Memory-bandwidth bound
**Symptoms:** high sustained DRAM throughput, low compute utilization, low arithmetic intensity, runtime proportional to bytes.
**Evidence:** roofline position, achieved bandwidth, requested versus transferred bytes, sectors/requests, cache hit rates.
**Treatments:** remove redundant passes, coalesce/vectorize, improve locality/reuse, fuse producer/consumer work, choose cache policy from measured reuse.
**Trap:** low compute utilization alone does not prove bandwidth saturation.

### 10.2 Memory-latency or dependency bound
**Symptoms:** low bandwidth with long-scoreboard stalls, sparse gathers, pointer chasing, graph/BVH traversal, few independent loads.
**Mechanism:** dependent accesses prevent enough memory-level parallelism to saturate bandwidth.
**Treatments:** improve layout/locality, reorder or spatially batch queries, cache reused nodes, prefetch predictable work, reduce dependency depth, increase active warp supply only when resources permit.

### 10.3 Compute or instruction bound
**Symptoms:** relevant compute pipelines near saturation, high arithmetic intensity, runtime tracks operation count, expensive transcendental/integer/address/conversion work.
**Treatments:** reduce instruction count, use appropriate precision/library paths, improve ILP, examine source-level instruction and stall attribution.

### 10.4 Occupancy, registers, shared memory, and spills
**Symptoms:** few active warps/blocks, register or shared-memory residency limit, spills/local memory, low eligible warps.
**Treatments:** shorten live ranges, tune tile/block size, reduce per-block state, split excessive fusion, tune shared memory, apply register limits only after measuring spill tradeoffs.
**Trap:** occupancy is a latency-hiding resource, not a score. NVIDIA notes that higher occupancy does not always improve performance and that register/shared-memory tradeoffs require per-kernel analysis. See [Occupancy](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy).

### 10.5 Coalescing and cache efficiency
**Symptoms:** excessive memory transactions, poor sector/request efficiency, misaligned or strided warp access, cache thrash.
**Treatments:** map adjacent lanes to adjacent aligned words, transpose/tile data, use shared memory for reuse or access remapping, avoid loading unused fields.
See [Coalesced Access to Global Memory](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory).

### 10.6 Divergence and irregular work
**Symptoms:** low branch efficiency, large warp/block duration variance, sparse/graph/ray/mesh work, a few CTAs run far longer.
**Treatments:** group similar work, compact active items, split common/exception paths, use dynamic/persistent scheduling when appropriate, redesign queues.

### 10.7 Tail effect, grid underfill, and wave quantization
**Symptoms:** kernel begins busy and ends with few active SMs, grid has too few blocks, runtime changes with block count modulo SM count.
**Treatments:** expose more blocks, change tile shape, combine small problems, use dynamic/persistent scheduling, inspect PM/time-series evidence rather than averages.
NVIDIA recommends enough blocks to keep the GPU busy but warns that block size and occupancy interact with resource limits. See [Thread and Block Heuristics](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#thread-and-block-heuristics).

### 10.8 Atomics and serialization
**Symptoms:** hot global counters/buckets, atomic pipeline pressure, performance collapses with contention.
**Treatments:** aggregate per lane/warp/block, shard counters, use hierarchical reduction, repartition hot keys. Preserve deterministic or audit semantics where required.

### 10.9 Shared-memory conflicts and barriers
**Symptoms:** shared accesses serialize, barrier stalls dominate, added staging regresses.
**Treatments:** pad/swizzle layouts, reduce barriers, verify producer/consumer phases, retain staging only when it increases reuse.

### 10.10 Excessive total work inside a healthy kernel
**Symptoms:** NCU metrics look healthy but device time is large; padded capacity dominates active work.
**Treatments:** measure useful work, remove padding/redundant passes, use sparse deltas/frontiers, reuse results, change the work decomposition before instruction tuning.

## 11. Avoidable custom kernels

Profile-first: a named kernel already owns material device time. The library table, when a custom kernel is warranted, and the call-pattern mistakes that make libraries look slow are in [cpu-mental-model-antipatterns.md](cpu-mental-model-antipatterns.md) section 5.

### Symptoms

- A hand-written kernel is a genuine top-device-time entry, and its NCU metrics look healthy.
- The kernel's name or body describes a standard operation: reduce, min/max, arg-max, prefix sum, sort, filter, compact, unique, histogram, segmented variants of those, GEMM, or transpose.
- Performance is fine at the shape it was developed on and poor at skewed, small, or very large shapes.
- It regressed relative to the rest of the program after a GPU or toolkit upgrade.
- Tuning it has already consumed several rounds for diminishing returns.

### Mechanism

The kernel is not slow because of how it was written; it is slow because it exists. It matched one shape and one GPU, then lost on the rest of the distribution and on later architectures. A second cost is invisible in the profile: every future engineer maintains and re-validates it.

### Evidence

- Benchmark the library equivalent on representative shapes as the baseline, in the real call site rather than a microbenchmark.
- Compare across the full shape distribution, including the small and skewed cases.
- Count launches, extra global-memory passes, and temporary bytes on both sides.
- Check whether the custom kernel's advantage is fusion with a neighbor rather than the primitive itself — if so, compare against the fused library sequence.
- When the library measures slower, inspect how it was called (temporary-storage reuse, stream policy, handle, layout) before concluding anything.

### Misdiagnoses

- "Our version is faster" — established once, on one shape, on the previous GPU.
- "The library is slower," when the real cause was per-call temporary-storage allocation, a default execution policy that synchronizes, or a layout mismatch forcing a transpose.
- "It is already written, so keeping it is free."
- "Fusion requires a custom kernel," before checking block-level library primitives inside your own kernel, or an existing fused framework operator.

### Treatments and risks

- Replace the kernel with the library primitive when the library wins or ties; a tie still wins on maintenance and future architectures.
- Keep the kernel but replace its internals with block- and warp-level library primitives when the fusion is the real value.
- Fix the call pattern — reused temporary storage, an explicit stream policy, a persistent handle — before accepting that the library is slower.
- Record the measurement either way — one line in the repository's project memory block, per [report-and-receipts.md](report-and-receipts.md) section 8.

Risk: library implementations may differ in reduction order, tie-breaking, and determinism, and their temporary-storage and stream contracts must fit the surrounding allocator. Both belong in the equivalence contract before the swap.

## 12. Common real-project patterns

Use these as scale examples, not universal thresholds:

- Millions of kernels with nearly all below 50 microseconds indicate transaction fragmentation even if several long kernels also exist.
- Hundreds of thousands of scalar reads can move only megabytes yet serialize an entire host-driven algorithm.
- Most copy calls can be tiny while a few bulk copies carry almost all bytes; count and bandwidth require separate treatments.
- A full large-buffer clear repeated for a sparse delta is an algorithmic traffic problem.
- A loop-invariant `cudaMemGetInfo` query repeated hundreds of thousands of times can become a driver/`ioctl` bottleneck.
- A CPU cleanup or preprocessing phase can dominate outside the CUDA algorithm boundary.
- Correcting inherited CPU thread settings can dwarf kernel tuning.
- Reusing an already-proven certificate can reduce GPU-active time and utilization while correctly improving wall time.
- A secondary-stream experiment can regress because of contention, event/allocator overhead, and greater lifetime even when it creates visible overlap.
- A hand-written reduction or sort that once beat the library on one shape and one GPU can lose severalfold on the current hardware, while looking healthy in every kernel metric.
