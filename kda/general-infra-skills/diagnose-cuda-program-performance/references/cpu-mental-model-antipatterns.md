# CUDA written with a CPU mental model

Source-level catalog. [issue-catalog.md](issue-catalog.md) starts from a profile; this page starts from code you can read before any profiler runs.

## Contents

1. How to use this catalog
2. Source-to-profile signature map
3. Host-side patterns
4. Device-side patterns
5. Reinvented library primitives
6. Fast source reconnaissance
7. Explaining a finding to the author

## 1. How to use this catalog

Most slow CUDA code was not written by someone tuning a kernel. It was written by someone applying a correct CPU picture — sequential control flow, cheap function calls, immediate access to any value, locks for coordination — to a device where none of those assumptions hold. The result is usually correct and usually slow. Two jobs: cheap hypotheses at the start of an investigation, and CUDA review where no profile exists yet.

1. **A source hit is a hypothesis, never a finding.** A `cudaDeviceSynchronize()` called twice per run costs nothing. The same call inside a per-item loop can own the whole runtime. Frequency and critical-path position decide; only measurement supplies them.
2. **Rank by measured wall time, not by how wrong the code looks.**
3. **Confirm each hypothesis against the section-2 signature** before proposing a change. If that signature is absent, the pattern is not the current bottleneck even though the code still contains it.

*Done when* every source hit is a hypothesis paired with its section-2 signature, and none is ranked as a finding.

## 2. Source-to-profile signature map

| Pattern | Predicted profile signature |
|---|---|
| Synchronize after every launch (3.1) | GPU-active bursts separated by gaps, one gap per launch; large `cudaDeviceSynchronize`/`cudaStreamSynchronize` count and host duration |
| Synchronizing error check (3.2) | Same as 3.1, with sync count matching launch count almost exactly |
| Device value drives host control (3.3) | Many `<=8 B` D2H copies; `_local_scalar_dense` or `cudaMemcpy` D2H immediately followed by a GPU gap |
| One launch per item (3.4) | Kernel count in the 10⁴–10⁷ range, nearly all kernels under 10–50 µs, launch API time material |
| Allocate or free in the loop (3.5) | High `cudaMalloc`/`cudaFree`/`cudaMemGetInfo` counts, allocator events, driver `ioctl` in gaps |
| Copy in and out per iteration (3.6) | Alternating H2D/D2H in a loop; bytes far exceeding the working set |
| Host timer without synchronization (3.8) | Implausibly small reported CUDA time; the following sync or copy owns the duration |
| Managed memory as ordinary memory (3.9) | Unified-memory page-fault and migration rows; kernel time varying with access order |
| One thread does the work (4.1) | One kernel, low occupancy, one active warp, duration scaling with the serial loop bound |
| Thread per object with inner loop (4.2) | Low branch efficiency, poor sector-per-request efficiency, large warp duration variance |
| Barrier inflation (4.3) | Barrier stall dominating a kernel with little shared-memory reuse |
| Atomic lock or spin barrier (4.4, 4.5) | Atomic pipeline pressure; duration collapsing as contention rises; occasional hangs |
| Single global counter (4.6) | Atomic throughput limit with runtime proportional to update count, not data size |
| Tiny grid (4.8) | Blocks far below SM count; kernel busy on a fraction of SMs for its whole duration |
| Reinvented primitive (5) | A custom kernel is a genuine top-device-time entry, with healthy-looking metrics and no fusion advantage |

## 3. Host-side patterns

### 3.1 Synchronize after every launch

```cpp
for (int i = 0; i < n; ++i) {
  step_kernel<<<grid, block>>>(...);
  cudaDeviceSynchronize();          // "make sure it finished"
}
```

Framework equivalent: `torch.cuda.synchronize()` inside a loop, or a blocking `cudaMemcpy` used purely as a wait.

A launch is asynchronous so the host can run ahead and keep the queue full. Stream order already guarantees that the next kernel on the same stream sees the previous kernel's writes, so the barrier usually buys nothing.

**Confirm.** Count syncs per iteration and measure the gap immediately after each one.
**Fix direction.** Delete the synchronization and let stream order carry the dependency. Keep one synchronization at the outer boundary where the host genuinely needs results. If the code synchronizes to detect errors, see 3.2.
**Risk.** Errors surface later and at a different call site; exception timing changes.

### 3.2 Error checking that synchronizes

```cpp
#define CHECK(x) do { x; CUDA_CHECK(cudaDeviceSynchronize()); } while (0)
```

Also: leaving `CUDA_LAUNCH_BLOCKING=1` set outside debugging, and calling `cudaGetLastError()` in a way that requires completion.

This is 3.1 wearing a safety label. It converts every launch into a round trip, so the program pays the cost permanently in exchange for a better stack trace during debugging.

**Fix direction.** Check the launch's own return code, which is synchronous and cheap, and check accumulated asynchronous errors once per outer iteration or phase. Keep the synchronizing form behind a debug build flag or an environment variable.
**Risk.** Attribution of an asynchronous fault becomes coarser. State this trade-off rather than removing the check.

### 3.3 A device value drives host control flow

```python
while residual.item() > tol:      # host asks the device a question every iteration
    step(state)
```

```cpp
int count;
cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
if (count > 0) { ... }
```

Also: `bool(tensor)`, `.cpu()`, `.numpy()`, `.tolist()`, `thrust::reduce` returning a host scalar, and a CUB reduction followed by a D2H copy of the result.

On a CPU any value is readable immediately; on a GPU reading a device-produced value forces every operation that value depends on to complete first. Four bytes can serialize milliseconds of queued work. The worked GPU debt / `.item()` example lives in [SKILL.md](../SKILL.md) section 5.

**Confirm.** Bucket D2H copies by size and correlate the small ones with the gap that follows.
**Fix direction.** Keep predicates, counters, convergence flags, and error codes device-resident. Run a fixed iteration count and check convergence every *k* iterations instead of every one. Pack several scalars into one publication. Where the loop structure is stable, move the control into the device.
**Risk.** Termination point, iteration count, exception timing, and logged decisions can change. Deferring a convergence check by *k* iterations changes results and needs the owner's agreement.

### 3.4 One launch per logical item

```cpp
for (const auto& obj : objects) {
  process_kernel<<<1, 128>>>(obj.data, obj.size);   // thousands of tiny launches
}
```

The loop reads as natural parallelism — each object goes to the GPU — but the parallelism is inside each launch, not across them. This also covers the "one kernel per step" habit, where normalize, scale, and add become three launches and two round trips through global memory.

**Fix direction.** Batch items into a packed, segmented, or CSR layout and launch once over all of them. Fuse adjacent elementwise and reduction steps. Once the launch sequence is stable and free of host decisions, CUDA Graphs can remove the remaining submission cost.
**Risk.** Batching changes reduction order, tie-breaking, error timing, and peak memory.

### 3.5 Allocate or free inside the loop

```cpp
for (...) {
  float* tmp; cudaMalloc(&tmp, bytes);
  kernel<<<...>>>(tmp);
  cudaFree(tmp);
}
```

Framework equivalent: constructing a temporary tensor per item, or querying free memory before every transaction.

`cudaMalloc` and `cudaFree` are not `malloc` and `free`. They can enter the driver, synchronize, and manage device-wide state. A caching allocator lowers the frequency but does not make allocation, fragmentation, or stream-lifetime bookkeeping free.

**Fix direction.** Allocate a bounded reusable workspace outside the loop, sized to the maximum shape. Hoist loop-invariant capacity queries to a real phase boundary.
**Risk.** A larger persistent workspace raises peak memory. A cached headroom value can go stale if another process shares the GPU; keep the refresh points explicit.

### 3.6 Copy in and copy out every iteration

```cpp
for (...) {
  cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);
  kernel<<<...>>>(d_in, d_out);
  cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
}
```

Intermediate results that only the next kernel consumes never need to visit the host. The blocking `cudaMemcpy` form also serializes, so this pattern usually carries 3.1 with it.

**Fix direction.** Keep intermediates device-resident for the whole phase and transfer only the inputs and the final outputs. Where an overlapping transfer is genuinely needed, use pinned host memory and a non-default stream, and verify real overlap before adding double buffering.
**Risk.** Longer device-side lifetimes raise peak memory; aliasing a reused buffer can expose races.

### 3.7 Logging, asserts, and progress output in the hot loop

Printing a tensor, formatting a device value into a log line, or asserting on a device-derived condition all read device memory. Each one is an instance of 3.3 hidden behind a call that looks like host-only bookkeeping.

**Fix direction.** Accumulate telemetry on the device and publish it once per phase, or gate it behind a verbosity level that production runs do not enable. Measure with it disabled to bound its contribution. Risk: deferred telemetry changes when a failure is observed.

### 3.8 Timing with a host clock and no synchronization

```cpp
auto t0 = clock::now();
kernel<<<...>>>(...);
auto t1 = clock::now();          // measures submission, not execution
```

The measurement is not slow; it is wrong. The kernel appears to take microseconds and whichever later call synchronizes appears to be the bottleneck. Expect this pattern in the benchmark used to argue a change was an improvement.

**Fix direction.** Synchronize at intentional outer boundaries around the region being timed, or use CUDA events for a stream interval. Time the schedule that production runs. Re-time any prior result produced this way before trusting it. Risk: inner-loop synchronization added for a timer changes the schedule being measured.

### 3.9 Managed memory used as ordinary memory

`cudaMallocManaged` removes the explicit copy, which is exactly why it attracts a CPU picture of memory. Alternating host and device access to the same pages then produces fault-driven migration that is slower and far harder to see than the copy it replaced.

**Fix direction.** Keep it only where the access pattern is genuinely phase-separated, and add prefetch hints at those phase boundaries. Where the working set and direction are known, an explicit copy is predictable and usually faster. Risk: prefetch or explicit copies change first-touch residency and can raise peak device memory.

### 3.10 Host threads used where streams belong

Multiple host threads submitting to the same default stream produce no device concurrency, only contention and nondeterministic ordering.

**Fix direction.** Establish whether device work is actually independent, then express it with streams and events. Before adding either, remove the host synchronization and submission gaps from 3.1 and 3.3 — they are usually the real reason the timeline looks serial. Profile-first treatments live in [issue-catalog.md](issue-catalog.md) section 8. Risk: independent streams can expose races that the default stream had hidden.

## 4. Device-side patterns

These are CPU idioms transplanted inside a kernel. Several are correctness hazards, not just performance problems.

### 4.1 One thread does the work

```cuda
__global__ void process(const T* data, int n, R* out) {
  if (threadIdx.x == 0) {
    for (int i = 0; i < n; ++i) out[0] = combine(out[0], data[i]);
  }
}
```

A CPU function was pasted into a kernel and guarded so it runs once. One thread of one block executes; the rest of the warp is masked and the rest of the device is idle. `<<<1, 1>>>`, or a grid with one block, is the same pattern.

**Fix direction.** Re-express the operation as a parallel reduction, scan, or map over elements — and check section 5 first, because operations written this way are almost always exactly what CUB and Thrust already provide. Risk: a parallel reduction can change accumulation order.

### 4.2 One thread per object with a serial inner loop

```cuda
int obj = blockIdx.x * blockDim.x + threadIdx.x;
for (int k = 0; k < objects[obj].count; ++k) { ... }   // divergent trip count
```

The decomposition mirrors a CPU loop over objects, so each thread walks its own object's elements. Adjacent threads then touch addresses that are far apart, and unequal trip counts make every warp wait for its slowest lane.

**Fix direction.** Assign a warp or block per object and threads per element, or flatten to a segmented layout with one thread per element and segment ids. Risk: segmented layouts change peak memory and the order of per-object updates.

### 4.3 Barrier inflation and misuse of `__syncthreads()`

Three distinct errors travel together:

- **A barrier after every shared-memory write**, added defensively. Each one stalls the block; most are unnecessary because the producing and consuming threads are in the same warp or the data is not shared at all.
- **A barrier inside divergent control flow.** `__syncthreads()` requires every non-exited thread in the block to reach it. Placing it inside an `if` that only some threads take is undefined behavior, not a slow-but-correct choice.
- **A barrier expected to synchronize the whole grid.** `__syncthreads()` is block-scoped. Blocks are not guaranteed to be resident simultaneously, so there is no block-to-block ordering to rely on.

**Fix direction.** Keep barriers only where one group of threads consumes what another produced. Use warp-level primitives when the exchange is within a warp. For grid-wide ordering, launch separate kernels or use cooperative groups with a launch that supports them. Risk: removing a necessary barrier is a data race; cooperative-group grid sync has its own occupancy contract.

### 4.4 Atomics used as a lock

```cuda
while (atomicCAS(&lock, 0, 1) != 0) { }   // spin
critical_section();
atomicExch(&lock, 0);
```

This is the CPU concurrency idiom, and it is worse than slow. Threads in a warp execute together; a lock held by one lane while other lanes of the same warp spin for it can deadlock. Where it does run, it serializes the work the kernel exists to parallelize.

**Fix direction.** Redesign around atomic accumulate, per-lane or per-block private accumulation followed by one merge, or a compaction pass that removes the conflict. Risk: removing the lock without a replacement ordering can change results or expose races.

### 4.5 Spinning on a flag to emulate a barrier or a queue

Polling global memory with `volatile` or `__threadfence()` to wait for another block reproduces 4.4 at grid scope, with the same residency-based deadlock risk.

**Fix direction.** Split the kernel at the synchronization point, or use cooperative groups. A persistent-kernel design is a deliberate choice with its own occupancy contract, not an incremental patch. Risk: kernel splits change when side effects become visible; a persistent kernel can deadlock if occupancy is below the intended resident set.

### 4.6 One global counter for everything

```cuda
atomicAdd(&global_count, 1);   // every thread, every iteration
```

Every thread in the grid serializes on one address. Runtime becomes proportional to the number of updates rather than the amount of data.

**Fix direction.** Aggregate within the warp, then the block, then perform one atomic per block. Shard the counter across many addresses when aggregation does not apply. Preserve deterministic or audit-visible ordering where the contract requires it. Risk: sharded or hierarchical counters can change the visible intermediate totals.

### 4.7 Host allocation and printing habits inside a kernel

Device-side `malloc`, `new`, and `printf` all exist and all cost far more than their host counterparts: the allocator serializes and draws from a fixed heap, and `printf` buffers per thread and forces the output to be drained. They are debugging tools.

**Fix direction.** Pass in a preallocated workspace. Write diagnostics to a device buffer and inspect it after the kernel. Risk: a fixed device heap can OOM; dropping `printf` changes when a fault is observed.

### 4.8 Launch configurations that ignore the device

`<<<1, 256>>>` uses one SM out of many. `<<<n, 1>>>` uses one lane of each warp, wasting 31 of 32. A block size that is not a multiple of the warp size leaves a partial warp in every block.

**Fix direction.** Size the grid so blocks substantially exceed the SM count, use a block size that is a multiple of 32 (commonly 128–512), and grid-stride the loop so one configuration handles any input size. Read the achievable configuration from the device properties rather than hard-coding it. Risk: a larger grid can raise occupancy enough to spill, or change which wave is the tail.

### 4.9 Array-of-structures layout carried over from host code

A `struct Particle { float x, y, z; ... }` array is the natural CPU layout and the wrong device layout: each warp lane reads one field of one struct, so a request that could have fetched 32 contiguous values fetches 32 scattered ones. The same reasoning applies to pointer-chasing structures ported directly from CPU code.

**Fix direction.** Store fields as separate arrays, or stage a tile through shared memory and transpose there. Confirm with sector-per-request efficiency before and after. Risk: structure-of-arrays changes host interop and can raise peak memory during conversion.

## 5. Reinvented library primitives

A team fluent in writing kernels will write one before checking whether NVIDIA already ships a better version. The custom kernel is usually correct, usually slower, and permanently owned by the team. [issue-catalog.md](issue-catalog.md) section 11 is the profile-first landing.

### 5.1 What already exists

| Operation | Use instead |
|---|---|
| Reduce, min/max, arg-min/arg-max | `cub::DeviceReduce`, `thrust::reduce` |
| Prefix sum, running total | `cub::DeviceScan`, `thrust::inclusive_scan` |
| Sort, sort by key | `cub::DeviceRadixSort`, `thrust::sort_by_key` |
| Filter, compact, unique, partition | `cub::DeviceSelect`, `cub::DeviceRunLengthEncode` |
| Histogram, binning | `cub::DeviceHistogram` |
| Per-segment reduce or scan (ragged batches) | `cub::DeviceSegmentedReduce`, `cub::DeviceSegmentedSort` |
| Dense GEMM, batched GEMM, epilogue fusion | cuBLAS, cuBLASLt, CUTLASS |
| Sparse matrix operations | cuSPARSE |
| FFT, convolution via FFT | cuFFT |
| Random number generation | cuRAND |
| Factorization, solve, eigen | cuSOLVER |
| Multi-GPU collectives | NCCL |
| Image and JPEG decode, data loading | nvJPEG, DALI |
| Reduce, scan, sort, exchange *inside your own kernel* | `cub::BlockReduce`, `cub::BlockScan`, `cub::WarpReduce`, `cub::BlockRadixSort` |

The last row matters most for a team that does need a custom kernel: the block- and warp-level building blocks let you keep your kernel and still stop hand-writing its internals. A framework's own operators count as libraries too — a fused operator that already exists beats a new kernel that reproduces it.

### 5.2 When a custom kernel is the right answer

Write one when the evidence supports at least one of these, and say which:

- **Fusion the library cannot express.** Several library calls each round-trip through global memory; one kernel keeps the intermediate in registers or shared memory. This is the most common legitimate reason.
- **A dtype, layout, or sparsity structure outside the library's API.**
- **Measured library overhead that dominates.** Temporary-storage allocation, an extra pass, or launch count matters at your problem size — measured, not assumed.
- **A stream, allocator, or synchronization contract the library cannot honor.** Note that this is often fixable by configuring the library instead.

Absent one of these, a custom kernel that duplicates a primitive is a maintenance liability with a performance regression attached.

### 5.3 Before writing the kernel

1. Search the CUB, Thrust, and framework operator lists for the operation, including its segmented and by-key variants. The operation is often present under unfamiliar terminology — "compaction" for filtering, "run-length encode" for grouping, "segmented" for ragged batches.
2. Benchmark the library call on representative shapes as the baseline. A custom kernel that does not beat it end to end has no case.
3. If it loses, check how it was called before concluding the library is slow. The usual causes are in 5.4.

*Done when* the library equivalent has been named and benchmarked on representative shapes, or a section-5.2 reason is stated.

### 5.4 Calling these libraries correctly

Most "the library was slower" claims come from one of these, all of which the profile shows:

- **CUB's two-call temporary-storage protocol.** The first call reports the required bytes; the second does the work. Allocating that storage on every call reintroduces the allocation churn of 3.5. Allocate once and reuse, or route it through the framework's allocator.
- **Thrust's synchronizing returns.** `thrust::reduce` returns a value to the host, which is 3.3 by definition. Prefer the CUB form that writes the result to a device pointer, and keep the value on the device.
- **Thrust's default execution policy.** It uses the default stream and its own allocator. Pass an explicit stream policy so the call joins your stream instead of serializing against it.
- **cuBLAS handle and layout.** Reusing one handle per stream, and matching the expected column-major layout instead of transposing around every call, both matter more than the kernel selection.

## 6. Fast source reconnaissance

Cheap, runs anywhere, no GPU required. Report counts and whether each hit sits inside a hot loop; then confirm against section 2 before drawing any conclusion.

```bash
# Host synchronization (3.1, 3.2)
rg -n 'cudaDeviceSynchronize|cudaStreamSynchronize|cudaEventSynchronize' -g '!*test*'
rg -n 'torch\.cuda\.synchronize|CUDA_LAUNCH_BLOCKING'
# Device values reaching the host (3.3)
rg -n '\.item\(\)|\.cpu\(\)|\.numpy\(\)|\.tolist\(\)'
rg -n 'cudaMemcpy\(' -g '*.{cu,cuh,cc,cpp,h,hpp}'   # blocking form
# Allocation and capacity queries in loops (3.5)
rg -n 'cudaMalloc|cudaFree|cudaMemGetInfo'
# Managed memory (3.9)
rg -n 'cudaMallocManaged|cudaMemPrefetchAsync'
# Serial or misconfigured kernels (4.1, 4.8)
rg -n 'if\s*\(\s*(threadIdx\.x|tid)\s*==\s*0\s*\)' -g '*.{cu,cuh}'
rg -n '<<<\s*1\s*,|,\s*1\s*>>>' -g '*.{cu,cuh}'
# Barriers, locks, and device-side host habits (4.3, 4.4, 4.7)
rg -n '__syncthreads|__threadfence' -g '*.{cu,cuh}'
rg -n 'atomicCAS|atomicExch' -g '*.{cu,cuh}'
rg -n 'printf|malloc|new\s' -g '*.{cu,cuh}'
# Candidate reinvented primitives (5)
rg -in 'reduce|scan|prefix|sort|histogram|unique|compact' -g '*.{cu,cuh}'
```

Then read structure rather than lines:

- Which loops contain a launch, a copy, or a synchronization? Estimate iteration count per run.
- Is the launch configuration derived from the input size and the device, or hard-coded?
- Do intermediate results visit the host between two kernels that could have passed them on the device?
- Does the project already own a batched or fused path that this call site bypasses?
- Which of the custom kernels duplicate a primitive from section 5.1?

*Done when* each hit reports a count and whether it sits inside a hot loop, and the matching section-2 signature is named before any conclusion.

## 7. Explaining a finding to the author

The author is a competent programmer whose picture of the machine is wrong in one specific place. That is what the explanation has to repair — and if it lands, they will find the next instance themselves, which is worth more than the fix.

- **Lead with the mechanism in plain language, not the label.** "The CPU can't continue until it knows the residual, so it waits for every queued kernel to finish before it can start the next iteration" teaches something. "Host sync bottleneck" does not. Define `D2H`, `occupancy`, or `coalescing` in the sentence where it first appears.
- **Show the arithmetic.** "180 µs per iteration × 40,000 iterations = 7.2 s of the 9.1 s total" makes the finding undeniable and puts the fix in proportion. A pattern with no such number does not belong in the report.
- **Show the evidence.** Name the timeline feature: the gap after this line, the count of these copies, the idle interval between these launches.
- **Separate what is measured from what you expect.** Give a conservative ceiling for the fix and say plainly if it is a guess.
- **Name the one or two patterns that own measured time.** The code was reasonable under the sequential-CPU assumptions in use; the useful correction is those assumptions, delivered once.
- **Propose the smallest change first**, with its risk stated: what could differ in results, ordering, error timing, or memory. Where a change alters behavior the author may be relying on — a deferred convergence check, a batched reduction order — ask before implementing.

*Done when* the report names the mechanism in plain language, shows the arithmetic and the timeline feature, states a conservative ceiling, and names the risk of the smallest change. Report format lives in [report-and-receipts.md](report-and-receipts.md).
