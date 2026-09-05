# Nsight Systems workflow

Nsight Systems answers where wall time goes and what the GPU was doing while the clock ran. It is the first profiler for a program nobody has measured yet. Nsight Compute answers why one kernel is slow and cannot see host code, gaps, launch rate, transfers, or phases. Reach for Compute only after Systems has proven that one kernel owns material wall time — sections [7](#7-verdicts-systems-can-support) and [8](#8-hand-off-to-kernel-level-profiling). First-pass order lives in [SKILL.md](../SKILL.md).

Flags here were current as of 2026-05. Verify a flag against `--help` before treating a mismatch as a skill bug.

## Contents

1. [What Systems answers, and what it cannot](#1-what-systems-answers-and-what-it-cannot)
2. [Start from the analyzer](#2-start-from-the-analyzer)
3. [Capture](#3-capture)
4. [Export to SQLite](#4-export-to-sqlite)
5. [Run the analyzer](#5-run-the-analyzer)
6. [Read the digest](#6-read-the-digest)
7. [Verdicts Systems can support](#7-verdicts-systems-can-support)
8. [Hand off to kernel-level profiling](#8-hand-off-to-kernel-level-profiling)
9. [Footguns](#9-footguns)
10. [Named SQL exceptions](#10-named-sql-exceptions)

## 1. What Systems answers, and what it cannot

Answers, from a single capture: how much of the wall interval the GPU was doing anything at all; which application phase owns each stretch of GPU idle; whether idle is a few long stalls or a flood of tiny ones; which kernels own device time, and which phase launched them; how many launches, copies, syncs, allocations, and queries the host issued, and what they cost the host; whether copies are a count problem or a bytes problem.

Cannot answer, at any capture setting: why a kernel is slow internally (no occupancy, stall reasons, memory-pipe utilization, or per-instruction attribution); whether a kernel's memory accesses coalesce; what a host function computed — only that it was on-CPU or blocked.

A kernel that Systems shows owning 30% of device time is a *candidate* for kernel-level work. Nothing Systems reports says whether that kernel can be made faster.

## 2. Start from the analyzer

Run [`../scripts/nsys_analyze.py`](../scripts/nsys_analyze.py) first. It computes every quantity section [6](#6-read-the-digest) describes, so the first pass costs one command instead of an hour of queries, and the hour goes to the diagnosis.

Quote `digest.txt` in the report before any custom SQL or a reimplementation of this analysis. Section [10](#10-named-sql-exceptions) is the only place a further query is allowed, and only after that quote.

The interval algebra is easy to describe and full of traps. Each of these has been gotten wrong by an agent writing fresh SQL against the same database:

1. Summing concurrent kernel durations instead of merging them, which double-counts every pair of concurrent kernels and can report more GPU-busy time than the window contains.
2. Adding CUDA API duration to kernel duration, when API duration overlaps the GPU work it launched.
3. Adding inclusive NVTX durations across nested ranges, which counts the same wall time once per level of nesting.
4. Attributing a kernel to whatever NVTX range was open on *some* thread at that timestamp, rather than on the thread that launched it.
5. Ranking gaps by length and concluding from the top of that list, when most idle time sits in gaps too short to appear on it.

Reimplementing it costs an hour and usually produces a confidently wrong number.

Speed is the other reason. The script uses prefix sums and binary search rather than nested scans, so a 238 MB export with 1.4 million launches analyzes in roughly 15 seconds; a direct implementation of the same measurements on that file takes about six minutes, which is long enough that an agent starts sampling instead of measuring.

*Done when the first-pass command in section [5](#5-run-the-analyzer) is the next action, and `digest.txt` is the first artifact to quote.*

## 3. Capture

### 3.1 A first capture

Trace CUDA, NVTX, and OS runtime. Leave framework-wide instrumentation off for the capture you intend to quote timings from.

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --cuda-memory-usage=true \
  --force-overwrite=false \
  --output=/path/to/artifacts/run_<something-that-identifies-this-capture> \
  <your program and arguments>
```

`--force-overwrite=false` with a fresh output name every time, per [profiler-environment-setup.md](profiler-environment-setup.md). A capture is evidence: overwriting one silently destroys the baseline a later comparison needs. The SQLite export in section [4](#4-export-to-sqlite) is derived, regenerable, and safe to overwrite.

Sampling and OS-runtime tracing distinguish "the CPU was busy computing" from "the CPU was blocked waiting." The digest attributes idle from NVTX ranges and in-flight CUDA API calls only; those OS-runtime tables stay available for a named exception in section [10](#10-named-sql-exceptions) or the GUI once the digest has named a specific gap.

*Done when a `.nsys-rep` exists under a unique `--output` name.*

### 3.2 Narrow the capture before widening the trace

A full capture of a long program produces a report too large to export and analysis dominated by warmup. Narrow it, in this order of preference:

1. **An NVTX range**, when the program is already annotated: `--capture-range=nvtx --nvtx-capture=<range-name> --capture-range-end=stop`. This is the only option that lines up exactly with a semantic phase.
2. **Time bounds**, when it is not: `--delay=<seconds> --duration=<seconds>`. Verify afterwards that the window landed on steady state and not in initialization.
3. **The profiler API**, when the region is neither annotated nor at a predictable time: `cudaProfilerStart`/`cudaProfilerStop` with `--capture-range=cudaProfilerApi`.

The first digest proceeds without adding ranges. `## No-CUDA time by host API in flight` classifies idle on an unannotated program. Recapture-with-annotation is [nvtx-bootstrap.md](nvtx-bootstrap.md) section 1, and only after that host-API table still leaves material idle unclassified. Discard and recapture when the window missed steady state, the wrong process was traced, or profiled wall time is far worse than clean wall time.

*Done when the window is on steady state, or a discarded capture has been replaced.*

### 3.3 Preserve the evidence

Keep, in one directory per capture: the `.nsys-rep`, the exact command and environment, the profiler version, the application's own correctness or receipt output, and the clean (unprofiled) timing this capture is meant to explain. Treat that directory as read-only afterwards. Write analysis outputs somewhere else — the analyzer defaults to `./nsys-analysis` and never writes beside the report.

*Done when that directory holds the `.nsys-rep`, command, environment, profiler version, and clean timing.*

## 4. Export to SQLite

Analysis reads an exported SQLite database, not the `.nsys-rep` directly:

```bash
nsys export --type sqlite --force-overwrite=true \
  --output /path/to/analysis/run.sqlite /path/to/artifacts/run.nsys-rep
```

The analyzer does this when handed a `.nsys-rep`, writing the export into `--out` and reusing an existing one. Hand-export a large report once: export is the slow step. Budget minutes and several GB for a multi-hundred-MB report. Failures are usually disk space or a version mismatch between the `nsys` that captured and the `nsys` that exports.

*Done when a `.sqlite` exists — analyzer-made or hand-exported.*

## 5. Run the analyzer

`nsys_analyze.py` lives in this skill's `scripts/` folder. Run it from the project under investigation, resolving the script from this skill directory (same command as [SKILL.md](../SKILL.md) section 2). First-pass form has **no** `--kernel`:

```bash
python3 .agents/skills/diagnose-cuda-program-performance/scripts/nsys_analyze.py REPORT --out ARTIFACTS/nsys-analysis
```

If that checkout-relative path is wrong, use the absolute path of `scripts/nsys_analyze.py` inside this skill. `REPORT` is a `.nsys-rep` or an existing `.sqlite`. It writes `digest.txt` (also printed to stdout) and `analysis.json` and needs only the Python standard library.

| Flag | Use it when |
|---|---|
| `--out DIR` | Write `digest.txt` and `analysis.json` here (default `nsys-analysis`). Never next to the capture. |
| `--window {auto,nvtx,cuda,api,trace}` | Default `auto` uses the NVTX span when the program is annotated and the CUDA span otherwise. Choose `cuda` to exclude host setup before the first launch; choose `trace` to include everything the capture saw. |
| `--range NAME` | Restrict every measurement to the union of NVTX ranges with that exact name. Use this instead of recapturing to look at one phase. |
| `--exclude-prefix PREFIX` | Drop framework auto-annotation such as `aten::`, or a per-launch native wrapper, that would otherwise dominate the NVTX rankings. Repeatable. |
| `--min-gap-us` | Raise it to stop the individual-gap list filling with sub-millisecond noise; totals are unaffected. |
| `--top N` | Rows per ranking (default 20). |
| `--kernel` | **Section 8 only.** First-pass command must not pass it. |

*Done when `digest.txt` and `analysis.json` exist from the command above.*

## 6. Read the digest

Read the sections in the order below. Each one exists to stop a wrong conclusion from an earlier one. Quote the heading and its columns.

### 6.1 `## Clocks in this report`

clock, wall s, offset from trace start. Four clocks, and their differences are findings: the CUDA-activity span, the NVTX span, the CUDA API span, and the whole trace. Wall time inside the trace but outside the CUDA span is host work with no GPU work at all. If the CUDA span is much shorter than the NVTX span, the program spends that difference on the host. A first-kernel-to-last-kernel window silently deletes annotated setup that ran before the first launch; check which window was chosen before quoting any percentage.

### 6.2 `## CUDA activity versus wall`

CUDA-activity union, no-CUDA gap, kernel span sum, copy engine sum, activity islands.

- The union is a **merge**, not a sum. Concurrent kernels count once.
- Kernel span sum and copy engine sum are printed for scale and are **not additive** with each other or with the union.
- The gap is time with no GPU activity. Classify it: active host computation, framework dispatch, blocked on a producer, driver query or allocation, lock wait, I/O, compilation, or still unclassified.
- Activity islands counts contiguous stretches of GPU work. Compare it against kernels plus memcpies plus memsets. As the ratio approaches one island per activity, the GPU is running one thing at a time with a hole after each: a launch-rate problem. Far fewer islands than activities means work is packing back-to-back.

### 6.3 `## No-CUDA time by owning NVTX range`

idle excl, wall excl, n, name. Absent when the program is unannotated. This ranking names a host-side bottleneck. It survives idle being spread thinly, so a phase can own enormous idle here while appearing nowhere in the largest-gaps list. When a single application phase tops this table, that phase is the answer.

### 6.4 `## No-CUDA time by host API in flight`

idle s, calls, family, api; plus idle with a CUDA call open vs idle with none open. The same idle, attributed to what the host was doing — and the section that still works when the program has no NVTX.

- **Idle with a CUDA call open** is idle the driver can account for. A `sync` or blocking `copy` at the top is the host waiting on the GPU: real, but a symptom — find what it waits for. `malloc`/`free` is allocator overhead a caching allocator should be hiding. A large figure against `cudaLaunchKernel` itself means submission cannot keep up, confirmed independently of [6.5](#65-no-cuda-gaps-by-duration).
- **Idle with none open** is host compute, I/O, lock wait, or import and compile time. No CUDA-side change touches it.

Per-api seconds are overlaps, not a partition: two threads waiting over the same idle nanosecond both get credit. Compare rows against each other; take totals from the two summary lines.

### 6.5 `## No-CUDA gaps by duration`

bucket, count, seconds, share. Where the idle *seconds* sit by gap size, not where the gap *count* sits. Two programs with identical total idle need opposite fixes:

- Seconds concentrated in a few long gaps are **serial host phases**. Overlap them with GPU work, move them off the critical path, or port them to the device.
- The same seconds spread across thousands of sub-millisecond gaps are **launch-bound**. Batch, fuse, or capture a CUDA graph.

Read this section before the next one. A picket fence of tiny gaps looks healthy in a list of largest gaps.

### 6.6 `## Largest individual no-CUDA gaps`

ms, t+ (s), owner. Individual holes with their owning phase. A single large gap is a blocking call or a genuine host phase, and the owner names it. Skip this list when [6.5](#65-no-cuda-gaps-by-duration) says the seconds are in the small buckets.

### 6.7 `## NVTX ranges by inclusive wall`

incl s, wall excl, busy s, idle incl, busy%, n, name. **Inclusive columns nest and must never be added.** Exclusive time subtracts nested children on the same thread and is the column that reconciles against the window. When a parent's inclusive time vastly exceeds the sum of its children, the difference is work in the parent that no child range covers.

### 6.8 `## Kernels by device time`

**total s, calls, mean us, max ms**, name.

### 6.9 `## Kernel duration distribution`

**bucket, calls, total s**. The digest has this histogram. The histogram decides the treatment; the ranking alone cannot:

- many short calls: fuse, batch, or capture a graph — the per-launch overhead is the cost;
- few long calls: the kernel itself is the cost, and kernel-level profiling is now justified.

A single name can be both, which is why the histogram is a separate heading.

### 6.10 `## Kernel device time by launching NVTX range`

device s, calls, name, plus the `(unattributed)` leftover. Credits each kernel to the host call that launched it, following the CUPTI correlation id, so a kernel is charged to the phase that submitted it even when it ran after that phase returned. Device time per owner overlaps across streams: compare owners against each other, never against the window. The owner is resolved on the *submitting thread*.

`(unattributed)` is the leftover row of this table. A large count is graph replay, a submitting thread with no NVTX, or a launch outside the window. It is not an idle owner.

### 6.11 `## Copies by direction and size`

direction, calls, MiB, engine s, size buckets. Separate count from bytes: a flood of tiny device-to-host copies is a control-flow problem; a few large ones are a payload problem. The fixes share nothing.

### 6.12 `## CUDA runtime API by family`

family, calls, host s. Families include launch, sync, copy, memset, allocation, query, event, graph, virtual-memory management.

### 6.13 `## CUDA runtime API by host time`

host s, calls, name. **API duration overlaps GPU work and must not be added to it.** A large `cudaStreamSynchronize` total is the host waiting for work already counted in the union. This table counts a call's whole duration; [6.4](#64-no-cuda-time-by-host-api-in-flight) counts only the part that overlapped GPU idle; the difference is the part that successfully overlapped GPU work. Sudden large totals in the virtual-memory family mean the program is mapping and unmapping device memory on the critical path.

`## Launch ordinals for kernels matching '…'` appears only after the command in section [8](#8-hand-off-to-kernel-level-profiling).

*Done when each heading the digest printed has been read in this order, with its columns quoted.*

## 7. Verdicts Systems can support

| Digest evidence | Verdict | Where to go next |
|---|---|---|
| Union is a high fraction of the window, kernels look healthy, and the algorithm still does redundant or padded work | Algorithmic work volume | [issue-catalog.md](issue-catalog.md) section 7 |
| Union is a high fraction of the window, and a few kernel names own most device time | GPU-bound on specific kernels | After [SKILL.md](../SKILL.md) section 7: [8](#8-hand-off-to-kernel-level-profiling) |
| Large gap, seconds in the long buckets, one phase owning it, host active during it | Host-bound on a serial phase | [issue-catalog.md](issue-catalog.md) section 6, then [cpu-mental-model-antipatterns.md](cpu-mental-model-antipatterns.md) |
| Large gap, seconds in the sub-millisecond buckets, islands ≈ kernel count, launch family large | Launch-bound | [issue-catalog.md](issue-catalog.md) section 3 |
| Large gap, host blocked in sync during it, small D2H copies frequent | Synchronization-bound | [issue-catalog.md](issue-catalog.md) sections 2 and 4 |
| Copy engine time material, or bytes dominated by a few large transfers | Transfer-bound | [issue-catalog.md](issue-catalog.md) section 4 |
| Allocation, query, or virtual-memory families material | Allocator-bound | [issue-catalog.md](issue-catalog.md) section 5 |
| Host-API idle table still leaves material idle unclassified | Idle owner unknown after host-API attribution | [nvtx-bootstrap.md](nvtx-bootstrap.md) section 1 |

Report unclassified critical-path time as a quantity rather than dropping it. `(unattributed)` in [6.10](#610-kernel-device-time-by-launching-nvtx-range) is the leftover kernel-owner row, not a row of this table. The digest footer that says "annotate the submitting thread" is a later-pass hint for that leftover row, not an idle-owner trigger and not a reason to skip the first digest.

*Done when one row is chosen and its Evidence/Misdiagnosis page is opened. Return to [SKILL.md](../SKILL.md) section 6. Do not apply a Treatment and do not open section 8 until section 7 of SKILL.md is answered.*

## 8. Hand off to kernel-level profiling

Only after the digest shows one kernel owning material wall time — not device time share alone, but a share of the window large enough that making that kernel free would matter. Take the exact kernel name and a launch ordinal whose grid matches production from `--kernel`:

```bash
python3 .agents/skills/diagnose-cuda-program-performance/scripts/nsys_analyze.py \
  REPORT --out ARTIFACTS/nsys-analysis --kernel <substring>
```

That prints `## Launch ordinals for kernels matching '…'`. Ordinals restart per exact kernel name. Pass the exact name from that table to `ncu -k`, or the ordinal will select a different kernel. Early launches are usually warmup; pick one whose grid matches the production shape. Then see [profiling-and-attribution.md](profiling-and-attribution.md) section 10, or the `ncu-report` skill if it is installed.

*Done when a named kernel plus a digest launch ordinal are in hand, or this section is skipped because no kernel owns material wall time.*

## 9. Footguns

- **The window decides every percentage.** Check [6.1](#61-clocks-in-this-report) before quoting one.
- **Ranking gaps by size answers a different question than where the idle is.** Read [6.5](#65-no-cuda-gaps-by-duration) first, every time.
- **Never add overlapping or nested totals.** Not API to kernel, not inclusive NVTX across levels, not per-owner device time to the window, not kernel span sum to copy engine sum.
- **A version mismatch between capture and analysis silently costs the report.** Record the profiler version with the capture.
- **The profiler's environment leaks into the application.** When profiled wall time is far worse than clean wall time, suspect an injected `LD_LIBRARY_PATH` or preload.
- **Framework auto-annotation buries the application's own ranges.** Use `--exclude-prefix`, and check the range count before concluding a phase is absent.
- **Per-launch native wrappers do the same thing.** Annotate phases, not launches.
- **CUDA graphs break kernel attribution.** Under graph replay the launch is the graph, so kernels land in `(unattributed)`.
- **Kernel names are short names.** Templated kernels that differ only in template arguments collapse to one row.
- **Thread count in the environment changes the profile.** Record it with the capture.
- **A phase that is 100% GPU-busy still may not be GPU-bound.** It may be one long kernel that is itself underfilling the device.

## 10. Named SQL exceptions

Quote `digest.txt` in the report first. After that quote, one of these questions may add a query against the SQLite export:

- correlating GPU activity with a subsystem the digest does not model (OS-runtime calls, page faults, a custom NVTX domain);
- per-stream or per-device breakdowns, which the digest merges;
- reconstructing an exact timeline ordering for a suspected race or dependency stall;
- following one specific gap or one specific launch to the API calls surrounding it;
- checking a number the digest reports against `nsys stats` or the application's own timing.

Keep the interval rules from section [9](#9-footguns). Cross-check every overlapping number against the digest so a disagreement surfaces while both sides are still in hand. If a query answers a question worth asking of the next capture too, add it to [`../scripts/nsys_analyze.py`](../scripts/nsys_analyze.py).

*Done when `digest.txt` is quoted, and any query is one of the exceptions above whose overlapping numbers match the digest.*
