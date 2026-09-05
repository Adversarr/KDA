# Final Report Template

The report is the deliverable. Everything else (`.ncu-rep`, Python artifacts, CSVs) is evidence. Structure matters: a busy reader should see the top findings in 30 seconds and be able to drill into details if they want.

Save as `$PROFILE_RUN_DIR/REPORT.md`.

---

## Template

```markdown
# `<kernel_name>` Profiling Report

**Kernel:** `<exact kernel name or template instantiation>`
**GPU:** `<device__attribute_display_name>` (`<SM count>` SMs, CC `<major>.<minor>`) — read from the report, not assumed
**Nsight Compute:** `<capture version>` captured, `<reader version>` read
**Capture token / regex:** `<kernel_token>` / `regex:.*<kernel_token>.*`
**Profile date:** YYYY-MM-DD
**Run directory:** `profile/<run_name>/`
**Scope:** this is one launch. It says nothing about how often the kernel runs; if the question is end-to-end time, that answer needs Nsight Systems.

---

## 0. Profiling setup

> How exactly did we get these numbers? Required for reproducibility.

- Prerequisite: correctness was already established before entering profiling.
- Profile driver: `profile/<run_name>/target/profile_<kernel>.py` — what it imports, how it compiles or JITs the kernel, and whether it performs a warmup launch.
- Workloads: which real tensors / shapes were used. Cite the workload UUID or shape tuple.
- Dispatch paths covered: list each specialization or generated-kernel family profiled.
- Metric-name caveats: which logical fields resolved to a non-default metric name, and which were absent from this report entirely. Absent is not zero — list it rather than reporting a number for it.

Minimal runnable command listing:

    # Warmup if needed
    python3 profile/<run_name>/target/profile_<kernel>.py --warmup-only [args]

    # Profile (full + PM)
    ncu --set full --section PmSampling --section PmSampling_WarpStates \
        -k "regex:.*<kernel_token>.*" -c 1 \
        -o profile/<run_name>/reports/full_<tag> \
        python3 profile/<run_name>/target/profile_<kernel>.py [args]

    # Profile (source-level). There is no `source` set on every install, and
    # --import-source yes is what makes source_info(pc) resolve to lines at all.
    ncu --section SourceCounters --import-source yes \
        -k "regex:.*<kernel_token>.*" -c 1 \
        -o profile/<run_name>/reports/source_<tag> \
        python3 profile/<run_name>/target/profile_<kernel>.py [args]

### Artifacts

    profile/<run_name>/
    ├── REPORT.md                       ← this file
    ├── target/...                      ← Python profile driver and notes
    ├── reports/                        ← raw .ncu-rep files (re-openable with ncu-ui)
    └── analysis/                       ← scripts + extracted metrics

---

## 1. Headline numbers

> A single table that tells the whole story at a glance.

| Metric | `<tag1>` | `<tag2>` | Source |
|---|---:|---:|---|
| **Duration** | X µs | Y µs | `gpu__time_duration.sum` |
| SM throughput (% peak) | X% | Y% | `sm__throughput.avg.pct_of_peak_sustained_elapsed` |
| Memory throughput (% peak) | X% | Y% | `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed` |
| DRAM throughput (% peak) | X% | Y% | `dram__bytes_read.sum.pct_of_peak_sustained_elapsed` |
| L1 hit rate | X% | Y% | `l1tex__t_sector_hit_rate.pct` |
| L2 hit rate | X% | Y% | `lts__t_sector_hit_rate.pct` |
| Tensor Core usage | X% | Y% | `sm__pipe_tensor_cycles_active.*` |
| Reg / thread | X | Y | `launch__registers_per_thread` |
| Theoretical / Achieved occupancy | X% / Y% | ... | |
| Waves / SM | X | Y | `launch__waves_per_multiprocessor` |

**One-line read:** <"The kernel runs at X% of peak SM throughput — it's latency-bound on Y, not DRAM-BW-bound."> — this is the punchline.

---

## 2. Per-dimension analysis

> Walk through the six analysis dimensions, cite metrics, state findings.

### 2.1 SM occupancy & launch geometry
<grid size, block size, waves/SM, occupancy, register/shared-mem limits, wave math>

### 2.2 Thread-block balance (tail effect)
<per-SM active cycles, PM timeline shape, input distribution imbalance ratios>

### 2.3 Instruction-level stall analysis
<stall breakdown %, top source-line hotspots (cite file:line + samples + stall type)>

### 2.4 Tensor Core utilization
<value or "0%, n/a">

### 2.5 SM utilization timeline
<shape: flat-high / flat-low / tail / sawtooth — reference the ASCII plot in analysis/>

### 2.6 Memory access pattern
<sectors/request, L1/L2 hits, DRAM throughput, store efficiency, register spill>

### 2.7 Additional findings
<items from NCU rule engine not otherwise mentioned — each with the rule's `Est. Speedup: X%`>

---

## 3. Summary diagnosis

| Factor | `<tag1>` | `<tag2>` | Impact |
|---|---|---|---|
| <factor 1> | <status> | <status> | <ranked impact> |
| <factor 2> | ... | ... | ... |

---

## 4. Optimization directions (ranked by impact)

> Each priority: name the change, cite evidence, estimate magnitude, flag effort.

### Priority 1 — <one-line name>

<what to do, concretely, with line numbers / function names from the existing kernel>

**Evidence:**
- <metric + value>
- <NCU rule + est. speedup>

**Expected impact:** <X% on this kernel's own duration>, <which workloads benefit>. State an end-to-end number only if an Nsight Systems capture supplied this kernel's share of wall time; an `.ncu-rep` is one launch and cannot say how often the kernel runs.

**Effort:** <low/medium/high + rough description of the code change>

### Priority 2 — ...

<same structure>

### Priority 3 — ...

(Stop at 3-5. More dilutes the signal.)

---

## 5. Confidence & caveats

- What I'm sure about: <list>
- What I'm uncertain about: <list + what would resolve the uncertainty>
- Anything the profile couldn't answer that the user should know: <list>

---

## 6. Reproduction

    cd /abs/path/to/repo
    export PROFILE_RUN_DIR=profile/<run_name>
    <one-block runnable script that warms up the Python runner, runs ncu, and parses>
```

---

## Style rules

- **Cite specific metric values for every claim.** "SM throughput X.X%" (with the actual number from your report) > "SM throughput is low".
- **Name files and line numbers.** Point to the generated or captured source file/line when available, and always name the Python profile driver used to produce the report.
- **Use NCU's own estimates.** Rule-engine `Est. Speedup: X%` numbers are usually in the right ballpark — use them instead of guessing. Never sum them; they overlap. Where a rule gives no estimate, write "no estimate" rather than inventing one.
- **Rank by magnitude.** Fix the 50% problem before the 5% problem — but keep the un-estimated rules in the list, because that is where Small Grid and its kind live.
- **Keep the top-line summary dense.** A reader should be able to get the #1 finding in 10 seconds of reading.
- **Link to artifacts.** Don't paste huge tables into the prose — link to `analysis/compare_<tag1>_vs_<tag2>.txt` etc.

## Anti-patterns

- ❌ Generic advice without evidence ("you might consider using shared memory").
- ❌ More than 5 "priorities" — you're probably padding.
- ❌ Re-running the same profile with different tags and copy-pasting the same analysis — consolidate.
- ❌ Reporting from the CLI table directly. Extract, interpret, write — don't dump.
- ❌ Omitting the setup section. Without it, nobody can reproduce or trust the numbers.
