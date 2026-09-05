---
name: torch-profile-reading
description: Analyzes PyTorch profiler Chrome/Kineto JSON traces (export_chrome_trace, tensorboard_trace_handler) for GPU utilization, idle gaps, kernel hotspots, sync/memcpy/NCCL issues, and launch latency. Use when a torch profiler trace already exists or can be captured — a .trace.json, .json.gz, Kineto or chrome trace, or a tensorboard_trace_handler directory. Not for .nsys-rep or .ncu-rep reports, and not for hardware counters such as occupancy, bandwidth, or Tensor Core utilization, which a Chrome trace does not contain.
---

# Torch Profile Reading

Read **Chrome/Kineto JSON** from `torch.profiler` — not raw Perfetto `.pftrace` files. The bundled script normalizes events and prints Nsight-like summaries.

Triage is text and JSON only. CSV, SQLite, PNG, Markdown, and HTML exist for the drill-down in step 5 of the workflow below; requesting them up front materializes every event and can cost minutes on a large trace.

## Trace inputs

| Source | Typical path |
|--------|----------------|
| `prof.export_chrome_trace(path)` | `{trace_name}.trace.json` |
| Training or inference entry point | Project-specific profiler flags or `torch.profiler` setup |
| Compressed | `.json.gz` supported |

Timestamps in the file are **microseconds**; the tool reports **milliseconds** unless a column says `_us`.

## Run the analyzer

`SCRIPT` below is [`scripts/torch_prof_hotspot.py`](scripts/torch_prof_hotspot.py) inside this
skill folder. Set it to that absolute path once, since the working directory is the project
under investigation, not the skill:

```bash
SCRIPT=/absolute/path/to/torch-profile-reading/scripts/torch_prof_hotspot.py
python3 "$SCRIPT" TRACE.json --mode text --mode json --top 30 --out /tmp/hotspot_run
```

The script uses the standard library plus optional `orjson`; matplotlib is needed only
for `--mode png`.

### Large traces

For uncompressed traces of at least 256 MiB, `text`, `agent`, and `json` summary runs automatically use the mmap-based compact scanner. A 1 GiB trace should take only a few seconds and bounded memory.

Do **not** request `csv`, `sqlite`, `png`, `md`, or `html` during initial triage. These modes require every event to be materialized and can take minutes or produce very large artifacts. `--zoom`, `--query-regex`, `--inspect`, `--dump-regex`, `--select`, SQL options, compressed `.json.gz`, and `--full` also select the full parser.

Compact summaries intentionally omit CPU operators, duration percentiles, launch correlation, counters, flows, and detailed `agent.*` findings. Use full mode only for small traces or when event-level data is essential:

```bash
python3 $SCRIPT TRACE.json --full --mode sqlite --out /tmp/hotspot_full
```

Malformed Kineto event names may contain invalid UTF-8, control bytes, or unescaped quotes. The full parser repairs those name fields; do not preprocess or rewrite the original trace.

## Agent workflow

Follow this order unless the user asks for something narrower:

1. **Fast triage + JSON** — run `--mode text --mode json` once (add `--top 50` for wide traces).
   *Done when you know per-device utilization, the largest idle gap in ms, and whether the run
   used the compact scanner (the report prints its analysis mode).*
2. **Act on summary signals** (in order):
   - `gpu_utilization` and `gpu_idle_gaps`
   - `top_gpu_kernels`, `top_cuda_runtime`, `sync_hotspots`, `launch_latency_summary`
   - `recommendations`

   *Done when the dominant cost is one of: GPU-busy work, idle gaps, or launch overhead —
   with the ms that back the choice.*
3. **Separate stage behavior** — compare GPU-active compute time with CPU-heavy
   decode/export spans. Whole-trace utilization can hide a healthy GPU stage behind
   CPU postprocessing. *Done when the number you quote covers the stage the user cares
   about, not the whole file.*
4. **Attribute large idle gaps** — inspect long `python_function` / `cpu_op` spans at the same timestamps, then read the named source. Treat nested CPU durations as overlapping, not additive. *Done when each of the top gaps has a named CPU span or an explicit "unattributed".*
5. **Full drill-down only when justified** — use `--full`, SQLite recipes, event selection, or a timeline after confirming the trace size and expected cost (see [reference.md](reference.md)). *Done when the drill-down answers the specific question that step 2–4 could not.*

For unknown event names, start with `--inspect` or `--dump-regex 'pattern'`.

List recipes or the embedded field guide without a trace:

```bash
python3 $SCRIPT --recipes
python3 $SCRIPT --field-guide
```

## Multi-rank NCCL diagnosis

A long NCCL kernel on one rank does not prove that the transfer is slow. It often means
that rank entered the collective early and waited inside NCCL for a straggler.

Compare the same kernel ordinal across traces from the same distributed run:

```bash
RANK_SCRIPT=/absolute/path/to/torch-profile-reading/scripts/torch_prof_rank_skew.py
python3 "$RANK_SCRIPT" rank0.trace.json rank1.trace.json rank2.trace.json rank3.trace.json \
  --kernel-regex 'ncclDevKernel_SendRecv' --top 20
```

Interpret one aligned collective as follows:

- Similar start times and long durations on every rank indicate real collective or
  topology cost.
- Early ranks with long durations plus one late rank with a short duration indicate
  arrival skew. Investigate work before the collective on the late rank.
- Use the late rank's duration / completion tail as the better transfer-cost estimate;
  do not attribute the early ranks' wait time to `all_to_all_single`.

Inspect the decisive ordinal to attribute the straggler:

```bash
MATCH_INDEX=123  # Replace with an index reported by the ranking command.
python3 "$RANK_SCRIPT" rank*.trace.json \
  --kernel-regex 'ncclDevKernel_SendRecv' --inspect-index "$MATCH_INDEX"
```

Look for long `python_function` / `cpu_op` spans and nested CUDA synchronization during
the arrival gap and the default two-second lookback. Increase `--lookback-ms` when the
enclosing operation starts earlier. Innocent-looking operators such as `aten::gelu` or
`aten::index` can contain allocator event reclamation
(`cudaEventSynchronize` / `cudaStreamSynchronize`) caused by earlier tensor lifetimes.
Fix the memory lifetime or rank-local work that creates the skew, then reprofile every
rank. Verify both collective total/max and end-to-end stage latency; a fix can merely
move allocator synchronization to the next large allocation.

Ordinal alignment requires the same capture window, control flow, and exact kernel
filter on every rank. Different match counts are a warning that alignment may become
invalid after the first missing event.

Unlike the hotspot script, the rank-skew script rejects `.json.gz`; decompress every rank
trace first.

## What to conclude (and what not to)

**Timeline GPU utilization** = fraction of window time with at least one GPU kernel/memcpy/NCCL active on that device. It is **not** SM occupancy, memory bandwidth, or Tensor Core utilization.

| Signal | Likely cause | Next step |
|--------|----------------|-----------|
| Low util + large idle gaps | CPU bound, sync, dataloader, serial launches | `gap_context`, `cpu_threads` recipe, `--select` on gap bookmark |
| High sync + low GPU util | `.item()`, logging CUDA tensors, explicit sync | Correlate sync calls with idle gaps and long CPU spans |
| High sync + high GPU util | Host waiting for real GPU work | Optimize dominant kernels first; sync time is not additive overhead |
| Many tiny kernels | Python dispatch, unfused ops | `tiny_kernels`, consider compile/graphs/batching |
| Material `top_gpu_memory` | H2D/D2H, layout copies | `memcpy`, pinned memory / overlap |
| High launch p95 | Host overhead, allocator | `kernel_launch_pairs`, runtime top list |
| NCCL dominates symmetrically | Collective/topology cost or poor overlap | filter with `--query-regex nccl`; inspect topology and overlap |
| NCCL long only on early ranks | Late-rank arrival skew | compare the same ordinal across ranks, then inspect the late rank |

A Chrome trace carries no hardware counters. When the question needs them, go to
**Nsight Systems** first to confirm which interval and which kernel actually dominate the
timeline, and only then to **Nsight Compute** for that one kernel. Jumping straight to
Nsight Compute profiles a kernel you have not yet shown to matter.

## CI / regression gates

Exit code **2** when guards fail (combine with normal analysis):

```bash
python3 $SCRIPT trace.json --mode json \
  --require-gpu-util 75 \
  --max-idle-gap-ms 5 \
  --fail-on-high-finding
```

## Output artifacts (`--out PREFIX`)

| Artifact | Role |
|----------|------|
| `PREFIX.json` | Compact summary by default for large traces; full report with `--full` |
| `PREFIX.events.csv` | Normalized slices (prefer over raw trace JSON) |
| `PREFIX.agent_*.csv` | Tracks, findings, bookmarks, flows, counters |
| `PREFIX.sqlite` | SQL console (`events`, `args`, `flows`, `findings`, …) |
| `PREFIX.md` / `.html` | Human-readable rollup |
| `PREFIX.timeline.png` | Lane timeline (`*.timeline.zoom.png` with `--zoom`) |

## Reporting back to the user

Structure answers as:

1. **Window** — trace duration and any `--zoom` used
2. **GPU health** — per-device util %, worst idle gap(s)
3. **Top offenders** — kernels, runtime, sync/memcpy/NCCL (with ms and call counts)
4. **Root-cause hypothesis** — tie long idle windows to stage/function spans and source behavior
5. **Actionable fixes** — concrete project-relative code or configuration changes

Distinguish wall time from nested event totals, and do not label synchronization as avoidable overhead without checking concurrent GPU utilization. Do not dump entire JSON traces into chat; cite only decisive rows and event ids.

## Additional resources

- SQLite schema, recipes, and JSON keys: [reference.md](reference.md)
- Worked command sequences: [examples.md](examples.md)
