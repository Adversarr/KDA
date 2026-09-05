# Torch profile reference

## SQLite schema (`PREFIX.sqlite`)

Created by `--mode sqlite` or implicitly when using `--recipe`, `--sql`, or `--sql-file`.

| Table | Contents |
|-------|----------|
| `events` | Normalized slices: `id`, `kind`, `name`, `lane`, `ts_us`, `dur_us`, `self_us`, `device`, `stream`, `corr_id`, `args_json` |
| `args` | Flattened `events.args` key/value rows |
| `counters` | Counter samples (C/ph=i/R/P style trace events) |
| `counter_args` | Args for counters |
| `instants` | Instant events |
| `flows` | Launch→kernel pairs (`latency_us`, `confidence`) |
| `track_summary` | Per-lane busy time and util in window |
| `phases` | Step/phase slices with nested GPU util |
| `findings` | Heuristic findings (severity, recommendation) |
| `bookmarks` | Suggested zoom ranges |
| `gap_context` | CPU/GPU events near GPU idle gaps |

Event `kind` values include: `GPU_KERNEL`, `GPU_MEMCPY`, `GPU_MEMSET`, `NCCL`, `CUDA_RUNTIME`, `CPU_OP`, `CPU_NCCL`, `OTHER`.

Recipe queries accept `:limit` (bound to `--top`). Ad-hoc `--sql` uses the same limit when printing.

## Cross-rank collective fields

`scripts/torch_prof_rank_skew.py` aligns matching GPU kernels by timestamp order and
reports:

- `arrival_skew_ms` — latest start minus earliest start across ranks.
- `latest_rank` — rank that entered the collective last.
- `latest_duration_ms` — duration observed by the last-arriving rank.
- `max_duration_ms` — longest rank-local NCCL duration, including peer wait.
- `completion_tail_ms` — last completion minus latest arrival.

For a synchronized collective, early-rank duration is approximately arrival wait plus
the completion tail. This is an attribution aid, not a network-bandwidth measurement:
NCCL algorithm behavior and GPU scheduling can add rank-local differences.

Alignment is trustworthy only while all traces contain the same matching-kernel
sequence. Use a narrow exact-name regex and traces from one run. A count mismatch,
different profiler schedule, divergent branch, or unrelated kernel matching the same
regex can shift subsequent ordinals.

## Built-in SQL recipes

Core recipes (also `name_contains_*` per heuristic pattern):

| Recipe | Purpose |
|--------|---------|
| `top_gpu_kernels` | Sum/avg/max ms by kernel name |
| `top_cuda_runtime` | CUDA API time |
| `gpu_streams` | Per device/stream activity |
| `tiny_kernels` | Kernels with `dur_us < 50` |
| `sync_calls` | Synchronize / alloc / memcpy-like runtime |
| `memcpy` | Memcpy/memset/copy events |
| `steps` | ProfilerStep / iteration-like names |
| `longest_events` | Longest slices by duration |
| `arg_keys` | Most common arg keys |
| `shape_args` | Shape/dtype/device args joined to events |
| `correlations` | Shared `corr_id` groups |
| `cpu_threads` | CPU-side time by process/thread |
| `kernel_launch_pairs` | Rows from `flows` |
| `findings` | Precomputed findings table |
| `bookmarks` | Precomputed bookmarks |
| `counter_summary` | Counter track stats |

## JSON report keys (`PREFIX.json`)

Top-level sections from `build_report`:

- `trace` — raw/complete/selected counts, window bounds, durations  
- `counts_by_kind` — event kind histogram  
- `gpu_utilization` — per device: `util_percent`, `busy_ms`, `window_ms`  
- `gpu_idle_gaps` — per device: ranked gaps with `dur_ms`, neighbors  
- `top_cpu_ops`, `top_cuda_runtime`, `top_gpu_kernels`, `top_gpu_memory`  
- `sync_hotspots`, `launch_latency_summary`, `top_launch_latencies`  
- `debug` — processes, threads, devices, streams  
- `recommendations` — short text hints  

`agent` object (when `--mode json`):

- `findings`, `bookmarks`, `track_summary`, `phases`, `gap_context`  
- `flow_count`, `counter_count`, `instant_count`  
- `available_sql_recipes`  

## CLI flags (quick index)

| Flag | Effect |
|------|--------|
| `--mode` | Repeatable: `text`, `json`, `csv`, `png`, `agent`, `sqlite`, `md`, `html` |
| `--out PREFIX` | Output stem (default `hotspot_report`) |
| `--top N` | Row limits for tables and recipe `:limit` |
| `--zoom START:END` | Relative window from trace start |
| `--query-regex` | Filter events before analysis |
| `--gap-threshold-us` | Minimum GPU idle gap to report |
| `--inspect` | Schema/category/lane/arg-key dump |
| `--dump-regex` / `--dump-limit` | Print matching normalized events |
| `--select ID` | Event detail + same-lane context |
| `--recipe NAME` | Run built-in SQL (repeatable) |
| `--sql` / `--sql-file` | Ad-hoc SQL |
| `--require-gpu-util` / `--max-idle-gap-ms` / `--fail-on-high-finding` | Exit 2 on failure |

## Perfetto UI mapping

Same content as `python3 torch_prof_hotspot.py --field-guide`:

- Timeline overview → text/json util + PNG  
- Track list → `agent_tracks.csv`, `track_summary`  
- Slice details → `--select`, `events`/`args`  
- SQL → `--mode sqlite`, recipes  
- Search → `--query-regex`, `--dump-regex`, SQL LIKE on `events`  
- Counters / flows / bookmarks → SQLite tables + agent CSVs  

## Application trace capture

Applications may expose helper functions or command-line flags around `torch.profiler`:

- A trace mode and output-directory option commonly write `{trace_name}.trace.json`.
- Ensure `ProfilerActivity.CUDA` is active on GPU runs; otherwise `gpu_utilization` may be empty and recommendations will say so.
