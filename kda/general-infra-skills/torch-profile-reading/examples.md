# Examples

Replace `TRACE.json` with your Chrome trace path. `$P` and `$R` below are the two scripts
in this skill's own `scripts/` folder; set them to absolute paths, since the working
directory is the project under investigation:

```bash
P=/absolute/path/to/torch-profile-reading/scripts/torch_prof_hotspot.py
R=/absolute/path/to/torch-profile-reading/scripts/torch_prof_rank_skew.py
```

Triage first — text and JSON only, per the workflow in [SKILL.md](SKILL.md):

```bash
OUT=/tmp/my_hotspot
python3 "$P" TRACE.json --mode text --mode json --out "$OUT"
```

Then read `/tmp/my_hotspot.json`. Every example below this point is a drill-down: run it
only after triage has pointed at something specific.

## Capture a trace from an application

Use the application's profiler flags when available, or add a standard
`torch.profiler` schedule and write a Chrome/Kineto JSON — for example
`export_chrome_trace("./profile_out/my_run.trace.json")` or
`tensorboard_trace_handler("./profile_out")`.

Analyze the written file:

```bash
python3 "$P" ./profile_out/my_run.trace.json \
  --mode agent --mode sqlite \
  --recipe top_gpu_kernels --recipe tiny_kernels --recipe sync_calls \
  --out ./profile_out/my_run_hotspot
```

## Zoom a suspicious gap

If `agent.bookmarks` or `gpu_idle_gaps` points to ~80–120 ms:

```bash
python3 "$P" TRACE.json --mode agent --mode png \
  --zoom 80ms:120ms --out /tmp/gap_zoom
```

## Ad-hoc SQL

```bash
python3 "$P" TRACE.json --mode sqlite --sql "
  select kind, name, count(*) calls, round(sum(dur_us)/1000.0, 3) total_ms
  from events
  group by kind, name
  order by sum(dur_us) desc
  limit 25
"
```

## Filter distributed or attention ops

```bash
python3 "$P" TRACE.json --mode text --query-regex 'nccl|flash|sdpa|matmul' --top 40
```

## Diagnose cross-rank collective skew

First rank the worst arrival skews for one exact NCCL kernel. This script rejects
`.json.gz`, so decompress every rank trace first:

```bash
python3 "$R" ./profile_out/run_rank{0,1,2,3}.trace.json \
  --kernel-regex 'ncclDevKernel_SendRecv' --top 20 \
  --json-out /tmp/rank_skew.json
```

If several ranks show a long kernel at the same index while another rank starts later
and finishes quickly, inspect the late rank during that arrival gap:

```bash
MATCH_INDEX=123  # Replace with an index reported by the ranking command.
python3 "$R" ./profile_out/run_rank{0,1,2,3}.trace.json \
  --kernel-regex 'ncclDevKernel_SendRecv' \
  --inspect-index "$MATCH_INDEX" --context-top 40
```

The late-arriving rank is the straggler. Attribute its delay to the reported CPU/CUDA spans
before changing the collective. After the fix, capture all ranks again and compare:

- arrival skew and maximum NCCL duration,
- summed NCCL time from `torch_prof_hotspot.py`,
- end-to-end workload or stage latency,
- whether allocator synchronization moved to a later operator.

## Inspect one event

After `longest_events` or CSV shows `id=12345`:

```bash
python3 "$P" TRACE.json --select 12345
```
