# Profile Collection Commands

This document lists the exact `ncu` commands to run for Python DSL profile drivers and generated CUDA kernels.

---

## Prerequisites recap

- `ncu` available on `PATH`
- writable `HOME`
- a Python profile driver under `$PROFILE_RUN_DIR/target/`
- a kernel token chosen from the Python-level symbol name or another stable identifier
- a validated selector, usually `regex:.*<kernel_token>.*`

Correctness is assumed to be settled before this phase begins.

Quick permission test:

```bash
ncu --section SpeedOfLight -k "regex:.*YOUR_KERNEL_TOKEN.*" -c 1 \
    python3 "$PROFILE_RUN_DIR/target/profile_<kernel>.py" [args]
```

If you see `ERR_NVGPUCTRPERM`, see [`09-common-issues.md`](09-common-issues.md).

---

## Discover the regex token first

The default discovery path for DSL-generated kernels is:

1. start from the Python entrypoint name, for example `some_kernel`
2. build `regex:.*some_kernel.*`
3. run a cheap inspection or first collection pass
4. confirm the captured demangled name in the resulting report

Use emitted/generated CUDA source or runtime debug output only if the obvious Python token is not specific enough.

---

## Recipe 0: Confirm what this install actually has

Set names, section names, and metric names all vary by Nsight Compute version and by architecture. Run these three before copying any recipe below, and believe them over this document:

```bash
ncu --version
ncu --list-sets        # which --set values exist here
ncu --list-sections    # which --section values exist here
```

The recipes below were written for B200 / sm_100 and a 2026.x Nsight Compute. On a 2024.1 install the set list is `basic`, `detailed`, `full`, `nvlink`, `pmsampling`, `roofline` — note there is **no `source` set** — and `full` already includes both `SourceCounters` and `PmSampling`. If a set named in this document does not exist, do not conclude the data is unavailable; find the section that carries it.

---

## Recipe 1: Full overview (first pass)

Collects all standard sections plus PM sampling. This is the mandatory first run.

```bash
ncu --set full \
    --section PmSampling_WarpStates \
    -k "regex:.*KERNEL_TOKEN.*" \
    -c 1 \
    --kill yes \
    -f \
    -o "$PROFILE_RUN_DIR/reports/full_<tag>" \
    python3 "$PROFILE_RUN_DIR/target/profile_<kernel>.py" [args]
```

| Flag | Meaning |
|---|---|
| `--set full` | Run all built-in sections. Already includes `PmSampling` and `SourceCounters` on the versions checked, so naming those again is redundant |
| `--section PmSampling_WarpStates` | Warp-stall time series. This one is **not** in `full`, so it is the section worth adding explicitly |
| `-k "regex:..."` | Only profile kernels whose **function** name matches (see below) |
| `-c 1` | Only profile one matching launch |
| `--kill yes` | Stop the application once `-c` is satisfied |
| `-f` | Overwrite an existing report of the same name |
| `-o ...` | Output path; `.ncu-rep` is appended automatically |

`-k` matches the function name by default (`--kernel-name-base function`), not the full demangled signature. So a bare token such as `regex:my_kernel` matches, and you only need `--kernel-name-base demangled` when you deliberately want to select on parameter types. Confirm what was actually captured with `action.name()` for the function name and `action.name(1)` for the demangled form.

`--kill yes` matters more than it looks. Without it, an application that launches the target kernel thousands of times keeps running to completion after the one launch you profiled, which on a long pipeline can mean hours of wall time bought for nothing.

Replay count is typically 45-50 passes; 47 was observed repeatedly on a real `--set full` capture. Budget for more than the arithmetic suggests: with kernel replay, each pass saves and restores the kernel's device working set, and a real capture reported roughly 42 GB backed up per pass. A "1.5 ms kernel × 47 passes" estimate can understate the true cost by orders of magnitude, and the application must still run to the target launch before profiling even starts.

---

## Recipe 2: Source-level profile (second pass)

Collects per-PC stall sampling data. This only works when the target path preserves usable source mapping.

```bash
ncu --section SourceCounters \
    --import-source yes \
    -k "regex:.*KERNEL_TOKEN.*" \
    -c 1 \
    --kill yes \
    -f \
    -o "$PROFILE_RUN_DIR/reports/source_<tag>" \
    python3 "$PROFILE_RUN_DIR/target/profile_<kernel>.py" [args]
```

Select the **section**, not a set: `--list-sets` on a 2024.1 install has no `source` entry, and `--set full` from Recipe 1 already collected `SourceCounters`. So this second pass is worth running only for `--import-source yes`, or to get source counters without paying for `full`.

`--import-source yes` is the part that is easy to skip and expensive to omit. Compiling with `-lineinfo` is necessary but not sufficient: without the import, the report still contains SASS and correct per-PC sample counts, but `source_info(pc)` returns `None` for every address, so anything that aggregates by source line collapses the whole kernel onto a single `?:0` row. Verify mapping exists before trusting a per-line table, and fall back to per-PC plus `sass_by_pc(pc)` when it does not.

Use this report with `extract_stall_hotspots.py`.

---

## Recipe 3: Warmup before collection

If the first launch triggers JIT or compile work, run the profile driver once without NCU first:

```bash
python3 "$PROFILE_RUN_DIR/target/profile_<kernel>.py" --warmup-only [args]
```

Then collect with NCU against the steady-state launch path. If the driver cannot separate warmup from steady-state, use `-s` and `-c` to target the right launch.

---

## Recipe 4: Skip compile-only or helper launches

If the profile driver launches matching kernels multiple times:

```bash
ncu --set full \
    -k "regex:.*KERNEL_TOKEN.*" \
    -s 1 -c 1 \
    --kill yes -f \
    -o "$PROFILE_RUN_DIR/reports/full_<tag>" \
    python3 "$PROFILE_RUN_DIR/target/profile_<kernel>.py" [args]
```

- `-s N` skips the first `N` matching launches
- `-c N` limits collection to `N` matching launches

This is the main way to avoid compile-only or warmup launches polluting the report.

---

## Recipe 5: Details page (quick rule summary)

No need to collect again:

```bash
ncu --import "$PROFILE_RUN_DIR/reports/full_<tag>.ncu-rep" --page details \
    --print-details all \
    --print-metric-name name \
    > "$PROFILE_RUN_DIR/analysis/details_<tag>.txt"
```

Both print flags are load-bearing. The default is `--print-details header`, which gives you the section headers and the rule bullets but **omits the body tables** — so the warp-state stall breakdown is simply absent, and grepping for a stall reason returns nothing on a report that measured it. And the default output shows display labels such as `Compute (SM) Throughput` rather than metric names, which cannot be copied into a script or into `action[...]`; `--print-metric-name name` prints `sm__throughput.avg.pct_of_peak_sustained_elapsed` instead.

Always read `details_<tag>.txt` early. NCU's rule engine is often directionally correct. Read its estimates as a ranking of where to look, not as a budget: they overlap, one may be relative to a section while the next is relative to the whole kernel, and on a real report they summed to far more than 100%.

---

## Recipe 5b: Recover how a report was captured

A report records the command line that produced it. This is the only reliable way to answer "what were the flags, which launch was this, was source imported" about an artifact you did not create in the last five minutes:

```bash
ncu --import "$REPORT" --page session
ncu --import "$REPORT" --print-summary per-kernel
```

Write the output next to the report. Reconstructing a capture from shell history is guesswork; reconstructing it from `--page session` is not.

---

## Recipe 6: CSV and raw export

```bash
ncu --import "$PROFILE_RUN_DIR/reports/full_<tag>.ncu-rep" --page raw --csv \
    > "$PROFILE_RUN_DIR/analysis/raw_<tag>.csv"

ncu --import "$PROFILE_RUN_DIR/reports/source_<tag>.ncu-rep" --page source \
    > "$PROFILE_RUN_DIR/analysis/source_<tag>.txt"
```

The Python API is usually easier, but CSV is useful for quick inspection.

---

## Recipe 7: Targeted metrics only

If you only need a few metrics:

```bash
ncu --metrics \
    sm__throughput.avg.pct_of_peak_sustained_elapsed,\
    sm__warps_active.avg.pct_of_peak_sustained_active,\
    gpu__time_duration.sum,\
    l1tex__t_sector_hit_rate.pct \
    -k "regex:.*KERNEL_TOKEN.*" -c 1 \
    python3 "$PROFILE_RUN_DIR/target/profile_<kernel>.py" [args]
```

---

## Recipe 8: A/B comparison

```bash
ncu --set full -k "regex:.*my_kernel.*" -c 1 --kill yes -f \
    -o "$PROFILE_RUN_DIR/reports/v1" \
    python3 "$PROFILE_RUN_DIR/target/profile_my_kernel_v1.py" [args]

ncu --set full -k "regex:.*my_kernel.*" -c 1 --kill yes -f \
    -o "$PROFILE_RUN_DIR/reports/v2" \
    python3 "$PROFILE_RUN_DIR/target/profile_my_kernel_v2.py" [args]
```

Compare with `analyze_reports.py`, passing both reports in one invocation:

```bash
python3 "$HELPERS/analyze_reports.py" --run-dir "$PROFILE_RUN_DIR" \
    --report "$PROFILE_RUN_DIR/reports/v1.ncu-rep" --tag v1 \
    --report "$PROFILE_RUN_DIR/reports/v2.ncu-rep" --tag v2
```

That writes `analysis/compare_v1_vs_v2.txt`, which leads with a comparability gate:
if the two reports disagree on kernel signature, device, block size, or grid size, they
measured different things and ranking their durations says nothing. Do not hand-roll a
comparison script — the gate is the point.

---

## What each `--set` contains

```bash
ncu --list-sets
ncu --list-sections
```

Rough mapping (B200, NCU 2026.1):

| Set | Sections included | Replay passes | Use when |
|---|---|---|---|
| `basic` | SOL, LaunchStats, Occupancy | ~3-5 | Smoke test |
| `detailed` | Middle-ground preset | ~15 | Faster but limited |
| `full` | Everything, including `SourceCounters` and `PmSampling` on the versions checked | ~45 | First-pass profile |

There is no portable `source` set — a 2024.1 install has none, and `full` already carries
`SourceCounters`. Use `--section SourceCounters` (Recipe 2) for per-line stall attribution,
and run `ncu --list-sets` before assuming any set name from this table exists.

---

## Common section additions

```bash
--section PmSampling_WarpStates   # not in --set full; the one worth adding
--section PmSampling              # already in --set full on the versions checked
--section SourceCounters          # already in --set full; add it to skip paying for full
--section Nvlink_Topology
--section Nvlink_Tables
```

---

## GPU frequency locking

```bash
nvidia-smi -q -d CLOCK
sudo nvidia-smi -lgc <boost_clock_mhz>
sudo nvidia-smi -rgc
```

Usually unnecessary for B200 full profiles, but useful if results jitter across repeated runs.

---

## Gotchas

- **`--set full` is slow**: each replay reruns the target launch.
- **`regex:...` matches nothing**: start from the Python/kernel token, then inspect the captured action name in a report.
- **`regex:...` matches too much**: refine the token to exclude helper kernels or unrelated specializations.
- **Report file is empty or 0 KB**: the target crashed before the selected launch or the regex never matched.
- **PM sampling returns nothing**: check the section flags and the environment.
- **First-run compile polluted the profile**: warm up once outside NCU or skip early launches with `-s`.
- **Source-level report lacks line mapping**: the runtime did not preserve usable source info; keep working at the report and metric level unless the runtime has another way to recover source attribution.
