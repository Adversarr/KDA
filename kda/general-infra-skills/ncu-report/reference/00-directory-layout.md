# Profile Directory Layout & Naming

**Read this first, before any collection.** Bad directory layout is the single most common cause of mixing results from different runs, overwriting prior profiles, or losing track of which `.ncu-rep` belongs to which kernel version. Follow the rules below for every profiling project.

---

## Top-level rule

**All profiling artifacts live under a single `profile/` directory at the repo root.** Never scatter `.ncu-rep` files across random locations. Never put profile artifacts under `src/`, `scripts/`, or other source directories.

```
<repo_root>/
├── profile/                        ← everything profiling-related lives here
│   ├── <run_1>/
│   ├── <run_2>/
│   └── ...
├── src/
└── ...
```

---

## One run = one subdirectory

Every time you profile a kernel — whether it's a new kernel, a new version of the same kernel, or the same kernel on a different workload — **create a new subdirectory under `profile/`**. Never write into an existing run's directory.

Rationale:

- Profiles of different implementations of the same kernel must not overwrite each other. If you profile `<kernel>_v1` today and `<kernel>_v2` tomorrow, both reports need to coexist for A/B comparison.
- The profile target itself is part of the profile: it encodes which Python runner was used, with which workload and capture token. Keeping that target under the run dir pins the provenance.
- Analysis artifacts (`metrics_*.json`, `compare_*.txt`, ASCII plots) are tied to a specific set of `.ncu-rep` files; they must not be mixed.

---

## Run directory naming

Use descriptive, short, kebab-case names. Include **what** was profiled and **when/why**, not how.

Good:
```
profile/<kernel>_v1_baseline/
profile/<kernel>_v2_optimized/
profile/<kernel>_v2_optimized_vs_v1/      # for comparison run
profile/<kernel>_v2_tma_prefetch/
profile/flash_attn_b200_h128_baseline/
```

Bad:
```
profile/test/                   # too vague
profile/run1/                   # meaningless
profile/20260413/               # dates with no context
profile/final/                  # there's never a "final"
```

If you genuinely have multiple runs on the same day for the same kernel/version combo, append a short distinguisher or a date suffix: `<kernel>_v1_baseline_20260413_am` / `<kernel>_v1_baseline_20260413_pm`.

---

## Standard run layout

Inside each run subdirectory, use this structure:

```
profile/<run_name>/
├── REPORT.md                       ← human-readable final report (Markdown)
├── target/
│   ├── profile_<kernel>.py         ← preferred Python runner used by ncu
│   └── target_notes.md             ← optional: kernel token, warmup notes, shapes
├── reports/
│   ├── full_<tag1>.ncu-rep         ← ncu --set full output (Recipe 1)
│   ├── full_<tag2>.ncu-rep
│   ├── source_<tag1>.ncu-rep       ← ncu --section SourceCounters --import-source yes (Recipe 2)
│   └── source_<tag2>.ncu-rep
└── analysis/
    ├── summary_<tag>.txt           ← the diagnosis-ready digest. READ THIS FIRST.
    ├── metrics_all_<tag>.json      ← 2000+ metrics, full archive
    ├── metrics_key_<tag>.{txt,json}← curated key metrics
    ├── compare_<a>_vs_<b>.txt      ← side-by-side, with the comparability gate
    ├── details_<tag>.txt           ← ncu --page details dump
    ├── stall_hotspots_<tag>.txt    ← per-line stall aggregation
    ├── pm_timeline_plots.txt       ← ASCII time-series
    └── raw_<tag>.csv               ← optional: ncu CSV export
```

Notes:

- `summary_<tag>.txt` is what `analyze_reports.py` writes and what everything else in this skill is written around. Read it before any other artifact in `analysis/`.
- The helper scripts stay in the skill's `helpers/` folder and are invoked by absolute path. Copying them into `analysis/` breaks their `ncu_utils` import and forks a second copy that will not get fixes.
- `<tag>` is the per-workload / per-dispatch-path label, e.g. `path_a_shapeA`, `path_b_shapeB`. Pick tags that are short and name the representative workload, not the file UUID.
- If you profile only one tag, you can omit the tag suffix from filenames. But as soon as you profile a second, backfill the tag to avoid ambiguity.
- Keep a copy of the exact Python profile runner used for collection under `target/`. That file is part of the evidence.

---

## Comparing two runs

For A/B comparisons (optimization-before vs after, or two dispatch variants on the same build), create a comparison run that *references* both underlying runs, and produce its numbers with `analyze_reports.py` rather than a bespoke script:

```bash
export CMP_DIR=/abs/path/to/profile/<kernel>_v2_vs_v1
mkdir -p "$CMP_DIR/analysis"

python3 /abs/path/to/ncu-report/helpers/analyze_reports.py --run-dir "$CMP_DIR" \
    --report /abs/path/to/profile/<kernel>_v1_baseline/reports/full_<tag>.ncu-rep  --tag v1 \
    --report /abs/path/to/profile/<kernel>_v2_optimized/reports/full_<tag>.ncu-rep --tag v2
```

```
profile/<kernel>_v2_vs_v1/
├── REPORT.md                       ← describes both runs + the comparison
└── analysis/
    ├── compare_v1_vs_v2.txt        ← side-by-side, led by the comparability gate
    ├── summary_v1.txt              ← digest for each side
    └── summary_v2.txt
    (No ncu-rep files — they live in the referenced runs)
```

A hand-written comparison script is the wrong move here: `compare_*.txt` opens with a comparability gate that flags a differing kernel signature, device, block size, or grid size, and that gate is what stops a ranking of two unlike measurements from being published as a speedup.

The comparison run does not re-profile; it only produces comparison artifacts and prose.

---

## What does NOT go in a run directory

- `.ncu-rep.old` backup files — if you need a prior version, you should have made it a separate run.
- Temporary scratch files — `/tmp` is for those.
- The dataset / workload files themselves — these belong in a shared dataset directory outside the run tree (for example, `/path/to/datasets/workload/`). Reference them through a configurable path in scripts.
- Compiler intermediates (`*.o`, `*.d`) generally do not belong in the run dir.
- `ncu_home/` or ncu cache directories — delete these after profiling, they're huge and regenerable. Set `HOME=$HOME` before running ncu rather than letting it cache inside the run dir.

Add a simple `.gitignore` inside `profile/` if you want to keep the run dirs out of git:
```
profile/*/
!profile/README.md
```

Or, if you want a few canonical runs tracked in git, `.gitignore` only the data-heavy subdirs:
```
profile/*/reports/
profile/*/analysis/metrics_all_*.json
profile/*/analysis/raw_*.csv
```

---

## Environment variable convention (optional but recommended)

Scripts and ncu invocations should pick up the run directory from a single env var, so they're easy to redirect to different runs:

```bash
export PROFILE_RUN_DIR=/abs/path/to/profile/<kernel>_v1_baseline
export HELPERS=/abs/path/to/ncu-report/helpers
mkdir -p "$PROFILE_RUN_DIR"/{target,reports,analysis}

# prepare the target
cp "$HELPERS/profile_template.py" "$PROFILE_RUN_DIR/target/profile_<kernel>.py"

# run ncu — Recipe 1 of 03-collection.md
ncu --set full --section PmSampling_WarpStates \
    -k "regex:.*my_kernel.*" -c 1 --kill yes -f \
    -o "$PROFILE_RUN_DIR/reports/full_<tag>" \
    python3 "$PROFILE_RUN_DIR/target/profile_<kernel>.py" [args]

# parse
python3 "$HELPERS/analyze_reports.py" --run-dir "$PROFILE_RUN_DIR" \
    --report "$PROFILE_RUN_DIR/reports/full_<tag>.ncu-rep" --tag <tag>
```

Every helper in `../helpers/` requires `--run-dir`, `--report`, and `--tag` explicitly; none of them defaults to the current directory, and outputs always land in `<run-dir>/analysis/`.

---

## Checklist before starting a profile run

1. `mkdir -p profile/<new_run_name>/{target,reports,analysis}` — make the three subdirs up front.
2. Copy or write the Python profile runner into `profile/<new_run_name>/target/`.
3. Record the kernel token and expected regex in that target or a nearby note.
4. Run ncu with `-o profile/<new_run_name>/reports/full_<tag>`.
5. Run `analyze_reports.py --run-dir profile/<new_run_name> --report … --tag <tag>`, and read `analysis/summary_<tag>.txt` before anything else.
6. Write `REPORT.md` at `profile/<new_run_name>/REPORT.md`.
7. Before starting a *new* run, go back to step 1 with a new name — never write into the existing one.
