# Helpers

Reusable code for Python DSL profile drivers and Nsight Compute report analysis. See `../SKILL.md` for context.

## Python / DSL

| File | Purpose |
|---|---|
| `profile_template.py` | Preferred starting point for a Python profile driver. Import the kernel, compile or JIT it, print the regex token, then launch. |
| `kernel_name_regex.py` | Tiny helper for building the canonical `regex:.*token.*` selector. |
| `ncu_utils.py` | Shared helpers: `Metrics` (alias resolution plus absent-vs-zero status), `fill_verdict`, `shared_memory`, `stall_reasons`, `derived_memory`, `rule_speedups`, `KEY_METRICS`, `load_report`, `per_pc_values`. |
| `analyze_reports.py` | The first thing to run on a report: one diagnosis-ready digest per launch, plus side-by-side comparison across reports. |
| `extract_stall_hotspots.py` | Aggregate per-PC stall samples by source line, or by PC with SASS when the capture has no source mapping. |
| `plot_timeline.py` | ASCII plot PM sampling timelines, discovering the series the report carries when the preferred names are empty. |
| `list_flashinfer_workloads.py` | Optional workload inspector for projects that use flashinfer-trace datasets. |

### Typical DSL-first workflow

```bash
export PROFILE_RUN_DIR=profile/<run_name>
HELPERS=/absolute/path/to/ncu-report/helpers

mkdir -p "$PROFILE_RUN_DIR"/{target,reports,analysis}
cp "$HELPERS/profile_template.py" "$PROFILE_RUN_DIR/target/profile_<kernel>.py"

# Collect. This is Recipe 1 of ../reference/03-collection.md — keep the two in
# step rather than editing the flags here.
ncu --set full --section PmSampling_WarpStates \
    -k "regex:.*<kernel_token>.*" -c 1 --kill yes -f \
    -o "$PROFILE_RUN_DIR/reports/full_<tag>" \
    python3 "$PROFILE_RUN_DIR/target/profile_<kernel>.py"

# Analyze. Read analysis/summary_<tag>.txt first.
python3 "$HELPERS/analyze_reports.py" --run-dir "$PROFILE_RUN_DIR" \
    --report "$PROFILE_RUN_DIR/reports/full_<tag>.ncu-rep" --tag <tag>
```

Only if the project uses a flashinfer-trace dataset, `list_flashinfer_workloads.py`
inspects the workload space before you write the driver. It needs
`export FIB_DATASET_PATH=/path/to/flashinfer-trace`, and nothing else here depends on it.

Reading a report needs no GPU and no `ncu` binary — only the `ncu_report` module from some Nsight Compute install, which `ncu_utils.py` locates including from the macOS app bundle.

`ncu_utils.py` tries to auto-locate `ncu_report` from common CUDA install paths. If that fails, set `PYTHONPATH`:

```bash
export PYTHONPATH=$PYTHONPATH:/usr/local/cuda-13.2/nsight-compute-2026.1.0/extras/python
```
