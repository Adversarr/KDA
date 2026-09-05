# Profiling Workflow — End-to-End

This is the complete checklist from "user asks to profile" to "final report". The default path is a small Python profile driver for a DSL kernel that compiles into CUDA.

This workflow starts only **after the kernel is already correct**. Correctness debugging and validation are out of scope for this skill.

---

## Phase 0 — Create a new run directory

**Always start here.** See [`00-directory-layout.md`](00-directory-layout.md) for the full convention.

```bash
PROFILE_RUN_DIR=profile/<descriptive_run_name>
mkdir -p "$PROFILE_RUN_DIR"/{target,reports,analysis}
```

- Pick a new, descriptive name for this run. Never reuse an existing directory.
- If you're profiling a new version of a kernel you've profiled before, that's a **new** run.
- If you're profiling the same version against a different workload or dispatch path, that's also a new run or at minimum a new tag set.

Every artifact produced in subsequent phases is written **only** under `$PROFILE_RUN_DIR`.

---

## Phase 0.5 — Frame the problem

Before typing any commands, answer these:

1. **What Python-level kernel token am I targeting?** This is usually the kernel function or wrapper name that survives inside the generated CUDA name.
2. **Which workload or shape is representative?** If the kernel is shape-sensitive, pick a representative real workload or exact shape tuple.
3. **Which dispatch path or specialization is active?** Many DSL stacks pick different generated kernels based on shapes, dtypes, or autotune results.
4. **What question am I answering?** "Why is this slow?" is weaker than "At shape X, is this specialization latency-bound, bandwidth-bound, or under-filled?"
5. **What is the baseline?** Previous version, alternate specialization, cuBLAS, PyTorch, or another reference implementation.

If any of 1-4 are unclear, clarify before profiling. Profiling the wrong specialization wastes time.

---

## Phase 1 — Environment check

```bash
ncu --version
nvidia-smi
python3 --version
find /usr/local/cuda* -name "ncu_report*" -type f 2>/dev/null
```

Put `ncu_report` on `PYTHONPATH` so the helpers work:

```bash
export PYTHONPATH=$PYTHONPATH:/usr/local/cuda-XX.X/nsight-compute-YYYY.X.0/extras/python
python3 -c "import ncu_report; print('OK')"
```

Also set a writable `HOME` before running ncu:

```bash
export HOME=/some/writable/dir
```

If you expect source-level analysis, make sure the DSL/runtime path preserves usable source mapping or generated CUDA line info.

---

## Phase 2 — Prepare the profile target

**Preferred path: a Python profile driver.** Build a small `profile_<kernel>.py` that:

- imports the kernel module directly
- compiles or JITs the kernel explicitly when needed
- constructs representative profiling inputs
- optionally performs one warmup launch if the first call triggers codegen
- launches the steady-state kernel in a reproducible way
- records the intended kernel token and regex, typically `regex:.*<kernel_token>.*`

Put that script under `$PROFILE_RUN_DIR/target/`.

Use [`02-harness-guide.md`](02-harness-guide.md) for the recommended profile-driver shape.

---

## Phase 3 — Validate the capture token

Before collecting expensive reports:

1. Pick the kernel token from the Python-level symbol name.
2. Build the NCU selector as `regex:.*<kernel_token>.*`.
3. Warm up once outside NCU if the first launch triggers compilation.
4. Run a cheap first-pass collection or inspection run and confirm the captured kernel name is the one you intended.
5. If the regex is too broad, refine the token using specialization hints, shape tags, or emitted generated code.

The regex is part of the workflow, not a last-minute tweak.

---

## Phase 4 — Collect profiles

Run two NCU invocations. Both outputs go under `$PROFILE_RUN_DIR/reports/`.

```bash
# Overview — all sections + PM sampling
ncu --set full \
    --section PmSampling --section PmSampling_WarpStates \
    -k "regex:.*<kernel_token>.*" \
    -c 1 \
    -o "$PROFILE_RUN_DIR/reports/full_<tag>" \
    python3 "$PROFILE_RUN_DIR/target/profile_<kernel>.py" [args]

# Source-level — per-PC stall sampling. No `source` set exists on every install,
# and without --import-source yes every PC resolves to no line at all.
ncu --section SourceCounters --import-source yes \
    -k "regex:.*<kernel_token>.*" \
    -c 1 \
    -o "$PROFILE_RUN_DIR/reports/source_<tag>" \
    python3 "$PROFILE_RUN_DIR/target/profile_<kernel>.py" [args]
```

Run the pair once per `(kernel token, dispatch path, representative workload)` combination.

If the profile driver launches the same kernel multiple times:

- use `-s` to skip compile-only or warmup launches
- use `-c 1` to capture one steady-state launch
- record exactly which iteration was profiled

---

## Phase 5 — Extract structured data

Do not eyeball the CLI output. Parse reports in Python so you can compare, aggregate, and archive. See [`04-python-api.md`](04-python-api.md) and use the helpers in [`../helpers/`](../helpers/).

Minimum analysis artifacts to produce:

| Artifact | Tool | What it tells you |
|---|---|---|
| `summary_<tag>.txt` | `analyze_reports.py` | **Read this first.** Device, fill verdict, stalls, coalescing, ranked rules, and what was not measured |
| `metrics_key_<tag>.txt` | `analyze_reports.py` | Curated key metrics, with aliases resolved in the digest |
| `metrics_all_<tag>.json` | `analyze_reports.py` | Full metric archive |
| `compare_<a>_vs_<b>.txt` | `analyze_reports.py` | Side-by-side metric comparison, with a comparability gate |
| `stall_hotspots_<tag>.txt` | `extract_stall_hotspots.py` | Top sites ranked by stall samples — source lines, or PC plus SASS |
| `pm_timeline_plots.txt` | `plot_timeline.py` | ASCII time-series plots of the series this report actually carries |
| `details_<tag>.txt` | `ncu --import ... --page details` | NCU rule-engine suggestions — skip if there is no `ncu` binary; the digest already ranks the rules |

Save everything under `$PROFILE_RUN_DIR/analysis/`.

---

## Phase 6 — Diagnose

Work through the six analysis dimensions in [`05-analysis-dimensions.md`](05-analysis-dimensions.md):

1. **SM occupancy and wave structure**
2. **Thread-block balance and tail effect**
3. **Instruction-level stall analysis**
4. **Tensor Core utilization**
5. **SM utilization timeline**
6. **Memory access pattern**

For each dimension, write down the signal and the metric value that supports it. Then consult [`06-diagnosis-playbook.md`](06-diagnosis-playbook.md).

For DSL-generated kernels, always include:

- the captured demangled kernel name from the report
- the Python-level token you targeted
- the workload or specialization tag that produced that kernel

---

## Phase 7 — Write the report

Structure described in [`07-report-template.md`](07-report-template.md). Key elements:

1. **Setup**: exact profile driver, workloads, regex, NCU commands, metric-name caveats
2. **Headline numbers**: duration, SM throughput, memory throughput, occupancy, Tensor Core usage
3. **Per-dimension analysis** with evidence
4. **Optimization directions** ranked by expected impact
5. **Confidence and caveats**

Keep the report short enough that a busy reader can see the top findings in 30 seconds.

---

## Anti-patterns to avoid

- ❌ Profiling with arbitrary synthetic shapes that don't match the real workload.
- ❌ Letting first-run JIT or compile work pollute the measured kernel launch.
- ❌ Using a regex so broad that helper kernels or unrelated specializations are captured.
- ❌ Dumping the full NCU CLI output into the report with no interpretation.
- ❌ Proposing optimizations without evidence.
- ❌ Missing the highest-impact finding because you got distracted by a smaller one.
