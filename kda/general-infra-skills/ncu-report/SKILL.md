---
name: ncu-report
description: Profile and diagnose a single CUDA kernel with Nsight Compute — capture a report, read an existing .ncu-rep, or turn one into a ranked optimization plan. Written against B200 / sm_100 and verified on sm_80, with per-architecture metric-name handling. Use only when a specific kernel is already proven to own material wall time (named kernel plus an Nsight Systems digest ordinal, or an explicit KDA handoff with verified workload/phase timing), or the user already has an .ncu-rep for that kernel. Not for whole-program slowness, GPU idle, launch storms, host synchronization, or choosing which kernel matters. Includes kernels emitted by Python DSLs such as TileLang, Triton, or torch.compile.
---

# Single-Kernel Profiling with Nsight Compute

**When to use:** user asks to profile a CUDA kernel emitted by a Python DSL or runtime-managed stack, analyze its performance, find its bottlenecks, or write an optimization plan based on Nsight Compute data. The kernel must already pass correctness checks before this skill begins. Common triggers include: "profile X", "为什么这个 kernel 慢", "ncu report 说...", "下一步怎么优化", "帮我看一下这份 ncu 报告", or requests involving TileLang, Triton, `torch.compile`, TVM-FFI, or other Python-side compilation flows.

**Written against:** NVIDIA B200 (sm_100, CC 10.0, 148 SMs) with Nsight Compute 2026.x, and re-verified on A800 (sm_80, 108 SMs) with Nsight Compute 2024.1.1. Most of the advice is generic; where a name or a default differs, the difference is recorded rather than assumed away.

**Spend one minute on these four checks before copying any command or quoting any number.** Set names, section names, metric names, and the Python API shape all vary by version and architecture.

1. **Read the device and its SM count from the report** — `device__attribute_display_name`, `device__attribute_multiprocessor_count` — never from this document. Every wave and occupancy number is graded against the SM count, so one assumed number silently mis-grades the whole diagnosis. `analyze_reports.py` prints this first for that reason.
2. **Enumerate `action.metric_names()`** (or `--page raw --csv`) instead of trusting any metric table, including [`reference/08-metric-names.md`](reference/08-metric-names.md). Keep a fallback for each field you need, or let `Metrics.resolve()` in [`helpers/ncu_utils.py`](helpers/ncu_utils.py) do it — it reports which name supplied each value.
3. **Distinguish absent from zero.** A metric that does not exist on this chip and a metric that measured zero both come back as `None` from a naive accessor. Reporting "0% tensor core" or "occupancy unknown" from an absent name is the most common way to be confidently wrong here. `Metrics.status()` separates `absent`, `novalue`, and `zero`.
4. **If you are only reading a report, most collection advice does not apply.** With no `ncu` binary on PATH there is no `--list-sets`, no `--page session`, and no CLI fallback; `helpers/ncu_utils.py` locates the `ncu_report` module from a host install (including the macOS app bundle) and that is enough. Note also that `rule_results_as_dicts()` availability follows the **reader** module, not the version that captured the file — a 2026.x reader exposes rules on a 2024.1 capture, so check `hasattr` rather than assuming.

If you *are* collecting: run `ncu --version` and `ncu --list-sets` first. There is no `source` set on older installs, and `--set source` there fails with `No metrics to collect found in sections`, which reads like "this kernel has no source counters". Use `--section SourceCounters`, which `--set full` already collects.

**Two report shapes need different handling.** Check `action.sections()` before diagnosing:

- **`--set full`** — many sections, rules populated. The normal path.
- **`--metrics` capture** — a single `Command line profiler metrics` section. Rules never ran, so an empty `rule_results_as_dicts()` means *nobody looked*, not "Nsight Compute found no problems". Whole dimensions are unavailable; say which questions the report cannot answer instead of filling them with `None`.

`analyze_reports.py` detects and labels both, and prints a "Not measured in this report" section so a gap is never read as a measurement.

---

## Golden rule

**Profile → Diagnose → Plan, in that order. Never guess.**

Most under-performing CUDA kernels are under-performing for exactly one reason that ncu can tell you in 10 seconds. Don't invent hypotheses before you have the report. Don't start coding a fix before you've matched the observed pattern to a known diagnosis. Don't write a wall of suggestions — rank them by evidence and expected impact.

For DSL workflows, the target is usually a small Python profile driver that imports the kernel module, compiles or JITs it, warms up if needed, and then launches the steady-state kernel that Nsight Compute should capture.

This skill starts **after correctness**. Do not use it to debug functional bugs, validation mismatches, or reference-check failures.

---

## Embedded KDA handoff

When KDA dispatches this skill, accept its already-correct named kernel, workload/phase,
isolated timing evidence and diagnostic question instead of requiring a Systems launch
ordinal. This establishes an isolated tuning target, not whole-model critical-path ownership.
Use the caller's `<op_dir>/_scratch/profile/<unique-run>` as the output root for the layout
below; keep evidence under the worker's permitted directory. Retain useful measurements in
its audit/implementation notes before scratch cleanup. For a specific counter question,
collect the smallest sufficient metric set; perform broader captures only if it remains
unresolved. Standalone collection keeps the defaults below.

## Quickstart (what to do when someone says "profile this kernel")

0. **Create a new run directory first** under the caller-selected output root, or `profile/<run_name>/` at the project root for standalone use — **one directory per run**, never reuse an existing one. Each run contains its own `target/`, `reports/`, `analysis/`, and `REPORT.md`. This separation is mandatory for reliable comparisons. See [`reference/00-directory-layout.md`](reference/00-directory-layout.md).

1. **Decide what you're profiling.** What Python-level kernel symbol or token identifies the generated CUDA kernel? Which dispatch path or specialization? What question do you want answered? If the kernel takes variable-sized inputs, pick specific representative shapes from the user's workload — don't profile with arbitrary inputs.

2. **Prepare a small Python profile driver** unless the user is already profiling through an existing Python runner. Use an importable kernel module plus `profile_<kernel>.py`. The profile driver should compile or JIT the kernel, launch the already-correct kernel in a steady-state way, and expose a stable kernel token for NCU regex capture such as `regex:.*some_kernel.*`. Put it under `profile/<run_name>/target/`. See [`reference/02-harness-guide.md`](reference/02-harness-guide.md) and the template in [`helpers/profile_template.py`](helpers/profile_template.py).

3. **Standalone overview: run two profiles** (a targeted KDA handoff may collect only its required metrics): `--set full` plus `--section PmSampling_WarpStates` for the overview, and `--section SourceCounters --import-source yes` for per-line stall attribution. Drive both through the Python profile script, using `-k "regex:.*<kernel_token>.*"` once the token is confirmed, and add `--kill yes` so a many-launch driver stops once the captured launch is done. Write outputs to `profile/<run_name>/reports/`. See [`reference/03-collection.md`](reference/03-collection.md).

4. **Parse with the `ncu_report` Python module** — not by eye-balling the CLI. Run [`helpers/analyze_reports.py`](helpers/analyze_reports.py) first: its digest answers most of the six dimensions below in one pass, and it states which metric name supplied each number so you can audit any of them. Go deeper from there with the other helpers and your own queries; the digest is a floor, not a ceiling. Write outputs to `profile/<run_name>/analysis/`. See [`reference/04-python-api.md`](reference/04-python-api.md).

5. **Work through the six analysis dimensions.** See [`reference/05-analysis-dimensions.md`](reference/05-analysis-dimensions.md). Every one matters, but on any given kernel only 1–2 will dominate.

6. **Match patterns to the diagnosis playbook.** See [`reference/06-diagnosis-playbook.md`](reference/06-diagnosis-playbook.md). It maps NCU signal → likely cause → concrete fix, with example counts for "how big is this".

7. **Write the report** at `profile/<run_name>/REPORT.md` with evidence-backed recommendations, ranked by expected impact. See [`reference/07-report-template.md`](reference/07-report-template.md).

---

## File index

### Reference docs (read these when you need details)

| File | Purpose |
|---|---|
| [`reference/00-directory-layout.md`](reference/00-directory-layout.md) | **Read first.** Directory / naming conventions — one run = one subdirectory, no cross-contamination |
| [`reference/01-workflow.md`](reference/01-workflow.md) | End-to-end checklist from "user request" to "final report" |
| [`reference/02-harness-guide.md`](reference/02-harness-guide.md) | Primary target-preparation guide for small Python profile drivers |
| [`reference/03-collection.md`](reference/03-collection.md) | NCU command recipes for Python-runner collection, regex targeting, PM sampling, and custom sections |
| [`reference/04-python-api.md`](reference/04-python-api.md) | `ncu_report` Python API patterns for reports emitted from DSL-generated kernels |
| [`reference/05-analysis-dimensions.md`](reference/05-analysis-dimensions.md) | Six analysis dimensions: occupancy, balance, stalls, tensor core, timeline, memory |
| [`reference/06-diagnosis-playbook.md`](reference/06-diagnosis-playbook.md) | Pattern → diagnosis → fix. Merges Blackwell programming principles with NCU signals |
| [`reference/07-report-template.md`](reference/07-report-template.md) | How to structure the final report for a Python-runner-driven profile |
| [`reference/08-metric-names.md`](reference/08-metric-names.md) | Metric names as observed on sm_100 / 2026.x and sm_80 / 2024.1, and the substitutions between them |
| [`reference/09-common-issues.md`](reference/09-common-issues.md) | Permissions, PM sampling gaps, DSL regex pitfalls, and runtime-specific gotchas |

### Helpers (reusable code)

| File | Purpose |
|---|---|
| [`helpers/profile_template.py`](helpers/profile_template.py) | Preferred template for Python DSL profile drivers — import, compile/JIT, warm up, print regex, launch |
| [`helpers/kernel_name_regex.py`](helpers/kernel_name_regex.py) | Tiny helper for turning a kernel token into the canonical `regex:.*token.*` selector |
| [`helpers/analyze_reports.py`](helpers/analyze_reports.py) | **Start here on any report.** One digest with device identity, the fill verdict, shared-memory breakdown, stalls in their correct form, coalescing, ranked rules, and an explicit list of what was not measured. Also side-by-side comparison across reports |
| [`helpers/extract_stall_hotspots.py`](helpers/extract_stall_hotspots.py) | Per-site stall aggregation — source lines when the capture has mapping, PC plus SASS when it does not |
| [`helpers/plot_timeline.py`](helpers/plot_timeline.py) | ASCII PM-sampling timeline plotter — makes tail effect visible. Discovers the series a report actually carries when the preferred names are empty |
| [`helpers/list_flashinfer_workloads.py`](helpers/list_flashinfer_workloads.py) | Optional workload inspector for projects that use flashinfer-trace datasets |
| [`helpers/ncu_utils.py`](helpers/ncu_utils.py) | Shared Python helpers: `Metrics` alias resolution with absent-vs-zero status, `fill_verdict`, `shared_memory`, `stall_reasons`, `derived_memory`, `rule_speedups`, report loading |

---

## Critical lessons (don't skip)

1. **Metric names drift in both directions, so enumerate.** `dram__bytes.sum` and `l1tex__average_t_sectors_per_request*.ratio` are absent on sm_100 *and* on sm_80 / 2024.1 — compute those from sums. The `smsp__sass_inst_executed_op_*` forms are not a Blackwell rename; they are what 2024.1 reports on Ampere too, so "adapting to an older GPU" by dropping `sass_` fails. What genuinely differs between the two verified pairings is tabulated in [`reference/08-metric-names.md`](reference/08-metric-names.md) and encoded as `FIELD_ALIASES`.

2. **Always preserve source mapping if you want source-level analysis.** Without `-lineinfo` or an equivalent generated-source path, ncu's source view is blank and you cannot do per-line stall analysis. `-lineinfo` alone is not enough: capture with `--import-source yes` as well, or `source_info(pc)` returns `None` for every address while SASS and sample counts still look healthy — and a per-line table then reports the entire kernel on one `?:0` row. When mapping is missing, attribute by PC and disassemble with `action.sass_by_pc(pc)`.

3. **PM sampling is the best way to see *when* utilization changes, but not the only way to see imbalance.** Static metrics average over the whole kernel, and only the time-series (`pmsampling:` metrics, or the ASCII plotter in `helpers/`) shows the shape of utilization over time. For imbalance specifically, the `WorkloadDistribution` section is collected even in `--set basic` and reports SM active-cycle spread directly. Also check which `pmsampling:` names your report actually contains: the set differs by version and architecture, and the plotter's defaults return nothing on an install that names them differently.

4. **Load-imbalance on variable-length inputs is often the #1 bottleneck.** If the workload has sequences of varying length, per-SM active-cycle variance will often dwarf every other effect. Always check the input distribution.

5. **NCU's rule engine (`--page details`) already does half the work.** Each rule comes with `Est. Speedup: X%`. Read them first — they often point straight at the answer. Read them as a ranking, not a budget: the estimates overlap and routinely sum past 100% on a real report, some are relative to one section while others are relative to the whole kernel, and not every rule offers one. Add `--print-details all --print-metric-name name`, or you get section headers without the body tables and display labels you cannot paste into a script.

6. **Regex targeting is part of the workflow, not an afterthought.** Most DSLs emit generated CUDA names that still contain the Python/kernel token. Start from that token and use `regex:.*<kernel_token>.*`, then verify the captured name before trusting the report. `-k` matches the **function** name by default (`--kernel-name-base function`), so compare against `action.name()`; `action.name(1)` gives the demangled signature and `--page session` records the selector that actually ran.

7. **Don't delegate understanding.** Run the profiles yourself, open the reports, cite specific metric values. Never write "the profile shows it's memory-bound" — instead, name the two or three metric values that back your conclusion (e.g., "`dram__bytes_read.sum.pct_of_peak_sustained_elapsed` well under 10%, and `long_scoreboard` stalls dominate the pcsamp histogram, so the kernel is **latency-bound on L1**, not DRAM-bandwidth-bound"). Fill in the actual numbers from your report. Specificity is the deliverable.

## Project file layout

- Use a simple file split: one importable kernel module, one `test_<kernel>.py`, one `profile_<kernel>.py`, and optional `debug_<kernel>.py`.
- For TileLang-style kernels, treat the Python entrypoint name as the starting capture token and keep profiling in a dedicated runner rather than inside the implementation module.

## Further reading

- [`blackwell-cuda-programming.md`](blackwell-cuda-programming.md) — Blackwell-specific backend optimization guidance for the generated CUDA kernel once NCU has shown what to fix.
