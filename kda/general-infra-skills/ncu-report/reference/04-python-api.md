# `ncu_report` Python API

Use the Python module, not CLI output, for anything beyond a quick look. This API works the same whether the `.ncu-rep` came from handwritten CUDA or from a Python DSL kernel that compiled into CUDA.

Module path (adjust for your CUDA version):
```bash
export PYTHONPATH=$PYTHONPATH:/usr/local/cuda-13.2/nsight-compute-2026.1.0/extras/python
python3 -c "import ncu_report; print('OK')"
```

---

## Basic loading

```python
import sys
sys.path.insert(0, "/usr/local/cuda-13.2/nsight-compute-2026.1.0/extras/python")
import ncu_report

report = ncu_report.load_report("path/to/full_<tag>.ncu-rep")

# A report can contain multiple "ranges" (each range = one profiled region).
# In practice, with -c 1 you have exactly one range containing one action (= one kernel launch).
rng = report.range_by_idx(0)
action = rng.action_by_idx(0)

print(f"Kernel demangled name: {action.name()}")       # e.g. some generated name containing "some_kernel"
print(f"Total metrics collected: {len(action.metric_names())}")
```

For DSL workflows, `action.name()` is the first thing to inspect after collection. Confirm that the captured demangled name contains the Python/kernel token you targeted with `-k "regex:.*<token>.*"`.

Example:

```python
captured = action.name()
token = "some_kernel"
print("token matched:", token in captured)
```

---

## Reading a single metric

Use `Metrics` from [`../helpers/ncu_utils.py`](../helpers/ncu_utils.py). A bare
`action[name].value()` raises on any name this chip or capture does not have, and the
obvious `try/except` fix returns `None` for three different situations — the chip never had
the metric, this capture did not collect it, and it read back empty. Only the middle one
is a reason to recapture, so a diagnosis has to tell them apart.

```python
from ncu_utils import Metrics, load_report

report, action = load_report("reports/full_<tag>.ncu-rep")
m = Metrics(action)

print(m.resolve("sm_sol").text())        # resolves aliases across ncu versions
print(m.status("dram__bytes_read.sum.per_second"))   # absent | novalue | zero | value
duration_ns = m.get("gpu__time_duration.sum")        # None when not measured
```

Metric names differ between GPU generations — see [`08-metric-names.md`](08-metric-names.md).
`Metrics.resolve` takes a logical field name and reports which concrete name it found, so
an sm_80 report and an sm_100 report can be read by the same code.

`ncu_utils.safe` still exists for a throwaway script where the absent-versus-zero
distinction does not change the answer. Do not build a diagnosis on it.

---

## Enumerating available metrics

```python
# Full list — 2000+ metrics for --set full
all_names = action.metric_names()

# Filter by pattern
for name in sorted(all_names):
    if "warps_issue_stalled" in name and "ratio" in name:
        print(name, "=", m.get(name))
```

This is how you discover the *actual* metric names available on your GPU instead of guessing.

---

## Per-instance (per-SM, per-PC, per-time-sample) values

Many metrics have multiple values per collection. `value()` returns the aggregate (sum / avg depending on `rollup_operation`), but you can also enumerate the individual samples.

```python
m = action["pmsampling:smsp__warps_issue_stalled_long_scoreboard.avg"]
n = m.num_instances()              # e.g., 1660 for a PM-sampled metric
print(f"instances: {n}")

vals = []
for i in range(n):
    try:
        v = m.as_double(i)
    except Exception:
        try:
            v = float(m.as_uint64(i))
        except Exception:
            v = None
    vals.append(v)
```

For PM sampling this is the timeline — index `i` is a time-ordered sample. Bucket these for an ASCII plot (see `helpers/plot_timeline.py`).

---

## Per-PC → per-source-line mapping

A report containing the `SourceCounters` section has per-PC samples. Map them to source lines via `action.source_info(pc)`:

```python
def per_pc_stalls(action, stall_metric):
    m = action[stall_metric]
    n = m.num_instances()
    if n == 0 or not m.has_correlation_ids():
        return []
    cor = m.correlation_ids()
    out = []
    for i in range(n):
        pc = cor.as_uint64(i)
        val = m.as_uint64(i)
        si = action.source_info(pc)
        if si is None:
            file, line = "?", 0
        else:
            file, line = si.file_name(), si.line()
        out.append((file, line, val))
    return out

stalls = per_pc_stalls(action, "smsp__pcsamp_warps_issue_stalled_long_scoreboard")
```

Aggregate by `(file, line)` and sort by total to get hottest stall lines. See `helpers/extract_stall_hotspots.py` for a complete implementation.

**If every PC maps to `?:0`, the report has no source mapping, and aggregating by line will pile the whole kernel onto one row.** Compiling with `-lineinfo` is necessary but not sufficient: a report captured without `--import-source yes` returns `None` from `source_info` for every PC even though the SASS and the PC-sampling counts are present and correct. Detect that case and fall back to attributing by PC, disassembling each hot address with `action.sass_by_pc(pc)`. An instruction-level answer is still actionable; a table of one line is not.

---

## Discovering Value Kind

Each metric has a value kind (uint64, double, float, string). Use `m.kind()` to check before calling the right accessor:

```python
def metric_val(m, i=None):
    k = m.kind()
    VK = m.ValueKind_UINT64, m.ValueKind_DOUBLE, m.ValueKind_FLOAT, m.ValueKind_STRING
    if i is None:
        return m.value()          # aggregate
    if k == m.ValueKind_UINT64:
        return m.as_uint64(i)
    if k in (m.ValueKind_DOUBLE, m.ValueKind_FLOAT):
        return m.as_double(i)
    if k == m.ValueKind_STRING:
        return m.as_string(i)
    # Try generic conversions as fallbacks
    try:
        return m.as_uint64(i)
    except Exception:
        return m.as_double(i)
```

---

## Useful `action` / `metric` methods

```python
# Action (= one kernel launch's profile data)
action.name()                   # FUNCTION name (matches -k regex: by default)
action.name(1)                  # demangled name, with the full signature
action.name(2)                  # mangled name
action.metric_names()           # list of all metrics
action.metric_by_name(name)     # same as action[name]
action.source_info(pc)          # PC → SourceInfo (file, line); see caveat below
action.sass_by_pc(pc)           # ONE address → SASS string (not a dict)
action.ptx_by_pc(pc)            # ONE address → PTX string
action.rule_results_as_dicts()  # rule-engine output; presence follows the reader version

# Metric
m.value()                       # aggregate value
m.unit()                        # string, e.g. "%" or "cycle"
m.kind()                        # value kind (UINT64 / DOUBLE / ...)
m.rollup_operation()            # AVG / MAX / MIN / SUM / NONE
m.num_instances()               # per-instance count (0 if aggregate only)
m.has_correlation_ids()         # True for per-PC metrics
m.correlation_ids()             # parallel array for num_instances
m.description()                 # human-readable metric description
```

---

## Exploring when you don't know the right metric name

```python
# Print all metric names sorted
for n in sorted(action.metric_names()):
    print(n)

# Print metrics matching a pattern with their current value
import re
pat = re.compile(r"dram__bytes.*sum")
for n in sorted(action.metric_names()):
    if pat.search(n):
        try:
            v = action[n].value()
            print(f"{n} = {v}  (unit: {action[n].unit()})")
        except Exception as e:
            print(f"{n} = ERROR {e}")
```

This is how [`08-metric-names.md`](08-metric-names.md) was built — by enumerating everything available on sm_100.

---

## Comparing two reports

Run [`../helpers/analyze_reports.py`](../helpers/analyze_reports.py) with both reports in
one invocation:

```bash
python3 "$HELPERS/analyze_reports.py" --run-dir "$CMP_DIR" \
    --report v1.ncu-rep --tag v1 \
    --report v2.ncu-rep --tag v2
```

It writes `analysis/compare_v1_vs_v2.txt`: a per-metric table over `KEY_METRICS`, led by a
comparability gate that flags a differing kernel signature, device, block size, or grid
size. A percentage change between two reports that fail that gate is a comparison of
different work, and reading it as a speedup is the most expensive mistake available here.

That is the reason not to write the twenty-line comparison loop yourself. Useful for:

- two autotuned variants
- two dispatch paths
- two shape-specific specializations
- before/after changes to the Python/kernel implementation

---

## Extracting NCU's rule suggestions

The rule engine results — the "OPT Est. Speedup: X%" bullets — are accessible as structured data. **The dict shape is version-specific**, so print the keys of one entry before writing a parser against it:

```python
results = list(action.rule_results_as_dicts())
print(sorted(results[0].keys()))
```

The shape verified on a real report is nested, and the speedup lives under its own key rather than at the top level:

```python
for result in action.rule_results_as_dicts():
    rule = result.get("rule_identifier", "?")     # e.g. "CPIStall"
    section = result.get("section_identifier", "?")
    message = (result.get("rule_message") or {})  # {"title", "message", "type"}
    estimate = result.get("speedup_estimation") or {}
    pct = estimate.get("speedup")                 # float, or absent
    kind = {1: "local", 2: "global"}.get(estimate.get("type"), "unknown")
    label = "n/a" if pct is None else f"{pct:.1f}% {kind}"
    print(f"[{label}] {rule} ({section}): {message.get('title') or ''}")
```

Three things this API will do to you:

- **Not every rule estimates a speedup.** On the report checked, 13 of 17 rules carried `speedup_estimation`; the rest have none. A missing estimate means the rule did not offer one, so do not coerce it to `0.0` and then sort — that silently buries rules and is indistinguishable from "no gain available".
- **Local and global estimates are not comparable.** `type: 1` is relative to the analyzed section, `type: 2` is relative to the whole kernel. Carry the kind alongside the number.
- **The estimates overlap and are not additive.** A real full report summed to well over 100% across rules. Treat them as a ranking of where to look, never as a budget.

Two conditions both return nothing here, and they call for different responses.

**The reader module is too old to expose the method.** Availability follows the version doing the *reading*, not the version that captured the file: a 2026.x module reads a 2024.1 capture and exposes all 17 rules. So check `hasattr(action, "rule_results_as_dicts")` rather than inferring from the capture version — assuming an old capture means no rules skips an API that works and sends you to a CLI you may not have.

**The report was captured with an explicit `--metrics` list**, so the rule engine never ran. An empty list here is then a fact about the capture, not about the kernel. Reporting it as "Nsight Compute found no problems" is the most expensive available misreading; say instead that rules were not collected and name what a `--set full` recapture would add.

When the method genuinely is unavailable, the CLI has the same data — but only on a host that has an `ncu` binary, which a report-reading machine often does not:

```bash
ncu --import "$REPORT" --page details --csv    # columns include Rule Name, Rule Type, Estimated Speedup
```

---

## Saving everything for later

Always archive the full metric dump:

```python
import json
from pathlib import Path

def dump_all(action, outpath):
    rows = []
    for name in sorted(action.metric_names()):
        try:
            m = action[name]
            rows.append({
                "name": name,
                "value": m.value(),
                "unit": m.unit() if hasattr(m, "unit") else "",
            })
        except Exception as e:
            rows.append({"name": name, "error": str(e)})
    Path(outpath).write_text(json.dumps(rows, indent=1, default=str))

dump_all(action, "analysis/metrics_all_<tag>.json")
```

This makes future re-analysis cheap: the raw data lives as JSON, you don't need to reopen the `.ncu-rep`.

---

## Gotchas

- **`KeyError` on a metric that "should" exist**: the metric has a different name on this GPU. Check [`08-metric-names.md`](08-metric-names.md) or enumerate with `action.metric_names()`.
- **`num_instances() == 0`** but you expected per-instance data: the metric wasn't collected in instanced mode, or the section that produces it wasn't requested. Re-run ncu with the right `--section`.
- **`has_correlation_ids() == False`** on a source-level metric: `-lineinfo` wasn't on the compile line. Rebuild.
- **`source_info(pc)` returns None**: same as above — rebuild with `-lineinfo`.
- **Metric value is a string** like `"PolicySpread"`: it's an enum, use `m.as_string()` or `m.value()` and expect a string.
