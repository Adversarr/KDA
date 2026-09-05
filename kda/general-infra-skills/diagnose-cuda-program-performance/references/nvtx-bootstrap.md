# NVTX bootstrap

Add NVTX to a program that has none: decide, install, wire, choose ranges, verify they appear, then select a capture or kernel collection with those names. Templates: [nvtx_helper.h](../code_templates/nvtx_helper.h) and [nvtx_helper.py](../code_templates/nvtx_helper.py).

Flags here were current as of 2026-05. NVTX capture and selection expressions are version-sensitive; verify a flag against `--help` before treating a mismatch as a skill bug.

## Contents

1. Decide whether to instrument
2. Install the templates
3. Build wiring
4. Choose the first ranges
5. Verify the ranges exist
6. Select with the annotation
7. CUDA graphs
8. Framework auto-annotation is a bounded mode

## 1. Decide whether to instrument

The first pass already classifies idle from CUDA rows. Recapture with annotation only after both hold:

1. A clean baseline exists — first-pass step 2 in [SKILL.md](../SKILL.md) section 6 (synchronized outer timer, no profiler).
2. First-pass step 4 quoted `## No-CUDA time by host API in flight` and still left **material idle unclassified** by that table.

`(unattributed)` in `## Kernel device time by launching NVTX range` is leftover launch attribution, not an idle owner and not this trigger. Stay on the unannotated digest when kernel and API rows already name the interval, when one source phase already owns it, or when step 2 has no number yet.

Two constraints before writing any range:

- **NVTX is not a timer.** Range durations are annotation, collected only under a tool, describing host intervals. Wall-time authority stays the step-2 outer timer.
- **A range measures the host call, not the GPU work it queues.** Keep the range around the dispatch; let Systems correlate enclosed launches. Both templates state this at the top; the header has a diagram.

*Done when* the investigation either continues from the host-API idle table, or names the unclassified-idle interval that annotation will own.

## 2. Install the templates

| File | Layer | Needs |
|---|---|---|
| [../code_templates/nvtx_helper.h](../code_templates/nvtx_helper.h) | native C++ / CUDA | `nvtx3/nvtx3.hpp` or the v3 C header `nvtx3/nvToolsExt.h`; C++11 |
| [../code_templates/nvtx_helper.py](../code_templates/nvtx_helper.py) | host Python | NVIDIA's `nvtx` package, or a CUDA-enabled PyTorch build |

Copy whichever layers the program has, then replace exactly two tokens in each: `project` with the project identifier in lowercase, and `PROJECT` with the same identifier uppercased. `grep -n 'project\|PROJECT'` on the copies must come back empty — an unreplaced token means two projects in one process collide in the same domain.

The Python module resolves its backend once, at import: NVIDIA `nvtx` first (domains and categories), then `torch.cuda.nvtx` (neither), then a no-op only if `PROJECT_NVTX` is `off`. With neither package importable and no waiver, it raises `NvtxBackendUnavailable`, naming both remedies. **That is an Ask, not a Detect** — installing a package and silencing instrumentation have different consequences for the next profile. Put the choice to the human and wait.

Report `NVTX_BACKEND_NAME` alongside any profile taken with the Python layer. It decides whether domains and categories were recorded.

*Done when* each copied file greps clean and, for Python, the human has answered the backend Ask.

## 3. Build wiring

NVTX v3 is header-only. There is nothing to link: `-lnvToolsExt` is a v2 leftover, and on glibc older than 2.34 `-ldl` may still be needed.

Usually no wiring is required, because the CUDA Toolkit include path already provides `nvtx3/`. A translation unit that already compiles against CUDA compiles against NVTX.

When the Toolkit headers are not on the path, vendor the supported branch:

```bash
git clone --depth 1 --branch release-v3 https://github.com/NVIDIA/NVTX.git third_party/NVTX
```

For CMake, NVIDIA's configuration exposes header-only interface targets:

```cmake
add_subdirectory(third_party/NVTX/c)
target_link_libraries(my_extension PRIVATE nvtx3-cpp)
```

For a PyTorch extension, `CUDAExtension` already adds the CUDA include path; vendoring only needs `include_dirs=["third_party/NVTX/c/include"]`.

**Two NVTX headers, shipped independently.** The C++ wrapper `nvtx3/nvtx3.hpp` and the C API `nvtx3/nvToolsExt.h` do not always travel together: some CUDA and PyTorch images carry the C header alone. The template prefers the wrapper, falls back to the C API, and presents the same names and macros either way. Which path compiled is `PROJECT_NVTX_CPP_API` or `PROJECT_NVTX_C_API`.

An image can support NVTX fully and still lack the C++ wrapper. Treat a missing wrapper as a fallback — the empty band looks like an idle program. If the build prints the template's `#pragma message`, *neither* header was found and every native annotation compiled to nothing: fix the include path. `PROJECT_NVTX_ENABLED=0` silences the message for a deliberate build, not a broken one. Grep the log for it; it stays a warning under `-Werror` and scrolls past a ninja or setuptools success.

*Done when* a native unit defines `PROJECT_NVTX_CPP_API` or `PROJECT_NVTX_C_API`, or the log's `#pragma message` has a fixed include path.

## 4. Choose the first ranges

Budget roughly six ranges for the first annotated capture and add only what a named unanswered question demands:

```text
stage / phase
  transaction / function
    selected subphase
      CUDA API and kernel rows   <- already named by the profiler
```

| Annotate | Leave unnamed |
|---|---|
| Request, batch, step, epoch boundary | Every scalar helper or accessor |
| Algorithm phase; native extension entry | Every kernel launch, allocation, or copy |
| A repeated semantic transaction | Anything below roughly 1 microsecond |
| A materialization, publication, audit, or synchronization boundary | A wrapper whose range is one-to-one with a CUDA row |
| A native internal phase owning several launches | Both a Python and a native range for the same boundary |

Then keep the annotation aggregable:

- **Stable, low-cardinality messages.** `"update_kv_cache"`, not `"update_kv_cache seq=317"`. A formatted message defeats aggregation and costs time with no profiler attached. A varying value belongs in a payload — `PROJECT_NVTX_PAYLOAD` in the native template. The Python template carries messages and categories only, so drop the value or keep it in a separate coarse range name.
- **One domain per layer.** The templates ship `project` for host and `project.native` for native, independently filterable, so the two do not nest into one visual stack. Section 6 uses that. To trade filtering for one hierarchy, move both layers to the global domain as each template's Domains note describes.
- **Categories subdivide a domain**, with a slash where hierarchy helps: `kernel/attention`, `memory/pool`. The templates ship a starter set to edit.
- **Push/pop is a per-thread stack.** Work that begins on one thread and finishes on another uses the handle form (`nvtx_range_start` / `nvtx_range_end`); push and pop stay in one layer.

Name threads and streams once during setup, with `nvtx_name_current_thread` and `nvtx_name_stream` from the native template. The Python template has no equivalent; in a Python-only program, call those through a native extension the project already loads, or accept numeric rows.

*Done when* each chosen range is a stable name at a semantic boundary, with a domain and (if used) a category.

## 5. Verify the ranges exist

See the annotation in a trace before treating it as evidence. Capture, then check these:

```bash
nsys profile --trace=cuda,nvtx --force-overwrite=false --output=/path/to/reports/nvtx-check COMMAND
```

- **Which path compiled, before reading the trace.** Print `PROJECT_NVTX_CPP_API` and `PROJECT_NVTX_C_API` for the native layer and `NVTX_BACKEND_NAME` for the host layer, once at startup on any unfamiliar image.
- **The rows appear.** No NVTX rows means a no-op backend, a compiled-out header, or a filtered domain — read the path print first.
- **Nesting matches the source.** A child that shows as a sibling is an unbalanced push/pop, or a cross-thread range that should use the handle form.
- **Counts are plausible.** A range expected once per step appearing thousands of times is instrumentation inside a loop you did not intend.
- **Clean wall time did not move.** Re-run the step-2 timing with the annotation compiled in and no profiler attached. A moved number invalidates comparisons against earlier baselines.

`nsys stats --help-reports` lists the report names this version installs; the NVTX summaries give per-range counts and totals without the GUI. Read a GPU-projection report as *range plus the GPU work its launches produced*.

*Done when* the check capture shows the intended names, nesting, and counts, and the unprofiled step-2 number is unchanged.

## 6. Select with the annotation

Stable range names are a machine-selectable interface. Capture only steady state:

```bash
nsys profile --trace=cuda,nvtx \
  --capture-range=nvtx --nvtx-capture='step@project' \
  COMMAND
```

The `range@domain` form selects one layer. With a repeated range, `--capture-range-end` controls whether collection stops after the first instance; the default ends the session. Verify both flags against `--help`.

`profiler_capture()` in the Python template covers `--capture-range=cudaProfilerApi`, which is easier when the window is chosen by iteration index rather than by name.

The same annotation restricts Nsight Compute to kernels launched inside a range:

```bash
ncu --nvtx --nvtx-include 'project.native@launch_attention/' COMMAND
```

`/` carries push/pop nesting meaning, and include and exclude rules compose. Check `ncu --help` and confirm the action count before trusting a filtered collection. Hand off to `ncu-report` if installed; otherwise see [profiling-and-attribution.md](profiling-and-attribution.md) section 10.

*Done when* the selected capture or NCU run contains only the intended range instances, and the action or range count matches expectation.

## 7. CUDA graphs

A graph separates host execution at *capture* from execution at *replay*. Ranges written around CUDA calls during capture describe building the graph once; they do not re-emit on each launch, so a replay-heavy workload can look almost unannotated while its host ranges look cheap.

Annotate the two separately — a `graph_capture` range and a `graph_launch` range — and attribute replay cost to the launch, not to the captured body. Recent Nsight Systems versions add graph-specific NVTX projection modes; check the version's own documentation before interpreting range placement inside a graph.

*Done when* capture and replay each have their own range, and replay cost is attributed to `graph_launch`.

## 8. Framework auto-annotation is a bounded mode

`torch.autograd.profiler.emit_nvtx()` and `python -m nvtx` annotate every operation or every Python function automatically. Both are diagnostics: they can emit millions of ranges, the overhead lands on the fragmented short-call paths under investigation, and a perturbed trace is still usable for *ownership* and *counts*. They are never the timing authority and never replace the hand-written semantic ranges that survive into the next investigation.

`emit_nvtx` adds sequence numbers correlating a forward operation with its backward counterpart. Reach for it when that correlation is the question. Keep routine and diagnostic captures separate; [profiling-and-attribution.md](profiling-and-attribution.md) section 9 is the framework-profiler rung.

*Done when* auto-annotation is scoped to the ownership or correlation question, and the step-2 timer remains the wall-time authority.
