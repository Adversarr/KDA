#!/usr/bin/env python3
"""Turn one or more .ncu-rep files into a diagnosis-ready summary.

The point of this script is that the first read of a report should not require
writing any code. It answers, for each launch: what device this ran on, whether
the grid fills it and how it fails to, what the kernel actually requested in
shared memory and registers, where the warps stalled, whether the accesses
coalesce, and what Nsight Compute's own rules concluded. Every number carries the
metric name it came from, and a field that was not measured says so rather than
printing a zero.

Produces in ``<run_dir>/analysis/``:
    summary_<tag>.txt          — the diagnosis-ready digest, read this first
    metrics_all_<tag>.json     — every metric, archival
    metrics_key_<tag>.txt/json — curated key metrics
    compare_<tag1>_vs_<tag2>.txt (when two or more reports given) — side-by-side

Usage examples:
    # Single report
    python3 analyze_reports.py --run-dir profile/myrun \\
            --report profile/myrun/reports/full_<tag>.ncu-rep --tag <tag>

    # Every launch in the report, not just the first
    python3 analyze_reports.py --run-dir profile/myrun \\
            --report profile/myrun/reports/full_<tag>.ncu-rep --tag <tag> --all-actions

    # Multiple reports -> side-by-side compare
    python3 analyze_reports.py --run-dir profile/myrun \\
            --report profile/myrun/reports/full_<tag1>.ncu-rep --tag <tag1> \\
            --report profile/myrun/reports/full_<tag2>.ncu-rep --tag <tag2>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make ncu_utils importable whether we're invoked from the skill dir or a run dir
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ncu_utils import (  # noqa: E402
    KEY_METRICS, Metrics, derived_memory, dump_all_metrics, fill_verdict,
    fmt_num, iter_actions, load_report, reader_version, rule_speedups, safe,
    shared_memory, stall_form, stall_reasons, uncollected_stalls,
)


def fmt_bytes(value):
    """Render a byte count with a unit, or a dash when absent."""

    if value is None:
        return "-"
    for unit, scale in (("GiB", 1 << 30), ("MiB", 1 << 20), ("KiB", 1 << 10)):
        if value >= scale:
            return f"{value / scale:,.2f} {unit}"
    return f"{value:,.0f} B"


def summarize(report, action, tag, label):
    """Render the diagnosis-ready digest for one launch.

    Args:
        report: The loaded report, used for the reader version.
        action: One profiled kernel launch.
        tag: Caller's short name for the report.
        label: Which launch this is, e.g. ``"range 0 action 0"``.

    Returns:
        List of output lines.
    """

    m = Metrics(action)
    out = []
    add = out.append

    add("=" * 78)
    add(f"{tag}  ({label})")
    add("=" * 78)

    # --- Identity. Every grading step below depends on the SM count, so it is
    # read from the report rather than assumed.
    device = m.get("device__attribute_display_name") or "unknown device"
    cc_major = m.get("device__attribute_compute_capability_major")
    cc_minor = m.get("device__attribute_compute_capability_minor")
    sm_count = m.get("device__attribute_multiprocessor_count")
    add("")
    add("## Device and report")
    add(f"  device        {device}  (CC {cc_major}.{cc_minor}, {sm_count} SMs)")
    add(f"  reader        ncu_report {reader_version(report)}  "
        f"(the module reading the file, not the capture)")
    try:
        sections = [s.name() for s in action.sections()]
    except Exception:
        sections = []
    add(f"  sections      {len(sections)}: {', '.join(sections)}")
    sparse = len(sections) == 1 and "profiler metrics" in sections[0].lower()
    if sparse:
        add("  NOTE          single 'command line profiler metrics' section: this is a")
        add("                --metrics capture, not --set full. Rules never ran, and")
        add("                whole sections are absent. Absent is not zero.")
    add(f"  kernel        {action.name()}")
    try:
        add(f"  signature     {action.name(1)}")
    except Exception:
        pass

    # --- Fill. The distinction between idle SMs and an unfilled occupancy wave
    # is the one the playbook used to blur, and it changes the prescription.
    fill = fill_verdict(m)
    add("")
    add("## Launch geometry and fill")
    add(f"  grid          {fmt_num(fill['grid'])} blocks x {fmt_num(fill['block'])} "
        f"threads = {fmt_num(m.get('launch__thread_count'))} threads")
    add(f"  registers     {fmt_num(m.get('launch__registers_per_thread'))} per thread")
    for name, (value, unit) in fill["limits"].items():
        short = name.replace("launch__occupancy_limit_", "")
        add(f"  limit {short:<12} {fmt_num(value)} {unit}")
    add(f"  blocks/SM     {fmt_num(fill['blocks_per_sm'])}  (min of the limits in blocks)")
    if fill["unconverted_limits"]:
        add(f"  WARNING       limits in an unexpected unit, excluded from the minimum "
            f"above, so blocks/SM may be too high: "
            f"{', '.join(fill['unconverted_limits'])}")
    add(f"  waves/SM      {fmt_num(fill['waves'])}")
    add(f"  VERDICT       {fill['verdict']}")

    smem = shared_memory(m)
    add(f"  shared mem    per_block={fmt_num(smem['per_block'])} "
        f"static={fmt_num(smem['static'])} dynamic={fmt_num(smem['dynamic'])} "
        f"driver={fmt_num(smem['driver'])}")
    add(f"                {smem['verdict']}")

    # --- Time and imbalance.
    add("")
    add("## Time")
    dur = m.resolve("duration", "gpu__time_duration.sum")
    if dur.ok and dur.unit == "ns":
        add(f"  duration      {dur.value / 1e6:,.3f} ms  (via {dur.name})")
    else:
        add(f"  duration      {dur.text()}  (via {dur.name})")
    cyc_avg = m.get("sm__cycles_active.avg")
    cyc_max = m.get("sm__cycles_active.max")
    cyc_min = m.get("sm__cycles_active.min")
    if cyc_avg:
        add(f"  SM cycles     avg {cyc_avg:,.0f}  min {cyc_min:,.0f}  max {cyc_max:,.0f}")
        if cyc_max:
            add(f"  imbalance     max/avg = {cyc_max / cyc_avg:.2f}  "
                f"(1.0 is perfectly balanced; the tail is the slowest SM)")

    # --- Throughput and occupancy, by logical field so an alias is transparent.
    add("")
    add("## Throughput and occupancy")
    for field in (
        "sm_sol", "memory_sol", "theoretical_occupancy", "achieved_occupancy",
        "fma_pipe", "tensor_pipe", "l1_hit_rate", "l2_hit_rate",
    ):
        r = m.resolve(field)
        add(f"  {field:<22} {r.text():<28} {'via ' + r.name if r.ok else '(' + r.name + ')'}")

    # --- Warp efficiency: divergence and predication, in threads per warp.
    add("")
    add("## Warp execution efficiency")
    eff = m.resolve("warp_efficiency")
    pred = m.resolve("warp_efficiency_pred_on")
    if eff.ok:
        add(f"  active threads/warp   {eff.value:.2f} of 32  "
            f"({eff.value / 32 * 100:.1f}% efficiency)")
    else:
        add(f"  active threads/warp   {eff.text()}")
    if pred.ok:
        add(f"  predicated-on/warp    {pred.value:.2f} of 32")
    uniform = m.resolve("branch_uniform", "smsp__sass_average_branch_targets_threads_uniform.pct")
    if uniform.ok:
        add(f"  uniform branches      {uniform.text()}  (100% means no divergent branching)")

    # --- Stalls. The form travels with the numbers because the two forms are
    # different quantities that must not be compared or added.
    rows = stall_reasons(m)
    add("")
    add(f"## Warp stall reasons — {stall_form(rows)}")
    if not rows:
        add("  none collected in this report")
    # The raw unit on these ratio metrics ("inst") describes the denominator, not
    # the quantity, so the header carries the meaning and the rows carry numbers.
    for r in rows[:8]:
        add(f"  {r.field:<22} {r.text(with_unit=False):<12} via {r.name}")
    if rows:
        add("  These are not concurrent warp counts. Rank them; do not sum them")
        add("  against a budget or normalize them to 100%.")
    absent_stalls = uncollected_stalls(m)
    if absent_stalls and rows:
        add(f"  {len(absent_stalls)} reason(s) have no value here, so the rows above are")
        add(f"  not a complete accounting: {', '.join(absent_stalls)}")

    # --- Coalescing, computed from sums because the average metrics are absent.
    add("")
    add("## Memory access quality")
    mem = derived_memory(m)
    for op, vals in mem.items():
        # A measured 0.0 is a fact about the kernel (no such access at all) and
        # must not render as the "-" that means "could not compute".
        spr = vals["sectors_per_request"]
        spr_text = f"{spr:.2f}" if spr is not None else "-"
        add(f"  global {op}    sectors={fmt_num(vals['sectors'])} "
            f"requests={fmt_num(vals['requests'])} sectors/request={spr_text}")
        bps = vals["bytes_per_sector"]
        if bps is not None:
            add(f"                bytes/sector={bps:.2f} of 32")
    add("  Fully coalesced 32-bit access is 4 sectors/request and 32 bytes/sector.")
    add("  32 sectors/request is one sector per thread — the worst case.")
    dram_read = m.get("dram__bytes_read.sum")
    dram_write = m.get("dram__bytes_write.sum")
    if dram_read is not None or dram_write is not None:
        add(f"  DRAM read={fmt_bytes(dram_read)} write={fmt_bytes(dram_write)}")
    # Bytes alone cannot separate a latency problem from a bandwidth one. The
    # percentage of peak can, and it is the number that decides whether moving
    # less data is even worth attempting.
    for field, label in (
        ("dram_read_pct", "DRAM read % of peak "),
        ("dram_write_pct", "DRAM write % of peak"),
    ):
        r = m.resolve(field)
        if r.ok:
            add(f"  {label} {r.text()}  (low % with high stalls means latency, "
                f"not bandwidth)")
    for field in ("local_ld_inst", "local_st_inst"):
        r = m.resolve(field)
        if r.ok and r.value:
            add(f"  {field:<16} {r.text()}  (non-zero local traffic means register spills)")

    # --- Rules. Ranked, never summed, and explicit when they did not run.
    add("")
    add("## Nsight Compute rules, ranked by estimated speedup")
    speedups = rule_speedups(action)
    if not speedups:
        if sparse:
            add("  no rule results, because this is a --metrics capture and rules never")
            add("  ran. Recapture with --set full to get them.")
        else:
            add("  no rule results, from a capture that does have sections. Either the")
            add("  reader module is too old to expose the rule API, or the rules ran and")
            add("  raised nothing. Check the reader version above before reading this as")
            add("  a clean bill of health.")
    # Print every rule, including the ones with no estimate. Truncating the list
    # to the highest percentages drops exactly the rules that decline to guess —
    # Small Grid among them — and those are often the ones naming the real
    # problem. A rule without a number is not a rule without a finding.
    for pct, kind, rule, text in speedups:
        shown = f"{pct:5.1f}% {kind:<7}" if pct is not None else "   no estimate"
        add(f"  {shown}  {rule}: {text}")
    if speedups:
        add("  Estimates overlap and are not additive; local applies to one section,")
        add("  global to the whole kernel. Read as a ranking, and note that a rule")
        add("  with no estimate is not a rule with no finding.")

    # --- What could not be measured, so a gap is never read as a zero.
    missing = []
    for field in (
        "achieved_occupancy", "memory_sol", "fma_pipe", "tensor_pipe",
        "warp_efficiency", "l1_hit_rate", "l2_hit_rate",
    ):
        r = m.resolve(field)
        if not r.ok:
            missing.append(f"  {field:<22} {r.status:<8} last name tried: {r.name}")
    if missing:
        add("")
        add("## Not measured in this report")
        add("  'absent' means the name is not in this report: either no such metric on")
        add("  this chip or ncu version, or a capture narrow enough not to include it.")
        add("  Check the section list above before concluding the chip lacks it.")
        add("  'novalue' means the name exists but this capture holds no value.")
        add("  Neither is zero, and neither is evidence about the kernel.")
        out.extend(missing)

    pm = m.pmsampling_names()
    add("")
    add(f"## PM sampling series with data: {len(pm)}")
    for name in pm[:12]:
        add(f"  {name}")
    if pm:
        add("  Plot these with plot_timeline.py to see the shape over time.")
    else:
        add("  This capture holds no PM time series, so utilization over time is")
        add("  unavailable. sm__cycles_active min/max still bounds the imbalance.")

    return out


def collect(report_path: Path, tag: str, analysis_dir: Path, all_actions: bool) -> dict:
    """Write the digest and metric archives for one report, returning key metrics."""

    report, first = load_report(report_path)
    actions = list(iter_actions(report)) if all_actions else [(0, 0, first)]
    print(f"[{tag}] {report_path.name}: {len(first.metric_names())} metrics, "
          f"{len(actions)} action(s) analyzed, kernel {first.name()}")

    lines = []
    for ri, ai, action in actions:
        lines.extend(summarize(report, action, tag, f"range {ri} action {ai}"))
        lines.append("")
    summary_path = analysis_dir / f"summary_{tag}.txt"
    summary_path.write_text("\n".join(lines))
    print(f"  -> summary_{tag}.txt   <- read this first")

    n = dump_all_metrics(first, analysis_dir / f"metrics_all_{tag}.json")
    print(f"  -> metrics_all_{tag}.json ({n} metrics)")
    if len(actions) > 1:
        # Only the digest is per-launch. Say so, or the metric archive and the
        # comparison table get read as covering every action they do not.
        print(f"     (metrics_all_ and metrics_key_ cover the first action only; "
              f"summary_{tag}.txt covers all {len(actions)})")

    key = {name: safe(first, name) for name in KEY_METRICS}
    key["__kernel_name__"] = first.name()
    # The demangled signature is what actually settles whether two reports profile
    # the same kernel: two variants can share a mangled name and take different
    # arguments, and ranking their durations would then compare different work.
    try:
        key["__signature__"] = first.name(1)
    except Exception:
        key["__signature__"] = None
    key["__derived_sectors_per_request_ld__"] = (
        derived_memory(Metrics(first))["ld"]["sectors_per_request"]
    )
    (analysis_dir / f"metrics_key_{tag}.json").write_text(
        json.dumps(key, indent=2, default=str)
    )
    with open(analysis_dir / f"metrics_key_{tag}.txt", "w") as f:
        f.write(f"===== {tag} =====\nKernel: {first.name()}\n\n")
        for name, value in key.items():
            if name.startswith("__"):
                continue
            f.write(f"{name:95s} = {value}\n")
    print(f"  -> metrics_key_{tag}.{{json,txt}}")
    return key


def compare(collected: dict, analysis_dir: Path):
    """Write a side-by-side table of key metrics across tags."""

    tags = list(collected.keys())
    if len(tags) < 2:
        return
    out_path = analysis_dir / f"compare_{'_vs_'.join(tags)}.txt"
    with open(out_path, "w") as f:
        col_w = max(20, max(len(t) for t in tags) + 2)
        # A comparison is only meaningful for the same kernel on the same device,
        # so state the identity before the numbers rather than after.
        f.write("Comparability gate — a duration ranking below is only meaningful if\n")
        f.write("these agree. Different signature, device, or work size means the\n")
        f.write("reports measure different things and the ranking says nothing.\n\n")
        for t in tags:
            f.write(f"  {t:<12} {collected[t].get('__kernel_name__')}\n")
            f.write(f"  {'':<12} signature={collected[t].get('__signature__')}\n")
            f.write(f"  {'':<12} device={collected[t].get('device__attribute_display_name')} "
                    f"grid={fmt_num(collected[t].get('launch__grid_size'))} "
                    f"block={fmt_num(collected[t].get('launch__block_size'))} "
                    f"threads={fmt_num(collected[t].get('launch__thread_count'))}\n")
            f.write(f"  {'':<12} work proxies: global_st_inst="
                    f"{fmt_num(collected[t].get('smsp__sass_inst_executed_op_global_st.sum'))}"
                    f" dram_read={fmt_num(collected[t].get('dram__bytes_read.sum'))}"
                    f" sectors/request(ld)="
                    f"{fmt_num(collected[t].get('__derived_sectors_per_request_ld__'))}\n")

        # Name the divergences rather than leaving them for the reader to spot in a
        # 100-row table: an unnoticed signature change is how a ranking of unlike
        # things gets published.
        for field, label in (
            ("__signature__", "signature"),
            ("device__attribute_display_name", "device"),
            ("launch__block_size", "block size"),
            # Grid size is the work proxy. Two reports of the same kernel at
            # different grids compare a small problem against a large one, and
            # the duration ranking says nothing about the code.
            ("launch__grid_size", "grid size"),
        ):
            values = {str(collected[t].get(field)) for t in tags}
            if len(values) > 1:
                f.write(f"\n  DIVERGENT {label}: these reports do not share one "
                        f"{label}. Do not rank durations across them without saying\n"
                        f"  what changed and why the comparison still holds.\n")
        f.write("\n")
        f.write(f"{'Metric':<95}")
        for t in tags:
            f.write(f"{t:>{col_w}}")
        f.write("\n" + "-" * (95 + col_w * len(tags)) + "\n")
        for name in KEY_METRICS:
            f.write(f"{name:<95}")
            for t in tags:
                v = collected[t].get(name, "N/A")
                if isinstance(v, (int, float)):
                    v = f"{v:.4g}"
                # Truncate so a long string value cannot run into the next column.
                cell = str(v)[: col_w - 1]
                f.write(f"{cell:>{col_w}}")
            f.write("\n")
    print(f"compare -> {out_path}")


def main():
    ap = argparse.ArgumentParser(
        description="Extract key NCU metrics and compare reports from Python DSL runners or fallback CUDA targets."
    )
    ap.add_argument("--run-dir", type=Path, required=True,
                    help="The profile run directory — outputs go to <run-dir>/analysis/")
    ap.add_argument("--report", type=Path, action="append", required=True,
                    help="Path to a .ncu-rep file. Can be passed multiple times.")
    ap.add_argument("--tag", type=str, action="append", required=True,
                    help="Short tag for each report. Must be passed once per --report.")
    ap.add_argument("--all-actions", action="store_true",
                    help="Summarize every launch in the report, not only the first.")
    args = ap.parse_args()

    if len(args.report) != len(args.tag):
        ap.error("--report and --tag counts must match")

    analysis_dir = args.run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    collected = {}
    for rep, tag in zip(args.report, args.tag):
        if not rep.exists():
            print(f"[skip] {rep} does not exist", file=sys.stderr)
            continue
        collected[tag] = collect(rep, tag, analysis_dir, args.all_actions)

    compare(collected, analysis_dir)


if __name__ == "__main__":
    main()
