#!/usr/bin/env python3
"""ASCII-plot PM sampling timeseries from .ncu-rep files.

PM sampling metrics (those prefixed `pmsampling:`) have per-instance values
that form a time-ordered series across the kernel's execution. Plotting the
series reveals tail effects, pipeline bubbles, and sawtooth patterns that
are invisible in aggregate metrics.

Produces `<run-dir>/analysis/pm_timeline_plots.txt` with ASCII plots for each
requested metric.

Usage:
    python3 plot_timeline.py --run-dir profile/myrun \\
            --report profile/myrun/reports/full_<tag>.ncu-rep --tag <tag>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ncu_utils import Metrics, load_action, per_instance_values  # noqa: E402


# Preferred metrics, in the order they are worth looking at. Which of these
# exist depends on both the architecture and the ncu version: stall-reason
# series are the most informative but are absent from many 2024.x captures,
# and DRAM throughput is `dramc__` on Ampere and `dram__` on Blackwell. When
# none of these populate, the plotter discovers whatever the report does carry
# rather than reporting "no instances" and stopping — an empty plot is not
# evidence that the kernel had flat utilization.
PREFERRED_METRICS = [
    "pmsampling:smsp__warps_issue_stalled_long_scoreboard.avg",
    "pmsampling:smsp__warps_issue_stalled_short_scoreboard.avg",
    "pmsampling:smsp__warps_issue_stalled_wait.avg",
    "pmsampling:smsp__warps_issue_stalled_dispatch_stall.avg",
    "pmsampling:smsp__warps_issue_stalled_math_pipe_throttle.avg",
    "pmsampling:smsp__warps_issue_stalled_mio_throttle.avg",
    "pmsampling:sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "pmsampling:sm__warps_active.avg.pct_of_peak_sustained_active",
    "pmsampling:dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "pmsampling:l1tex__throughput.avg.pct_of_peak_sustained_active",
    # Names seen on Ampere / 2024.1 captures, where the ones above are absent.
    "pmsampling:sm__cycles_active.avg",
    "pmsampling:sm__inst_executed_realtime.avg.pct_of_peak_sustained_elapsed",
    "pmsampling:dramc__throughput.avg.pct_of_peak_sustained_elapsed",
    "pmsampling:l1tex__data_pipe_lsu_wavefronts.avg",
    "pmsampling:sm__ctas_launched.sum",
]


def choose_metrics(action, requested=None):
    """Pick which PM series to plot, discovering names when the defaults are empty.

    Args:
        action: One profiled kernel launch.
        requested: Explicit metric names from the caller, which are honored as-is.

    Returns:
        ``(names, note)`` where ``note`` explains the choice for the output header.
    """

    if requested:
        return requested, "caller-specified metrics"

    available = set(Metrics(action).pmsampling_names(populated_only=True))
    preferred = [name for name in PREFERRED_METRICS if name in available]
    if preferred:
        extra = len(available) - len(preferred)
        note = f"{len(preferred)} preferred series populated"
        if extra > 0:
            note += f"; {extra} further pmsampling series in this report"
        return preferred, note
    if available:
        return sorted(available), (
            "none of the preferred names are populated in this report, so these "
            "were discovered from it (typical of Ampere / 2024.x captures)"
        )
    return [], "this report carries no populated pmsampling series at all"


def ascii_plot(vals, label, max_rows=20, max_cols=80):
    """Render one PM-sampling series as an ASCII plot.

    The whole series is plotted, zeros included. A run of zeros at the end of
    ``sm__cycles_active`` is a tail where most SMs had already finished — the
    exact shape this plot exists to reveal — so trimming it would delete the
    finding.

    Args:
        vals: Per-instance values in sample order; ``None`` is read as zero.
        label: Metric name for the header.
        max_rows: Plot height.
        max_cols: Plot width; samples are averaged into this many buckets.

    Returns:
        List of output lines.
    """

    if not vals:
        return [f"{label}: no data"]

    vals = [v if v is not None else 0.0 for v in vals]
    n = len(vals)
    lead = next((i for i, v in enumerate(vals) if v > 0), n)
    trail = next((i for i, v in enumerate(reversed(vals)) if v > 0), n)
    if lead == n:
        return [f"{label}: all zero across {n} samples"]

    # Boundaries are computed from the sample index so the last bucket reaches
    # the final sample. Sizing every bucket at n // ncols instead drops the
    # remainder, which is the tail.
    ncols = min(max_cols, n)
    buckets = []
    for c in range(ncols):
        s = (c * n) // ncols
        e = ((c + 1) * n) // ncols
        chunk = vals[s:e] or vals[s:s + 1]
        buckets.append(sum(chunk) / len(chunk))
    mx = max(buckets) if buckets else 1.0
    if mx == 0:
        mx = 1.0

    lines = [f"\n{label}",
             f"  (n={n} samples, leading_zero={lead}, trailing_zero={trail}, max={mx:.3g})"]
    for r in range(max_rows, 0, -1):
        threshold = mx * r / max_rows
        row = "".join("#" if b >= threshold else " " for b in buckets)
        lines.append(f"  {threshold:8.2g} | {row}")
    lines.append("  " + " " * 10 + "-" * len(buckets))
    lines.append("  " + " " * 10 + " (time →)")
    return lines


def main():
    ap = argparse.ArgumentParser(
        description="Render PM-sampling timelines from reports captured through a Python runner or fallback CUDA target."
    )
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--report", type=Path, action="append", required=True)
    ap.add_argument("--tag", type=str, action="append", required=True)
    ap.add_argument("--metric", type=str, action="append", default=None,
                    help="Override default metric list.")
    ap.add_argument("--rows", type=int, default=20)
    ap.add_argument("--cols", type=int, default=80)
    args = ap.parse_args()

    if len(args.report) != len(args.tag):
        ap.error("--report and --tag counts must match")

    analysis_dir = args.run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    out_lines = []
    for rep, tag in zip(args.report, args.tag):
        if not rep.exists():
            print(f"[skip] {rep} not found", file=sys.stderr)
            continue
        action = load_action(rep)
        metrics, note = choose_metrics(action, args.metric)
        out_lines.append(f"\n{'=' * 60}\n{tag}: {rep.name}\n{'=' * 60}")
        out_lines.append(f"metric selection: {note}")
        if not metrics:
            print(f"[{tag}] no populated pmsampling series", file=sys.stderr)
        for m in metrics:
            vals = per_instance_values(action, m)
            if vals is None:
                out_lines.append(f"\n{m}: no instances")
                continue
            out_lines.extend(ascii_plot(vals, m, args.rows, args.cols))

    out_path = analysis_dir / "pm_timeline_plots.txt"
    out_path.write_text("\n".join(out_lines))
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
