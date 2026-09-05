#!/usr/bin/env python3
"""Aggregate per-PC stall samples into per-source-line hotspots.

Requires:
    - .ncu-rep containing the `SourceCounters` section (`--set full` collects it,
      as does `--section SourceCounters`; there is no `--set source` on every
      Nsight Compute version)
    - for source lines rather than instruction addresses, a target compiled with
      `-lineinfo` AND a capture that passed `--import-source yes`; without the
      import, sample counts are still correct but every PC resolves to nothing,
      so this script falls back to per-instruction attribution

Produces in `<run-dir>/analysis/`:
    stall_hotspots_<tag>.txt — top lines ranked by total stall samples,
                               with per-stall-type breakdown, plus per-stall
                               top lines.

Usage:
    python3 extract_stall_hotspots.py --run-dir profile/myrun \\
            --report profile/myrun/reports/source_<tag>.ncu-rep --tag <tag>
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ncu_utils import load_action, metric_value_at  # noqa: E402


# Known per-PC stall counters, in a stable order for the output columns. The
# set is not closed: `stall_metric_names` adds any `smsp__pcsamp_warps_issue_stalled_*`
# the report carries but this list omits, which is how Ampere's `imc_miss`
# constant-cache stall shows up without editing the list.
STALL_METRICS = [
    "smsp__pcsamp_warps_issue_stalled_long_scoreboard",
    "smsp__pcsamp_warps_issue_stalled_short_scoreboard",
    "smsp__pcsamp_warps_issue_stalled_wait",
    "smsp__pcsamp_warps_issue_stalled_math_pipe_throttle",
    "smsp__pcsamp_warps_issue_stalled_mio_throttle",
    "smsp__pcsamp_warps_issue_stalled_lg_throttle",
    "smsp__pcsamp_warps_issue_stalled_not_selected",
    "smsp__pcsamp_warps_issue_stalled_dispatch_stall",
    "smsp__pcsamp_warps_issue_stalled_drain",
    "smsp__pcsamp_warps_issue_stalled_no_instructions",
    "smsp__pcsamp_warps_issue_stalled_selected",
    "smsp__pcsamp_warps_issue_stalled_branch_resolving",
    "smsp__pcsamp_warps_issue_stalled_barrier",
    "smsp__pcsamp_warps_issue_stalled_tex_throttle",
    "smsp__pcsamp_warps_issue_stalled_sleeping",
    "smsp__pcsamp_warps_issue_stalled_misc",
    "smsp__pcsamp_warps_issue_stalled_membar",
]


def stall_metric_names(action):
    """Return the per-PC stall counters this report actually carries.

    Hard-coding the list would silently drop architecture-specific reasons, and a
    missing reason reads as "this stall did not happen" rather than "nobody
    looked for it".
    """

    known = set(STALL_METRICS)
    discovered = sorted(
        name
        for name in action.metric_names()
        if name.startswith("smsp__pcsamp_warps_issue_stalled_")
        and name not in known
        # The `_not_issued` counters are the subset of each reason's samples where
        # no warp issued at all. They are already inside their parent counter, so
        # adding them would double-count every stall.
        and not name.endswith("_not_issued")
    )
    return STALL_METRICS + discovered


def is_stall(name):
    """Return whether a per-PC counter represents a stall.

    ``..._selected`` counts cycles where the warp *did* issue, so it belongs in
    the per-site breakdown but not in a stall total.
    """

    return not name.endswith("_selected") and not name.endswith("_not_issued")


def collect_per_pc(action):
    """Return dict[pc] -> dict[stall_name -> count]."""
    per_pc = defaultdict(lambda: defaultdict(int))
    for sn in stall_metric_names(action):
        try:
            m = action[sn]
        except Exception:
            continue
        try:
            n = m.num_instances()
        except Exception:
            continue
        if n == 0 or not m.has_correlation_ids():
            continue
        cor = m.correlation_ids()
        for i in range(n):
            try:
                pc = cor.as_uint64(i)
            except Exception:
                try:
                    pc = int(cor.as_double(i))
                except Exception:
                    continue
            v = metric_value_at(m, i)
            if v:
                per_pc[pc][sn] += int(v)
    return per_pc


def sass_text(action, pc):
    """Return the SASS instruction at one address, or "" if unavailable.

    Note the signature: ``sass_by_pc`` takes a single address and returns a
    string. It is not a ``{pc: instruction}`` mapping.
    """

    try:
        return (action.sass_by_pc(pc) or "").strip()
    except Exception:
        return ""


def aggregate_by_site(action, per_pc):
    """Group per-PC stall samples into display sites.

    Prefers ``file:line`` sites. When the report carries no source mapping, every
    PC would otherwise resolve to a single ``?:0`` row and the output would claim
    the whole kernel stalls in one place, so this falls back to one site per
    instruction address annotated with its SASS. That is still actionable.

    Args:
        action: An ``ncu_report`` action (one profiled kernel launch).
        per_pc: Mapping of PC to per-stall-metric sample counts.

    Returns:
        Tuple of ``(per_site, mapped_pcs)`` where ``per_site`` maps a display
        string to per-stall-metric counts, and ``mapped_pcs`` is how many PCs
        resolved to a source line. ``mapped_pcs == 0`` means the fallback ran.
    """

    resolved = {}
    mapped_pcs = 0
    for pc in per_pc:
        file_, line = None, 0
        try:
            info = action.source_info(pc)
            if info is not None:
                file_, line = info.file_name(), info.line()
        except Exception:
            file_, line = None, 0
        if file_:
            mapped_pcs += 1
        resolved[pc] = (file_, line)

    per_site = defaultdict(lambda: defaultdict(int))
    for pc, stalls in per_pc.items():
        file_, line = resolved[pc]
        if file_:
            site = f"{Path(str(file_)).name}:{line}"
        else:
            # Partial source mapping is normal — inlined library code and
            # compiler-generated instructions have no line. Collapsing all of
            # them into one "<unmapped>" row would hide a hotspot behind a label
            # nobody can act on, so each keeps its address and SASS.
            instruction = sass_text(action, pc)
            site = f"0x{pc:x}" + (f"  {instruction}" if instruction else "")
        for name, value in stalls.items():
            per_site[site][name] += value
    return per_site, mapped_pcs


def short_stall_name(full):
    return full.replace("smsp__pcsamp_warps_issue_stalled_", "")


def write_report(per_site, out_path, tag, mapped_pcs, top_n=30):
    """Write the ranked stall-hotspot table for one report.

    Args:
        per_site: Mapping of display site to per-stall-metric counts.
        out_path: Destination text file.
        tag: Caller's short name for the report.
        mapped_pcs: How many PCs resolved to a source line, for the header.
        top_n: How many sites to rank.
    """

    totals = []
    for site, stalls in per_site.items():
        # Rank on stall samples only. `..._selected` counts cycles where the warp
        # did issue, so including it ranks the busiest site above the most stalled
        # one — the opposite of the question.
        totals.append((
            sum(v for k, v in stalls.items() if is_stall(k)),
            site,
            dict(stalls),
        ))
    totals.sort(key=lambda row: -row[0])

    with open(out_path, "w") as f:
        f.write(f"===== Stall hotspots for {tag} =====\n")
        f.write("Ranked by stall samples; the '_selected' column is issued cycles "
                "and is shown but not ranked.\n")
        if mapped_pcs:
            f.write(f"Attribution: source lines ({mapped_pcs} PCs mapped; "
                    f"unmapped PCs appear as address + SASS)\n")
        else:
            f.write(
                "Attribution: instruction addresses. This report has no source "
                "mapping, so sites below are PCs with their SASS. Recapture with "
                "--import-source yes (and compile with -lineinfo) for source lines.\n"
            )
        f.write(f"Total distinct sites: {len(totals)}\n")

        # A one-line headline so a campaign comparison can quote the dominant
        # reason without re-parsing the table below.
        by_reason = defaultdict(int)
        for _, _, stalls in totals:
            for name, value in stalls.items():
                by_reason[name] += value
        stall_only = {k: v for k, v in by_reason.items() if is_stall(k)}
        grand = sum(stall_only.values())
        if grand:
            top_reason, top_count = max(stall_only.items(), key=lambda kv: kv[1])
            f.write(
                f"Dominant stall: {short_stall_name(top_reason)} "
                f"({top_count:,} of {grand:,} stall samples, "
                f"{top_count / grand * 100:.1f}%)\n"
            )
        f.write("\n")
        f.write(f"{'Rank':>4} {'Stalls':>10}  {'Site':<48}  Breakdown (stall: count)\n")
        f.write("-" * 150 + "\n")
        for i, (total, site, stalls) in enumerate(totals[:top_n]):
            breakdown = ", ".join(
                f"{short_stall_name(k)}: {v}"
                for k, v in sorted(stalls.items(), key=lambda x: -x[1]) if v
            )
            f.write(f"{i:>4} {total:>10}  {site:<48}  {breakdown}\n")

        f.write("\n\n===== Per-stall-type top sites =====\n")
        # Iterate the reasons present in the data, so a discovered counter that is
        # not in STALL_METRICS still gets its own section.
        for name in sorted(by_reason, key=lambda k: -by_reason[k]):
            items = [(site, s.get(name, 0)) for site, s in per_site.items()]
            items = [it for it in items if it[1] > 0]
            items.sort(key=lambda x: -x[1])
            if not items:
                continue
            f.write(f"\n--- {short_stall_name(name)} ---\n")
            for site, value in items[:10]:
                f.write(f"  {value:>8}  {site}\n")


def main():
    ap = argparse.ArgumentParser(
        description="Aggregate source-level stall samples from reports captured through a Python runner or fallback CUDA target."
    )
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--report", type=Path, action="append", required=True,
                    help="Path(s) to source-level .ncu-rep file(s). Pass multiple with repeated flag.")
    ap.add_argument("--tag", type=str, action="append", required=True)
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()

    if len(args.report) != len(args.tag):
        ap.error("--report and --tag counts must match")

    analysis_dir = args.run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    for rep, tag in zip(args.report, args.tag):
        if not rep.exists():
            print(f"[skip] {rep} not found", file=sys.stderr)
            continue
        # First action only. A report with several launches needs one invocation
        # per launch; ranking a multi-launch report as if it were one kernel is
        # the failure this note exists to prevent.
        action = load_action(rep)
        per_pc = collect_per_pc(action)
        if not per_pc:
            print(
                f"[skip] {rep} has no per-PC stall samples; was SourceCounters collected?",
                file=sys.stderr,
            )
            continue
        per_site, mapped_pcs = aggregate_by_site(action, per_pc)
        out = analysis_dir / f"stall_hotspots_{tag}.txt"
        write_report(per_site, out, tag, mapped_pcs, top_n=args.top)
        kind = "source lines" if mapped_pcs else "instructions (no source mapping)"
        print(f"[{tag}] -> {out} ({len(per_site)} sites by {kind})")


if __name__ == "__main__":
    main()
