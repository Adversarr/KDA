"""Shared helpers for parsing Nsight Compute reports.

Read metrics through ``Metrics``, not ``safe``::

    from ncu_utils import Metrics, load_report

    report, action = load_report(path)
    metrics = Metrics(action)
    metrics.status("sm__throughput.avg.pct_of_peak_sustained_elapsed")

``safe`` returns ``None`` for a metric that this chip never had, a metric this
capture did not collect, and a metric that read back empty — three different
facts, one of which is a reason to recapture. ``Metrics.status`` keeps them
apart. ``safe`` remains for one-off scripts where that distinction does not
change the answer.

The caller may set PYTHONPATH to the ``extras/python`` directory of the
Nsight Compute install, e.g.:
    export PYTHONPATH=$PYTHONPATH:/usr/local/cuda-13.2/nsight-compute-2026.1.0/extras/python

If ncu_report is not importable, ``_locate_ncu_report`` probes for it. The
python module ships inside the profiler install, so its path always encodes a
version; never hard-code one version and assume it is present.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

# --- Attempt to locate ncu_report --------------------------------------------
def _locate_ncu_report():
    """Return a directory containing ``ncu_report.py``, or None.

    Deriving the path from the ``ncu`` on PATH is the most reliable probe: the
    python module lives in the same install as the binary, whatever the version
    and wherever the toolkit was unpacked. The globs below are fallbacks for the
    case where ``ncu`` is not on PATH, and they must glob the toolkit directory
    itself (``/opt/cuda-12.4``), not only children of a ``/opt/cuda`` parent
    that may not exist. Every glob is sorted newest-first: with several installs
    present and no ``ncu`` to disambiguate, the newest reader is the one that can
    read the widest range of captures.
    """

    candidates = []

    # 1. Relative to the installed binary, e.g. /opt/cuda-12.4/bin/ncu ->
    #    /opt/cuda-12.4/nsight-compute-*/extras/python.
    ncu_bin = shutil.which("ncu")
    if ncu_bin:
        base = Path(ncu_bin).resolve().parent
        for parent in (base, base.parent):
            candidates.append(str(parent / "extras" / "python"))
            candidates.extend(
                str(sub)
                for sub in sorted(parent.glob("nsight-compute-*/extras/python"), reverse=True)
            )

    # 2. Common toolkit and standalone install roots.
    for pattern in (
        "/usr/local/cuda*/nsight-compute*/extras/python",
        "/usr/local/cuda*/nsight-compute/extras/python",
        "/opt/cuda*/nsight-compute*/extras/python",
        "/opt/nvidia/nsight-compute*/extras/python",
        "/opt/nvidia/nsight-compute/*/extras/python",
    ):
        candidates.extend(
            str(sub) for sub in sorted(Path("/").glob(pattern.lstrip("/")), reverse=True)
        )

    # 3. macOS host install, which is useful for reading reports off-target.
    candidates.append(
        "/Applications/NVIDIA Nsight Compute.app/Contents/MacOS/python"
    )

    for c in candidates:
        if Path(c).is_dir() and (Path(c) / "ncu_report.py").exists():
            return c
    return None

try:
    import ncu_report  # noqa: F401
except ImportError:
    found = _locate_ncu_report()
    if found:
        sys.path.insert(0, found)
        import ncu_report  # noqa: F401
    else:
        raise


import ncu_report  # noqa: E402


# --- Loading -----------------------------------------------------------------

def load_report(path):
    """Load a .ncu-rep file and return the first action (= first kernel launch).

    Returns (report, action) tuple — keep the report alive while using the action.
    """
    r = ncu_report.load_report(str(path))
    rng = r.range_by_idx(0)
    action = rng.action_by_idx(0)
    return r, action


def load_action(path):
    """Shortcut when you don't need the report object separately."""
    _, action = load_report(path)
    return action


def iter_actions(report):
    """Yield ``(range_index, action_index, action)`` for every launch in a report.

    A report with ``-c N`` or several kernels holds more than one action, and
    analyzing only the first silently answers a question about a different launch
    than the one the user asked about.
    """

    for ri in range(report.num_ranges()):
        rng = report.range_by_idx(ri)
        for ai in range(rng.num_actions()):
            yield ri, ai, rng.action_by_idx(ai)


def reader_version():
    """Return the ``ncu_report`` module version that is reading the file.

    This is the *reader*, not the version that captured the report. The two
    differ routinely — reading a 2024.1 capture with a 2026.x module works — and
    conflating them leads to skipping API calls that are in fact available. That
    is why this deliberately never asks the report object: which API calls exist
    is a property of the module.
    """

    for getter in ("get_version", "version"):
        fn = getattr(ncu_report, getter, None)
        if callable(fn):
            try:
                return str(fn())
            except Exception:
                continue
    return "unknown"


# --- Safe metric access ------------------------------------------------------

def safe(action, name, default=None):
    """Return metric value, or `default` if the metric is missing or errors."""
    try:
        return action[name].value()
    except Exception:
        return default


def safe_many(action, names, default=None):
    """Bulk-fetch multiple metrics. Returns a dict name -> value-or-default."""
    return {n: safe(action, n, default) for n in names}


def metric_or_none(action, *candidates):
    """Try each candidate name, return first that works. Useful for
    GPU-gen-specific names: some metric names differ on sm_100 vs sm_90."""
    for n in candidates:
        v = safe(action, n, None)
        if v is not None:
            return v
    return None


# --- Missing versus zero -----------------------------------------------------
#
# ``safe`` collapses three different situations into ``None``: the metric name
# does not exist on this architecture or ncu version, the name exists but the
# report never collected a value, and the value is legitimately absent. A
# diagnosis needs to tell those apart from a measured zero. Reporting "None" for
# an occupancy that was in fact collected under a different name invents
# ignorance, and reporting 0 for a metric that was never collected invents a
# measurement.

#: Status returned by ``Metrics.resolve``: name is not in this report at all.
ABSENT = "absent"
#: Status: name exists in the report but reading a value failed or returned None.
NOVALUE = "novalue"
#: Status: value was collected and is exactly zero.
ZERO = "zero"
#: Status: value was collected and is non-zero.
VALUE = "value"


def fmt_num(value):
    """Render a metric value without float noise.

    ncu returns whole counts as doubles, so a grid size prints as ``568.0`` and a
    wave count as ``0.6574074074074074`` unless something intervenes.
    """

    if value is None:
        return "-"
    if isinstance(value, float):
        if value.is_integer() and abs(value) < 1e15:
            return f"{int(value):,}"
        if abs(value) >= 1000:
            return f"{value:,.1f}"
        return f"{value:.4g}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


class Resolved:
    """One logical field, the metric name that supplied it, and its status."""

    __slots__ = ("absent_means", "field", "name", "status", "unit", "value")

    def __init__(self, field, name, value, status, unit="", absent_means=""):
        self.field = field
        self.name = name
        self.value = value
        self.status = status
        self.unit = unit
        #: Why an absent name is absent, which differs by capture kind.
        self.absent_means = absent_means or "not on this chip/version"

    @property
    def ok(self) -> bool:
        """Whether a number was actually measured (zero counts as measured)."""

        return self.status in (ZERO, VALUE)

    def text(self, with_unit=True) -> str:
        """Render the value for a report line, never printing a fake number."""

        if self.status == ABSENT:
            return self.absent_means
        if self.status == NOVALUE:
            return "name present, not collected"
        body = fmt_num(self.value)
        return f"{body} {self.unit}".strip() if with_unit else body

    def __repr__(self) -> str:
        return f"Resolved({self.field}={self.text()!r} via {self.name!r})"


class Metrics:
    """Metric access over one action that resolves aliases and reports status.

    Metric names drift with both architecture and ncu version, so a single
    hard-coded name is a coin flip on any machine other than the one the caller
    was written against. Each logical field therefore carries a candidate list,
    and the resolved name travels with the value so a report can say which name
    it actually read.
    """

    def __init__(self, action):
        self.action = action
        self.names = set(action.metric_names())
        #: Whether this is a ``--metrics`` capture rather than a section set.
        self.sparse = self._detect_sparse()

    def _detect_sparse(self) -> bool:
        """Detect a ``--metrics`` capture from its single synthetic section.

        This changes what a missing name means. In a section capture, a name that
        is not present is a name this chip or ncu version does not have. In a
        ``--metrics`` capture it is far more likely a name nobody asked for, and
        saying "not on this chip" about it is simply false.
        """

        try:
            sections = [s.name() for s in self.action.sections()]
        except Exception:
            return False
        return len(sections) == 1 and "profiler metrics" in sections[0].lower()

    @property
    def absent_means(self) -> str:
        """Phrase explaining absence, matched to this report's capture kind."""

        if self.sparse:
            return "not requested in this --metrics capture"
        return "not on this chip/version"

    def has(self, name) -> bool:
        """Return whether ``name`` exists in this report."""

        return name in self.names

    def status(self, name) -> str:
        """Classify one metric name as absent, uncollected, zero, or a value."""

        if name not in self.names:
            return ABSENT
        value = safe(self.action, name, None)
        if value is None:
            return NOVALUE
        if isinstance(value, (int, float)) and value == 0:
            return ZERO
        return VALUE

    def unit(self, name) -> str:
        """Return the metric's unit, or an empty string.

        Units matter for more than presentation: ``launch__occupancy_limit_warps``
        carries unit ``block`` on the reports checked, and treating it as warps
        would corrupt the blocks-per-SM minimum that occupancy grading rests on.
        """

        try:
            return self.action[name].unit() or ""
        except Exception:
            return ""

    def get(self, name, default=None):
        """Return one metric's value by exact name."""

        return safe(self.action, name, default)

    def resolve(self, field, *candidates) -> Resolved:
        """Resolve ``field`` from the first candidate name that has a value.

        Args:
            field: Logical field name used in output, e.g. ``"achieved_occupancy"``.
            *candidates: Metric names in preference order. When omitted, the
                candidates come from ``FIELD_ALIASES[field]``.

        Returns:
            A ``Resolved``. When every candidate is absent the status is
            ``ABSENT``; when a name existed but held no value the status is
            ``NOVALUE`` and ``name`` records the last such name, so the caller can
            distinguish "this chip has no such counter" from "this capture did
            not collect it".
        """

        names = candidates or FIELD_ALIASES.get(field, ())
        seen_but_empty = None
        for name in names:
            st = self.status(name)
            if st in (ZERO, VALUE):
                return Resolved(field, name, self.get(name), st, self.unit(name))
            if st == NOVALUE and seen_but_empty is None:
                seen_but_empty = name
        if seen_but_empty:
            return Resolved(field, seen_but_empty, None, NOVALUE)
        return Resolved(
            field, names[-1] if names else field, None, ABSENT,
            absent_means=self.absent_means,
        )

    def pmsampling_names(self, populated_only=True):
        """List the ``pmsampling:`` series in this report.

        The populated set is version- and architecture-specific: a 2024.1 Ampere
        capture typically carries ``dramc__`` and ``*_realtime*`` series and no
        stall-reason series at all, so any fixed default list plots nothing.

        Args:
            populated_only: When true, drop names whose instance array is empty,
                which is what "the metric exists but has no samples" looks like.

        Returns:
            Sorted list of metric names.
        """

        out = []
        for name in self.names:
            if not name.startswith("pmsampling:"):
                continue
            if populated_only:
                try:
                    if self.action[name].num_instances() == 0:
                        continue
                except Exception:
                    continue
            out.append(name)
        return sorted(out)


# --- Value-kind robust accessor (for per-instance data) ---------------------

def metric_value_at(m, i):
    """Read the i-th instance value regardless of value kind."""
    k = m.kind()
    if k == m.ValueKind_UINT64:
        return m.as_uint64(i)
    if k in (m.ValueKind_DOUBLE, m.ValueKind_FLOAT):
        return m.as_double(i)
    if k == m.ValueKind_STRING:
        return m.as_string(i)
    # Fallbacks
    try:
        return m.as_uint64(i)
    except Exception:
        try:
            return m.as_double(i)
        except Exception:
            return None


def per_instance_values(action, metric_name):
    """Return a list of per-instance values, or None if the metric has none."""
    try:
        m = action[metric_name]
    except Exception:
        return None
    try:
        n = m.num_instances()
    except Exception:
        return None
    if n == 0:
        return None
    return [metric_value_at(m, i) for i in range(n)]


# --- Archive all metrics -----------------------------------------------------

def dump_all_metrics(action, outfile):
    """Dump every metric name + value to a JSON file for later analysis.

    Returns the number of entries written.
    """
    out = []
    for n in sorted(action.metric_names()):
        try:
            m = action[n]
            rec = {"name": n}
            try:
                rec["value"] = m.value()
            except Exception as e:
                rec["error"] = str(e)
            try:
                rec["unit"] = m.unit()
            except Exception:
                pass
            out.append(rec)
        except Exception as e:
            out.append({"name": n, "error": str(e)})
    Path(outfile).write_text(json.dumps(out, indent=1, default=str))
    return len(out)


# --- PC → source line mapping ------------------------------------------------

def per_pc_values(action, metric_name):
    """For a source-level metric (with correlation_ids = PCs), return list of (pc, value)."""
    try:
        m = action[metric_name]
    except Exception:
        return []
    try:
        n = m.num_instances()
    except Exception:
        return []
    if n == 0 or not m.has_correlation_ids():
        return []
    cor = m.correlation_ids()
    out = []
    for i in range(n):
        try:
            pc = cor.as_uint64(i)
        except Exception:
            try:
                pc = int(cor.as_double(i))
            except Exception:
                pc = None
        try:
            v = metric_value_at(m, i)
        except Exception:
            v = 0
        out.append((pc, v))
    return out


def pc_to_source_line(action, pc):
    """Return (file, line) for a given PC, or ('?', 0) if unavailable.

    Compiling with ``-lineinfo`` is necessary but not sufficient: a report
    captured without ``--import-source yes`` returns None here for every PC even
    though SASS and PC-sampling counts are present. When this returns ``('?', 0)``
    for everything, attribute to the PC and disassemble with
    ``action.sass_by_pc(pc)`` instead of collapsing every sample onto one line.
    """
    try:
        si = action.source_info(pc)
        if si is None:
            return "?", 0
        return si.file_name(), si.line()
    except Exception:
        return "?", 0


# --- Curated metric sets -----------------------------------------------------
#
# Verified to exist and carry meaningful values on B200 / sm_100 with Nsight
# Compute 2026.x, and re-checked against A800 / sm_80 captured with 2024.1.1.
# Most of the list is portable; the entries that are not have aliases in
# FIELD_ALIASES below. See ../reference/08-metric-names.md for the per-name
# story. Always confirm with action.metric_names() rather than trusting any
# list here — including this one.

KEY_METRICS = [
    # Device identity. Read these before grading anything: waves-per-SM is
    # graded against the SM count, so an assumed one mis-grades occupancy.
    "device__attribute_display_name",
    "device__attribute_compute_capability_major",
    "device__attribute_compute_capability_minor",
    "device__attribute_multiprocessor_count",
    "device__attribute_max_warps_per_multiprocessor",
    "device__attribute_reserved_shared_memory_per_block",
    # Launch geometry
    "launch__grid_size",
    "launch__block_size",
    "launch__grid_dim_x",
    "launch__grid_dim_y",
    "launch__grid_dim_z",
    "launch__block_dim_x",
    "launch__block_dim_y",
    "launch__block_dim_z",
    "launch__waves_per_multiprocessor",
    "launch__registers_per_thread",
    "launch__thread_count",
    "launch__occupancy_limit_blocks",
    "launch__occupancy_limit_registers",
    "launch__occupancy_limit_shared_mem",
    "launch__occupancy_limit_warps",
    # Shared memory. Read all four: `_per_block` alone is often the driver's
    # reserved carve-out on a kernel that requests none.
    "launch__shared_mem_per_block",
    "launch__shared_mem_per_block_static",
    "launch__shared_mem_per_block_dynamic",
    "launch__shared_mem_per_block_driver",
    "launch__shared_mem_per_block_allocated",
    # Timing
    "gpu__time_duration.sum",
    "smsp__cycles_active.avg",
    "sm__cycles_active.avg",
    "sm__cycles_active.max",
    "sm__cycles_active.min",
    "sm__cycles_elapsed.avg",
    # SOL
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__compute_memory_access_throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__throughput.avg.pct_of_peak_sustained_active",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    # Occupancy
    "sm__maximum_warps_per_active_cycle_pct",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "smsp__warps_active.avg.pct_of_peak_sustained_active",
    "sm__warps_active.avg.per_cycle_active",
    "sm__warps_active.max.per_cycle_active",
    "sm__warps_active.min.per_cycle_active",
    "smsp__warps_active.avg.per_cycle_active",
    "smsp__warps_eligible.avg.per_cycle_active",
    "smsp__warps_eligible.max.per_cycle_active",
    # IPC
    "sm__inst_executed.avg.per_cycle_active",
    "smsp__issue_active.avg.per_cycle_active",
    "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "smsp__inst_executed.avg",
    "smsp__inst_executed.sum",
    "smsp__average_warp_latency_per_inst_issued.ratio",
    # Warp execution efficiency. Answers "how much of each warp is doing work"
    # directly, and exists on both sm_80 and sm_100.
    "smsp__thread_inst_executed_per_inst_executed.ratio",
    "smsp__thread_inst_executed_pred_on_per_inst_executed.ratio",
    "smsp__sass_average_branch_targets_threads_uniform.pct",
    # Compute pipes
    "sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active",
    "sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_elapsed",
    "sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active",
    "sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active",
    "sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_elapsed",
    "sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_active",
    "sm__inst_executed_pipe_adu.avg.pct_of_peak_sustained_active",
    # Tensor core
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "sm__ops_path_tensor_op_hmma_src_bf16_dst_fp32_sparsity_off.avg",
    # DRAM
    "dram__bytes_read.sum",
    "dram__bytes_read.sum.pct_of_peak_sustained_elapsed",
    "dram__bytes_read.sum.per_second",
    "dram__bytes_write.sum",
    "dram__bytes_write.sum.pct_of_peak_sustained_elapsed",
    "dram__sectors_read.sum",
    "dram__sectors_write.sum",
    # Caches
    "l1tex__t_sector_hit_rate.pct",
    "lts__t_sector_hit_rate.pct",
    "l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct",
    "l1tex__t_sector_pipe_lsu_mem_global_op_st_hit_rate.pct",
    # Memory instruction counts
    "smsp__sass_inst_executed_op_global_ld.sum",
    "smsp__sass_inst_executed_op_global_st.sum",
    "smsp__sass_inst_executed_op_local_ld.sum",
    "smsp__sass_inst_executed_op_local_st.sum",
    "smsp__sass_inst_executed_op_shared.sum",
    "smsp__sass_inst_executed_op_shared_ld.sum",
    "smsp__sass_inst_executed_op_shared_st.sum",
    # Sectors / requests (for coalescing analysis)
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum",
    "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.ratio",
    "smsp__sass_average_data_bytes_per_sector_mem_global_op_st.ratio",
    # Stall reasons — aggregate ratios. Each of these is a CPI component: cycles
    # a warp spends in that stall between two issues. They are not concurrent
    # warp counts. Use stall_reasons() rather than this list when the report may
    # carry the per_warp_active percentage form instead.
    "smsp__average_warps_issue_stalled_imc_miss_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_membar_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_tex_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_drain_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_sleeping_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_misc_per_issue_active.ratio",
    # Stall reasons — per-PC. These come from the SourceCounters section, which
    # `--set full` already includes on the versions checked; there is no
    # `--set source` on every install. Run `ncu --list-sets` before assuming.
    "smsp__pcsamp_sample_count",
    "smsp__pcsamp_warps_issue_stalled_long_scoreboard",
    "smsp__pcsamp_warps_issue_stalled_short_scoreboard",
    "smsp__pcsamp_warps_issue_stalled_wait",
    "smsp__pcsamp_warps_issue_stalled_barrier",
    "smsp__pcsamp_warps_issue_stalled_math_pipe_throttle",
    "smsp__pcsamp_warps_issue_stalled_mio_throttle",
    "smsp__pcsamp_warps_issue_stalled_lg_throttle",
    "smsp__pcsamp_warps_issue_stalled_not_selected",
    "smsp__pcsamp_warps_issue_stalled_dispatch_stall",
    "smsp__pcsamp_warps_issue_stalled_drain",
    "smsp__pcsamp_warps_issue_stalled_no_instructions",
    "smsp__pcsamp_warps_issue_stalled_selected",
    "smsp__pcsamp_warps_issue_stalled_branch_resolving",
    "smsp__pcsamp_warps_issue_stalled_membar",
]


# --- Logical fields and their per-architecture names --------------------------
#
# Every entry here is a field whose metric name was observed to differ between
# B200 / 2026.x and A800 / 2024.1. Candidates are in preference order, and
# Metrics.resolve reports which one it used, so a diagnosis can be audited
# without re-running anything.

FIELD_ALIASES = {
    "achieved_occupancy": (
        "sm__warps_active.avg.pct_of_peak_sustained_active",
        "smsp__warps_active.avg.pct_of_peak_sustained_active",
    ),
    "theoretical_occupancy": ("sm__maximum_warps_per_active_cycle_pct",),
    "sm_sol": ("sm__throughput.avg.pct_of_peak_sustained_elapsed",),
    "memory_sol": (
        "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
        "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    ),
    "fma_pipe": (
        "sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active",
        "sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_elapsed",
        "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active",
    ),
    # Every alias in a group must share a unit, or a caller that formats the
    # resolved value as a percentage prints a raw op count as one. The hmma
    # op-path counter therefore lives in its own group rather than as a fallback
    # here. Absence of either means "not measured", not "no tensor cores used".
    "tensor_pipe": (
        "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
        "sm__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active",
    ),
    "tensor_ops": (
        "sm__ops_path_tensor_op_hmma_src_bf16_dst_fp32_sparsity_off.avg",
        "sm__ops_path_tensor_src_fp16_dst_fp32.sum",
    ),
    "global_ld_inst": (
        "smsp__sass_inst_executed_op_global_ld.sum",
        "smsp__inst_executed_op_global_ld.sum",
    ),
    "global_st_inst": (
        "smsp__sass_inst_executed_op_global_st.sum",
        "smsp__inst_executed_op_global_st.sum",
    ),
    "local_ld_inst": (
        "smsp__sass_inst_executed_op_local_ld.sum",
        "smsp__inst_executed_op_local_ld.sum",
    ),
    "local_st_inst": (
        "smsp__sass_inst_executed_op_local_st.sum",
        "smsp__inst_executed_op_local_st.sum",
    ),
    "dram_read_pct": (
        "dram__bytes_read.sum.pct_of_peak_sustained_elapsed",
        "dram__read_throughput.avg.pct_of_peak_sustained_elapsed",
    ),
    "dram_write_pct": (
        "dram__bytes_write.sum.pct_of_peak_sustained_elapsed",
        "dram__write_throughput.avg.pct_of_peak_sustained_elapsed",
    ),
    "l1_hit_rate": ("l1tex__t_sector_hit_rate.pct",),
    "l2_hit_rate": ("lts__t_sector_hit_rate.pct",),
    "warp_efficiency": ("smsp__thread_inst_executed_per_inst_executed.ratio",),
    "warp_efficiency_pred_on": (
        "smsp__thread_inst_executed_pred_on_per_inst_executed.ratio",
    ),
}

#: Stall reasons in the order they are worth reading, worst-first by experience.
STALL_REASONS = (
    "long_scoreboard",
    "short_scoreboard",
    "wait",
    "not_selected",
    "no_instruction",
    "mio_throttle",
    "lg_throttle",
    "math_pipe_throttle",
    "tex_throttle",
    "barrier",
    "membar",
    "branch_resolving",
    "dispatch_stall",
    "drain",
    "imc_miss",
    "sleeping",
    "misc",
)


# Both stall forms belong in the metric archive. A report carries one or the
# other, so a KEY_METRICS list that knows only the CPI form leaves the whole
# stall block empty for every capture that reports percentages instead. Rows from
# the two forms must never be compared or added to each other; ``stall_form``
# names which one a given report used.
KEY_METRICS.extend(
    f"smsp__warp_issue_stalled_{reason}_per_warp_active.pct" for reason in STALL_REASONS
)


def stall_reasons(metrics):
    """Resolve the per-reason stall table, recording which form was collected.

    Two incompatible forms exist. ``smsp__average_warps_issue_stalled_<r>_per_issue_active.ratio``
    is a CPI component in cycles, and the reasons plus the issuing cycle sum to
    the warp CPI. ``smsp__warp_issue_stalled_<r>_per_warp_active.pct`` is a
    percentage of warp-active cycles. Mixing them, or labelling either as "warps
    stalled", produces a number that cannot be reasoned about, so the form
    travels with every row.

    Args:
        metrics: A ``Metrics`` wrapper around the action.

    Returns:
        List of ``Resolved``, descending by value, for the reasons that were
        collected. ``Resolved.unit`` is ``cycle`` for the CPI form and ``%`` for
        the percentage form; the caller must not add the two.
    """

    rows = []
    for reason in STALL_REASONS:
        row = metrics.resolve(
            reason,
            f"smsp__average_warps_issue_stalled_{reason}_per_issue_active.ratio",
            f"smsp__warp_issue_stalled_{reason}_per_warp_active.pct",
            f"smsp__warps_issue_stalled_{reason}_per_issue_active.pct",
        )
        if row.ok:
            rows.append(row)
    rows.sort(key=lambda r: r.value or 0.0, reverse=True)
    return rows


def uncollected_stalls(metrics):
    """List the stall reasons this report does not carry, in any of the forms.

    A capture that collected three of seventeen reasons will show three rows that
    look like a complete accounting, and normalizing those three to 100% moves the
    unmeasured remainder into the measured buckets. Naming the absent reasons is
    what stops that.

    Args:
        metrics: A ``Metrics`` wrapper around the action.

    Returns:
        List of reason names with no value in this report.
    """

    missing = []
    for reason in STALL_REASONS:
        row = metrics.resolve(
            reason,
            f"smsp__average_warps_issue_stalled_{reason}_per_issue_active.ratio",
            f"smsp__warp_issue_stalled_{reason}_per_warp_active.pct",
            f"smsp__warps_issue_stalled_{reason}_per_issue_active.pct",
        )
        if not row.ok:
            missing.append(reason)
    return missing


def stall_form(rows):
    """Describe the stall form in one phrase, for a table header."""

    if not rows:
        return "no stall metrics collected"
    if rows[0].name.startswith("smsp__average_warps_issue_stalled_"):
        return "CPI component (cycles between issues; sums with issue to warp CPI)"
    return "percent of warp-active cycles"


def fill_verdict(metrics):
    """Answer whether the launch fills the device, and how it fails to.

    Two different failures get confused constantly. A grid smaller than the SM
    count leaves SMs with no work at all for the whole kernel. A grid larger than
    the SM count but under one occupancy wave keeps every SM busy while leaving
    too few resident warps to hide latency. The first is idle silicon; the second
    is a latency-hiding problem. Both are launch-configuration problems rather
    than kernel-body problems, which is the point worth making before anyone
    edits the kernel.

    Args:
        metrics: A ``Metrics`` wrapper around the action.

    Returns:
        Dict with the raw inputs (``sm_count``, ``grid``, ``block``, ``waves``),
        the resolved per-limit blocks-per-SM (``limits``, name -> (value, unit)),
        the derived ``blocks_per_sm``, the booleans ``fills_every_sm`` and
        ``fills_occupancy``, and a one-line ``verdict``.
    """

    sm_count = metrics.get("device__attribute_multiprocessor_count")
    grid = metrics.get("launch__grid_size")
    block = metrics.get("launch__block_size")
    waves = metrics.get("launch__waves_per_multiprocessor")

    limits = {}
    for name in (
        "launch__occupancy_limit_blocks",
        "launch__occupancy_limit_registers",
        "launch__occupancy_limit_shared_mem",
        "launch__occupancy_limit_warps",
    ):
        if metrics.status(name) in (ZERO, VALUE):
            limits[name] = (metrics.get(name), metrics.unit(name))

    # The minimum is only meaningful over limits in the same unit. All four have
    # carried unit `block` on the reports checked, including
    # launch__occupancy_limit_warps despite its name, but a version that reported
    # one in warps would corrupt blocks_per_sm if mixed in blindly. Convert those
    # rather than dropping them: dropping the binding constraint overestimates
    # occupancy, which is the direction that hides a problem.
    warps_per_block = (block / 32) if block else None
    block_limits = []
    unconverted = []
    for name, (value, unit) in limits.items():
        if unit in ("block", "", None):
            block_limits.append(value)
        elif unit == "warp" and warps_per_block:
            block_limits.append(value / warps_per_block)
        else:
            unconverted.append(f"{name} ({unit})")
    blocks_per_sm = min(block_limits) if block_limits else None

    fills_every_sm = None if (grid is None or not sm_count) else grid >= sm_count
    fills_occupancy = None if waves is None else waves >= 1.0

    if fills_every_sm is False:
        verdict = (
            f"grid {fmt_num(grid)} < {fmt_num(sm_count)} SMs: some SMs have no work "
            f"for the entire kernel. Fix the launch or the decomposition, not the "
            f"kernel body."
        )
    elif fills_occupancy is False and fills_every_sm:
        verdict = (
            f"{waves:.2f} waves/SM: every SM has work, but under one full wave of "
            f"resident blocks, so there are too few warps to hide latency and the "
            f"tail is not amortized. This is a launch-configuration problem."
        )
    elif fills_occupancy is False:
        # waves < 1 with an unknown grid or SM count. Under-filled either way, but
        # "every SM has work" is exactly the claim that is not established here.
        verdict = (
            f"{waves:.2f} waves/SM: under one full wave of resident blocks. Grid or "
            f"SM count is missing from this report, so whether some SMs are idle "
            f"outright cannot be separated from too-few-resident-warps."
        )
    elif fills_occupancy:
        verdict = f"{waves:.2f} waves/SM: the launch fills the device."
    else:
        verdict = "launch geometry incomplete in this report"

    # Occupancy has two different ceilings, and only one of them responds to
    # __launch_bounds__ or a smaller register budget. If the grid cannot even
    # place blocks_per_sm blocks on every SM, resident warps are capped by how
    # much work was launched, and raising the per-SM resource limit changes
    # nothing. Prescribing launch bounds here is the classic wasted round.
    grid_blocks_per_sm = grid / sm_count if (grid and sm_count) else None
    grid_limited = (
        grid_blocks_per_sm is not None
        and blocks_per_sm is not None
        and grid_blocks_per_sm < blocks_per_sm
    )
    if grid_limited:
        verdict += (
            f" Occupancy here is grid-limited, not resource-limited: the grid "
            f"places only {grid_blocks_per_sm:.2f} of the {fmt_num(blocks_per_sm)} "
            f"blocks/SM the resources allow, so raising theoretical occupancy "
            f"(__launch_bounds__, fewer registers) cannot add resident warps. "
            f"Launch more blocks instead."
        )

    return {
        "sm_count": sm_count,
        "grid": grid,
        "block": block,
        "waves": waves,
        "limits": limits,
        "blocks_per_sm": blocks_per_sm,
        "grid_blocks_per_sm": grid_blocks_per_sm,
        "grid_limited": grid_limited,
        "unconverted_limits": unconverted,
        "fills_every_sm": fills_every_sm,
        "fills_occupancy": fills_occupancy,
        "verdict": verdict,
    }


def shared_memory(metrics):
    """Separate kernel-requested shared memory from the driver's reservation.

    ``launch__shared_mem_per_block`` is often exactly the driver's reserved
    carve-out (1024 B on Ampere) for a kernel that requests no shared memory at
    all, so quoting it as the kernel's usage invents a shared-memory tile that
    does not exist and hides the fact that the kernel never used the resource.

    Args:
        metrics: A ``Metrics`` wrapper around the action.

    Returns:
        Dict of the four ``launch__shared_mem_per_block*`` values plus the
        device reservation, and ``verdict`` naming what the kernel actually asked
        for.
    """

    fields = {
        "per_block": metrics.get("launch__shared_mem_per_block"),
        "static": metrics.get("launch__shared_mem_per_block_static"),
        "dynamic": metrics.get("launch__shared_mem_per_block_dynamic"),
        "driver": metrics.get("launch__shared_mem_per_block_driver"),
        "allocated": metrics.get("launch__shared_mem_per_block_allocated"),
        "device_reserved": metrics.get(
            "device__attribute_reserved_shared_memory_per_block"
        ),
    }
    static = fields["static"]
    dynamic = fields["dynamic"]
    if static is None and dynamic is None:
        # A sparse capture that omitted both breakdown metrics looks identical to
        # a kernel that requested nothing, if absent is folded into zero. Say so
        # instead of claiming the kernel used no shared memory.
        fields["verdict"] = (
            "static/dynamic breakdown not in this report, so per_block cannot be "
            "split from the driver reservation; recapture with the Launch Statistics "
            "section or these metrics"
        )
    elif (static or 0) == 0 and (dynamic or 0) == 0:
        fields["verdict"] = (
            "no kernel-requested shared memory; any non-zero per_block here is "
            "the driver reservation"
        )
    else:
        fields["verdict"] = (
            f"kernel requested {static if static is not None else '?'} B static + "
            f"{dynamic if dynamic is not None else '?'} B dynamic"
        )
    return fields


def derived_memory(metrics):
    """Compute sectors-per-request and bytes-per-sector from sums.

    The ``l1tex__average_t_sectors_per_request_*`` metrics are absent on both
    sm_80 and sm_100 captures checked, so the coalescing question has to be
    answered from the sector and request sums. A fully coalesced 32-bit access
    pattern gives 4 sectors per request and 32 bytes per sector; 32 sectors per
    request is one sector per thread, the worst case.

    Args:
        metrics: A ``Metrics`` wrapper around the action.

    Returns:
        Dict keyed ``ld`` and ``st``, each with ``sectors``, ``requests``,
        ``sectors_per_request``, and ``bytes_per_sector`` where computable.
    """

    out = {}
    for op in ("ld", "st"):
        sectors = metrics.get(f"l1tex__t_sectors_pipe_lsu_mem_global_op_{op}.sum")
        requests = metrics.get(f"l1tex__t_requests_pipe_lsu_mem_global_op_{op}.sum")
        per_request = (
            sectors / requests if sectors is not None and requests else None
        )
        out[op] = {
            "sectors": sectors,
            "requests": requests,
            "sectors_per_request": per_request,
            "bytes_per_sector": metrics.get(
                f"smsp__sass_average_data_bytes_per_sector_mem_global_op_{op}.ratio"
            ),
        }
    return out


# --- Convenience: NCU rule results --------------------------------------------

def rule_results(action):
    """Return the NCU rule-engine results as a list of dicts, or [] if unavailable.

    Returns [] on two different conditions that both matter, and neither is a
    statement about the kernel.

    Availability follows the *reader* — the ``ncu_report`` module doing the
    loading — not the version that captured the report. A 2026.x module reads a
    2024.1 capture and exposes the rules fine, so check ``hasattr`` rather than
    assuming an old capture means no rules. Only a genuinely old reader module
    lacks the method, and on a host with no ``ncu`` binary the CLI fallback
    ``ncu --import REPORT --page details --csv`` is not available either.

    An empty list also comes back from a report captured with an explicit
    ``--metrics`` list, because rules never ran. That is not "Nsight Compute
    found no issues", and reporting it that way is the most expensive mistake
    available here.
    """

    try:
        return list(action.rule_results_as_dicts())
    except Exception:
        return []


def rule_speedups(action):
    """Return ``(speedup_pct, kind, rule_identifier, message)`` sorted descending.

    The dict shape is version-specific and was verified against a real report as
    ``{rule_identifier, name, section_identifier, rule_message: {title, message,
    type}, focus_metrics, speedup_estimation: {type, speedup}}``. Only some rules
    carry ``speedup_estimation`` at all (13 of 17 on the report checked), so a
    missing estimate means "this rule did not estimate one", not zero gain.

    ``kind`` is ``"local"`` when the estimate applies to the analyzed section
    and ``"global"`` when it applies to the whole kernel. The two are not
    comparable, and neither is additive across rules: the estimates overlap and
    routinely sum past 100%.

    Args:
        action: An ``ncu_report`` action (one profiled kernel launch).

    Returns:
        List of ``(speedup_pct, kind, rule_identifier, message)``, descending by
        speedup. Rules without an estimate sort last with ``None``.
    """

    out = []
    for rr in rule_results(action):
        estimate = rr.get("speedup_estimation") or {}
        pct = estimate.get("speedup")
        try:
            pct = float(pct) if pct is not None else None
        except (TypeError, ValueError):
            pct = None
        # Observed encoding: 1 = local to the section, 2 = global to the kernel.
        kind = {1: "local", 2: "global", None: "unknown"}.get(
            estimate.get("type"), "unknown"
        )
        message = rr.get("rule_message") or {}
        # Keep the body, not just the headline. The title is a label like "Low
        # Utilization"; the body carries the counts and the specific advice, which
        # is what a report needs to quote.
        text = " — ".join(
            part for part in (message.get("title"), message.get("message")) if part
        ) or "(rule fired with no message)"
        out.append((pct, kind, rr.get("rule_identifier", "?"), text))
    out.sort(key=lambda row: (row[0] is not None, row[0] or 0.0), reverse=True)
    return out
