#!/usr/bin/env python3
"""Turn one Nsight Systems report into the attribution numbers a diagnosis needs."""

from __future__ import annotations

import argparse
import bisect
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Kernel duration buckets, in nanoseconds. The boundary that matters is
# "would this kernel be dominated by its own launch cost".
_KERNEL_BUCKETS: tuple[tuple[str, int], ...] = (
    ("<1us", 1_000),
    ("<5us", 5_000),
    ("<10us", 10_000),
    ("<50us", 50_000),
    ("<100us", 100_000),
    ("<1ms", 1_000_000),
    (">=1ms", None),  # type: ignore[arg-type]
)

# Copy size buckets, in bytes. Tiny copies are control decisions; large ones
# are payload. They are different bottlenecks and must never be summed.
_COPY_BUCKETS: tuple[tuple[str, int | None], ...] = (
    ("<=8B", 8),
    ("<=64B", 64),
    ("<=256B", 256),
    ("<=4KiB", 4_096),
    ("<=1MiB", 1_048_576),
    (">1MiB", None),
)

# CUDA runtime API families worth calling out by name. Each one answers a
# different question than "how much host time did the API take".
_API_FAMILIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("launch", ("cudaLaunchKernel", "cudaLaunchCooperativeKernel", "cuLaunchKernel")),
    ("sync", ("Synchronize",)),
    ("copy", ("cudaMemcpy",)),
    ("memset", ("cudaMemset",)),
    # Virtual-memory management is its own bottleneck and is easy to miss: the
    # driver-style names live in the runtime table, so without this family they
    # land in "other" and a stage that spends seconds mapping memory looks idle.
    (
        "vmm",
        (
            "cuMemCreate",
            "cuMemMap",
            "cuMemUnmap",
            "cuMemSetAccess",
            "cuMemRelease",
            "cuMemAddressReserve",
            "cuMemAddressFree",
            "cuMemGetAllocationGranularity",
        ),
    ),
    ("alloc", ("cudaMalloc", "cudaFree", "cudaHostAlloc", "cudaHostRegister")),
    ("query", ("cudaMemGetInfo", "cudaStreamQuery", "cudaEventQuery", "cudaPointerGetAttributes")),
    ("event", ("cudaEventRecord", "cudaEventElapsedTime")),
    ("graph", ("cudaGraph",)),
    ("module", ("cuModuleLoad", "cuLibraryLoad", "cudaFuncGetAttributes")),
)

# CUPTI records runtime symbols with an ABI-version suffix such as
# ``cudaLaunchKernel_v7000``. ``nsys stats`` strips it, so a digest that keeps it
# cannot be cross-checked against the profiler's own tables.
_API_VERSION_SUFFIX = re.compile(r"_v\d+$")


@dataclass
class Interval:
    """One half-open ``[start, end)`` span in nanoseconds."""

    start: int
    end: int


@dataclass
class NvtxRange:
    """One NVTX push/pop range, with the thread that owns it."""

    name: str
    start: int
    end: int
    tid: int

    @property
    def duration(self) -> int:
        """Return the inclusive wall duration in nanoseconds."""

        return self.end - self.start


@dataclass
class RangeStats:
    """Per-name NVTX aggregate: wall, GPU overlap, and nesting-corrected wall."""

    name: str
    instances: int = 0
    inclusive_ns: int = 0
    exclusive_ns: int = 0
    """Inclusive wall minus the wall of strictly nested child ranges."""

    gpu_busy_ns: int = 0
    """Overlap between this range's spans and the merged CUDA-activity union."""

    idle_exclusive_ns: int = 0
    """No-CUDA time this range owns directly, with nested ranges subtracted."""

    spans: list[Interval] = field(default_factory=list)
    exclusive_spans: list[Interval] = field(default_factory=list)

    @property
    def idle_ns(self) -> int:
        """Return wall inside this range with no kernel, copy, or memset."""

        return max(0, self.inclusive_ns - self.gpu_busy_ns)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Build the CLI parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Compute CUDA-activity union, no-CUDA gaps and their NVTX owners, "
            "NVTX exclusive time, kernel and copy distributions, and launch "
            "ordinals from an Nsight Systems report."
        )
    )
    parser.add_argument(
        "report",
        help=".nsys-rep (exported to SQLite on demand) or an existing .sqlite",
    )
    parser.add_argument(
        "--out",
        metavar="DIR",
        default="nsys-analysis",
        help=(
            "Write digest.txt and analysis.json here, relative to the current "
            "directory (default: nsys-analysis). Never defaults to the report's "
            "own directory: captures are evidence and stay untouched."
        ),
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Rows per ranking (default: 20).",
    )
    parser.add_argument(
        "--range",
        metavar="NAME",
        help=(
            "Clip every measurement to the union of NVTX ranges with this exact "
            "name. Use this instead of re-capturing with --capture-range."
        ),
    )
    parser.add_argument(
        "--window",
        choices=("auto", "nvtx", "cuda", "api", "trace"),
        default="auto",
        help=(
            "Which clock bounds the measurement (default: auto, meaning the NVTX "
            "span when the program is annotated and the CUDA span otherwise). "
            "'cuda' is first-kernel-to-last-kernel and excludes host setup before "
            "the first launch; 'trace' is everything the capture observed. All of "
            "them are printed in the header so the choice is visible."
        ),
    )
    parser.add_argument(
        "--exclude-prefix",
        metavar="PREFIX",
        action="append",
        default=[],
        help=(
            "Drop NVTX names starting with PREFIX from range rankings. Repeatable. "
            "Use for library or framework auto-annotation that would otherwise bury "
            "the application's own ranges: 'aten::', 'cub::', 'thrust::', or a "
            "native wrapper opened once per launch."
        ),
    )
    parser.add_argument(
        "--min-gap-us",
        type=float,
        default=100.0,
        help=(
            "Ignore no-CUDA gaps shorter than this when listing individual gaps "
            "(default: 100). Short gaps still count in the by-owner totals, "
            "which is where a launch-bound picket fence shows up."
        ),
    )
    parser.add_argument(
        "--kernel",
        metavar="SUBSTRING",
        help=(
            "Also print per-launch ordinals, durations and grids for kernels whose "
            "name contains SUBSTRING, so a launch can be selected for ncu "
            "--launch-skip."
        ),
    )
    parser.add_argument(
        "--nsys",
        default=os.environ.get("NSYS", "nsys"),
        help="nsys binary used for the SQLite export (default: $NSYS or 'nsys').",
    )
    return parser.parse_args(argv)


## Report loading


def ensure_sqlite(report: Path, nsys: str, out_dir: Path) -> Path:
    """Return a SQLite path for ``report``, exporting it into ``out_dir`` if needed.

    A capture is evidence, so nothing is written next to it: an existing sibling
    ``.sqlite`` is reused when it is newer than the report, but a fresh export
    goes to ``out_dir``. That also keeps the tool usable when the capture sits on
    a read-only mount.

    Args:
        report: Either a ``.sqlite`` file or a ``.nsys-rep`` to export.
        nsys: Name or path of the ``nsys`` binary to use for the export.
        out_dir: Directory that receives a new export and its log.

    Returns:
        Path to a SQLite database.

    Raises:
        FileNotFoundError: ``report`` does not exist, or ``nsys`` is not on PATH
            while an export is required.
        RuntimeError: The export subprocess failed.
    """

    if not report.exists():
        raise FileNotFoundError(f"no such report: {report}")
    if report.suffix == ".sqlite":
        return report

    sibling = report.with_suffix(".sqlite")
    if sibling.exists() and sibling.stat().st_mtime >= report.stat().st_mtime:
        return sibling

    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{report.stem}.sqlite"
    if target.exists() and target.stat().st_mtime >= report.stat().st_mtime:
        return target
    if shutil.which(nsys) is None:
        raise FileNotFoundError(
            f"{nsys} not found; export manually with "
            f"'nsys export --type sqlite --output {target} {report}'"
        )
    # Export is the long pole on a large report. Keep the log beside the export so
    # a failure is diagnosable without re-running the capture.
    log = target.with_suffix(".export.log")
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(
            [
                nsys,
                "export",
                "--type",
                "sqlite",
                "--force-overwrite=true",
                "--output",
                str(target),
                str(report),
            ],
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0 or not target.exists():
        raise RuntimeError(f"nsys export failed (exit {result.returncode}); see {log}")
    return target


def open_readonly(path: Path) -> sqlite3.Connection:
    """Open a SQLite database read-only so analysis cannot corrupt evidence."""

    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    """Return whether a table is present.

    Table availability depends on which ``--trace`` options the capture used, so
    every read must tolerate absence rather than assume a full schema.
    """

    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return row is not None


def column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    """Return whether ``table`` has ``column``.

    Column sets drift between `nsys` versions. Checking beats letting a query
    raise, because one absent column would otherwise cost the whole analysis
    instead of the one measurement that needs it.
    """

    return any(
        row[1] == column for row in conn.execute(f"PRAGMA table_info({table})")
    )


## Interval algebra


def merge_intervals(spans: Iterable[tuple[int, int]]) -> list[Interval]:
    """Merge overlapping spans into a disjoint, ascending union.

    This is the operation behind every honest "GPU was busy" number: kernels,
    copies, and memsets overlap each other, so their durations must never be
    added.
    """

    ordered = sorted(spans)
    merged: list[Interval] = []
    for start, end in ordered:
        if end <= start:
            continue
        if merged and start <= merged[-1].end:
            merged[-1].end = max(merged[-1].end, end)
            continue
        merged.append(Interval(start, end))
    return merged


def total_ns(intervals: Sequence[Interval]) -> int:
    """Return the summed length of a disjoint interval list."""

    return sum(iv.end - iv.start for iv in intervals)


def clip(intervals: Sequence[Interval], window: Sequence[Interval]) -> list[Interval]:
    """Restrict ``intervals`` to ``window``, both disjoint and ascending."""

    if not window:
        return list(intervals)
    starts = [iv.start for iv in window]
    out: list[Interval] = []
    for iv in intervals:
        idx = max(0, bisect.bisect_right(starts, iv.start) - 1)
        # Walk by index rather than slicing: ``window[idx:]`` would copy the tail of
        # a million-element gap list once per interval.
        while idx < len(window):
            candidate = window[idx]
            if candidate.start >= iv.end:
                break
            lo = max(iv.start, candidate.start)
            hi = min(iv.end, candidate.end)
            if hi > lo:
                out.append(Interval(lo, hi))
            idx += 1
    return out


class Coverage:
    """A disjoint interval set with prefix sums, for fast overlap queries.

    "How much of this span was the GPU busy" gets asked once per NVTX range and
    once per gap, against a union that can hold hundreds of thousands of islands
    on a real capture. Walking the islands inside each span makes the total cost
    quadratic, so covered length is differenced out of a prefix sum instead:
    every query is two binary searches regardless of how long the span is.
    """

    def __init__(self, intervals: Sequence[Interval]) -> None:
        self.starts = [iv.start for iv in intervals]
        self.ends = [iv.end for iv in intervals]
        # prefix[i] is the covered length of the first i intervals.
        self.prefix = [0] * (len(intervals) + 1)
        for i, iv in enumerate(intervals):
            self.prefix[i + 1] = self.prefix[i] + (iv.end - iv.start)

    def before(self, point: int) -> int:
        """Return covered length strictly before ``point``."""

        idx = bisect.bisect_right(self.starts, point) - 1
        if idx < 0:
            return 0
        total = self.prefix[idx + 1]
        # The interval straddling ``point`` is counted whole by the prefix sum.
        if self.ends[idx] > point:
            total -= self.ends[idx] - point
        return total

    def covered(self, start: int, end: int) -> int:
        """Return covered length inside ``[start, end)``."""

        if end <= start:
            return 0
        return self.before(end) - self.before(start)


class Window:
    """The measurement window, as disjoint bounds with fast point lookup.

    Row filtering asks "is this timestamp in the window" once per kernel, copy,
    and API call. Testing every bound per row is linear in the number of bounds,
    and ``--range`` on a per-request annotation routinely yields thousands of
    them, so the bound is found by binary search.
    """

    def __init__(self, bounds: Sequence[Interval]) -> None:
        self.bounds = list(bounds)
        self.starts = [iv.start for iv in self.bounds]

    def contains(self, point: int) -> bool:
        """Return whether ``point`` falls inside any bound."""

        idx = bisect.bisect_right(self.starts, point) - 1
        return idx >= 0 and point < self.bounds[idx].end

    def first_overlap(self, start: int, end: int) -> tuple[int, int] | None:
        """Return the earliest sub-span of ``[start, end)`` inside the window."""

        idx = max(0, bisect.bisect_right(self.starts, start) - 1)
        for bound in self.bounds[idx:]:
            if bound.start >= end:
                break
            lo, hi = max(start, bound.start), min(end, bound.end)
            if hi > lo:
                return lo, hi
        return None


def span_of(bounds: Iterable[tuple[int, int]]) -> Interval | None:
    """Return the smallest interval enclosing ``bounds``, or ``None`` if empty."""

    lo: int | None = None
    hi: int | None = None
    for start, end in bounds:
        if lo is None or start < lo:
            lo = start
        if hi is None or end > hi:
            hi = end
    return None if lo is None or hi is None else Interval(lo, hi)


def overlap_ns(spans: Sequence[Interval], coverage: Coverage) -> int:
    """Return how much of ``spans`` is covered by ``coverage``."""

    return sum(coverage.covered(iv.start, iv.end) for iv in spans)


def gaps_between(union: Sequence[Interval], window: Sequence[Interval]) -> list[Interval]:
    """Return sub-windows of ``window`` that ``union`` does not cover."""

    starts = [iv.start for iv in union]
    out: list[Interval] = []
    for bound in window:
        cursor = bound.start
        # Seek to the last island starting at or before this bound, so a window
        # of many bounds does not rescan the whole union for each one.
        for busy in union[max(0, bisect.bisect_right(starts, bound.start) - 1) :]:
            if busy.start >= bound.end:
                break
            if busy.end <= cursor:
                continue
            if busy.start > cursor:
                out.append(Interval(cursor, min(busy.start, bound.end)))
            cursor = busy.end
            if cursor >= bound.end:
                break
        if cursor < bound.end:
            out.append(Interval(cursor, bound.end))
    return [iv for iv in out if iv.end > iv.start]


## Extraction


def read_activity(conn: sqlite3.Connection) -> dict[str, list[tuple[int, int]]]:
    """Read kernel, memcpy, and memset spans that define GPU-busy time."""

    kinds = {
        "kernel": "CUPTI_ACTIVITY_KIND_KERNEL",
        "memcpy": "CUPTI_ACTIVITY_KIND_MEMCPY",
        "memset": "CUPTI_ACTIVITY_KIND_MEMSET",
    }
    out: dict[str, list[tuple[int, int]]] = {}
    for label, table in kinds.items():
        if not table_exists(conn, table):
            out[label] = []
            continue
        out[label] = [
            (int(a), int(b))
            for a, b in conn.execute(
                f"SELECT start, end FROM {table} "
                "WHERE start IS NOT NULL AND end IS NOT NULL"
            )
        ]
    return out


def read_nvtx(conn: sqlite3.Connection) -> list[NvtxRange]:
    """Read closed NVTX push/pop ranges, resolving names through ``StringIds``.

    NVTX text is stored inline in ``NVTX_EVENTS.text`` for some producers and as
    a ``textId`` into ``StringIds`` for others, so both must be coalesced.
    """

    if not table_exists(conn, "NVTX_EVENTS"):
        return []
    rows = conn.execute(
        """
        SELECT COALESCE(n.text, s.value) AS name, n.start, n.end, n.globalTid
        FROM NVTX_EVENTS n
        LEFT JOIN StringIds s ON s.id = n.textId
        WHERE n.end IS NOT NULL AND n.start IS NOT NULL
        """
    )
    return [
        NvtxRange(name=name or "(unnamed)", start=int(a), end=int(b), tid=int(tid or 0))
        for name, a, b, tid in rows
    ]


def read_kernels(conn: sqlite3.Connection) -> list[tuple[str, int, int, int, int]]:
    """Return ``(name, start, duration, launch_start, launch_tid)`` per kernel.

    ``launch_start`` is the host timestamp of the runtime call that submitted the
    kernel, recovered through the CUPTI correlation id. Attribution to an
    application phase must use it rather than the kernel's own start: submission
    is asynchronous, so a kernel launched inside one stage routinely executes
    after that stage's NVTX range has closed. Matching device timestamps against
    host ranges therefore credits whichever stage happened to be open when the
    GPU got round to the work, which is a different question than who asked for
    it, and the two answers diverge exactly when a pipeline is deep.

    ``launch_tid`` is the submitting thread, needed because NVTX ranges are
    per-thread: the innermost range open on the launching thread is the owner,
    and another thread's range that merely overlaps in time is not.

    Kernels with no correlated runtime row report ``launch_start = -1`` so the
    caller can declare them unattributed rather than guess. That happens for CUDA
    graph replay, where the launch is the graph, and for captures taken without
    CUDA API tracing.
    """

    if not table_exists(conn, "CUPTI_ACTIVITY_KIND_KERNEL"):
        return []
    correlated = table_exists(conn, "CUPTI_ACTIVITY_KIND_RUNTIME") and all(
        column_exists(conn, table, "correlationId")
        for table in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_KERNEL")
    )
    if correlated:
        sql = """
            SELECT s.value, k.start, k.end - k.start, r.start, r.globalTid
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN StringIds s ON s.id = k.shortName
            LEFT JOIN CUPTI_ACTIVITY_KIND_RUNTIME r
                   ON r.correlationId = k.correlationId
            WHERE k.end IS NOT NULL
        """
    else:
        sql = """
            SELECT s.value, k.start, k.end - k.start, NULL, NULL
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN StringIds s ON s.id = k.shortName
            WHERE k.end IS NOT NULL
        """
    return [
        (
            name or "(unknown)",
            int(start),
            int(dur),
            int(launch) if launch is not None else -1,
            int(tid) if tid is not None else 0,
        )
        for name, start, dur, launch, tid in conn.execute(sql)
    ]


def read_copies(conn: sqlite3.Connection) -> list[tuple[str, int, int, int]]:
    """Return ``(direction, start, duration, bytes)`` for every copy."""

    if not table_exists(conn, "CUPTI_ACTIVITY_KIND_MEMCPY"):
        return []
    labels: dict[int, str] = {}
    if table_exists(conn, "ENUM_CUDA_MEMCPY_OPER"):
        labels = {
            int(i): str(label) for i, label in conn.execute("SELECT id, label FROM ENUM_CUDA_MEMCPY_OPER")
        }
    rows = conn.execute(
        "SELECT copyKind, start, end - start, bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY "
        "WHERE end IS NOT NULL"
    )
    return [
        (labels.get(int(kind), f"kind{kind}"), int(start), int(dur), int(nbytes or 0))
        for kind, start, dur, nbytes in rows
    ]


def read_runtime_api(conn: sqlite3.Connection) -> list[tuple[str, int, int]]:
    """Return ``(name, start, duration)`` for every CUDA runtime API call.

    Names have their ``_v<digits>`` ABI suffix removed so the table can be
    compared against ``nsys stats``, which strips it too.
    """

    if not table_exists(conn, "CUPTI_ACTIVITY_KIND_RUNTIME"):
        return []
    rows = conn.execute(
        """
        SELECT s.value, r.start, r.end - r.start
        FROM CUPTI_ACTIVITY_KIND_RUNTIME r
        JOIN StringIds s ON s.id = r.nameId
        WHERE r.end IS NOT NULL
        """
    )
    return [
        (_API_VERSION_SUFFIX.sub("", name or "(unknown)"), int(start), int(dur))
        for name, start, dur in rows
    ]


def read_kernel_launches(
    conn: sqlite3.Connection, substring: str
) -> list[tuple[str, int, int, int, int, int]]:
    """Return ``(name, ordinal, start, duration, gridX, blockX)`` for matched kernels.

    Ordinals restart per exact kernel name. ``ncu --launch-skip`` counts launches
    that pass ncu's own ``-k`` filter, so numbering a substring match as one
    sequence would hand back an ordinal that selects a different launch whenever
    the substring matches more than one kernel.
    """

    if not table_exists(conn, "CUPTI_ACTIVITY_KIND_KERNEL"):
        return []
    rows = conn.execute(
        """
        SELECT s.value, k.start, k.end - k.start, k.gridX, k.blockX
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON s.id = k.shortName
        WHERE s.value LIKE ? AND k.end IS NOT NULL
        ORDER BY k.start
        """,
        (f"%{substring}%",),
    )
    seen: dict[str, int] = {}
    launches: list[tuple[str, int, int, int, int, int]] = []
    for name, start, dur, grid, block in rows:
        name = name or "(unknown)"
        ordinal = seen.get(name, 0)
        seen[name] = ordinal + 1
        launches.append((name, ordinal, int(start), int(dur), int(grid or 0), int(block or 0)))
    return launches


## Aggregation


def _credit_exclusive(
    stats: dict[str, RangeStats],
    owners: list[tuple[int, int, int, str]],
    target: NvtxRange,
    start: int,
    end: int,
) -> None:
    """Credit ``[start, end)`` to ``target`` as time no nested range was open."""

    if end <= start:
        return
    entry = stats.setdefault(target.name, RangeStats(name=target.name))
    entry.exclusive_ns += end - start
    entry.exclusive_spans.append(Interval(start, end))
    owners.append((target.tid, start, end, target.name))


def _close_through(
    stack: list[list[Any]],
    stats: dict[str, RangeStats],
    owners: list[tuple[int, int, int, str]],
    until: int,
) -> None:
    """Pop ranges ending at or before ``until``, crediting each its tail time."""

    while stack and stack[-1][0].end <= until:
        top, cursor = stack.pop()
        _credit_exclusive(stats, owners, top, cursor, top.end)
        if stack:
            stack[-1][1] = max(stack[-1][1], top.end)


def aggregate_ranges(
    ranges: Sequence[NvtxRange],
    union: Coverage,
    gaps: Coverage,
    exclude_prefixes: Sequence[str],
) -> tuple[list[RangeStats], OwnerIndex]:
    """Aggregate NVTX ranges by name with inclusive, exclusive, busy, and idle time.

    Exclusive time subtracts nested child ranges on the same thread. Without it,
    adding a parent stage to its own substeps double-counts the same wall clock,
    which is the most common way a phase table stops reconciling.

    ``idle_exclusive_ns`` is the number that names a host-side bottleneck: the
    no-CUDA time a range owns after its children have taken their share. It is
    exact for single-threaded submission, and can double-count across threads
    that overlap, which is why the digest labels it per-owner rather than
    presenting it as a partition of the gap.

    Args:
        ranges: Closed NVTX ranges, already clipped to the measurement window.
        union: Coverage over the merged CUDA-activity intervals.
        gaps: Coverage over the merged no-CUDA intervals inside the window.
        exclude_prefixes: Name prefixes to drop from both ranking and ownership.

    Returns:
        Per-name stats descending by inclusive wall, and the per-thread index of
        exclusive spans that answers "which range was open here". The index is
        built during the same sweep because it needs the owning thread, which the
        per-name aggregate has already merged away.
    """

    kept = [
        r
        for r in ranges
        if not any(r.name.startswith(prefix) for prefix in exclude_prefixes)
    ]
    by_tid: dict[int, list[NvtxRange]] = {}
    for r in kept:
        by_tid.setdefault(r.tid, []).append(r)

    stats: dict[str, RangeStats] = {}
    owner_rows: list[tuple[int, int, int, str]] = []
    for group in by_tid.values():
        # Sorting by (start, -end) puts every enclosing range before the ranges
        # it contains, so one pass with a stack of open ranges is enough. Looking
        # ahead for each range's children instead costs a scan per range, which
        # is quadratic on exactly the deep per-step annotation this table exists
        # to read.
        group.sort(key=lambda r: (r.start, -r.end))
        # Each entry is ``[range, cursor]``: the timestamp since which that range
        # has been the innermost one open.
        stack: list[list[Any]] = []
        for r in group:
            _close_through(stack, stats, owner_rows, r.start)
            if stack:
                parent, cursor = stack[-1]
                _credit_exclusive(stats, owner_rows, parent, cursor, r.start)
                stack[-1][1] = max(cursor, r.start)
            entry = stats.setdefault(r.name, RangeStats(name=r.name))
            entry.instances += 1
            entry.inclusive_ns += r.duration
            entry.spans.append(Interval(r.start, r.end))
            stack.append([r, r.start])
        _close_through(stack, stats, owner_rows, sys.maxsize)

    for entry in stats.values():
        entry.spans = merge_intervals((iv.start, iv.end) for iv in entry.spans)
        entry.exclusive_spans = merge_intervals(
            (iv.start, iv.end) for iv in entry.exclusive_spans
        )
        entry.gpu_busy_ns = overlap_ns(entry.spans, union)
        entry.idle_exclusive_ns = overlap_ns(entry.exclusive_spans, gaps)
    ordered = sorted(stats.values(), key=lambda s: s.inclusive_ns, reverse=True)
    return ordered, OwnerIndex(owner_rows)


class OwnerIndex:
    """Exclusive NVTX spans per thread, searchable by timestamp.

    Exclusive spans have their nested children removed, so the span covering a
    timestamp is already the innermost range open at that moment. Answering "the
    request was running" is true and useless; this answers which step was running.

    The index is kept per thread because NVTX ranges are per thread. Flattening
    every thread into one list looks simpler and then fails in two ways on a
    multi-threaded submitter: a span from an unrelated thread can sit between the
    query point and the range that actually owns it, which reports no owner at
    all, and where threads genuinely overlap there is no single right answer.
    Searching within the asking thread makes the first case exact, and idle time
    is reported per owner rather than as a partition of the gap for the second.
    """

    def __init__(self, rows: Iterable[tuple[int, int, int, str]]) -> None:
        grouped: dict[int, list[tuple[int, int, str]]] = {}
        for tid, start, end, name in rows:
            grouped.setdefault(tid, []).append((start, end, name))
        self.by_tid: dict[int, tuple[list[int], list[int], list[str]]] = {}
        for tid, spans in grouped.items():
            spans.sort()
            self.by_tid[tid] = (
                [row[0] for row in spans],
                [row[1] for row in spans],
                [row[2] for row in spans],
            )

    def _in_thread(self, point: int, tid: int) -> tuple[int, str] | None:
        """Return ``(span_length, name)`` for the range open on ``tid``."""

        entry = self.by_tid.get(tid)
        if entry is None:
            return None
        starts, ends, names = entry
        # Exclusive spans are disjoint within a thread, so the nearest span
        # starting at or before the point is the only candidate.
        idx = bisect.bisect_right(starts, point) - 1
        if idx >= 0 and point < ends[idx]:
            return ends[idx] - starts[idx], names[idx]
        return None

    def at(self, point: int, tid: int | None = None) -> str:
        """Return the innermost range open at ``point``, or ``"(none)"``.

        Args:
            point: Timestamp to resolve, on the host clock.
            tid: Asking thread. Given one, only that thread's ranges can own the
                point, which is what kernel attribution needs. Without one, every
                thread is searched and the shortest containing span wins, which is
                what a GPU-side gap needs since a gap has no owning thread.
        """

        if tid is not None:
            found = self._in_thread(point, tid)
            return found[1] if found is not None else "(none)"
        best: tuple[int, str] | None = None
        for candidate in self.by_tid:
            found = self._in_thread(point, candidate)
            if found is not None and (best is None or found < best):
                best = found
        return best[1] if best is not None else "(none)"


@dataclass
class KernelSummary:
    """Kernel aggregates collected in a single pass over the window's launches."""

    calls: int = 0
    device_ns: int = 0
    """Sum of kernel durations. Streams overlap, so this is not wall time."""

    by_name: dict[str, list[int]] = field(default_factory=dict)
    """Kernel name to ``[calls, total_ns, max_ns]``."""

    buckets: dict[str, list[int]] = field(default_factory=dict)
    """Duration bucket label to ``[calls, total_ns]``."""

    by_owner: dict[str, list[int]] = field(default_factory=dict)
    """Launching NVTX range to ``[calls, device_ns]``."""

    unattributed: list[int] = field(default_factory=lambda: [0, 0])
    """``[calls, device_ns]`` for launches with no recoverable owner."""


def summarize_kernels(
    rows: Iterable[tuple[str, int, int, int, int]],
    window: Coverage,
    owners: OwnerIndex,
) -> KernelSummary:
    """Aggregate kernels by name, duration bucket, and launching NVTX range.

    All of it happens in one pass. This is the largest table in the report, so
    each extra traversal of it costs more than the rest of the analysis together.

    Time is clipped to the window rather than counted whole for anything that
    starts inside it. A kernel straddling the boundary would otherwise contribute
    time the window never contained, which is how a ``--range`` digest ends up
    reporting more device time than the range has wall time. Bucket and ``max``
    still use the kernel's true length, because a long kernel sliced by a boundary
    is still a long kernel and belongs in the row that says so.

    Args:
        rows: ``(name, start, duration, launch_start, launch_tid)`` from
            :func:`read_kernels`.
        window: Measurement window as a coverage set; kernels are clipped to it
            and those with no overlap are skipped.
        owners: Index used to credit each kernel to the range that launched it.
    """

    summary = KernelSummary(buckets={label: [0, 0] for label, _ in _KERNEL_BUCKETS})
    for name, start, dur, launch, launch_tid in rows:
        inside = window.covered(start, start + dur)
        if inside <= 0:
            continue
        summary.calls += 1
        summary.device_ns += inside
        entry = summary.by_name.get(name)
        if entry is None:
            summary.by_name[name] = [1, inside, dur]
        else:
            entry[0] += 1
            entry[1] += inside
            entry[2] = max(entry[2], dur)
        for label, limit in _KERNEL_BUCKETS:
            if limit is None or dur < limit:
                bucket = summary.buckets[label]
                bucket[0] += 1
                bucket[1] += inside
                break
        owner = owners.at(launch, launch_tid) if launch >= 0 else "(none)"
        target = (
            summary.unattributed
            if owner == "(none)"
            else summary.by_owner.setdefault(owner, [0, 0])
        )
        target[0] += 1
        target[1] += inside
    return summary


def bucket_copies(
    copies: Sequence[tuple[str, int, int, int]]
) -> dict[str, dict[str, Any]]:
    """Return per-direction copy counts, bytes, engine time, and size buckets."""

    out: dict[str, dict[str, Any]] = {}
    for direction, _, dur, nbytes in copies:
        entry = out.setdefault(
            direction,
            {
                "calls": 0,
                "bytes": 0,
                "engine_ns": 0,
                "buckets": {label: 0 for label, _ in _COPY_BUCKETS},
            },
        )
        entry["calls"] += 1
        entry["bytes"] += nbytes
        entry["engine_ns"] += dur
        for label, limit in _COPY_BUCKETS:
            if limit is None or nbytes <= limit:
                entry["buckets"][label] += 1
                break
    return out


def attribute_gaps_to_api(
    gaps: Sequence[Interval], apis: Sequence[tuple[str, int, int]]
) -> dict[str, Any]:
    """Credit no-CUDA time to the host API calls that were in flight during it.

    This is the idle classification that survives an unannotated program. NVTX
    ownership says which application phase was idle, but most code has no NVTX at
    all, and then the only witness to what the host was doing is the API call it
    was sitting in: time inside ``cudaStreamSynchronize`` or a blocking
    ``cudaMemcpy`` is the host waiting, time inside ``cudaMalloc`` or ``cudaFree``
    is allocator overhead, and idle with no CUDA call open at all is host compute
    or something outside CUDA entirely.

    Per-name numbers are overlaps, not a partition: calls on different threads can
    be open over the same idle nanosecond, so the column sums to more than the
    idle total. ``api_open_ns`` merges the calls first and is the honest total.
    """

    if not gaps:
        return {"rows": [], "api_open_ns": 0, "no_api_ns": 0, "gap_ns": 0}
    gap_cov = Coverage(gaps)
    per_name: dict[str, list[int]] = {}
    spans: list[tuple[int, int]] = []
    for name, start, dur in apis:
        end = start + dur
        inside = gap_cov.covered(start, end)
        if inside <= 0:
            continue
        entry = per_name.setdefault(name, [0, 0])
        entry[0] += inside
        entry[1] += 1
        spans.append((start, end))

    # Merged spans are disjoint, so their gap-covered lengths add up to exactly the
    # intersection of "some API open" with "GPU idle" — no interval list to build.
    api_open_ns = sum(gap_cov.covered(iv.start, iv.end) for iv in merge_intervals(spans))
    gap_ns = total_ns(gaps)
    rows = sorted(
        (
            {"name": name, "idle_ns": idle, "calls": calls, "family": classify_api(name)}
            for name, (idle, calls) in per_name.items()
        ),
        key=lambda row: row["idle_ns"],
        reverse=True,
    )
    return {
        "rows": rows,
        "api_open_ns": api_open_ns,
        "no_api_ns": gap_ns - api_open_ns,
        "gap_ns": gap_ns,
    }


def classify_api(name: str) -> str:
    """Return the API family for a CUDA runtime call name."""

    for family, needles in _API_FAMILIES:
        if any(needle in name for needle in needles):
            return family
    return "other"


## Rendering


def fmt_s(ns: int) -> str:
    """Format nanoseconds as seconds with millisecond resolution."""

    return f"{ns / 1e9:.3f}"


def render(result: dict[str, Any], top: int) -> str:
    """Render the analysis as a plain-text digest."""

    lines: list[str] = []
    add = lines.append

    add("=" * 78)
    add(f"report        {result['report']}")
    add(f"window        {result['window_label']}")
    add(f"window wall   {fmt_s(result['window_ns'])} s")
    add("=" * 78)
    add("")

    if result.get("clocks"):
        add("## Clocks in this report")
        add("")
        add(f"  {'clock':<6} {'wall s':>9}  {'offset from trace start (s)':>28}")
        for name, span in result["clocks"].items():
            add(
                f"  {name:<6} {span['wall_s']:>9.3f}  "
                f"{span['from_s']:>13.3f} .. {span['to_s']:<12.3f}"
            )
        add("")
        add("  These differ, and the difference is a finding. Wall time inside the")
        add("  trace span but outside the CUDA span is host work with no GPU work at")
        add("  all, which no kernel table can show. If the cuda span is much shorter")
        add("  than the nvtx span, the program spends that difference on the host.")
        add("")

    add("## CUDA activity versus wall")
    add("")
    add(f"  CUDA-activity union   {fmt_s(result['union_ns'])} s  "
        f"({result['union_pct']:.1f}% of window)")
    add(f"  no-CUDA gap           {fmt_s(result['gap_ns'])} s  "
        f"({100 - result['union_pct']:.1f}%)")
    add(f"  kernel span sum       {fmt_s(result['kernel_sum_ns'])} s  (overlaps; not additive)")
    add(f"  copy engine sum       {fmt_s(result['copy_sum_ns'])} s  (overlaps; not additive)")
    activities = result.get("activity_count") or 0
    add(f"  activity islands      {result['islands']} of {activities} activities")
    if activities and result["islands"] >= 0.99 * activities:
        add("                        (every activity is its own island: nothing on")
        add("                        the device overlapped, so kernels, copies, and")
        add("                        memsets all ran strictly one at a time)")
    add("")
    add("  The gap is wall minus the union of kernels, copies, and memsets. It is")
    add("  not 'CPU compute': classify it before proposing a fix. Host API")
    add("  duration overlaps GPU work and must not be added into it.")
    add("")

    if result.get("clip_hint"):
        hint = result["clip_hint"]
        add("  BEFORE QUOTING THE PERCENTAGE ABOVE: a phase inside this window is far")
        add(f"  less busy than the window as a whole. {hint['name']!r} runs")
        add(f"  {hint['inclusive_s']:.3f} s at {hint['busy_pct']:.1f}% busy, against "
            f"{result['union_pct']:.1f}% here.")
        add("  Setup that saturates the GPU is averaging away the phase that idles.")
        add("  Re-run clipped to it before classifying the program:")
        add(f"    --range {hint['name']!r}")
        add("")

    if result.get("idle_owners"):
        add(f"## No-CUDA time by owning NVTX range (top {min(top, len(result['idle_owners']))})")
        add("")
        add(f"  {'idle excl':>9} {'wall excl':>9} {'n':>8}  name")
        for owner in result["idle_owners"][:top]:
            add(
                f"  {owner['idle_exclusive_s']:>9.3f} {owner['exclusive_s']:>9.3f} "
                f"{owner['instances']:>8}  {owner['name']}"
            )
        add("")
        add("  This is the ranking that names a host-side bottleneck, because it")
        add("  survives being spread over many small gaps. A range can own large")
        add("  idle here while appearing nowhere in the list of largest gaps below.")
        add("")

    idle_api = result.get("idle_api") or {}
    if idle_api.get("gap_ns"):
        add(f"## No-CUDA time by host API in flight (top {min(top, len(idle_api['rows']))})")
        add("")
        add(f"  {'idle s':>9} {'calls':>9} {'family':<9}  api")
        for row in idle_api["rows"][:top]:
            add(
                f"  {row['idle_ns'] / 1e9:>9.3f} {row['calls']:>9} {row['family']:<9}  "
                f"{row['name']}"
            )
        add("")
        gap_ns = idle_api["gap_ns"]
        add(
            f"  idle with a CUDA call open   {fmt_s(idle_api['api_open_ns'])}"
            f"  ({idle_api['api_open_ns'] / gap_ns * 100:.1f}% of idle)"
        )
        add(
            f"  idle with none open          {fmt_s(idle_api['no_api_ns'])}"
            f"  ({idle_api['no_api_ns'] / gap_ns * 100:.1f}%)"
        )
        add("")
        add("  This classifies the idle without needing NVTX. A sync or a blocking")
        add("  copy at the top means the host is waiting on the GPU, so look for what")
        add("  it waits for; malloc or free means allocator overhead, and a caching")
        add("  allocator should be hiding it; idle with no CUDA call open at all is")
        add("  host compute or I/O, which no CUDA-side change will fix.")
        add("")
        add("  Per-api seconds are overlaps and sum to more than the idle total when")
        add("  threads wait concurrently. The two lines above are the partition.")
        add("")

    if result["gap_buckets"]:
        add("## No-CUDA gaps by duration")
        add("")
        add(f"  {'bucket':<11} {'count':>9} {'seconds':>9} {'share':>7}")
        for row in result["gap_buckets"]:
            add(
                f"  {row['bucket']:<11} {row['count']:>9} {row['seconds']:>9.3f} "
                f"{row['share'] * 100:>6.1f}%"
            )
        add("")
        add("  Read where the seconds sit, not where the count sits. Seconds in the")
        add("  long buckets are serial host phases: overlap or port them. The same")
        add("  seconds in the sub-millisecond buckets are launch-bound: batch or fuse")
        add("  the launches. Fixing the wrong shape buys nothing. Capturing a CUDA")
        add("  graph comes last: a graph cannot replay a sequence whose shape depends")
        add("  on values the host reads back, so the small copies and syncs in the")
        add("  tables below have to go first.")
        add("")

    if result["gaps"]:
        listed = min(top, len(result["gaps"]))
        add(
            f"## Largest individual no-CUDA gaps (showing {listed} of "
            f"{result['gap_count']} gaps, those over {result.get('min_gap_us', 0):.0f} us)"
        )
        add("")
        add(f"  {'ms':>10}  {'t+ (s)':>10}  owner")
        for gap in result["gaps"][:top]:
            add(f"  {gap['seconds'] * 1e3:>10.3f}  {gap['at_s']:>10.3f}  {gap['owner']}")
        add("")
        add("  One big gap is a blocking call or a host phase. Thousands of tiny")
        add("  gaps are a launch-rate problem; check the count above and the")
        add("  by-owner table, not this list.")
        add("")

    if result["ranges"]:
        add(f"## NVTX ranges by inclusive wall (top {min(top, len(result['ranges']))})")
        add("")
        add(
            f"  {'incl s':>9} {'wall excl':>9} {'busy s':>9} {'idle incl':>9} "
            f"{'busy%':>6} {'n':>8}  name"
        )
        for r in result["ranges"][:top]:
            add(
                f"  {r['inclusive_s']:>9.3f} {r['exclusive_s']:>9.3f} "
                f"{r['gpu_busy_s']:>9.3f} {r['idle_s']:>9.3f} "
                f"{r['busy_pct']:>6.1f} {r['instances']:>8}  {r['name']}"
            )
        add("")
        add("  Inclusive columns nest: do not add them. Exclusive subtracts nested")
        add("  children on the same thread and is the column that reconciles.")
        add("")

    if result["kernels"]:
        add(f"## Kernels by device time (top {min(top, len(result['kernels']))})")
        add("")
        add(f"  {'total s':>9} {'calls':>9} {'mean us':>10} {'max ms':>9}  name")
        for k in result["kernels"][:top]:
            add(
                f"  {k['total_s']:>9.3f} {k['calls']:>9} {k['mean_us']:>10.1f} "
                f"{k['max_ms']:>9.3f}  {k['name']}"
            )
        add("")
        add("## Kernel duration distribution")
        add("")
        add(f"  {'bucket':>8} {'calls':>10} {'total s':>10}")
        for bucket in result["kernel_buckets"]:
            add(f"  {bucket['bucket']:>8} {bucket['calls']:>10} {bucket['total_s']:>10.3f}")
        add("")
        add("  Many short calls means fuse, batch, or capture. Few long calls means")
        add("  profile the kernel. The treatments are different.")
        add("")

    if result.get("kernel_owners"):
        add(f"## Kernel device time by launching NVTX range (top {min(top, len(result['kernel_owners']))})")
        add("")
        add(f"  {'device s':>10} {'calls':>10}  name")
        for owner in result["kernel_owners"][:top]:
            add(f"  {owner['device_s']:>10.3f} {owner['calls']:>10}  {owner['name']}")
        unattributed = result["kernel_unattributed"]
        if unattributed["calls"]:
            add(
                f"  {unattributed['device_s']:>10.3f} {unattributed['calls']:>10}  "
                "(unattributed)"
            )
        add("")
        add("  Ownership follows the correlation id from the kernel back to the host")
        add("  call that launched it, so a kernel is credited to the stage that")
        add("  submitted it even when it ran after that stage returned. Device time")
        add("  is summed per owner and overlaps across streams, so compare owners")
        add("  against each other and not against the window. The owner is looked")
        add("  up on the submitting thread, so launches from a thread that carries")
        add("  no NVTX ranges land in the unattributed row rather than borrowing")
        add("  another thread's stage. That row is also graph replay and launch")
        add("  sites outside the window; a large count there means annotate the")
        add("  submitting thread before reasoning from this table.")
        add("")

    if result["copies"]:
        add("## Copies by direction and size")
        add("")
        header = f"  {'direction':<12} {'calls':>9} {'MiB':>10} {'engine s':>9}"
        for label, _ in _COPY_BUCKETS:
            header += f" {label:>8}"
        add(header)
        for direction, entry in result["copies"].items():
            row = (
                f"  {direction:<12} {entry['calls']:>9} "
                f"{entry['bytes'] / 1048576:>10.1f} {entry['engine_s']:>9.3f}"
            )
            for label, _ in _COPY_BUCKETS:
                row += f" {entry['buckets'][label]:>8}"
            add(row)
        add("")
        add("  Separate count from bytes. A flood of tiny device-to-host copies is a")
        add("  control-flow problem; a few large ones are a payload problem.")
        add("")

    if result["api_families"]:
        add("## CUDA runtime API by family")
        add("")
        add(f"  {'family':<10} {'calls':>10} {'host s':>10}")
        for family, entry in result["api_families"].items():
            add(f"  {family:<10} {entry['calls']:>10} {entry['host_s']:>10.3f}")
        add("")
        add(f"## CUDA runtime API by host time (top {min(top, len(result['apis']))})")
        add("")
        add(f"  {'host s':>10} {'calls':>10}  name")
        for api in result["apis"][:top]:
            add(f"  {api['host_s']:>10.3f} {api['calls']:>10}  {api['name']}")
        add("")

    if result.get("launches"):
        add(f"## Launch ordinals for kernels matching '{result['launch_filter']}'")
        add("")
        add(f"  {'ordinal':>8} {'t+ (s)':>10} {'ms':>9} {'gridX':>10} {'blockX':>7}  name")
        for launch in result["launches"][:top]:
            add(
                f"  {launch['ordinal']:>8} {launch['at_s']:>10.3f} "
                f"{launch['ms']:>9.3f} {launch['gridX']:>10} {launch['blockX']:>7}  "
                f"{launch['name']}"
            )
        add("")
        add("  Ordinals restart per exact kernel name and are what ncu --launch-skip")
        add("  counts once ncu's -k filter selects that same one name. Pass the exact")
        add("  name from this table to -k, or the ordinal will select another kernel.")
        add("  Pick a launch whose grid matches production, not the first match: early")
        add("  launches are usually warmup.")
        add("")

    return "\n".join(lines) + "\n"


## Driver


_GAP_BUCKETS: tuple[tuple[str, int], ...] = (
    ("<10us", 10_000),
    ("10-100us", 100_000),
    ("100us-1ms", 1_000_000),
    ("1-10ms", 10_000_000),
    ("10-100ms", 100_000_000),
    ("100ms-1s", 1_000_000_000),
    (">=1s", 1 << 62),
)


def bucket_gaps(gaps: Sequence[Interval]) -> list[dict[str, Any]]:
    """Bucket no-CUDA gaps by duration.

    This separates the two idle shapes that a ranked list of largest gaps cannot
    tell apart, because they call for opposite fixes. Seconds concentrated in a
    handful of long gaps are serial host phases: overlap or port them. The same
    seconds spread over thousands of sub-millisecond gaps are a launch-bound
    picket fence: batch, fuse, or graph the launches. A list of largest gaps shows
    only the first shape, so a program that is entirely the second one reads as
    healthy.
    """

    counts = [0] * len(_GAP_BUCKETS)
    totals = [0] * len(_GAP_BUCKETS)
    for gap in gaps:
        duration = gap.end - gap.start
        for index, (_, upper) in enumerate(_GAP_BUCKETS):
            if duration < upper:
                counts[index] += 1
                totals[index] += duration
                break
    grand_total = sum(totals)
    return [
        {
            "bucket": label,
            "count": counts[index],
            "seconds": totals[index] / 1e9,
            "share": (totals[index] / grand_total) if grand_total else 0.0,
        }
        for index, (label, _) in enumerate(_GAP_BUCKETS)
        if counts[index]
    ]


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    """Run every measurement and return one JSON-serializable result."""

    sqlite_path = ensure_sqlite(
        Path(args.report).expanduser(), args.nsys, Path(args.out).expanduser()
    )
    conn = open_readonly(sqlite_path)
    try:
        activity = read_activity(conn)
        nvtx = read_nvtx(conn)
        kernels = read_kernels(conn)
        copies = read_copies(conn)
        apis = read_runtime_api(conn)
        launches = read_kernel_launches(conn, args.kernel) if args.kernel else []
    finally:
        conn.close()

    all_spans = [span for spans in activity.values() for span in spans]
    union = merge_intervals(all_spans)

    clocks = {
        "cuda": Interval(union[0].start, union[-1].end) if union else None,
        "nvtx": span_of((r.start, r.end) for r in nvtx),
        "api": span_of((start, start + dur) for _, start, dur in apis),
    }
    clocks["trace"] = span_of(
        (iv.start, iv.end) for iv in (clocks["cuda"], clocks["nvtx"], clocks["api"]) if iv
    )

    # The window decides what "percent of wall" divides by, so choosing it
    # silently is how host work vanishes from a whole-program answer. Annotated
    # setup that runs before the first kernel is real wall time the program spent,
    # and a first-kernel-to-last-kernel window scores it as if it never happened.
    if args.range:
        named = [r for r in nvtx if r.name == args.range]
        if not named:
            raise SystemExit(
                f"error: no NVTX range named {args.range!r} in this report"
            )
        window = merge_intervals((r.start, r.end) for r in named)
        window_label = f"NVTX range {args.range!r} ({len(named)} instances)"
    else:
        choice = args.window
        if choice == "auto":
            # NVTX is the application stating its own scope, so prefer it. Falling
            # back to the CUDA span only when nothing is annotated.
            choice = "nvtx" if clocks["nvtx"] else "cuda"
        chosen = clocks[choice]
        if chosen is None:
            available = ", ".join(sorted(k for k, v in clocks.items() if v)) or "none"
            raise SystemExit(
                f"error: report has no {choice} span to use as a window "
                f"(available: {available})"
            )
        window = [chosen]
        window_label = f"{choice} span"

    union = clip(union, window)
    window_ns = total_ns(window)
    union_ns = total_ns(union)

    window_index = Window(window)
    window_cov = Coverage(window)
    # Islands only mean something next to the number of activities that formed
    # them: when the two are equal, nothing on the device overlapped anything
    # else, so the engines ran strictly one at a time.
    if len(window) == 1:
        bound = window[0]
        activity_count = sum(1 for s, e in all_spans if e > bound.start and s < bound.end)
    else:
        activity_count = sum(1 for s, e in all_spans if window_cov.covered(s, e) > 0)
    # Clip durations instead of filtering on start time, so a call that straddles
    # a boundary contributes only the part the window actually holds.
    copies = [
        row
        for kind, start, dur, nbytes in copies
        if (row := (kind, start, window_cov.covered(start, start + dur), nbytes))[2] > 0
    ]
    apis = [
        row
        for name, start, dur in apis
        if (row := (name, start, window_cov.covered(start, start + dur)))[2] > 0
    ]

    # Clip NVTX to the window by overlap rather than by start time. A stage range
    # that opened microseconds before the first kernel still owns work inside the
    # window, and a start-only test would silently drop exactly the outermost
    # ranges that answer "which phase is this".
    nvtx_in: list[NvtxRange] = []
    for r in nvtx:
        bounded = window_index.first_overlap(r.start, r.end)
        if bounded is not None:
            nvtx_in.append(
                NvtxRange(name=r.name, start=bounded[0], end=bounded[1], tid=r.tid)
            )

    gap_union = gaps_between(union, window)
    range_stats, owners = aggregate_ranges(
        nvtx_in, Coverage(union), Coverage(gap_union), args.exclude_prefix
    )

    # Offsets are reported from the window start. Raw capture timestamps are a
    # session-relative clock that can exceed the window length, which reads as a
    # contradiction next to "window wall".
    origin = window[0].start
    trace_span = clocks["trace"]
    trace_origin = trace_span.start if trace_span else origin

    min_gap_ns = int(args.min_gap_us * 1_000)
    ranked_gaps = sorted(gap_union, key=lambda iv: iv.end - iv.start, reverse=True)
    gaps = [
        {
            "seconds": (iv.end - iv.start) / 1e9,
            "at_s": (iv.start - origin) / 1e9,
            "abs_ns": iv.start,
            "owner": owners.at(iv.start + (iv.end - iv.start) // 2),
        }
        for iv in ranked_gaps[: max(args.top, 1)]
        if iv.end - iv.start >= min_gap_ns
    ]

    kernel_summary = summarize_kernels(kernels, window_cov, owners)
    kernel_rows = sorted(
        (
            {
                "name": name,
                "calls": calls,
                "total_s": total / 1e9,
                "mean_us": (total / calls) / 1e3 if calls else 0.0,
                "max_ms": longest / 1e6,
            }
            for name, (calls, total, longest) in kernel_summary.by_name.items()
        ),
        key=lambda row: row["total_s"],
        reverse=True,
    )

    per_api: dict[str, list[int]] = {}
    for name, _, dur in apis:
        entry = per_api.setdefault(name, [0, 0])
        entry[0] += 1
        entry[1] += dur

    # Classify once per distinct name, not once per call: a launch-bound capture
    # has hundreds of thousands of calls and a couple of dozen names.
    families: dict[str, list[int]] = {}
    for name, (calls, host) in per_api.items():
        fam = families.setdefault(classify_api(name), [0, 0])
        fam[0] += calls
        fam[1] += host

    # A window that averages a saturated setup phase together with an idle one
    # reports a healthy busy percentage for a program that is not healthy. Name the
    # worst offender so the reader clips to it instead of trusting the average.
    window_busy_pct = (union_ns / window_ns * 100) if window_ns else 0.0
    clip_hint: dict[str, Any] | None = None
    if not args.range:
        candidates = [
            r
            for r in range_stats
            if r.inclusive_ns >= 0.10 * window_ns
            and (r.gpu_busy_ns / r.inclusive_ns * 100) <= window_busy_pct - 15
        ]
        if candidates:
            worst = max(candidates, key=lambda r: r.idle_ns)
            clip_hint = {
                "name": worst.name,
                "inclusive_s": worst.inclusive_ns / 1e9,
                "busy_pct": worst.gpu_busy_ns / worst.inclusive_ns * 100,
                "idle_s": worst.idle_ns / 1e9,
            }

    result: dict[str, Any] = {
        "report": str(sqlite_path),
        "window_label": window_label,
        "window_ns": window_ns,
        "union_ns": union_ns,
        "union_pct": (union_ns / window_ns * 100) if window_ns else 0.0,
        "gap_ns": window_ns - union_ns,
        "islands": len(union),
        "activity_count": activity_count,
        "kernel_sum_ns": kernel_summary.device_ns,
        "copy_sum_ns": sum(dur for _, _, dur, _ in copies),
        "clocks": {
            name: {
                "wall_s": (span.end - span.start) / 1e9,
                "from_s": (span.start - trace_origin) / 1e9,
                "to_s": (span.end - trace_origin) / 1e9,
            }
            for name, span in clocks.items()
            if span is not None
        },
        "clip_hint": clip_hint,
        "gap_count": len(gap_union),
        "min_gap_us": args.min_gap_us,
        "gap_buckets": bucket_gaps(gap_union),
        "idle_api": attribute_gaps_to_api(gap_union, apis),
        "gaps": gaps,
        "ranges": [
            {
                "name": r.name,
                "instances": r.instances,
                "inclusive_s": r.inclusive_ns / 1e9,
                "exclusive_s": r.exclusive_ns / 1e9,
                "gpu_busy_s": r.gpu_busy_ns / 1e9,
                "idle_s": r.idle_ns / 1e9,
                "idle_exclusive_s": r.idle_exclusive_ns / 1e9,
                "busy_pct": (r.gpu_busy_ns / r.inclusive_ns * 100) if r.inclusive_ns else 0.0,
            }
            for r in range_stats
        ],
        "idle_owners": sorted(
            (
                {
                    "name": r.name,
                    "idle_exclusive_s": r.idle_exclusive_ns / 1e9,
                    "instances": r.instances,
                    "exclusive_s": r.exclusive_ns / 1e9,
                }
                for r in range_stats
                if r.idle_exclusive_ns > 0
            ),
            key=lambda row: row["idle_exclusive_s"],
            reverse=True,
        ),
        "kernels": kernel_rows,
        "kernel_owners": sorted(
            (
                {"name": name, "calls": calls, "device_s": device / 1e9}
                for name, (calls, device) in kernel_summary.by_owner.items()
            ),
            key=lambda row: row["device_s"],
            reverse=True,
        ),
        "kernel_unattributed": {
            "calls": kernel_summary.unattributed[0],
            "device_s": kernel_summary.unattributed[1] / 1e9,
        },
        "kernel_buckets": [
            {
                "bucket": label,
                "calls": kernel_summary.buckets[label][0],
                "total_s": kernel_summary.buckets[label][1] / 1e9,
            }
            for label, _ in _KERNEL_BUCKETS
        ],
        "copies": {
            direction: {
                "calls": entry["calls"],
                "bytes": entry["bytes"],
                "engine_s": entry["engine_ns"] / 1e9,
                "buckets": entry["buckets"],
            }
            for direction, entry in sorted(
                bucket_copies(copies).items(),
                key=lambda item: item[1]["calls"],
                reverse=True,
            )
        },
        "api_families": {
            family: {"calls": calls, "host_s": host / 1e9}
            for family, (calls, host) in sorted(
                families.items(), key=lambda item: item[1][1], reverse=True
            )
        },
        "apis": sorted(
            (
                {"name": name, "calls": calls, "host_s": host / 1e9}
                for name, (calls, host) in per_api.items()
            ),
            key=lambda row: row["host_s"],
            reverse=True,
        ),
    }
    if args.kernel:
        result["launch_filter"] = args.kernel
        result["launches"] = sorted(
            (
                {
                    "name": name,
                    "ordinal": ordinal,
                    "at_s": (start - origin) / 1e9,
                    "abs_ns": start,
                    "ms": dur / 1e6,
                    "gridX": grid,
                    "blockX": block,
                }
                for name, ordinal, start, dur, grid, block in launches
            ),
            key=lambda row: row["ms"],
            reverse=True,
        )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Analyze one report and write the digest and JSON."""

    args = parse_args(argv)
    try:
        result = analyze(args)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    digest = render(result, args.top)
    sys.stdout.write(digest)

    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "digest.txt").write_text(digest, encoding="utf-8")
    (out_dir / "analysis.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {out_dir / 'digest.txt'} and {out_dir / 'analysis.json'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
