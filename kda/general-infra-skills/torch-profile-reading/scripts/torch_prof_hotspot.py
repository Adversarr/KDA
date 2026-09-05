#!/usr/bin/env python3
"""
torch_prof_hotspot.py

A dependency-light hotspot analyzer for PyTorch profiler Chrome/Kineto JSON traces.

Input: JSON exported by torch.profiler.profile(...).export_chrome_trace(path)
       or trace files created by torch.profiler.tensorboard_trace_handler.

Outputs:
  - text: Nsight Systems-like summary for humans
  - json: machine-readable report for agents/CI
  - csv: normalized event table + summaries
  - png: matplotlib timeline, with explicit zoom windows

Examples:
  python torch_prof_hotspot.py trace.json --mode text
  python torch_prof_hotspot.py trace.json.gz --mode text --mode png --out run1 --zoom 0:50ms
  python torch_prof_hotspot.py trace.json --mode json --require-gpu-util 80 --max-idle-gap-ms 2
  python torch_prof_hotspot.py trace.json --inspect --top 30
  python torch_prof_hotspot.py trace.json --mode csv --query-regex 'nccl|cudaMemcpy|aten::matmul'

Notes:
  Chrome trace timestamps are conventionally microseconds. This tool reports ms unless noted.
  GPU SM occupancy/achieved occupancy cannot be inferred from a Chrome trace alone; use Nsight Compute
  or profiler counters for that. This tool computes timeline occupancy/utilization: percent of trace time
  with at least one GPU-side activity active on a device.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import json
import math
import mmap
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import orjson  # type: ignore
except Exception:  # pragma: no cover
    orjson = None

US_PER_MS = 1000.0
US_PER_S = 1_000_000.0

_FAST_SCAN_MIN_BYTES = 256 * 1024 * 1024
"""Minimum uncompressed trace size that enables compact summary scanning."""

GPU_CATS = {"kernel", "gpu_kernel", "gpu_memcpy", "memcpy", "memset", "gpu_memset", "nccl"}
RUNTIME_NAMES = (
    "cudalaunch", "cudamemcpy", "cudamemset", "cudafree", "cudamalloc",
    "cudastream", "cudevent", "cudadevicesynchronize", "cudastreamsynchronize",
    "culaunch", "cumemcpy", "cumemset", "cuoccupancy", "cuget", "cudagraph",
)
LAUNCH_NAMES = (
    "cudalaunchkernel", "cudalaunchcooperativekernel", "culaunchkernel",
    "cudagraphlaunch", "cudalaunchhostfunc",
)
SYNC_NAME_PAT = re.compile(r"cuda(Device|Stream|Event)?Synchronize|cudaMemcpy|cudaFree|cudaMalloc|cudaHostAlloc", re.I)
MEMORY_PAT = re.compile(r"memcpy|memset|DtoH|HtoD|DtoD|HtoH|copy", re.I)
# Anchored on collective vocabulary that does not appear in ordinary elementwise or
# copy kernels. Bare "send"/"recv"/"broadcast" are deliberately absent: they match
# broadcasting elementwise kernels and would inflate the communication share.
NCCL_PAT = re.compile(
    r"nccl|allreduce|all_reduce|allgather|all_gather|reduce_scatter|alltoall|all_to_all"
    r"|c10d|ProcessGroup",
    re.I,
)


@dataclass
class Event:
    idx: int
    name: str
    cat: str
    ph: str
    ts: float
    dur: float
    pid: Any
    tid: Any
    args: Dict[str, Any] = field(default_factory=dict)
    process: str = ""
    thread: str = ""
    kind: str = "OTHER"
    device: str = ""
    stream: str = ""
    corr_id: str = ""
    ext_id: str = ""
    parent_idx: int = -1
    child_time_us: float = 0.0
    self_us: float = 0.0

    @property
    def end(self) -> float:
        return self.ts + max(0.0, self.dur)

    @property
    def lane(self) -> str:
        if self.kind in {"GPU_KERNEL", "GPU_MEMCPY", "GPU_MEMSET", "NCCL"}:
            dev = self.device or "?"
            stream = self.stream or "?"
            return f"GPU{dev}/stream{stream}"
        label = self.thread or str(self.tid)
        proc = self.process or str(self.pid)
        return f"CPU {proc}:{label}"

    def overlaps(self, start: Optional[float], end: Optional[float]) -> bool:
        if start is not None and self.end < start:
            return False
        if end is not None and self.ts > end:
            return False
        return True


@dataclass
class MatchLatency:
    runtime_idx: int
    gpu_idx: int
    runtime_name: str
    gpu_name: str
    corr_id: str
    launch_end_to_gpu_start_us: float
    launch_start_to_gpu_start_us: float
    exact: bool


@dataclass
class TraceData:
    events: List[Event]
    metadata: Dict[Tuple[Any, Any], Dict[str, str]]
    trace_start_us: float
    trace_end_us: float
    raw_event_count: int


# --------------------------- loading and normalization ---------------------------

def _read_bytes(path: Path) -> bytes:
    if str(path).endswith(".gz"):
        with gzip.open(path, "rb") as f:
            return f.read()
    return path.read_bytes()


def _repair_event_name_fields(text: str) -> Tuple[str, int]:
    """Escape malformed characters in Kineto event-name fields."""
    field_pattern = re.compile(r'("name"\s*:\s*")(.*?)(",\s*"pid"\s*:\s*)')
    repaired_count = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal repaired_count
        value = match.group(2)
        escaped = []
        backslashes = 0
        for char in value:
            if char == "\\":
                escaped.append(char)
                backslashes += 1
                continue
            if char == '"' and backslashes % 2 == 0:
                escaped.append('\\"')
            elif ord(char) < 0x20:
                escaped.append(f"\\u{ord(char):04x}")
            else:
                escaped.append(char)
            backslashes = 0
        repaired = "".join(escaped)
        if repaired == value:
            return match.group(0)
        repaired_count += 1
        return f"{match.group(1)}{repaired}{match.group(3)}"

    return field_pattern.sub(replace, text), repaired_count


def _decode_trace_text(path: Path, data: bytes) -> str:
    """Decode trace bytes, replacing invalid UTF-8 when needed."""
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as error:
        print(
            f"warning: {path}: invalid UTF-8 at byte {error.start}; "
            "replacing malformed event-name bytes",
            file=sys.stderr,
        )
        return data.decode("utf-8", errors="replace")


def _loads_lenient(path: Path, text: str) -> Any:
    """Parse JSON text, repairing malformed event-name fields when needed."""
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        repaired_text, repaired_count = _repair_event_name_fields(text)
        if repaired_count:
            try:
                obj = json.loads(repaired_text)
            except json.JSONDecodeError:
                obj = json.loads(repaired_text, strict=False)
            print(
                f"warning: {path}: repaired {repaired_count} malformed event-name "
                "field(s)",
                file=sys.stderr,
            )
            return obj
        try:
            obj = json.loads(text, strict=False)
        except json.JSONDecodeError:
            raise error
        print(
            f"warning: {path}: unescaped control character at "
            f"line {error.lineno} column {error.colno}; accepting malformed event name",
            file=sys.stderr,
        )
        return obj


def load_json(path: Path) -> Any:
    data = _read_bytes(path)
    if orjson is not None:
        try:
            return orjson.loads(data)
        except orjson.JSONDecodeError:
            pass
    # Kineto can preserve non-UTF-8 bytes from native event names.
    return _loads_lenient(path, _decode_trace_text(path, data))


def deep_get_casefold(args: Dict[str, Any], candidates: Sequence[str]) -> str:
    if not args:
        return ""
    wanted = {c.casefold().replace("_", " ").replace("-", " ") for c in candidates}
    stack = [args]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            for k, v in cur.items():
                kn = str(k).casefold().replace("_", " ").replace("-", " ")
                if kn in wanted and v is not None:
                    return str(v)
                if isinstance(v, dict):
                    stack.append(v)
                elif isinstance(v, list):
                    for x in v:
                        if isinstance(x, dict):
                            stack.append(x)
    return ""


def stringify_maybe(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, (str, int, float, bool)):
        return str(x)
    try:
        return json.dumps(x, sort_keys=True, ensure_ascii=False)[:500]
    except Exception:
        return str(x)[:500]


def metadata_from_events(raw_events: Sequence[Dict[str, Any]]) -> Dict[Tuple[Any, Any], Dict[str, str]]:
    meta: Dict[Tuple[Any, Any], Dict[str, str]] = defaultdict(dict)
    proc_by_pid: Dict[Any, str] = {}
    for e in raw_events:
        if e.get("ph") != "M":
            continue
        name = e.get("name", "")
        pid = e.get("pid", "")
        tid = e.get("tid", "")
        args = e.get("args") or {}
        if name == "process_name":
            pname = stringify_maybe(args.get("name"))
            proc_by_pid[pid] = pname
        elif name == "thread_name":
            tname = stringify_maybe(args.get("name"))
            meta[(pid, tid)]["thread"] = tname
        elif name == "thread_sort_index":
            meta[(pid, tid)]["sort_index"] = stringify_maybe(args.get("sort_index"))
    for (pid, tid), m in list(meta.items()):
        if pid in proc_by_pid:
            m["process"] = proc_by_pid[pid]
    for pid, pname in proc_by_pid.items():
        meta[(pid, "*")]["process"] = pname
    return meta


def infer_device_stream(e: Dict[str, Any], process: str, thread: str) -> Tuple[str, str]:
    args = e.get("args") or {}
    device = deep_get_casefold(args, ["device", "device id", "device_id", "gpu", "gpu id", "gpu_id"])
    stream = deep_get_casefold(args, ["stream", "stream id", "stream_id"])

    text = " ".join([str(e.get("cat", "")), str(e.get("name", "")), process, thread])
    if not device:
        m = re.search(r"(?:GPU|Device)\s*#?\s*(\d+)|device\s*(\d+)", text, re.I)
        if m:
            device = next(g for g in m.groups() if g is not None)
    if not stream:
        m = re.search(r"stream\s*#?\s*(\d+)", text, re.I)
        if m:
            stream = m.group(1)
    return device, stream


def infer_ids(args: Dict[str, Any]) -> Tuple[str, str]:
    corr = deep_get_casefold(args, [
        "correlation", "correlation id", "correlation_id", "Correlation ID",
        "cuda correlation id", "cuda_correlation_id", "cbid",
    ])
    ext = deep_get_casefold(args, ["external id", "external_id", "External id", "External ID"])
    return corr, ext


def classify_event(name: str, cat: str, args: Dict[str, Any], process: str, thread: str, device: str, stream: str) -> str:
    low_name = name.casefold()
    low_cat = cat.casefold()
    text = " ".join([low_name, low_cat, process.casefold(), thread.casefold()])

    if NCCL_PAT.search(name) or "nccl" in low_cat:
        # NCCL can appear as CPU op or GPU kernel. If it has stream/device, treat as GPU-side NCCL.
        if device or stream or "kernel" in low_cat:
            return "NCCL"
        return "CPU_NCCL"
    if "memset" in text:
        return "GPU_MEMSET" if (device or stream or "gpu" in low_cat) else "CUDA_RUNTIME"
    if MEMORY_PAT.search(text):
        if device or stream or "gpu" in low_cat or "memcpy" in low_cat:
            return "GPU_MEMCPY"
        if low_name.startswith(("cuda", "cu")):
            return "CUDA_RUNTIME"
    if "kernel" in low_cat or low_cat in GPU_CATS or device or stream:
        if not low_name.startswith(("cuda", "cu")):
            return "GPU_KERNEL"
    if low_name.startswith(RUNTIME_NAMES) or "runtime" in low_cat or "driver" in low_cat:
        return "CUDA_RUNTIME"
    if "cpu_op" in low_cat or "operator" in low_cat or "user_annotation" in low_cat or "python" in low_cat:
        return "CPU_OP"
    if low_name.startswith("aten::") or low_name.startswith("torch::") or low_name.startswith("autograd::"):
        return "CPU_OP"
    return "OTHER"


def normalize_events(raw_events: Sequence[Dict[str, Any]]) -> TraceData:
    meta = metadata_from_events(raw_events)
    complete: List[Event] = []
    begin_stack: Dict[Tuple[Any, Any], List[Dict[str, Any]]] = defaultdict(list)
    idx = 0

    def proc_thread(pid: Any, tid: Any) -> Tuple[str, str]:
        m = meta.get((pid, tid), {})
        process = m.get("process") or meta.get((pid, "*"), {}).get("process") or ""
        thread = m.get("thread") or ""
        return process, thread

    for e in raw_events:
        ph = e.get("ph")
        if ph == "X":
            dur = float(e.get("dur") or 0.0)
            if dur < 0:
                continue
            pid, tid = e.get("pid", ""), e.get("tid", "")
            process, thread = proc_thread(pid, tid)
            args = e.get("args") or {}
            device, stream = infer_device_stream(e, process, thread)
            corr, ext = infer_ids(args)
            name = str(e.get("name", ""))
            cat = str(e.get("cat", ""))
            kind = classify_event(name, cat, args, process, thread, device, stream)
            complete.append(Event(
                idx=idx, name=name, cat=cat, ph="X", ts=float(e.get("ts") or 0.0), dur=dur,
                pid=pid, tid=tid, args=args, process=process, thread=thread,
                kind=kind, device=device, stream=stream, corr_id=corr, ext_id=ext,
            ))
            idx += 1
        elif ph == "B":
            key = (e.get("pid", ""), e.get("tid", ""))
            begin_stack[key].append(e)
        elif ph == "E":
            key = (e.get("pid", ""), e.get("tid", ""))
            if not begin_stack[key]:
                continue
            b = begin_stack[key].pop()
            ts0 = float(b.get("ts") or 0.0)
            ts1 = float(e.get("ts") or ts0)
            dur = max(0.0, ts1 - ts0)
            pid, tid = key
            process, thread = proc_thread(pid, tid)
            args = b.get("args") or {}
            device, stream = infer_device_stream(b, process, thread)
            corr, ext = infer_ids(args)
            name = str(b.get("name", ""))
            cat = str(b.get("cat", ""))
            kind = classify_event(name, cat, args, process, thread, device, stream)
            complete.append(Event(
                idx=idx, name=name, cat=cat, ph="B/E", ts=ts0, dur=dur,
                pid=pid, tid=tid, args=args, process=process, thread=thread,
                kind=kind, device=device, stream=stream, corr_id=corr, ext_id=ext,
            ))
            idx += 1

    if complete:
        t0 = min(e.ts for e in complete)
        t1 = max(e.end for e in complete)
    else:
        t0 = 0.0
        t1 = 0.0
    compute_self_times(complete)
    return TraceData(complete, meta, t0, t1, len(raw_events))


def read_trace(path: Path) -> TraceData:
    obj = load_json(path)
    if isinstance(obj, dict):
        raw_events = obj.get("traceEvents", [])
    elif isinstance(obj, list):
        raw_events = obj
    else:
        raise ValueError(f"Unsupported JSON root: {type(obj).__name__}")
    if not isinstance(raw_events, list):
        raise ValueError("Expected a Chrome trace with a traceEvents list or a root list of events.")
    raw_events = [x for x in raw_events if isinstance(x, dict)]
    return normalize_events(raw_events)


def _fast_group_rows(
    groups: Dict[Tuple[str, bytes], List[float]], kind: str, top: int
) -> List[Dict[str, Any]]:
    rows = []
    for (event_kind, name), (calls, total_us, max_us) in groups.items():
        if event_kind != kind:
            continue
        rows.append({
            "name": name.decode("utf-8", errors="replace"),
            "calls": int(calls),
            "total_ms": total_us / US_PER_MS,
            "self_ms": total_us / US_PER_MS,
            "avg_ms": total_us / calls / US_PER_MS,
            "max_ms": max_us / US_PER_MS,
        })
    rows.sort(key=lambda row: row["total_ms"], reverse=True)
    return rows[:top]


def _gap_row(
    start_us: float,
    end_us: float,
    window_start_us: float,
    before: str,
    after: str,
) -> Dict[str, Any]:
    """Build one idle-gap summary row.

    Carries both time domains on purpose: ``*_ms`` is window-relative and is what
    a reader wants to see, while ``*_us`` stays in the trace's absolute clock so
    gap context, bookmarks, and ``--zoom`` can be matched against ``Event.ts``.

    Args:
        start_us: Gap start on the trace clock.
        end_us: Gap end on the trace clock.
        window_start_us: Analysis window start, subtracted for the display values.
        before: Name of the GPU event that ended the previous busy span.
        after: Name of the GPU event that ends the gap.

    Returns:
        Gap row with absolute ``start_us`` / ``end_us`` and relative ``start_ms`` /
        ``end_ms`` / ``dur_ms``.
    """
    return {
        "start_us": start_us,
        "end_us": end_us,
        "start_ms": (start_us - window_start_us) / US_PER_MS,
        "end_ms": (end_us - window_start_us) / US_PER_MS,
        "dur_ms": (end_us - start_us) / US_PER_MS,
        "before": before,
        "after": after,
    }


def _fast_interval_analysis(
    intervals: List[Tuple[float, float, bytes, str]],
    window_start: float,
    window_end: float,
    gap_threshold_us: float,
    top: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    ordered = sorted(intervals, key=lambda item: item[0])
    clipped = [
        (max(start, window_start), min(end, window_end), name, stream)
        for start, end, name, stream in ordered
        if end > window_start and start < window_end
    ]
    clipped = [item for item in clipped if item[1] > item[0]]
    window_us = window_end - window_start
    if not clipped:
        return {
            "events": 0, "busy_ms": 0.0, "window_ms": 0.0,
            "util_percent": 0.0, "sum_event_ms": 0.0, "streams": [],
        }, []

    busy_us = interval_total((start, end) for start, end, _, _ in clipped)

    # Gap labeling needs the merge walk (before/after kernel names).
    gaps = []
    current_start, current_end, _, current_name = (
        clipped[0][0], clipped[0][1], clipped[0][3], clipped[0][2]
    )
    if current_start - window_start >= gap_threshold_us:
        gaps.append(_gap_row(
            window_start,
            current_start,
            window_start,
            "TRACE_START",
            current_name.decode("utf-8", errors="replace"),
        ))
    for start, end, name, _ in clipped[1:]:
        if start > current_end:
            if start - current_end >= gap_threshold_us:
                gaps.append(_gap_row(
                    current_end,
                    start,
                    window_start,
                    current_name.decode("utf-8", errors="replace"),
                    name.decode("utf-8", errors="replace"),
                ))
            current_start, current_end, current_name = start, end, name
        elif end > current_end:
            current_end, current_name = end, name
    if window_end - current_end >= gap_threshold_us:
        gaps.append(_gap_row(
            current_end,
            window_end,
            window_start,
            current_name.decode("utf-8", errors="replace"),
            "TRACE_END",
        ))
    gaps.sort(key=lambda row: row["dur_ms"], reverse=True)

    stream_rows = []
    streams: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for start, end, _, stream in clipped:
        streams[stream].append((start, end))
    for stream, stream_intervals in streams.items():
        stream_busy_us = interval_total(stream_intervals)
        stream_rows.append({
            "stream": stream,
            "events": len(stream_intervals),
            "busy_ms": stream_busy_us / US_PER_MS,
            "util_percent": 100.0 * stream_busy_us / window_us if window_us else 0.0,
            "sum_event_ms": sum(end - start for start, end in stream_intervals) / US_PER_MS,
        })
    stream_rows.sort(key=lambda row: row["busy_ms"], reverse=True)
    return {
        "events": len(clipped),
        "busy_ms": busy_us / US_PER_MS,
        "window_ms": window_us / US_PER_MS,
        "util_percent": 100.0 * busy_us / window_us if window_us else 0.0,
        "sum_event_ms": sum(end - start for start, end, _, _ in clipped) / US_PER_MS,
        "streams": stream_rows,
    }, gaps[:top]


def build_fast_report(path: Path, top: int, gap_threshold_us: float) -> Dict[str, Any]:
    """Build a compact GPU/runtime summary without materializing the JSON tree.

    Args:
        path: Uncompressed Kineto JSON trace.
        top: Maximum rows in each summary.
        gap_threshold_us: Minimum GPU idle gap to report.

    Returns:
        Summary report compatible with :func:`render_text`.
    """
    groups: Dict[Tuple[str, bytes], List[float]] = {}
    sync_groups: Dict[Tuple[str, bytes], List[float]] = {}
    intervals_by_device: Dict[str, List[Tuple[float, float, bytes, str]]] = defaultdict(list)
    counts: Counter[str] = Counter()
    needles = (
        (b'"cat": "kernel"', "GPU_KERNEL"),
        (b'"cat": "gpu_', "GPU_MEMORY"),
        (b'"cat": "cuda_runtime"', "CUDA_RUNTIME"),
    )
    name_key = b'"name": "'
    name_end_key = b'", "pid": '
    timestamp_key = b'"ts": '
    duration_key = b'"dur": '

    with path.open("rb") as file:
        with mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as data:
            find = data.find
            rfind = data.rfind
            trace_pos = find(b'"cat": "Trace"')
            if trace_pos < 0:
                raise ValueError(
                    f"{path}: fast path requires a top-level "
                    '"cat": "Trace" event; rerun with --full'
                )
            trace_line_end = find(b"\n", trace_pos)
            if trace_line_end < 0:
                raise ValueError(f"{path}: truncated Trace event line")
            trace_ts_start = find(timestamp_key, trace_pos, trace_line_end)
            if trace_ts_start < 0:
                raise ValueError(f"{path}: Trace event missing ts field")
            trace_ts_start += len(timestamp_key)
            trace_ts_end = find(b",", trace_ts_start, trace_line_end)
            if trace_ts_end < 0:
                raise ValueError(f"{path}: Trace event has malformed ts field")
            trace_dur_start = find(duration_key, trace_ts_end, trace_line_end)
            if trace_dur_start < 0:
                raise ValueError(f"{path}: Trace event missing dur field")
            trace_dur_start += len(duration_key)
            trace_dur_end = find(b",", trace_dur_start, trace_line_end)
            if trace_dur_end < 0:
                trace_dur_end = trace_line_end
            window_start = float(data[trace_ts_start:trace_ts_end])
            window_end = window_start + float(data[trace_dur_start:trace_dur_end])

            for needle, default_kind in needles:
                pos = 0
                while True:
                    pos = find(needle, pos)
                    if pos < 0:
                        break
                    line_end = find(b"\n", pos)
                    if line_end < 0:
                        break
                    next_line_end = find(b"\n", line_end + 1)
                    if next_line_end < 0:
                        next_line_end = len(data)
                    name_start = find(name_key, pos, line_end)
                    if name_start < 0:
                        pos = line_end
                        continue
                    name_start += len(name_key)
                    name_end = rfind(name_end_key, name_start, line_end)
                    if name_end < name_start:
                        pos = line_end
                        continue
                    duration_start = find(duration_key, line_end, next_line_end)
                    if duration_start < 0:
                        pos = line_end
                        continue
                    duration_start += len(duration_key)
                    duration_end = find(b",", duration_start, next_line_end)
                    if duration_end < 0:
                        duration_end = next_line_end
                    name = data[name_start:name_end]
                    duration = float(data[duration_start:duration_end])

                    if default_kind == "GPU_MEMORY":
                        category_start = pos + len(b'"cat": "')
                        category_end = find(b'"', category_start, line_end)
                        category = data[category_start:category_end]
                        kind = "GPU_MEMSET" if category == b"gpu_memset" else "GPU_MEMCPY"
                    elif default_kind == "GPU_KERNEL" and b"nccl" in name.lower():
                        kind = "NCCL"
                    else:
                        kind = default_kind

                    key = (kind, name)
                    row = groups.get(key)
                    if row is None:
                        groups[key] = [1.0, duration, duration]
                    else:
                        row[0] += 1.0
                        row[1] += duration
                        row[2] = max(row[2], duration)
                    counts[kind] += 1

                    if kind == "CUDA_RUNTIME":
                        if SYNC_NAME_PAT.search(name.decode("utf-8", errors="replace")):
                            sync_row = sync_groups.get(key)
                            if sync_row is None:
                                sync_groups[key] = [1.0, duration, duration]
                            else:
                                sync_row[0] += 1.0
                                sync_row[1] += duration
                                sync_row[2] = max(sync_row[2], duration)
                    else:
                        timestamp_start = find(timestamp_key, line_end, next_line_end)
                        if timestamp_start < 0:
                            pos = line_end
                            continue
                        timestamp_start += len(timestamp_key)
                        timestamp_end = find(b",", timestamp_start, next_line_end)
                        if timestamp_end < 0:
                            pos = line_end
                            continue
                        timestamp = float(data[timestamp_start:timestamp_end])
                        pid_start = name_end + len(name_end_key)
                        pid_end = find(b",", pid_start, line_end)
                        tid_start = find(b'"tid": ', pid_end, line_end)
                        if pid_end < 0 or tid_start < 0:
                            pos = line_end
                            continue
                        tid_start += len(b'"tid": ')
                        tid_end = find(b",", tid_start, line_end)
                        if tid_end < 0:
                            tid_end = line_end
                        device = data[pid_start:pid_end].strip().decode()
                        stream = data[tid_start:tid_end].strip().decode()
                        intervals_by_device[device].append(
                            (timestamp, timestamp + duration, name, stream)
                        )
                    pos = line_end

    gpu_utilization = {}
    gpu_idle_gaps = {}
    for device, intervals in intervals_by_device.items():
        utilization, gaps = _fast_interval_analysis(
            intervals, window_start, window_end, gap_threshold_us, top
        )
        gpu_utilization[device] = utilization
        gpu_idle_gaps[device] = gaps

    kernel_rows = _fast_group_rows(groups, "GPU_KERNEL", top)
    nccl_rows = _fast_group_rows(groups, "NCCL", top)
    kernel_rows = sorted(
        kernel_rows + nccl_rows, key=lambda row: row["total_ms"], reverse=True
    )[:top]
    memory_rows = sorted(
        _fast_group_rows(groups, "GPU_MEMCPY", top)
        + _fast_group_rows(groups, "GPU_MEMSET", top),
        key=lambda row: row["total_ms"],
        reverse=True,
    )[:top]
    event_count = sum(counts.values())
    report = {
        "trace": {
            "analysis_mode": "fast_summary",
            "raw_event_count": event_count,
            "complete_event_count": event_count,
            "selected_event_count": event_count,
            "trace_start_us": window_start,
            "trace_end_us": window_end,
            "trace_duration_ms": (window_end - window_start) / US_PER_MS,
            "window_start_us": window_start,
            "window_end_us": window_end,
            "window_duration_ms": (window_end - window_start) / US_PER_MS,
        },
        "counts_by_kind": dict(counts),
        "gpu_utilization": gpu_utilization,
        "gpu_idle_gaps": gpu_idle_gaps,
        "top_cpu_ops": [],
        "top_cuda_runtime": _fast_group_rows(groups, "CUDA_RUNTIME", top),
        "top_gpu_kernels": kernel_rows,
        "top_gpu_memory": memory_rows,
        "sync_hotspots": _fast_group_rows(sync_groups, "CUDA_RUNTIME", top),
        "launch_latency_summary": {"count": 0},
        "top_launch_latencies": [],
        "debug": {
            "devices": sorted(intervals_by_device),
            "fast_summary_limitations": [
                "CPU operators, percentiles, launch latency, counters, and flows require --full.",
                "Raw/complete event counts include only scanned GPU and CUDA runtime events.",
            ],
        },
        "recommendations": [],
    }
    report["recommendations"] = make_recommendations(report)
    return report


# --------------------------- interval math ---------------------------

def compute_self_times(events: List[Event]) -> None:
    # Per thread nesting. GPU streams often overlap; self-time is most useful for CPU threads.
    by_thread: Dict[Tuple[Any, Any], List[int]] = defaultdict(list)
    for i, e in enumerate(events):
        if e.dur > 0:
            by_thread[(e.pid, e.tid)].append(i)
    for ids in by_thread.values():
        ids.sort(key=lambda i: (events[i].ts, -events[i].end, -events[i].dur))
        stack: List[int] = []
        for i in ids:
            ev = events[i]
            while stack and events[stack[-1]].end <= ev.ts + 1e-9:
                stack.pop()
            if stack and ev.end <= events[stack[-1]].end + 1e-9:
                parent = stack[-1]
                ev.parent_idx = parent
                events[parent].child_time_us += ev.dur
            stack.append(i)
    for e in events:
        e.self_us = max(0.0, e.dur - e.child_time_us)


def merge_intervals(intervals: Iterable[Tuple[float, float]]) -> List[Tuple[float, float]]:
    clean = sorted((float(a), float(b)) for a, b in intervals if b > a)
    if not clean:
        return []
    out = [clean[0]]
    for a, b in clean[1:]:
        la, lb = out[-1]
        if a <= lb:
            if b > lb:
                out[-1] = (la, b)
        else:
            out.append((a, b))
    return out


def interval_total(intervals: Iterable[Tuple[float, float]]) -> float:
    return sum(b - a for a, b in merge_intervals(intervals))


def clip_interval(a: float, b: float, start: Optional[float], end: Optional[float]) -> Optional[Tuple[float, float]]:
    if start is not None:
        a = max(a, start)
    if end is not None:
        b = min(b, end)
    if b <= a:
        return None
    return a, b


def percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    k = (len(xs) - 1) * p / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return xs[int(k)]
    return xs[f] * (c - k) + xs[c] * (k - f)


# --------------------------- stats ---------------------------

def group_stats(events: Sequence[Event], metric: str = "dur", by: Sequence[str] = ("name",), top: int = 30) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], List[Event]] = defaultdict(list)
    for e in events:
        key = tuple(getattr(e, attr) for attr in by)
        groups[key].append(e)
    rows: List[Dict[str, Any]] = []
    for key, xs in groups.items():
        durs = [max(0.0, e.dur) for e in xs]
        selfs = [max(0.0, e.self_us) for e in xs]
        row: Dict[str, Any] = {attr: key[i] for i, attr in enumerate(by)}
        row.update({
            "calls": len(xs),
            "total_ms": sum(durs) / US_PER_MS,
            "self_ms": sum(selfs) / US_PER_MS,
            "avg_ms": (sum(durs) / len(durs) / US_PER_MS) if durs else 0.0,
            "p50_ms": percentile(durs, 50) / US_PER_MS,
            "p95_ms": percentile(durs, 95) / US_PER_MS,
            "max_ms": max(durs) / US_PER_MS if durs else 0.0,
        })
        rows.append(row)
    rows.sort(key=lambda r: (r["total_ms"], r.get("self_ms", 0.0)), reverse=True)
    return rows[:top]


def filtered_events(data: TraceData, start: Optional[float], end: Optional[float], regex: Optional[re.Pattern[str]]) -> List[Event]:
    out = []
    for e in data.events:
        if start is not None or end is not None:
            if not e.overlaps(start, end):
                continue
        if regex and not regex.search(" ".join([e.name, e.cat, e.kind, e.process, e.thread])):
            continue
        out.append(e)
    return out


def gpu_events(events: Sequence[Event]) -> List[Event]:
    return [e for e in events if e.kind in {"GPU_KERNEL", "GPU_MEMCPY", "GPU_MEMSET", "NCCL"} and e.dur > 0]


def cpu_events(events: Sequence[Event]) -> List[Event]:
    return [e for e in events if e.kind in {"CPU_OP", "CPU_NCCL", "CUDA_RUNTIME", "OTHER"} and e.dur > 0 and e.kind not in {"GPU_KERNEL", "GPU_MEMCPY", "GPU_MEMSET", "NCCL"}]


def runtime_events(events: Sequence[Event]) -> List[Event]:
    return [e for e in events if e.kind == "CUDA_RUNTIME" and e.dur > 0]


def analyze_gpu_utilization(events: Sequence[Event], window_start: float, window_end: float) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    ge = gpu_events(events)
    devices = sorted({e.device or "?" for e in ge}, key=lambda x: (x == "?", x))
    window_us = max(0.0, window_end - window_start)
    for dev in devices:
        dev_events = [e for e in ge if (e.device or "?") == dev]
        intervals = []
        for e in dev_events:
            clipped = clip_interval(e.ts, e.end, window_start, window_end)
            if clipped:
                intervals.append(clipped)
        busy_us = interval_total(intervals)
        stream_rows = []
        for stream in sorted({e.stream or "?" for e in dev_events}, key=str):
            se = [e for e in dev_events if (e.stream or "?") == stream]
            si = []
            for e in se:
                clipped = clip_interval(e.ts, e.end, window_start, window_end)
                if clipped:
                    si.append(clipped)
            s_busy = interval_total(si)
            stream_rows.append({
                "stream": stream,
                "events": len(se),
                "busy_ms": s_busy / US_PER_MS,
                "util_percent": (100.0 * s_busy / window_us) if window_us else 0.0,
                "sum_event_ms": sum(e.dur for e in se) / US_PER_MS,
            })
        stream_rows.sort(key=lambda x: x["busy_ms"], reverse=True)
        out[dev] = {
            "events": len(dev_events),
            "busy_ms": busy_us / US_PER_MS,
            "window_ms": window_us / US_PER_MS,
            "util_percent": (100.0 * busy_us / window_us) if window_us else 0.0,
            "sum_event_ms": sum(e.dur for e in dev_events) / US_PER_MS,
            "streams": stream_rows,
        }
    return out


def analyze_gaps(events: Sequence[Event], window_start: float, window_end: float, threshold_us: float, top: int) -> Dict[str, List[Dict[str, Any]]]:
    ge = gpu_events(events)
    result: Dict[str, List[Dict[str, Any]]] = {}
    for dev in sorted({e.device or "?" for e in ge}, key=lambda x: (x == "?", x)):
        dev_events = [e for e in ge if (e.device or "?") == dev]
        clipped_events = []
        for e in dev_events:
            c = clip_interval(e.ts, e.end, window_start, window_end)
            if c:
                clipped_events.append((c[0], c[1], e))
        merged = merge_intervals((a, b) for a, b, _ in clipped_events)
        gaps = []
        prev_end = window_start
        for a, b in merged:
            if a - prev_end >= threshold_us:
                before = max((x for x in clipped_events if x[1] <= a + 1e-9), key=lambda x: x[1], default=None)
                after = min((x for x in clipped_events if x[0] >= a - 1e-9), key=lambda x: x[0], default=None)
                gaps.append(_gap_row(
                    prev_end,
                    a,
                    window_start,
                    before[2].name if before else "TRACE_START",
                    after[2].name if after else "TRACE_END",
                ))
            prev_end = max(prev_end, b)
        if window_end - prev_end >= threshold_us:
            before = max((x for x in clipped_events if x[1] <= window_end + 1e-9), key=lambda x: x[1], default=None)
            gaps.append(_gap_row(
                prev_end,
                window_end,
                window_start,
                before[2].name if before else "TRACE_START",
                "TRACE_END",
            ))
        gaps.sort(key=lambda r: r["dur_ms"], reverse=True)
        result[dev] = gaps[:top]
    return result


def is_launch_runtime(e: Event) -> bool:
    n = e.name.casefold().replace("_", "")
    return any(x in n for x in LAUNCH_NAMES)


def analyze_launch_latency(events: Sequence[Event], max_heuristic_us: float = 100_000.0) -> List[MatchLatency]:
    """Pair launch-side runtime calls with the GPU work they produced.

    Args:
        events: Normalized events for the selected window.
        max_heuristic_us: Ceiling on a heuristic (non-correlated) match; a larger
            gap is treated as an unrelated kernel rather than a slow launch.

    Returns:
        Every match found, slowest first. Callers truncate for display; the
        percentile summary needs the full population to be meaningful.
    """
    runtimes = [e for e in runtime_events(events) if is_launch_runtime(e)]
    kernels = [e for e in gpu_events(events) if e.kind in {"GPU_KERNEL", "NCCL"}]
    kernel_by_corr: Dict[str, List[Event]] = defaultdict(list)
    for k in kernels:
        if k.corr_id:
            kernel_by_corr[k.corr_id].append(k)
    out: List[MatchLatency] = []
    matched_runtime = set()
    matched_kernel = set()
    for r in runtimes:
        if r.corr_id and r.corr_id in kernel_by_corr:
            cand = sorted(kernel_by_corr[r.corr_id], key=lambda x: abs(x.ts - r.end))
            k = cand[0]
            out.append(MatchLatency(
                r.idx, k.idx, r.name, k.name, r.corr_id,
                k.ts - r.end, k.ts - r.ts, True,
            ))
            matched_runtime.add(r.idx)
            matched_kernel.add(k.idx)

    # Fallback heuristic: nearest following kernel by timestamp. This is useful for traces with no correlation IDs,
    # but can be wrong when many streams/threads enqueue concurrently.
    remaining_kernels = sorted([k for k in kernels if k.idx not in matched_kernel], key=lambda e: e.ts)
    starts = [k.ts for k in remaining_kernels]
    # Walking launches in submission order with a monotone cursor means each kernel
    # is claimed at most once. Without it, N consecutive launches all match the same
    # next kernel and N-1 of the reported latencies are fabricated.
    cursor = 0
    for r in sorted((r for r in runtimes if r.idx not in matched_runtime), key=lambda e: e.end):
        pos = max(cursor, bisect.bisect_left(starts, r.end))
        if pos < len(remaining_kernels):
            k = remaining_kernels[pos]
            latency = k.ts - r.end
            if 0 <= latency <= max_heuristic_us:
                cursor = pos + 1
                out.append(MatchLatency(
                    r.idx, k.idx, r.name, k.name, r.corr_id or "", latency, k.ts - r.ts, False,
                ))
    out.sort(key=lambda x: x.launch_end_to_gpu_start_us, reverse=True)
    return out


def summarize_latencies(matches: Sequence[MatchLatency]) -> Dict[str, Any]:
    vals = [m.launch_end_to_gpu_start_us for m in matches]
    if not vals:
        return {"count": 0}
    return {
        "count": len(vals),
        "exact_count": sum(1 for m in matches if m.exact),
        "heuristic_count": sum(1 for m in matches if not m.exact),
        "avg_ms": statistics.mean(vals) / US_PER_MS,
        "p50_ms": percentile(vals, 50) / US_PER_MS,
        "p95_ms": percentile(vals, 95) / US_PER_MS,
        "max_ms": max(vals) / US_PER_MS,
    }


def find_sync_hotspots(events: Sequence[Event], top: int) -> List[Dict[str, Any]]:
    xs = [e for e in runtime_events(events) if SYNC_NAME_PAT.search(e.name)]
    return group_stats(xs, by=("name",), top=top)


def build_report(data: TraceData, start: Optional[float], end: Optional[float], regex: Optional[re.Pattern[str]], top: int, gap_threshold_us: float) -> Dict[str, Any]:
    events = filtered_events(data, start, end, regex)
    win_start = start if start is not None else data.trace_start_us
    win_end = end if end is not None else data.trace_end_us

    ge = gpu_events(events)
    ce = cpu_events(events)
    revents = runtime_events(events)
    kernels = [e for e in ge if e.kind in {"GPU_KERNEL", "NCCL"}]
    mem = [e for e in ge if e.kind in {"GPU_MEMCPY", "GPU_MEMSET"}]
    lat = analyze_launch_latency(events)

    report = {
        "trace": {
            "raw_event_count": data.raw_event_count,
            "complete_event_count": len(data.events),
            "selected_event_count": len(events),
            "trace_start_us": data.trace_start_us,
            "trace_end_us": data.trace_end_us,
            "trace_duration_ms": (data.trace_end_us - data.trace_start_us) / US_PER_MS,
            "window_start_us": win_start,
            "window_end_us": win_end,
            "window_duration_ms": (win_end - win_start) / US_PER_MS,
        },
        "counts_by_kind": dict(Counter(e.kind for e in events)),
        "gpu_utilization": analyze_gpu_utilization(events, win_start, win_end),
        "gpu_idle_gaps": analyze_gaps(events, win_start, win_end, gap_threshold_us, top),
        "top_cpu_ops": group_stats([e for e in ce if e.kind in {"CPU_OP", "CPU_NCCL"}], by=("name",), top=top),
        "top_cuda_runtime": group_stats(revents, by=("name",), top=top),
        "top_gpu_kernels": group_stats(kernels, by=("name",), top=top),
        "top_gpu_memory": group_stats(mem, by=("name",), top=top),
        "sync_hotspots": find_sync_hotspots(events, top),
        "launch_latency_summary": summarize_latencies(lat),
        "top_launch_latencies": [m.__dict__ for m in lat[:top]],
        "debug": {
            "processes": sorted({e.process or str(e.pid) for e in events}),
            "threads": sorted({e.thread or str(e.tid) for e in events})[:200],
            "devices": sorted({e.device or "?" for e in ge}, key=str),
            "streams": sorted({f"GPU{e.device or '?'}:stream{e.stream or '?'}" for e in ge}, key=str)[:500],
        },
        "recommendations": [],
    }
    report["recommendations"] = make_recommendations(report)
    return report


def make_recommendations(report: Dict[str, Any]) -> List[str]:
    recs = []
    utils = report.get("gpu_utilization", {})
    if not utils:
        recs.append("No GPU-side kernel/memcpy events found. Ensure torch.profiler uses ProfilerActivity.CUDA and CUPTI/Kineto GPU tracing is available.")
        return recs
    for dev, u in utils.items():
        util = u.get("util_percent", 0.0)
        if util < 50:
            recs.append(f"GPU{dev} timeline utilization is low ({util:.1f}%). Look for CPU dataloader stalls, blocking cudaMemcpy/cudaFree, too-small kernels, or missing overlap.")
        elif util < 80:
            recs.append(f"GPU{dev} has moderate timeline utilization ({util:.1f}%). Inspect idle gaps and launch latency; batching or CUDA Graphs may help if kernels are tiny.")
    gaps = report.get("gpu_idle_gaps", {})
    for dev, gs in gaps.items():
        if gs and gs[0]["dur_ms"] > 1.0:
            recs.append(f"GPU{dev} has a largest idle gap of {gs[0]['dur_ms']:.3f} ms before {gs[0]['after']}. Correlate with CPU threads around that timestamp.")
    lat = report.get("launch_latency_summary", {})
    if lat.get("count", 0) and lat.get("p95_ms", 0.0) > 0.2:
        recs.append(f"Launch p95 latency is {lat['p95_ms']:.3f} ms. Check Python overhead, synchronization, allocator activity, or consider torch.compile/CUDA Graphs for stable shapes.")
    runtime = report.get("top_cuda_runtime", [])
    if any("synchronize" in r.get("name", "").casefold() for r in runtime[:10]):
        recs.append("Synchronization APIs appear in top CUDA runtime calls. Remove unnecessary .item(), tensor.cpu(), print/logging of CUDA tensors, or explicit synchronizations in the hot path.")
    mem = report.get("top_gpu_memory", [])
    if mem and mem[0].get("total_ms", 0.0) > 1.0:
        recs.append("GPU memcpy/memset time is material. Use pinned memory, non_blocking=True, prefetching, larger transfers, and overlap H2D copies with compute where possible.")
    kernels = report.get("top_gpu_kernels", [])
    if kernels and kernels[0].get("avg_ms", 0.0) < 0.05 and sum(k.get("calls", 0) for k in kernels[:10]) > 1000:
        recs.append("Many tiny kernels detected. Fuse operations with torch.compile, use vectorized ops, increase batch size, or reduce per-step Python dispatch.")
    if not recs:
        recs.append("No obvious trace-level bottleneck detected. For SM occupancy, memory bandwidth, warp stalls, or tensor-core utilization, run Nsight Compute on the top kernels.")
    return recs


# --------------------------- output ---------------------------

def fmt_ms(x: float) -> str:
    return f"{x:,.3f}"


def print_table(title: str, rows: Sequence[Dict[str, Any]], cols: Sequence[Tuple[str, str]], limit: Optional[int] = None) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    rows = rows[:limit] if limit is not None else rows
    if not rows:
        print("  <none>")
        return
    widths = []
    for key, label in cols:
        vals = [label] + [fmt_cell(r.get(key, "")) for r in rows]
        widths.append(min(max(len(v) for v in vals), 80))
    header = "  ".join(label.ljust(widths[i]) for i, (_, label) in enumerate(cols))
    print(header)
    print("  ".join("-" * widths[i] for i in range(len(cols))))
    for r in rows:
        parts = []
        for i, (key, _) in enumerate(cols):
            v = fmt_cell(r.get(key, ""))
            if len(v) > widths[i]:
                v = v[: max(0, widths[i] - 1)] + "…"
            align = ">" if isinstance(r.get(key), (int, float)) else "<"
            parts.append(f"{v:{align}{widths[i]}}")
        print("  ".join(parts))


def fmt_cell(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:,.3f}"
    return str(v)


def render_text(report: Dict[str, Any]) -> None:
    tr = report["trace"]
    print("PyTorch Profiler Hotspot Report")
    print("==============================")
    if tr.get("analysis_mode") == "fast_summary":
        print("Analysis mode:    fast_summary (GPU and CUDA runtime events only; rerun with --full for the rest)")
    print(f"Raw events:       {tr['raw_event_count']:,}")
    print(f"Complete events:  {tr['complete_event_count']:,}")
    print(f"Selected events:  {tr['selected_event_count']:,}")
    print(f"Trace duration:   {fmt_ms(tr['trace_duration_ms'])} ms")
    print(f"Window duration:  {fmt_ms(tr['window_duration_ms'])} ms")
    print(f"Kinds:            {json.dumps(report['counts_by_kind'], sort_keys=True)}")

    print("\nGPU Utilization (timeline busy, not SM occupancy)")
    print("------------------------------------------------")
    if not report["gpu_utilization"]:
        print("  <no GPU-side activities found>")
    for dev, u in report["gpu_utilization"].items():
        print(f"GPU{dev}: util={u['util_percent']:.2f}% busy={u['busy_ms']:.3f}ms / window={u['window_ms']:.3f}ms events={u['events']:,} sum_event_ms={u['sum_event_ms']:.3f}")
        for s in u["streams"][:8]:
            print(f"  stream {s['stream']}: util={s['util_percent']:.2f}% busy={s['busy_ms']:.3f}ms events={s['events']:,} sum_event_ms={s['sum_event_ms']:.3f}")

    print_table("Top CPU operators / annotations", report["top_cpu_ops"], [
        ("name", "name"), ("calls", "calls"), ("total_ms", "total_ms"), ("self_ms", "self_ms"),
        ("avg_ms", "avg_ms"), ("p95_ms", "p95_ms"), ("max_ms", "max_ms"),
    ])
    print_table("Top CUDA runtime APIs", report["top_cuda_runtime"], [
        ("name", "name"), ("calls", "calls"), ("total_ms", "total_ms"),
        ("avg_ms", "avg_ms"), ("p95_ms", "p95_ms"), ("max_ms", "max_ms"),
    ])
    print_table("Top GPU kernels / NCCL", report["top_gpu_kernels"], [
        ("name", "name"), ("calls", "calls"), ("total_ms", "total_ms"),
        ("avg_ms", "avg_ms"), ("p50_ms", "p50_ms"), ("p95_ms", "p95_ms"), ("max_ms", "max_ms"),
    ])
    print_table("Top GPU memory ops", report["top_gpu_memory"], [
        ("name", "name"), ("calls", "calls"), ("total_ms", "total_ms"),
        ("avg_ms", "avg_ms"), ("p95_ms", "p95_ms"), ("max_ms", "max_ms"),
    ])
    print_table("Synchronization hotspots", report["sync_hotspots"], [
        ("name", "name"), ("calls", "calls"), ("total_ms", "total_ms"),
        ("avg_ms", "avg_ms"), ("p95_ms", "p95_ms"), ("max_ms", "max_ms"),
    ])

    lat = report["launch_latency_summary"]
    print("\nCUDA launch latency")
    print("-------------------")
    if not lat.get("count"):
        print("  <no launch-to-kernel matches found>")
    else:
        print(
            f"matches={lat['count']} exact={lat['exact_count']} heuristic={lat['heuristic_count']} "
            f"avg={lat['avg_ms']:.3f}ms p50={lat['p50_ms']:.3f}ms p95={lat['p95_ms']:.3f}ms max={lat['max_ms']:.3f}ms"
        )
        rows = []
        for m in report["top_launch_latencies"][:10]:
            rows.append({
                "latency_ms": m["launch_end_to_gpu_start_us"] / US_PER_MS,
                "runtime": m["runtime_name"],
                "kernel": m["gpu_name"],
                "exact": m["exact"],
                "corr_id": m["corr_id"],
            })
        print_table("Worst launch latencies", rows, [
            ("latency_ms", "latency_ms"), ("exact", "exact"), ("corr_id", "corr_id"),
            ("runtime", "runtime"), ("kernel", "kernel"),
        ])

    for dev, gaps in report["gpu_idle_gaps"].items():
        print_table(f"Largest GPU{dev} idle gaps", gaps, [
            ("dur_ms", "dur_ms"), ("start_ms", "start_ms"), ("end_ms", "end_ms"),
            ("before", "before"), ("after", "after"),
        ], limit=10)

    print("\nRecommendations")
    print("---------------")
    for r in report["recommendations"]:
        print(f"- {r}")


def write_json(report: Dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def write_csvs(data: TraceData, report: Dict[str, Any], out_prefix: Path, events: Sequence[Event]) -> None:
    event_path = out_prefix.with_suffix(".events.csv")
    fields = [
        "idx", "kind", "name", "cat", "pid", "tid", "process", "thread", "device", "stream",
        "corr_id", "ext_id", "ts_us", "dur_us", "end_us", "dur_ms", "self_ms", "lane",
    ]
    with event_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for e in events:
            w.writerow({
                "idx": e.idx, "kind": e.kind, "name": e.name, "cat": e.cat, "pid": e.pid, "tid": e.tid,
                "process": e.process, "thread": e.thread, "device": e.device, "stream": e.stream,
                "corr_id": e.corr_id, "ext_id": e.ext_id, "ts_us": e.ts, "dur_us": e.dur,
                "end_us": e.end, "dur_ms": e.dur / US_PER_MS, "self_ms": e.self_us / US_PER_MS,
                "lane": e.lane,
            })
    # Summary tables
    for key in ["top_cpu_ops", "top_cuda_runtime", "top_gpu_kernels", "top_gpu_memory", "sync_hotspots"]:
        rows = report.get(key, [])
        if not rows:
            continue
        p = out_prefix.with_suffix(f".{key}.csv")
        all_fields = sorted({k for r in rows for k in r.keys()})
        with p.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=all_fields)
            w.writeheader()
            w.writerows(rows)


def plot_timeline(events: Sequence[Event], data: TraceData, output: Path, start: Optional[float], end: Optional[float], max_lanes: int = 48, min_dur_us: float = 0.0) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        raise RuntimeError("matplotlib is required for PNG output. Install with: pip install matplotlib") from exc

    win_start = start if start is not None else data.trace_start_us
    win_end = end if end is not None else data.trace_end_us
    selected = [e for e in events if e.dur >= min_dur_us and e.overlaps(win_start, win_end)]
    if not selected:
        raise RuntimeError("No events in selected timeline window.")

    # Pick busiest lanes, preserving GPU lanes first when comparable.
    lane_busy: Dict[str, float] = defaultdict(float)
    lane_kind: Dict[str, str] = {}
    for e in selected:
        c = clip_interval(e.ts, e.end, win_start, win_end)
        if c:
            lane_busy[e.lane] += c[1] - c[0]
            lane_kind[e.lane] = "GPU" if e.kind in {"GPU_KERNEL", "GPU_MEMCPY", "GPU_MEMSET", "NCCL"} else "CPU"
    lanes = sorted(lane_busy, key=lambda l: (lane_kind.get(l) != "GPU", -lane_busy[l], l))[:max_lanes]
    lane_to_y = {lane: i for i, lane in enumerate(reversed(lanes))}

    fig_h = max(6, 0.28 * len(lanes) + 2)
    fig_w = 18
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    for e in selected:
        if e.lane not in lane_to_y:
            continue
        c = clip_interval(e.ts, e.end, win_start, win_end)
        if not c:
            continue
        x0 = (c[0] - win_start) / US_PER_MS
        width = (c[1] - c[0]) / US_PER_MS
        y = lane_to_y[e.lane]
        ax.barh(y, width, left=x0, height=0.76, align="center")
        if width > (win_end - win_start) / US_PER_MS * 0.06:
            label = e.name[:48]
            ax.text(x0 + width / 2, y, label, ha="center", va="center", fontsize=6, clip_on=True)

    ax.set_xlabel("time within selected window (ms)")
    ax.set_ylabel("lane")
    ax.set_yticks(list(lane_to_y.values()))
    ax.set_yticklabels(list(lane_to_y.keys()), fontsize=7)
    ax.set_title(f"PyTorch profiler timeline: {((win_end - win_start) / US_PER_MS):.3f} ms window")
    ax.grid(True, axis="x", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


# --------------------------- inspect/debug ---------------------------

def inspect_trace(data: TraceData, top: int) -> None:
    print("Trace inspection")
    print("================")
    print(f"raw events:      {data.raw_event_count:,}")
    print(f"complete events: {len(data.events):,}")
    print(f"duration:        {(data.trace_end_us - data.trace_start_us) / US_PER_MS:.3f} ms")
    print("\nKinds:")
    for k, n in Counter(e.kind for e in data.events).most_common():
        print(f"  {k:14} {n:,}")
    print("\nCategories:")
    for k, n in Counter(e.cat for e in data.events).most_common(top):
        print(f"  {k[:70]:70} {n:,}")
    print("\nProcesses / threads:")
    lane_counts = Counter((e.process or str(e.pid), e.thread or str(e.tid)) for e in data.events)
    for (p, t), n in lane_counts.most_common(top):
        print(f"  {p[:35]:35} | {t[:45]:45} {n:,}")
    print("\nExample arg keys:")
    keys = Counter()
    for e in data.events:
        for k in flatten_keys(e.args):
            keys[k] += 1
    for k, n in keys.most_common(top):
        print(f"  {k[:80]:80} {n:,}")
    print("\nDevices / streams:")
    for k, n in Counter((e.device or "?", e.stream or "?") for e in gpu_events(data.events)).most_common(top):
        print(f"  GPU{k[0]} stream{k[1]}: {n:,}")


def flatten_keys(obj: Any, prefix: str = "") -> Iterable[str]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            yield p
            yield from flatten_keys(v, p)
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:5]):
            yield from flatten_keys(v, f"{prefix}[{i}]")


def dump_matching_events(events: Sequence[Event], regex: re.Pattern[str], limit: int) -> None:
    rows = []
    for e in events:
        hay = " ".join([e.name, e.cat, e.kind, e.process, e.thread, json.dumps(e.args, ensure_ascii=False, default=str)[:1000]])
        if regex.search(hay):
            rows.append({
                "idx": e.idx, "kind": e.kind, "name": e.name, "cat": e.cat,
                "ts_ms": e.ts / US_PER_MS, "dur_ms": e.dur / US_PER_MS,
                "lane": e.lane, "corr": e.corr_id, "ext": e.ext_id,
            })
            if len(rows) >= limit:
                break
    print_table("Matching events", rows, [
        ("idx", "idx"), ("kind", "kind"), ("ts_ms", "ts_ms"), ("dur_ms", "dur_ms"),
        ("lane", "lane"), ("corr", "corr"), ("ext", "ext"), ("name", "name"), ("cat", "cat"),
    ])



# --------------------------- perfetto-for-agents extensions ---------------------------

import sqlite3
import html as _html
import textwrap as _textwrap

@dataclass
class AgentCounter:
    idx: int
    name: str
    cat: str
    ts: float
    pid: Any
    tid: Any
    value: float
    args: Dict[str, Any]
    process: str = ""
    thread: str = ""

@dataclass
class AgentInstant:
    idx: int
    name: str
    cat: str
    ts: float
    pid: Any
    tid: Any
    args: Dict[str, Any]
    process: str = ""
    thread: str = ""

@dataclass
class AgentFlow:
    src_idx: int
    dst_idx: int
    flow_id: str
    kind: str
    src_name: str
    dst_name: str
    src_ts: float
    dst_ts: float
    latency_us: float
    confidence: str

@dataclass
class AgentFinding:
    severity: str
    category: str
    title: str
    detail: str
    start_us: float = 0.0
    end_us: float = 0.0
    evidence: str = ""
    recommendation: str = ""

@dataclass
class AgentBookmark:
    name: str
    start_us: float
    end_us: float
    reason: str
    severity: str = "info"

@dataclass
class AgentArtifacts:
    counters: List[AgentCounter]
    instants: List[AgentInstant]
    flows: List[AgentFlow]
    findings: List[AgentFinding]
    bookmarks: List[AgentBookmark]
    track_rows: List[Dict[str, Any]]
    phases: List[Dict[str, Any]]
    gap_context: List[Dict[str, Any]]

HEURISTIC_CATALOG: List[Dict[str, str]] = [
    {
        "category": "SYNC",
        "pattern": "cudaDeviceSynchronize",
        "description": "Device-wide synchronization blocks host progress and all visible streams.",
        "severity": "high",
    },
    {
        "category": "SYNC",
        "pattern": "cudaStreamSynchronize",
        "description": "Stream synchronization serializes the CPU with a CUDA stream.",
        "severity": "medium",
    },
    {
        "category": "SYNC",
        "pattern": "cudaEventSynchronize",
        "description": "Event synchronization blocks the host until GPU reaches an event.",
        "severity": "medium",
    },
    {
        "category": "ALLOCATOR",
        "pattern": "cudaMalloc",
        "description": "Frequent cudaMalloc calls imply allocator churn or missing warmup/preallocation.",
        "severity": "medium",
    },
    {
        "category": "ALLOCATOR",
        "pattern": "cudaFree",
        "description": "cudaFree may synchronize; frequent frees can stall the host and GPU timeline.",
        "severity": "medium",
    },
    {
        "category": "MEMORY",
        "pattern": "Memcpy DtoH",
        "description": "Device-to-host copies often imply metric/logging/scalar extraction and may synchronize.",
        "severity": "medium",
    },
    {
        "category": "MEMORY",
        "pattern": "Memcpy HtoD",
        "description": "Host-to-device copies should be overlapped with compute using pinned memory and non_blocking=True where safe.",
        "severity": "medium",
    },
    {
        "category": "MEMORY",
        "pattern": "Memcpy DtoD",
        "description": "Device-to-device copies can indicate layout, dtype, or contiguous conversion overhead.",
        "severity": "medium",
    },
    {
        "category": "MEMORY",
        "pattern": "memset",
        "description": "Frequent memset activity can indicate repeated zero initialization or allocator clearing.",
        "severity": "low",
    },
    {
        "category": "DISTRIBUTED",
        "pattern": "nccl",
        "description": "NCCL activity dominates distributed jobs when communication is not overlapped or bucket sizes/topology are poor.",
        "severity": "medium",
    },
    {
        "category": "CPU",
        "pattern": "aten::item",
        "description": "item() extracts a scalar and can introduce an implicit GPU synchronization.",
        "severity": "high",
    },
    {
        "category": "CPU",
        "pattern": "aten::nonzero",
        "description": "CUDA nonzero is data-dependent and can introduce synchronization or irregular kernels.",
        "severity": "medium",
    },
    {
        "category": "CPU",
        "pattern": "aten::_to_copy",
        "description": "Tensor conversion copies can hide dtype/device/layout transitions.",
        "severity": "medium",
    },
    {
        "category": "CPU",
        "pattern": "aten::copy_",
        "description": "Copies often indicate dtype/device/layout conversion or implicit materialization.",
        "severity": "medium",
    },
    {
        "category": "CPU",
        "pattern": "aten::contiguous",
        "description": "contiguous() can materialize full tensor copies; inspect strides and memory format.",
        "severity": "medium",
    },
    {
        "category": "CPU",
        "pattern": "DataLoader",
        "description": "DataLoader work near GPU gaps points to input pipeline starvation.",
        "severity": "medium",
    },
]

AGENT_FIELD_GUIDE = """
Perfetto-for-agents field guide
===============================

The tool intentionally maps common Perfetto UI actions to deterministic artifacts:

- Timeline overview -> text/json utilization, stream overlap, idle gaps, and PNG timeline.
- Track list -> agent_tracks.csv and track_summary SQLite table.
- Slice details -> --select EVENT_ID and the events/args SQLite tables.
- SQL console -> --mode sqlite plus --sql, --sql-file, and --recipe.
- Search -> --query-regex, --dump-regex, SQL LIKE/GLOB over event names and args.
- Counters -> counters and counter_args tables from C/ph=i/R/P trace events.
- Flows/arrows -> flows table from correlation ids and launch-to-kernel matching.
- Bookmarks -> generated windows around idle gaps, syncs, long kernels, memory bursts.
- Screenshots -> --mode png with --zoom START:END for stable visual review.
- CI checks -> --require-gpu-util, --max-idle-gap-ms, --fail-on-high-finding.

Boundary: Chrome/Kineto JSON can show timeline utilization and causal gaps, but not
true SM occupancy, warp stalls, DRAM throughput, or Tensor Core utilization. Use
Nsight Compute or Nsight Systems GPU metrics for those counters.
"""

SQL_RECIPES: Dict[str, str] = {
    "top_gpu_kernels": """
        select name, count(*) calls,
               round(sum(dur_us)/1000.0, 3) total_ms,
               round(avg(dur_us)/1000.0, 6) avg_ms,
               round(max(dur_us)/1000.0, 3) max_ms
        from events
        where kind in ('GPU_KERNEL','NCCL')
        group by name
        order by sum(dur_us) desc
        limit :limit
    """,
    "top_cuda_runtime": """
        select name, count(*) calls,
               round(sum(dur_us)/1000.0, 3) total_ms,
               round(avg(dur_us)/1000.0, 6) avg_ms,
               round(max(dur_us)/1000.0, 3) max_ms
        from events
        where kind = 'CUDA_RUNTIME'
        group by name
        order by sum(dur_us) desc
        limit :limit
    """,
    "gpu_streams": """
        select device, stream, count(*) events,
               round(sum(dur_us)/1000.0, 3) summed_ms,
               round(min(ts_us)/1000.0, 3) first_ms,
               round(max(end_us)/1000.0, 3) last_ms
        from events
        where kind in ('GPU_KERNEL','GPU_MEMCPY','GPU_MEMSET','NCCL')
        group by device, stream
        order by sum(dur_us) desc
        limit :limit
    """,
    "tiny_kernels": """
        select name, count(*) calls,
               round(sum(dur_us)/1000.0, 3) total_ms,
               round(avg(dur_us), 3) avg_us,
               round(max(dur_us), 3) max_us
        from events
        where kind in ('GPU_KERNEL','NCCL') and dur_us < 50
        group by name
        order by count(*) desc, sum(dur_us) desc
        limit :limit
    """,
    "sync_calls": """
        select name, count(*) calls,
               round(sum(dur_us)/1000.0, 3) total_ms,
               round(max(dur_us)/1000.0, 3) max_ms
        from events
        where lower(name) like '%synchronize%'
           or lower(name) like '%cudafree%'
           or lower(name) like '%cudamalloc%'
           or lower(name) like '%memcpy%'
        group by name
        order by sum(dur_us) desc
        limit :limit
    """,
    "memcpy": """
        select name, kind, count(*) calls,
               round(sum(dur_us)/1000.0, 3) total_ms,
               round(avg(dur_us)/1000.0, 6) avg_ms,
               round(max(dur_us)/1000.0, 3) max_ms
        from events
        where lower(name) like '%memcpy%'
           or lower(name) like '%copy%'
           or kind in ('GPU_MEMCPY','GPU_MEMSET')
        group by name, kind
        order by sum(dur_us) desc
        limit :limit
    """,
    "steps": """
        select name, count(*) calls,
               round(sum(dur_us)/1000.0, 3) total_ms,
               round(avg(dur_us)/1000.0, 3) avg_ms,
               round(min(ts_us)/1000.0, 3) first_ms,
               round(max(end_us)/1000.0, 3) last_ms
        from events
        where lower(name) like '%profilerstep%'
           or lower(name) like '%iteration%'
           or lower(name) like '%train%step%'
           or lower(name) like '%validation%step%'
        group by name
        order by min(ts_us)
        limit :limit
    """,
    "longest_events": """
        select id, kind, name, lane,
               round(ts_us/1000.0, 3) ts_ms,
               round(dur_us/1000.0, 3) dur_ms,
               round(self_us/1000.0, 3) self_ms
        from events
        order by dur_us desc
        limit :limit
    """,
    "arg_keys": """
        select key, count(*) rows
        from args
        group by key
        order by count(*) desc, key
        limit :limit
    """,
    "shape_args": """
        select e.id, e.kind, e.name, a.key, a.value,
               round(e.ts_us/1000.0, 3) ts_ms,
               round(e.dur_us/1000.0, 3) dur_ms
        from args a join events e on a.event_id = e.id
        where lower(a.key) like '%shape%'
           or lower(a.key) like '%input%'
           or lower(a.key) like '%dtype%'
           or lower(a.key) like '%device%'
        order by e.ts_us
        limit :limit
    """,
    "correlations": """
        select corr_id, count(*) events,
               group_concat(distinct kind) kinds,
               round(min(ts_us)/1000.0, 3) start_ms,
               round(max(end_us)/1000.0, 3) end_ms
        from events
        where corr_id != ''
        group by corr_id
        having count(*) > 1
        order by min(ts_us)
        limit :limit
    """,
    "cpu_threads": """
        select process, thread, count(*) events,
               round(sum(dur_us)/1000.0, 3) summed_ms,
               round(sum(self_us)/1000.0, 3) self_ms,
               round(max(dur_us)/1000.0, 3) max_ms
        from events
        where kind not in ('GPU_KERNEL','GPU_MEMCPY','GPU_MEMSET','NCCL')
        group by process, thread
        order by sum(dur_us) desc
        limit :limit
    """,
    "kernel_launch_pairs": """
        select f.flow_id, f.kind, f.src_name runtime, f.dst_name kernel,
               round(f.latency_us/1000.0, 6) latency_ms,
               f.confidence
        from flows f
        order by f.latency_us desc
        limit :limit
    """,
    "findings": """
        select severity, category, title, detail, recommendation,
               round(start_us/1000.0, 3) start_ms,
               round(end_us/1000.0, 3) end_ms
        from findings
        order by case severity when 'high' then 0 when 'medium' then 1 else 2 end, start_us
        limit :limit
    """,
    "bookmarks": """
        select severity, name, reason,
               round(start_us/1000.0, 3) start_ms,
               round(end_us/1000.0, 3) end_ms,
               round((end_us-start_us)/1000.0, 3) dur_ms
        from bookmarks
        order by start_us
        limit :limit
    """,
    "counter_summary": """
        select name, count(*) samples, round(min(value), 3) min_value,
               round(avg(value), 3) avg_value, round(max(value), 3) max_value,
               round(min(ts_us)/1000.0, 3) first_ms,
               round(max(ts_us)/1000.0, 3) last_ms
        from counters
        group by name
        order by samples desc
        limit :limit
    """,
}

# Add name-focused recipes for catalog patterns. These are useful for agents that want a stable
# vocabulary without writing SQL. Names are intentionally deterministic.
for _i, _rule in enumerate(HEURISTIC_CATALOG):
    _safe = re.sub(r"[^a-z0-9]+", "_", _rule["pattern"].casefold()).strip("_") or f"rule_{_i}"
    SQL_RECIPES.setdefault(f"name_contains_{_safe[:64]}", """
        select id, kind, name, lane, round(ts_us/1000.0, 3) ts_ms,
               round(dur_us/1000.0, 3) dur_ms, corr_id, ext_id
        from events
        where lower(name) like '%' || lower(:pattern) || '%'
           or lower(cat) like '%' || lower(:pattern) || '%'
           or lower(args_json) like '%' || lower(:pattern) || '%'
        order by dur_us desc
        limit :limit
    """)


def load_raw_events(path: Path) -> List[Dict[str, Any]]:
    obj = load_json(path)
    if isinstance(obj, dict):
        raw = obj.get("traceEvents", [])
    elif isinstance(obj, list):
        raw = obj
    else:
        raw = []
    return [x for x in raw if isinstance(x, dict)]


def flatten_args(obj: Any, prefix: str = "") -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, (dict, list)):
                rows.append((key, stringify_maybe(v)))
                rows.extend(flatten_args(v, key))
            else:
                rows.append((key, stringify_maybe(v)))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            key = f"{prefix}[{i}]" if prefix else f"[{i}]"
            if isinstance(v, (dict, list)):
                rows.append((key, stringify_maybe(v)))
                rows.extend(flatten_args(v, key))
            else:
                rows.append((key, stringify_maybe(v)))
    return rows


def numeric_values_from_args(args: Dict[str, Any]) -> List[Tuple[str, float]]:
    out: List[Tuple[str, float]] = []
    def rec(x: Any, prefix: str = "") -> None:
        if isinstance(x, dict):
            for k, v in x.items():
                key = f"{prefix}.{k}" if prefix else str(k)
                rec(v, key)
        elif isinstance(x, list):
            for i, v in enumerate(x[:64]):
                rec(v, f"{prefix}[{i}]")
        elif isinstance(x, bool):
            out.append((prefix, 1.0 if x else 0.0))
        elif isinstance(x, (int, float)) and math.isfinite(float(x)):
            out.append((prefix, float(x)))
        elif isinstance(x, str):
            try:
                val = float(x)
            except Exception:
                return
            if math.isfinite(val):
                out.append((prefix, val))
    rec(args)
    return out


def extract_counters_and_instants(raw_events: Sequence[Dict[str, Any]], data: TraceData) -> Tuple[List[AgentCounter], List[AgentInstant]]:
    meta = metadata_from_events(raw_events)
    counters: List[AgentCounter] = []
    instants: List[AgentInstant] = []
    cidx = 0
    iidx = 0
    for e in raw_events:
        ph = e.get("ph")
        if ph not in {"C", "i", "I", "R", "s", "t", "f", "n"}:
            continue
        pid, tid = e.get("pid", ""), e.get("tid", "")
        m = meta.get((pid, tid), {})
        process = m.get("process") or meta.get((pid, "*"), {}).get("process") or ""
        thread = m.get("thread") or ""
        args = e.get("args") or {}
        ts = float(e.get("ts") or 0.0)
        if ph == "C":
            nums = numeric_values_from_args(args)
            if not nums:
                counters.append(AgentCounter(cidx, str(e.get("name", "counter")), str(e.get("cat", "")), ts, pid, tid, 0.0, args, process, thread))
                cidx += 1
            else:
                for key, val in nums:
                    name = str(e.get("name", "counter"))
                    if key:
                        name = f"{name}.{key}"
                    counters.append(AgentCounter(cidx, name, str(e.get("cat", "")), ts, pid, tid, val, args, process, thread))
                    cidx += 1
        else:
            instants.append(AgentInstant(iidx, str(e.get("name", "")), str(e.get("cat", "")), ts, pid, tid, args, process, thread))
            iidx += 1
    return counters, instants


def compute_track_rows(events: Sequence[Event], start: Optional[float], end: Optional[float]) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Event]] = defaultdict(list)
    for e in events:
        c = clip_interval(e.ts, e.end, start, end)
        if c:
            groups[e.lane].append(e)
    rows: List[Dict[str, Any]] = []
    for lane, xs in groups.items():
        intervals = []
        summed = 0.0
        kinds = Counter()
        names = Counter()
        first = min(e.ts for e in xs)
        last = max(e.end for e in xs)
        for e in xs:
            c = clip_interval(e.ts, e.end, start, end)
            if c:
                intervals.append(c)
                summed += c[1] - c[0]
            kinds[e.kind] += 1
            names[e.name] += 1
        busy = interval_total(intervals)
        span = max(1e-9, (end if end is not None else last) - (start if start is not None else first))
        rows.append({
            "lane": lane,
            "kind_hint": "GPU" if any(k in {"GPU_KERNEL", "GPU_MEMCPY", "GPU_MEMSET", "NCCL"} for k in kinds) else "CPU",
            "events": len(xs),
            "busy_ms": busy / US_PER_MS,
            "summed_ms": summed / US_PER_MS,
            "util_percent_in_window": 100.0 * busy / span,
            "first_ms": first / US_PER_MS,
            "last_ms": last / US_PER_MS,
            "top_kind": kinds.most_common(1)[0][0] if kinds else "",
            "top_name": names.most_common(1)[0][0] if names else "",
        })
    rows.sort(key=lambda r: (r["kind_hint"] != "GPU", -r["busy_ms"], r["lane"]))
    return rows


def compute_phase_rows(events: Sequence[Event], start: Optional[float], end: Optional[float]) -> List[Dict[str, Any]]:
    candidates = []
    phase_re = re.compile(r"profilerstep|iteration|train.*step|valid.*step|eval.*step|forward|backward|optimizer|dataloader|data loader|zero_grad", re.I)
    for e in events:
        if e.kind in {"CPU_OP", "OTHER"} and phase_re.search(e.name):
            if clip_interval(e.ts, e.end, start, end):
                candidates.append(e)
    candidates.sort(key=lambda e: (e.ts, -e.dur))
    rows = []
    for e in candidates[:500]:
        gpu_inside = [g for g in gpu_events(events) if g.ts >= e.ts and g.end <= e.end]
        gpu_busy = interval_total((g.ts, g.end) for g in gpu_inside)
        rows.append({
            "event_id": e.idx,
            "phase": e.name,
            "start_ms": e.ts / US_PER_MS,
            "dur_ms": e.dur / US_PER_MS,
            "gpu_events_inside": len(gpu_inside),
            "gpu_busy_inside_ms": gpu_busy / US_PER_MS,
            "gpu_util_inside_percent": 100.0 * gpu_busy / e.dur if e.dur > 0 else 0.0,
            "lane": e.lane,
        })
    return rows


def build_flows(events: Sequence[Event], latencies: Sequence[MatchLatency]) -> List[AgentFlow]:
    flows: List[AgentFlow] = []
    by_idx = {e.idx: e for e in events}
    for m in latencies:
        src = by_idx.get(m.runtime_idx)
        dst = by_idx.get(m.gpu_idx)
        if not src or not dst:
            continue
        flows.append(AgentFlow(
            src_idx=src.idx,
            dst_idx=dst.idx,
            flow_id=m.corr_id or f"heuristic:{src.idx}->{dst.idx}",
            kind="cuda_launch_to_gpu",
            src_name=src.name,
            dst_name=dst.name,
            src_ts=src.ts,
            dst_ts=dst.ts,
            latency_us=m.launch_end_to_gpu_start_us,
            confidence="exact" if m.exact else "heuristic",
        ))
    return flows


def gap_context_rows(events: Sequence[Event], report: Dict[str, Any], radius_us: float = 1000.0, limit_per_gap: int = 8) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    cpu = [e for e in events if e.kind not in {"GPU_KERNEL", "GPU_MEMCPY", "GPU_MEMSET", "NCCL"}]
    for dev, gaps in report.get("gpu_idle_gaps", {}).items():
        for gi, g in enumerate(gaps[:20]):
            start_us = float(g["start_us"])
            end_us = float(g["end_us"])
            window_start = start_us - radius_us
            window_end = end_us + radius_us
            near = [e for e in cpu if e.end >= window_start and e.ts <= window_end]
            near.sort(key=lambda e: (-e.dur, e.ts))
            for e in near[:limit_per_gap]:
                rows.append({
                    "device": dev,
                    "gap_rank": gi + 1,
                    "gap_start_ms": g["start_ms"],
                    "gap_dur_ms": g["dur_ms"],
                    "event_id": e.idx,
                    "event_kind": e.kind,
                    "event_name": e.name,
                    "event_lane": e.lane,
                    "event_start_ms": e.ts / US_PER_MS,
                    "event_dur_ms": e.dur / US_PER_MS,
                })
    return rows


def percentile_ms_from_events(events: Sequence[Event], p: float) -> float:
    return percentile([e.dur for e in events], p) / US_PER_MS


def severity_rank(s: str) -> int:
    return {"high": 0, "medium": 1, "low": 2, "info": 3}.get(s, 4)


def make_finding(severity: str, category: str, title: str, detail: str, recommendation: str = "", evidence: str = "", start_us: float = 0.0, end_us: float = 0.0) -> AgentFinding:
    return AgentFinding(severity, category, title, detail, start_us, end_us, evidence, recommendation)


def generate_findings(events: Sequence[Event], report: Dict[str, Any], flows: Sequence[AgentFlow], counters: Sequence[AgentCounter]) -> List[AgentFinding]:
    findings: List[AgentFinding] = []
    gpu = gpu_events(events)
    cpu = [e for e in events if e.kind not in {"GPU_KERNEL", "GPU_MEMCPY", "GPU_MEMSET", "NCCL"}]
    total_window_ms = report.get("trace", {}).get("window_duration_ms", 0.0)
    for dev, util in report.get("gpu_utilization", {}).items():
        pct = float(util.get("util_percent", 0.0))
        if pct < 50.0:
            findings.append(make_finding("high", "gpu_utilization", f"GPU{dev} is severely underutilized", f"Timeline utilization is {pct:.2f}% over the selected window.", "Inspect largest idle gaps and CPU gap context; look for DataLoader waits, Python overhead, synchronization, or small batches."))
        elif pct < 80.0:
            findings.append(make_finding("medium", "gpu_utilization", f"GPU{dev} has headroom", f"Timeline utilization is {pct:.2f}% over the selected window.", "Increase overlap, batch size, data pipeline throughput, or reduce CPU launch overhead."))
    for dev, gaps in report.get("gpu_idle_gaps", {}).items():
        if gaps:
            g = gaps[0]
            severity = "high" if g.get("dur_ms", 0.0) > 10.0 else "medium"
            findings.append(make_finding(severity, "gpu_idle_gap", f"Largest GPU{dev} idle gap is {g.get('dur_ms',0.0):.3f} ms", f"Gap from {g.get('start_ms',0.0):.3f} ms to {g.get('end_ms',0.0):.3f} ms, before={g.get('before','')}, after={g.get('after','')}.", "Use gap context, zoomed PNG, and track summary to locate why the GPU was not fed.", start_us=g.get("start_us", 0.0), end_us=g.get("end_us", 0.0)))
    tiny = [e for e in gpu if e.kind in {"GPU_KERNEL", "NCCL"} and e.dur < 50.0]
    if len(tiny) >= 100 or (gpu and len(tiny) / max(1, len(gpu)) > 0.25):
        findings.append(make_finding("medium", "tiny_kernels", "Many tiny GPU kernels", f"{len(tiny)} kernels are shorter than 50 us; p95 GPU event duration is {percentile_ms_from_events(gpu,95):.4f} ms.", "Consider torch.compile, CUDA graphs, fused optimizers, larger batch sizes, or reducing eager-mode fragmentation."))
    syncs = [e for e in cpu if SYNC_NAME_PAT.search(e.name)]
    if syncs:
        worst = max(syncs, key=lambda e: e.dur)
        findings.append(make_finding("high" if worst.dur > 5000 else "medium", "synchronization", "Host calls that can block the CPU on the GPU", f"{len(syncs)} blocking-capable CPU calls; worst is {worst.name} at {worst.dur/US_PER_MS:.3f} ms. This set includes cudaMalloc/cudaFree/cudaMemcpy, which block only in some forms — confirm against the trace before calling it a synchronization.", "Remove explicit synchronizations from the training step; avoid item()/blocking DtoH copies; batch metrics asynchronously when possible.", start_us=worst.ts, end_us=worst.end, evidence=f"event_id={worst.idx}"))
    mem = [e for e in events if e.kind in {"GPU_MEMCPY", "GPU_MEMSET"} or MEMORY_PAT.search(e.name)]
    if mem:
        total_mem_ms = sum(e.dur for e in mem) / US_PER_MS
        if total_window_ms and total_mem_ms / total_window_ms > 0.15:
            findings.append(make_finding("medium", "memory_transfer", "Memory operations consume significant timeline", f"Memory-like events sum to {total_mem_ms:.3f} ms over a {total_window_ms:.3f} ms window.", "Check pinned memory, non_blocking transfers, unnecessary .to()/.copy_()/contiguous(), and host-device scalar extraction."))
    nccl = [e for e in gpu if e.kind == "NCCL" or NCCL_PAT.search(e.name)]
    if nccl:
        nccl_ms = sum(e.dur for e in nccl) / US_PER_MS
        gpu_ms = sum(e.dur for e in gpu) / US_PER_MS
        if gpu_ms and nccl_ms / gpu_ms > 0.20:
            findings.append(make_finding("medium", "distributed", "NCCL/communication is a large GPU component", f"NCCL-like work sums to {nccl_ms:.3f} ms ({100*nccl_ms/gpu_ms:.1f}% of summed GPU event time).", "Inspect rank skew, bucket sizes, gradient accumulation, overlap with backward, topology, and collective algorithm choices."))
    if flows:
        lats = [f.latency_us for f in flows if f.latency_us >= 0]
        if lats and percentile(lats, 95) > 1000:
            findings.append(make_finding("medium", "launch_latency", "High CUDA launch-to-kernel latency", f"p95 launch-to-kernel latency is {percentile(lats,95)/US_PER_MS:.3f} ms across {len(lats)} launch flows.", "Look for CPU thread saturation, synchronization, driver overhead, graph breaks, or oversubscription."))
    if counters:
        names = Counter(c.name for c in counters)
        if names:
            findings.append(make_finding("info", "counters", "Trace contains counter tracks", f"Detected {len(counters)} counter samples across {len(names)} counter names.", "Query the counters table for memory, queue, or custom utilization metrics."))
    # Catalog-driven findings. These are deterministic, searchable, and useful for agents.
    hay_cache = [(e, " ".join([e.name, e.cat, e.kind, stringify_maybe(e.args)]).casefold()) for e in events]
    for rule in HEURISTIC_CATALOG:
        pattern = rule["pattern"].casefold()
        # Reserved hooks should not spam unless present.
        matched = [e for e, hay in hay_cache if pattern in hay]
        if not matched:
            continue
        total_ms = sum(e.dur for e in matched) / US_PER_MS
        worst = max(matched, key=lambda e: e.dur)
        sev = rule.get("severity", "medium")
        if total_ms < 0.05 and len(matched) < 3 and not pattern.startswith("agent_rule_"):
            continue
        findings.append(make_finding(sev, rule["category"], f"Pattern detected: {rule['pattern']}", f"{len(matched)} matching events, summed duration {total_ms:.3f} ms. {rule['description']}", rule["description"], evidence=f"worst_event_id={worst.idx}", start_us=worst.ts, end_us=worst.end))
    findings.sort(key=lambda f: (severity_rank(f.severity), -(f.end_us - f.start_us), f.category, f.title))
    # Keep report manageable.
    return findings[:300]


def generate_bookmarks(events: Sequence[Event], report: Dict[str, Any], findings: Sequence[AgentFinding], pad_us: float = 1000.0) -> List[AgentBookmark]:
    bookmarks: List[AgentBookmark] = []
    trace_start = min((e.ts for e in events), default=0.0)
    trace_end = max((e.end for e in events), default=0.0)
    def add(name: str, start: float, end: float, reason: str, severity: str = "info") -> None:
        s = max(trace_start, start - pad_us)
        t = min(trace_end, end + pad_us)
        if t <= s:
            t = min(trace_end, s + 100.0)
        bookmarks.append(AgentBookmark(name, s, t, reason, severity))
    for dev, gaps in report.get("gpu_idle_gaps", {}).items():
        for i, g in enumerate(gaps[:10]):
            add(f"GPU{dev} idle gap #{i+1}", g.get("start_us", 0.0), g.get("end_us", 0.0), f"Idle gap {g.get('dur_ms',0.0):.3f} ms", "high" if g.get("dur_ms",0.0) > 10 else "medium")
    syncs = [e for e in events if SYNC_NAME_PAT.search(e.name)]
    syncs.sort(key=lambda e: e.dur, reverse=True)
    for i, e in enumerate(syncs[:10]):
        add(f"Sync hotspot #{i+1}: {e.name[:60]}", e.ts, e.end, f"Synchronization-like event duration {e.dur/US_PER_MS:.3f} ms", "high" if e.dur > 5000 else "medium")
    long_gpu = [e for e in gpu_events(events) if e.kind in {"GPU_KERNEL", "NCCL"}]
    long_gpu.sort(key=lambda e: e.dur, reverse=True)
    for i, e in enumerate(long_gpu[:10]):
        add(f"Long GPU event #{i+1}: {e.name[:60]}", e.ts, e.end, f"GPU event duration {e.dur/US_PER_MS:.3f} ms on {e.lane}", "info")
    for i, f in enumerate(findings[:20]):
        if f.start_us or f.end_us:
            add(f"Finding #{i+1}: {f.title[:70]}", f.start_us, f.end_us, f.detail[:200], f.severity)
    bookmarks.sort(key=lambda b: (b.start_us, severity_rank(b.severity), b.name))
    return bookmarks[:200]


def build_agent_artifacts(raw_events: Sequence[Dict[str, Any]], data: TraceData, report: Dict[str, Any], events: Sequence[Event], start: Optional[float], end: Optional[float]) -> AgentArtifacts:
    counters, instants = extract_counters_and_instants(raw_events, data)
    # Recompute launch latencies over selected events for flow table.
    flows = build_flows(events, analyze_launch_latency(events))
    track_rows = compute_track_rows(events, start, end)
    phases = compute_phase_rows(events, start, end)
    findings = generate_findings(events, report, flows, counters)
    bookmarks = generate_bookmarks(events, report, findings)
    gaps = gap_context_rows(events, report)
    return AgentArtifacts(counters, instants, flows, findings, bookmarks, track_rows, phases, gaps)


def event_args_json(e: Event) -> str:
    try:
        return json.dumps(e.args, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        return stringify_maybe(e.args)


def create_sqlite(path: Path, data: TraceData, events: Sequence[Event], artifacts: AgentArtifacts, overwrite: bool = True) -> None:
    if overwrite and path.exists():
        path.unlink()
    con = sqlite3.connect(str(path))
    try:
        cur = con.cursor()
        cur.executescript("""
        pragma journal_mode = off;
        pragma synchronous = off;
        create table if not exists events (
            id integer primary key,
            kind text, name text, cat text,
            pid text, tid text, process text, thread text,
            device text, stream text, corr_id text, ext_id text,
            ts_us real, dur_us real, end_us real, self_us real,
            parent_id integer, lane text, args_json text
        );
        create table if not exists args (
            event_id integer, key text, value text
        );
        create table if not exists counters (
            id integer primary key,
            name text, cat text, pid text, tid text, process text, thread text,
            ts_us real, value real, args_json text
        );
        create table if not exists counter_args (
            counter_id integer, key text, value text
        );
        create table if not exists instants (
            id integer primary key,
            name text, cat text, pid text, tid text, process text, thread text,
            ts_us real, args_json text
        );
        create table if not exists flows (
            src_id integer, dst_id integer, flow_id text, kind text,
            src_name text, dst_name text, src_ts_us real, dst_ts_us real,
            latency_us real, confidence text
        );
        create table if not exists track_summary (
            lane text, kind_hint text, events integer, busy_ms real, summed_ms real,
            util_percent_in_window real, first_ms real, last_ms real, top_kind text, top_name text
        );
        create table if not exists phases (
            event_id integer, phase text, start_ms real, dur_ms real,
            gpu_events_inside integer, gpu_busy_inside_ms real, gpu_util_inside_percent real, lane text
        );
        create table if not exists findings (
            severity text, category text, title text, detail text,
            start_us real, end_us real, evidence text, recommendation text
        );
        create table if not exists bookmarks (
            name text, start_us real, end_us real, reason text, severity text
        );
        create table if not exists gap_context (
            device text, gap_rank integer, gap_start_ms real, gap_dur_ms real,
            event_id integer, event_kind text, event_name text, event_lane text,
            event_start_ms real, event_dur_ms real
        );
        """)
        cur.executemany("""
            insert into events values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, [
            (e.idx, e.kind, e.name, e.cat, stringify_maybe(e.pid), stringify_maybe(e.tid), e.process, e.thread,
             e.device, e.stream, e.corr_id, e.ext_id, e.ts, e.dur, e.end, e.self_us, e.parent_idx, e.lane, event_args_json(e))
            for e in events
        ])
        arg_rows = []
        for e in events:
            for k, v in flatten_args(e.args):
                arg_rows.append((e.idx, k, v))
        cur.executemany("insert into args values (?,?,?)", arg_rows)
        cur.executemany("insert into counters values (?,?,?,?,?,?,?,?,?,?)", [
            (c.idx, c.name, c.cat, stringify_maybe(c.pid), stringify_maybe(c.tid), c.process, c.thread, c.ts, c.value, stringify_maybe(c.args))
            for c in artifacts.counters
        ])
        counter_arg_rows = []
        for c in artifacts.counters:
            for k, v in flatten_args(c.args):
                counter_arg_rows.append((c.idx, k, v))
        cur.executemany("insert into counter_args values (?,?,?)", counter_arg_rows)
        cur.executemany("insert into instants values (?,?,?,?,?,?,?,?,?)", [
            (i.idx, i.name, i.cat, stringify_maybe(i.pid), stringify_maybe(i.tid), i.process, i.thread, i.ts, stringify_maybe(i.args))
            for i in artifacts.instants
        ])
        cur.executemany("insert into flows values (?,?,?,?,?,?,?,?,?,?)", [
            (f.src_idx, f.dst_idx, f.flow_id, f.kind, f.src_name, f.dst_name, f.src_ts, f.dst_ts, f.latency_us, f.confidence)
            for f in artifacts.flows
        ])
        cur.executemany("insert into track_summary values (?,?,?,?,?,?,?,?,?,?)", [
            (r["lane"], r["kind_hint"], r["events"], r["busy_ms"], r["summed_ms"], r["util_percent_in_window"], r["first_ms"], r["last_ms"], r["top_kind"], r["top_name"])
            for r in artifacts.track_rows
        ])
        cur.executemany("insert into phases values (?,?,?,?,?,?,?,?)", [
            (r["event_id"], r["phase"], r["start_ms"], r["dur_ms"], r["gpu_events_inside"], r["gpu_busy_inside_ms"], r["gpu_util_inside_percent"], r["lane"])
            for r in artifacts.phases
        ])
        cur.executemany("insert into findings values (?,?,?,?,?,?,?,?)", [
            (f.severity, f.category, f.title, f.detail, f.start_us, f.end_us, f.evidence, f.recommendation)
            for f in artifacts.findings
        ])
        cur.executemany("insert into bookmarks values (?,?,?,?,?)", [
            (b.name, b.start_us, b.end_us, b.reason, b.severity)
            for b in artifacts.bookmarks
        ])
        cur.executemany("insert into gap_context values (?,?,?,?,?,?,?,?,?,?)", [
            (r["device"], r["gap_rank"], r["gap_start_ms"], r["gap_dur_ms"], r["event_id"], r["event_kind"], r["event_name"], r["event_lane"], r["event_start_ms"], r["event_dur_ms"])
            for r in artifacts.gap_context
        ])
        cur.executescript("""
        create index if not exists idx_events_kind on events(kind);
        create index if not exists idx_events_name on events(name);
        create index if not exists idx_events_time on events(ts_us, end_us);
        create index if not exists idx_events_corr on events(corr_id);
        create index if not exists idx_args_key on args(key);
        create index if not exists idx_counters_name on counters(name);
        create index if not exists idx_flows_src on flows(src_id);
        create index if not exists idx_flows_dst on flows(dst_id);
        """)
        con.commit()
    finally:
        con.close()


def sqlite_query_rows(db_path: Path, sql: str, limit: int = 50, pattern: str = "") -> List[Dict[str, Any]]:
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    try:
        cur = con.execute(sql, {"limit": limit, "pattern": pattern})
        rows = [dict(r) for r in cur.fetchall()]
        return rows
    finally:
        con.close()


def print_sql_rows(title: str, rows: List[Dict[str, Any]], limit: int = 50) -> None:
    if not rows:
        print(f"\n{title}\n" + "-" * len(title))
        print("  <no rows>")
        return
    fields = list(rows[0].keys())
    print_table(title, rows, [(f, f) for f in fields], limit=limit)


def run_recipes(db_path: Path, recipes: Sequence[str], limit: int) -> None:
    for name in recipes:
        sql = SQL_RECIPES.get(name)
        if not sql:
            print(f"Unknown recipe: {name}", file=sys.stderr)
            continue
        pattern = ""
        if name.startswith("name_contains_"):
            # map back best-effort to a catalog pattern
            suffix = name[len("name_contains_"):]
            for rule in HEURISTIC_CATALOG:
                safe = re.sub(r"[^a-z0-9]+", "_", rule["pattern"].casefold()).strip("_")[:64]
                if safe == suffix:
                    pattern = rule["pattern"]
                    break
        rows = sqlite_query_rows(db_path, sql, limit=limit, pattern=pattern)
        print_sql_rows(f"SQL recipe: {name}", rows, limit)


def write_agent_csvs(prefix: Path, artifacts: AgentArtifacts) -> None:
    tables: Dict[str, List[Dict[str, Any]]] = {
        "agent_tracks": artifacts.track_rows,
        "agent_phases": artifacts.phases,
        "agent_gap_context": artifacts.gap_context,
        "agent_findings": [f.__dict__ for f in artifacts.findings],
        "agent_bookmarks": [b.__dict__ for b in artifacts.bookmarks],
        "agent_flows": [f.__dict__ for f in artifacts.flows],
        "agent_counters": [c.__dict__ for c in artifacts.counters],
        "agent_instants": [i.__dict__ for i in artifacts.instants],
    }
    for name, rows in tables.items():
        if not rows:
            continue
        path = prefix.with_suffix(f".{name}.csv")
        fields = sorted({k for r in rows for k in r.keys()})
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                rr = {}
                for k in fields:
                    v = r.get(k, "")
                    if isinstance(v, (dict, list)):
                        v = stringify_maybe(v)
                    rr[k] = v
                w.writerow(rr)


def agent_report_dict(report: Dict[str, Any], artifacts: AgentArtifacts) -> Dict[str, Any]:
    d = dict(report)
    d["agent"] = {
        "findings": [f.__dict__ for f in artifacts.findings],
        "bookmarks": [b.__dict__ for b in artifacts.bookmarks],
        "track_summary": artifacts.track_rows,
        "phases": artifacts.phases,
        "gap_context": artifacts.gap_context,
        "flow_count": len(artifacts.flows),
        "counter_count": len(artifacts.counters),
        "instant_count": len(artifacts.instants),
        "available_sql_recipes": sorted(SQL_RECIPES),
    }
    return d


def render_agent_text(report: Dict[str, Any], artifacts: AgentArtifacts, top: int) -> None:
    print("\nAgent Perfetto-style analysis")
    print("=============================")
    print_table("Findings", [f.__dict__ for f in artifacts.findings], [
        ("severity", "sev"), ("category", "category"), ("title", "title"), ("recommendation", "recommendation"),
    ], limit=top)
    print_table("Generated bookmarks", [b.__dict__ for b in artifacts.bookmarks], [
        ("severity", "sev"), ("name", "name"), ("start_us", "start_us"), ("end_us", "end_us"), ("reason", "reason"),
    ], limit=top)
    print_table("Track summary", artifacts.track_rows, [
        ("kind_hint", "kind"), ("lane", "lane"), ("events", "events"), ("busy_ms", "busy_ms"),
        ("util_percent_in_window", "util%"), ("top_kind", "top_kind"), ("top_name", "top_name"),
    ], limit=top)
    print_table("Gap context", artifacts.gap_context, [
        ("device", "dev"), ("gap_rank", "gap"), ("gap_dur_ms", "gap_ms"), ("event_kind", "kind"),
        ("event_dur_ms", "event_ms"), ("event_name", "event"), ("event_lane", "lane"),
    ], limit=top)
    print_table("Phases / steps", artifacts.phases, [
        ("event_id", "id"), ("phase", "phase"), ("dur_ms", "dur_ms"),
        ("gpu_busy_inside_ms", "gpu_ms"), ("gpu_util_inside_percent", "gpu_util%"), ("lane", "lane"),
    ], limit=top)
    if artifacts.counters:
        counter_summary = []
        by_name: Dict[str, List[float]] = defaultdict(list)
        for c in artifacts.counters:
            by_name[c.name].append(c.value)
        for name, vals in by_name.items():
            counter_summary.append({"name": name, "samples": len(vals), "min": min(vals), "avg": sum(vals)/len(vals), "max": max(vals)})
        counter_summary.sort(key=lambda r: r["samples"], reverse=True)
        print_table("Counter tracks", counter_summary, [("name", "name"), ("samples", "samples"), ("min", "min"), ("avg", "avg"), ("max", "max")], limit=top)


def markdown_table(rows: List[Dict[str, Any]], fields: Sequence[str], limit: int = 20) -> str:
    if not rows:
        return "\n<no rows>\n"
    rows = rows[:limit]
    lines = []
    lines.append("| " + " | ".join(fields) + " |")
    lines.append("| " + " | ".join(["---"] * len(fields)) + " |")
    for r in rows:
        vals = []
        for f in fields:
            v = r.get(f, "")
            if isinstance(v, float):
                vals.append(f"{v:.6g}")
            else:
                vals.append(str(v).replace("|", "\\|")[:160])
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def write_markdown(path: Path, report: Dict[str, Any], artifacts: AgentArtifacts, top: int) -> None:
    lines = []
    lines.append("# PyTorch profiler hotspot report")
    lines.append("")
    w = report.get("trace", {})
    lines.append(
        f"Window: **{w.get('window_start_us', 0) / US_PER_MS:.3f} ms → "
        f"{w.get('window_end_us', 0) / US_PER_MS:.3f} ms** "
        f"({w.get('window_duration_ms', 0):.3f} ms on the trace clock)"
    )
    lines.append("")
    lines.append("## GPU utilization")
    # gpu_utilization is keyed by device, and the device is not a field of the row.
    util_rows = [dict(row, device=dev) for dev, row in report.get("gpu_utilization", {}).items()]
    lines.append(markdown_table(util_rows, ["device", "util_percent", "busy_ms", "window_ms"], top))
    lines.append("## Findings")
    lines.append(markdown_table([f.__dict__ for f in artifacts.findings], ["severity", "category", "title", "recommendation"], top))
    lines.append("## Track summary")
    lines.append(markdown_table(artifacts.track_rows, ["kind_hint", "lane", "events", "busy_ms", "util_percent_in_window", "top_name"], top))
    lines.append("## Gap context")
    lines.append(markdown_table(artifacts.gap_context, ["device", "gap_rank", "gap_dur_ms", "event_kind", "event_dur_ms", "event_name"], top))
    lines.append("## Built-in SQL recipes")
    for r in sorted(SQL_RECIPES)[:500]:
        lines.append(f"- `{r}`")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_html(path: Path, report: Dict[str, Any], artifacts: AgentArtifacts, top: int) -> None:
    md_path = path.with_suffix(".tmp.md")
    write_markdown(md_path, report, artifacts, top)
    md = md_path.read_text(encoding="utf-8")
    try:
        md_path.unlink()
    except Exception:
        pass
    body = "<pre>" + _html.escape(md) + "</pre>"
    html = f"""<!doctype html>
<html><head><meta charset='utf-8'><title>PyTorch profiler hotspot report</title>
<style>body{{font-family:ui-monospace,Menlo,Consolas,monospace;max-width:1400px;margin:2rem auto;line-height:1.35}} pre{{white-space:pre-wrap}}</style></head>
<body>{body}</body></html>"""
    path.write_text(html, encoding="utf-8")


def select_event_text(events: Sequence[Event], idx: int, context: int = 5) -> str:
    by_idx = {e.idx: e for e in events}
    e = by_idx.get(idx)
    if not e:
        return f"No event with id={idx}"
    near = [x for x in events if x.lane == e.lane and x.ts <= e.end + 1000 and x.end >= e.ts - 1000]
    near.sort(key=lambda x: x.ts)
    pos = near.index(e) if e in near else 0
    around = near[max(0, pos-context):pos+context+1]
    lines = []
    lines.append(f"Selected event id={e.idx}")
    lines.append(f"  name: {e.name}")
    lines.append(f"  kind: {e.kind}")
    lines.append(f"  lane: {e.lane}")
    lines.append(f"  ts_ms: {e.ts/US_PER_MS:.6f}")
    lines.append(f"  dur_ms: {e.dur/US_PER_MS:.6f}")
    lines.append(f"  self_ms: {e.self_us/US_PER_MS:.6f}")
    lines.append(f"  corr_id: {e.corr_id}")
    lines.append(f"  ext_id: {e.ext_id}")
    lines.append("  args:")
    for k, v in flatten_args(e.args)[:80]:
        lines.append(f"    {k}: {v}")
    lines.append("\nContext on same lane:")
    for x in around:
        mark = "=>" if x.idx == e.idx else "  "
        lines.append(f"{mark} {x.idx:7d} {x.ts/US_PER_MS:12.6f} {x.dur/US_PER_MS:10.6f} {x.kind:14s} {x.name[:120]}")
    return "\n".join(lines)


def print_recipes() -> None:
    try:
        sys.stdout.write("Available SQL recipes\n=====================\n")
        for name in sorted(SQL_RECIPES):
            sys.stdout.write(name + "\n")
    except BrokenPipeError:
        return


def print_field_guide() -> None:
    try:
        sys.stdout.write(AGENT_FIELD_GUIDE + "\n")
    except BrokenPipeError:
        return


# --------------------------- extended CLI ---------------------------

def parse_time_value(s: str) -> float:
    s = s.strip()
    m = re.fullmatch(r"([+-]?\d+(?:\.\d+)?)(us|ms|s)?", s)
    if not m:
        raise argparse.ArgumentTypeError(f"Bad time value: {s!r}. Use forms like 100us, 2.5ms, 1s, or bare ms.")
    v = float(m.group(1))
    unit = m.group(2) or "ms"
    if unit == "us":
        return v
    if unit == "ms":
        return v * US_PER_MS
    if unit == "s":
        return v * US_PER_S
    raise argparse.ArgumentTypeError(f"Unsupported unit: {unit}")


def parse_zoom(s: str) -> Tuple[Optional[float], Optional[float]]:
    if ":" not in s:
        raise argparse.ArgumentTypeError("--zoom must be START:END, e.g. 0:50ms or 1s:1.1s")
    a, b = s.split(":", 1)
    start = parse_time_value(a) if a.strip() else None
    end = parse_time_value(b) if b.strip() else None
    if start is not None and end is not None and end <= start:
        raise argparse.ArgumentTypeError("--zoom end must be greater than start")
    return start, end


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Analyze PyTorch profiler Chrome/Kineto JSON traces for hotspots, GPU utilization, gaps, flows, counters, SQL, and agent-ready Perfetto-style summaries.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("trace", type=Path, nargs="?", help="Path to PyTorch profiler Chrome trace JSON or .json.gz")
    p.add_argument("--mode", action="append", choices=["text", "json", "csv", "png", "agent", "sqlite", "md", "html"], default=None,
                   help="Output mode. Can be repeated. Default: text")
    p.add_argument("--out", type=Path, default=Path("hotspot_report"), help="Output prefix for json/csv/png/sqlite/md/html")
    p.add_argument("--top", type=int, default=30, help="Top rows per table")
    p.add_argument("--zoom", type=parse_zoom, default=None,
                   help="Analyze/plot a relative time window from trace start, e.g. 0:100ms, 1s:1.2s. Bare numbers are ms.")
    p.add_argument("--query-regex", type=str, default=None,
                   help="Restrict analysis to events whose name/category/kind/process/thread/args matches regex")
    p.add_argument("--gap-threshold-us", type=float, default=100.0,
                   help="Minimum GPU idle gap to report, in microseconds")
    p.add_argument("--max-heuristic-launch-latency-us", type=float, default=100_000.0,
                   help="Only used internally for heuristic launch matching when correlation ids are absent")
    p.add_argument("--png-max-lanes", type=int, default=48, help="Max lanes in PNG timeline")
    p.add_argument("--png-min-dur-us", type=float, default=0.0, help="Hide shorter events in PNG")
    p.add_argument("--inspect", action="store_true", help="Print schema/category/lane/arg-key inspection")
    p.add_argument("--dump-regex", type=str, default=None, help="Print first matching normalized events for debugging")
    p.add_argument("--dump-limit", type=int, default=50, help="Limit for --dump-regex")
    p.add_argument("--select", type=int, default=None, help="Print details and same-lane context for a normalized event id")
    p.add_argument("--recipe", action="append", default=[], help="Run a built-in SQL recipe after creating a SQLite DB. Repeatable.")
    p.add_argument("--recipes", action="store_true", help="List available SQL recipes and exit")
    p.add_argument("--sql", action="append", default=[], help="Run ad-hoc SQL against the generated SQLite DB. Repeatable.")
    p.add_argument("--sql-file", type=Path, action="append", default=[], help="Run SQL file against the generated SQLite DB. Repeatable.")
    p.add_argument("--field-guide", action="store_true", help="Print Perfetto-for-agents field guide and exit")
    p.add_argument("--full", action="store_true",
                   help="Disable compact scanning and materialize every event")
    p.add_argument("--require-gpu-util", type=float, default=None,
                   help="CI/agent guard: exit 2 if any detected GPU has utilization below this percent")
    p.add_argument("--max-idle-gap-ms", type=float, default=None,
                   help="CI/agent guard: exit 2 if any reported GPU idle gap exceeds this many ms")
    p.add_argument("--fail-on-high-finding", action="store_true", help="CI/agent guard: exit 2 if any high-severity finding is generated")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = make_parser().parse_args(argv)
    if args.recipes:
        print_recipes()
        return 0
    if args.field_guide:
        print_field_guide()
        return 0
    if args.trace is None:
        print("error: trace path is required unless --recipes or --field-guide is used", file=sys.stderr)
        return 2
    modes = args.mode or ["text"]
    deep_modes = {"csv", "png", "sqlite", "md", "html"}
    deep_options = (
        args.zoom or args.query_regex or args.inspect or args.dump_regex
        or args.select is not None or args.recipe or args.sql or args.sql_file
        # The fast path never produces findings, so this guard would always pass.
        or args.fail_on_high_finding
    )
    use_fast_scan = (
        not args.full
        and args.trace.suffix != ".gz"
        and args.trace.stat().st_size >= _FAST_SCAN_MIN_BYTES
        and not deep_modes.intersection(modes)
        and not deep_options
    )
    if use_fast_scan:
        report = build_fast_report(args.trace, args.top, args.gap_threshold_us)
        if "text" in modes or "agent" in modes:
            render_text(report)
        if "agent" in modes:
            print(
                "\nFast summary omits detailed findings, flow correlation, counters, "
                "and CPU operator attribution; rerun a focused window with --full."
            )
        if "json" in modes:
            full_report = dict(report)
            full_report["agent"] = {
                "analysis_mode": "fast_summary",
                "findings": [],
                "bookmarks": [],
                "limitations": report["debug"]["fast_summary_limitations"],
            }
            out = args.out.with_suffix(".json")
            write_json(full_report, out)
            print(f"\nWrote JSON report: {out}", file=sys.stderr)

        failed = False
        if args.require_gpu_util is not None:
            if not report["gpu_utilization"]:
                print(
                    "FAIL: no GPU activity found, so --require-gpu-util cannot be checked",
                    file=sys.stderr,
                )
                failed = True
            for dev, utilization in report["gpu_utilization"].items():
                if utilization["util_percent"] < args.require_gpu_util:
                    print(
                        f"FAIL: GPU{dev} utilization {utilization['util_percent']:.2f}% "
                        f"< required {args.require_gpu_util:.2f}%",
                        file=sys.stderr,
                    )
                    failed = True
        if args.max_idle_gap_ms is not None:
            for dev, gaps in report["gpu_idle_gaps"].items():
                if gaps and gaps[0]["dur_ms"] > args.max_idle_gap_ms:
                    print(
                        f"FAIL: GPU{dev} idle gap {gaps[0]['dur_ms']:.3f} ms "
                        f"> max {args.max_idle_gap_ms:.3f} ms",
                        file=sys.stderr,
                    )
                    failed = True
        return 2 if failed else 0

    raw_events = load_raw_events(args.trace)
    data = normalize_events(raw_events)

    start = end = None
    if args.zoom:
        z0, z1 = args.zoom
        start = data.trace_start_us + z0 if z0 is not None else None
        end = data.trace_start_us + z1 if z1 is not None else None

    regex = re.compile(args.query_regex, re.I) if args.query_regex else None
    if args.inspect:
        inspect_trace(data, args.top)
    if args.dump_regex:
        dump_matching_events(data.events, re.compile(args.dump_regex, re.I), args.dump_limit)
    selected = filtered_events(data, start, end, regex)
    report = build_report(data, start, end, regex, args.top, args.gap_threshold_us)
    artifacts = build_agent_artifacts(raw_events, data, report, selected, start, end)
    full_report = agent_report_dict(report, artifacts)

    if args.select is not None:
        print(select_event_text(selected if selected else data.events, args.select))

    db_path = args.out.with_suffix(".sqlite")
    needs_sqlite = "sqlite" in modes or args.recipe or args.sql or args.sql_file
    if needs_sqlite:
        create_sqlite(db_path, data, selected, artifacts)
        print(f"Wrote SQLite database: {db_path}", file=sys.stderr)

    if "text" in modes:
        render_text(report)
    if "agent" in modes:
        render_agent_text(report, artifacts, args.top)
    if "json" in modes:
        out = args.out.with_suffix(".json")
        write_json(full_report, out)
        print(f"\nWrote JSON report: {out}", file=sys.stderr)
    if "csv" in modes:
        write_csvs(data, report, args.out, selected)
        write_agent_csvs(args.out, artifacts)
        print(f"Wrote CSV files with prefix: {args.out}", file=sys.stderr)
    if "png" in modes:
        suffix = ".timeline.png"
        if args.zoom:
            suffix = ".timeline.zoom.png"
        out = args.out.with_suffix(suffix)
        plot_timeline(selected, data, out, start, end, args.png_max_lanes, args.png_min_dur_us)
        print(f"Wrote PNG timeline: {out}", file=sys.stderr)
    if "md" in modes:
        out = args.out.with_suffix(".md")
        write_markdown(out, full_report, artifacts, args.top)
        print(f"Wrote Markdown report: {out}", file=sys.stderr)
    if "html" in modes:
        out = args.out.with_suffix(".html")
        write_html(out, full_report, artifacts, args.top)
        print(f"Wrote HTML report: {out}", file=sys.stderr)
    if args.recipe:
        run_recipes(db_path, args.recipe, args.top)
    for sql in args.sql:
        rows = sqlite_query_rows(db_path, sql, limit=args.top)
        print_sql_rows("Ad-hoc SQL", rows, args.top)
    for sql_file in args.sql_file:
        sql = sql_file.read_text(encoding="utf-8")
        rows = sqlite_query_rows(db_path, sql, limit=args.top)
        print_sql_rows(f"SQL file: {sql_file}", rows, args.top)

    failed = False
    if args.require_gpu_util is not None:
        if not report.get("gpu_utilization"):
            print("FAIL: no GPU activity found, so --require-gpu-util cannot be checked", file=sys.stderr)
            failed = True
        for dev, u in report.get("gpu_utilization", {}).items():
            if u.get("util_percent", 0.0) < args.require_gpu_util:
                print(f"FAIL: GPU{dev} utilization {u.get('util_percent', 0.0):.2f}% < required {args.require_gpu_util:.2f}%", file=sys.stderr)
                failed = True
    if args.max_idle_gap_ms is not None:
        for dev, gaps in report.get("gpu_idle_gaps", {}).items():
            for g in gaps:
                if g.get("dur_ms", 0.0) > args.max_idle_gap_ms:
                    print(f"FAIL: GPU{dev} idle gap {g.get('dur_ms', 0.0):.3f} ms > max {args.max_idle_gap_ms:.3f} ms", file=sys.stderr)
                    failed = True
                    break
    if args.fail_on_high_finding and any(f.severity == "high" for f in artifacts.findings):
        print("FAIL: high-severity finding generated", file=sys.stderr)
        failed = True
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
