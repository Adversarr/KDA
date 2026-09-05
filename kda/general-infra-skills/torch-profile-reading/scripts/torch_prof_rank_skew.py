"""Cross-rank NCCL arrival-skew analysis for Kineto JSON traces."""

from __future__ import annotations

import argparse
import json
import mmap
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class TraceEvent:
    """One matched trace event."""

    name: str
    ts_us: float
    dur_us: float


@dataclass(frozen=True)
class SkewRow:
    """Arrival-skew summary for one aligned collective ordinal."""

    index: int
    arrival_skew_ms: float
    latest_rank: str
    latest_duration_ms: float
    max_duration_ms: float
    completion_tail_ms: float


def _line_bounds(data: mmap.mmap, position: int) -> tuple[int, int]:
    """Return the current line's start and end offsets."""

    start = data.rfind(b"\n", 0, position) + 1
    end = data.find(b"\n", position)
    return start, len(data) if end < 0 else end


def _parse_number(line: bytes, key: bytes) -> float | None:
    """Parse one numeric JSON field from a trace line."""

    start = line.find(key)
    if start < 0:
        return None
    start += len(key)
    end = line.find(b",", start)
    if end < 0:
        end = line.find(b"}", start)
    if end < 0:
        return None
    return float(line[start:end])


def _parse_name(line: bytes) -> str | None:
    """Parse an event name without decoding the full JSON document."""

    key = b'"name": "'
    start = line.find(key)
    if start < 0:
        return None
    start += len(key)
    end = line.rfind(b'", "pid": ', start)
    if end < start:
        return None
    return line[start:end].decode("utf-8", errors="replace")


def scan_kernel_events(path: Path, pattern: re.Pattern[str]) -> list[TraceEvent]:
    """Scan matching GPU kernels with bounded host memory.

    Args:
        path: Uncompressed Kineto JSON trace.
        pattern: Kernel-name pattern to retain.

    Returns:
        Matching events sorted by timestamp.

    Raises:
        ValueError: The trace is compressed or contains no matching events.
    """

    if path.suffix == ".gz":
        raise ValueError(f"{path}: rank-skew scanning requires uncompressed JSON")
    events: list[TraceEvent] = []
    needle = b'"cat": "kernel"'
    with path.open("rb") as file, mmap.mmap(
        file.fileno(), 0, access=mmap.ACCESS_READ
    ) as data:
        position = 0
        while True:
            position = data.find(needle, position)
            if position < 0:
                break
            line_start, line_end = _line_bounds(data, position)
            next_end = data.find(b"\n", line_end + 1)
            if next_end < 0:
                next_end = len(data)
            event_line = data[line_start:line_end]
            fields_line = data[line_end + 1 : next_end]
            name = _parse_name(event_line)
            ts_us = _parse_number(fields_line, b'"ts": ')
            dur_us = _parse_number(fields_line, b'"dur": ')
            if (
                name is not None
                and ts_us is not None
                and dur_us is not None
                and pattern.search(name)
            ):
                events.append(TraceEvent(name, ts_us, dur_us))
            position = line_end
    if not events:
        raise ValueError(f"{path}: no kernel names match {pattern.pattern!r}")
    events.sort(key=lambda event: event.ts_us)
    return events


def scan_cpu_context(
    path: Path,
    start_us: float,
    end_us: float,
    *,
    top: int,
    lookback_us: float,
) -> list[TraceEvent]:
    """Find CPU spans that can explain one late-rank arrival window.

    Args:
        path: Straggler-rank trace.
        start_us: Earliest collective arrival across ranks.
        end_us: Latest collective arrival across ranks.
        top: Maximum number of spans to return.
        lookback_us: Time before the earliest arrival to include.

    Returns:
        Longest overlapping CPU spans, excluding unhelpful trace-wide parents.
    """

    categories = (
        b'"cat": "python_function"',
        b'"cat": "cpu_op"',
        b'"cat": "cuda_runtime"',
        b'"cat": "user_annotation"',
    )
    events: list[TraceEvent] = []
    window_us = max(end_us - start_us, 1.0)
    max_parent_us = max(window_us * 10.0, 1_000_000.0)
    scan_start_us = start_us - lookback_us
    with path.open("rb") as file, mmap.mmap(
        file.fileno(), 0, access=mmap.ACCESS_READ
    ) as data:
        for needle in categories:
            position = 0
            while True:
                position = data.find(needle, position)
                if position < 0:
                    break
                line_start, line_end = _line_bounds(data, position)
                next_end = data.find(b"\n", line_end + 1)
                if next_end < 0:
                    next_end = len(data)
                event_line = data[line_start:line_end]
                fields_line = data[line_end + 1 : next_end]
                name = _parse_name(event_line)
                ts_us = _parse_number(fields_line, b'"ts": ')
                dur_us = _parse_number(fields_line, b'"dur": ')
                if (
                    name is not None
                    and ts_us is not None
                    and dur_us is not None
                    and dur_us <= max_parent_us
                    and ts_us < end_us
                    and ts_us + dur_us > scan_start_us
                ):
                    events.append(TraceEvent(name, ts_us, dur_us))
                position = line_end
    events.sort(key=lambda event: event.dur_us, reverse=True)
    sync_events = [
        event
        for event in events
        if "synchronize" in event.name.lower()
        or "cudaeventquery" in event.name.lower()
    ][: max(1, top // 3)]
    sync_ids = {id(event) for event in sync_events}
    general_events = [
        event for event in events if id(event) not in sync_ids
    ][: top - len(sync_events)]
    return sorted(
        general_events + sync_events,
        key=lambda event: event.dur_us,
        reverse=True,
    )


def compare_collectives(
    rank_events: dict[str, list[TraceEvent]],
) -> list[SkewRow]:
    """Align matched kernels by ordinal and calculate rank arrival skew.

    Args:
        rank_events: Matching kernels keyed by rank label.

    Returns:
        One row per ordinal available in every trace.
    """

    shared_count = min(len(events) for events in rank_events.values())
    rows = []
    for index in range(shared_count):
        aligned = {
            rank: events[index] for rank, events in rank_events.items()
        }
        latest_rank, latest = max(
            aligned.items(), key=lambda item: item[1].ts_us
        )
        earliest_us = min(event.ts_us for event in aligned.values())
        latest_us = latest.ts_us
        completion_us = max(
            event.ts_us + event.dur_us for event in aligned.values()
        )
        rows.append(
            SkewRow(
                index=index,
                arrival_skew_ms=(latest_us - earliest_us) / 1000.0,
                latest_rank=latest_rank,
                latest_duration_ms=latest.dur_us / 1000.0,
                max_duration_ms=max(
                    event.dur_us for event in aligned.values()
                )
                / 1000.0,
                completion_tail_ms=(completion_us - latest_us) / 1000.0,
            )
        )
    return rows


def _rank_labels(paths: Sequence[Path]) -> dict[str, Path]:
    """Assign stable rank labels from filenames or argument order."""

    labels: dict[str, Path] = {}
    for index, path in enumerate(paths):
        match = re.search(r"rank[_-]?(\d+)", path.name, re.IGNORECASE)
        label = f"rank{match.group(1)}" if match else f"rank{index}"
        if label in labels:
            raise ValueError(f"duplicate inferred rank label {label!r}")
        labels[label] = path
    return labels


def _print_rows(rows: Sequence[SkewRow]) -> None:
    """Print aligned collective skew rows."""

    print(
        f"{'index':>7} {'arrival_ms':>11} {'latest':>8} "
        f"{'late_dur_ms':>12} {'max_dur_ms':>11} {'tail_ms':>9}"
    )
    for row in rows:
        print(
            f"{row.index:7d} {row.arrival_skew_ms:11.3f} "
            f"{row.latest_rank:>8} {row.latest_duration_ms:12.3f} "
            f"{row.max_duration_ms:11.3f} {row.completion_tail_ms:9.3f}"
        )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Compare aligned NCCL kernel arrival times across rank traces."
    )
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument(
        "--kernel-regex",
        default=r"ncclDevKernel_SendRecv",
        help="Kernel-name regex used for ordinal alignment.",
    )
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument(
        "--inspect-index",
        type=int,
        help="Show rank details and straggler CPU context for one ordinal.",
    )
    parser.add_argument("--context-top", type=int, default=30)
    parser.add_argument(
        "--lookback-ms",
        type=float,
        default=2000.0,
        help="CPU context to include before the earliest rank arrival.",
    )
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def main() -> int:
    """Compare rank traces and report collective arrival skew."""

    args = parse_args()
    if len(args.traces) < 2:
        raise ValueError("rank-skew analysis requires at least two traces")
    paths = _rank_labels(args.traces)
    pattern = re.compile(args.kernel_regex, re.IGNORECASE)
    rank_events = {
        rank: scan_kernel_events(path, pattern)
        for rank, path in paths.items()
    }
    counts = {rank: len(events) for rank, events in rank_events.items()}
    count_summary = ", ".join(
        f"{rank}={count}" for rank, count in counts.items()
    )
    print("matched kernels:", count_summary)
    if len(set(counts.values())) != 1:
        print(
            "WARNING: counts differ; ordinal alignment is valid only before "
            "the first missing or extra matching kernel."
        )

    rows = compare_collectives(rank_events)
    worst = sorted(rows, key=lambda row: row.arrival_skew_ms, reverse=True)[
        : args.top
    ]
    _print_rows(worst)

    context = []
    if args.inspect_index is not None:
        index = args.inspect_index
        if index < 0 or index >= len(rows):
            raise ValueError(
                f"inspect index {index} is outside shared range [0, {len(rows)})"
            )
        aligned = {
            rank: events[index] for rank, events in rank_events.items()
        }
        print(f"\ncollective index {index}:")
        for rank, event in aligned.items():
            print(
                f"  {rank}: start_ms={event.ts_us / 1000.0:.3f} "
                f"dur_ms={event.dur_us / 1000.0:.3f}"
            )
        latest_rank = max(
            aligned, key=lambda rank: aligned[rank].ts_us
        )
        start_us = min(event.ts_us for event in aligned.values())
        end_us = aligned[latest_rank].ts_us
        context = scan_cpu_context(
            paths[latest_rank],
            start_us,
            end_us,
            top=args.context_top,
            lookback_us=args.lookback_ms * 1000.0,
        )
        print(f"\n{latest_rank} CPU context during arrival gap:")
        print("  (negative at means the span began before the earliest arrival)")
        for event in context:
            relative_ms = (event.ts_us - start_us) / 1000.0
            print(
                f"  at={relative_ms:9.3f}ms dur={event.dur_us / 1000.0:9.3f}ms "
                f"{event.name}"
            )

    if args.json_out is not None:
        payload = {
            "kernel_regex": args.kernel_regex,
            "counts": counts,
            "worst": [asdict(row) for row in worst],
            "inspect_index": args.inspect_index,
            "context": [asdict(event) for event in context],
        }
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
