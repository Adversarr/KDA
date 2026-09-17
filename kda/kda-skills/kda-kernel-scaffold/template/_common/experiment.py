"""Unprofiled paired experiments and explicit comparison summaries."""

import math
import statistics
import time


def summarize_ab(records, *, repeats=5):
    """Summarize every retained attempt without treating sample count as significance."""
    measured = [r for r in records if r.get("warmup") is False]
    def valid(row):
        return (row.get("status") == "success" and isinstance(row.get("wall_ms"), (int, float))
                and math.isfinite(row["wall_ms"]) and row["wall_ms"] > 0
                and all(isinstance(row.get(k), (int, float)) and math.isfinite(row[k]) for k in ("started_at", "finished_at"))
                and row["started_at"] <= row["finished_at"]
                and row.get("outputs", {}).get("passed") is True)
    values = {name: [r["wall_ms"] for r in measured if r.get("variant") == name and valid(r)]
              for name in ("current", "candidate")}
    expected_order = [name for index in range(repeats) for name in
                      (("current", "candidate") if index % 2 == 0 else ("candidate", "current"))]
    warmed = {r.get("variant") for r in records[:len(records)-len(measured)] if r.get("warmup") is True}
    complete = (repeats >= 1 and len(measured) == 2 * repeats and all(valid(r) for r in records)
                and warmed == {"current", "candidate"}
                and [r.get("variant") for r in measured] == expected_order
                and [r.get("pair") for r in measured] == [i for i in range(repeats) for _ in range(2)]
                and [r.get("order") for r in records] == list(range(len(records)))
                and all(a["finished_at"] <= b["started_at"] for a, b in zip(records, records[1:]))
                and all(len(v) == repeats for v in values.values()))
    summary = {"repeats": repeats, "measurement_valid": complete, "statistical_significance": "not_established",
               "metric": "request_wall_ms", "samples": records, "variants": {}}
    for name, samples in values.items():
        if samples:
            median = statistics.median(samples)
            summary["variants"][name] = {"n": len(samples), "median_ms": median,
                "min_ms": min(samples), "max_ms": max(samples),
                "mad_ms": statistics.median(abs(v - median) for v in samples)}
    if complete:
        current = summary["variants"]["current"]["median_ms"]
        candidate = summary["variants"]["candidate"]["median_ms"]
        summary["deployment_delta_ms"] = current - candidate
        summary["deployment_reduction"] = (current - candidate) / current if current else None
        pairs = []
        for index in range(repeats):
            pair = {r["variant"]: r["wall_ms"] for r in measured if r["pair"] == index}
            if set(pair) != {"current", "candidate"}:
                summary["measurement_valid"] = False
                break
            pairs.append(pair["current"] - pair["candidate"])
        summary["paired_savings_ms"] = pairs
    return summary


def run_ab(current, candidate, inspect_output, *, warmup=1, repeats=5, invocation=None):
    """Run balanced A/B calls and inspect every output, including warmups.

    Args:
        current: Synchronous original request callable, including completion waits.
        candidate: Synchronous candidate request callable with equivalent boundaries.
        inspect_output: Return a mapping containing ``passed`` and saved-output evidence.
        warmup: Warmup calls for each variant, excluded from timing summaries.
        repeats: Number of balanced measurement pairs.
        invocation: Immutable source, input, configuration and command identifiers.

    Returns:
        Raw attempts and a descriptive deployment comparison, including failures.
    """
    if warmup < 0 or repeats < 1:
        raise ValueError("invalid experiment counts")
    records = []
    functions = {"current": current, "candidate": candidate}
    schedule = [(True, i, name) for i in range(warmup) for name in functions]
    schedule += [(False, i, name) for i in range(repeats)
                 for name in (("current", "candidate") if i % 2 == 0 else ("candidate", "current"))]
    for is_warmup, pair, name in schedule:
        row = {"variant": name, "warmup": is_warmup, "pair": pair, "order": len(records),
               "invocation": invocation, "started_at": time.time()}
        start = time.perf_counter()
        try:
            output = functions[name]()
            row["wall_ms"] = (time.perf_counter() - start) * 1000
            row["finished_at"] = time.time()
            row["outputs"] = inspect_output(output)
            if not isinstance(row["outputs"], dict) or row["outputs"].get("passed") is not True:
                raise ValueError("output validation did not pass")
            if not math.isfinite(row["wall_ms"]) or row["wall_ms"] <= 0:
                raise ValueError("invalid wall timing")
            row["status"] = "success"
        except Exception as exc:
            row.setdefault("finished_at", time.time())
            row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        records.append(row)
    return summarize_ab(records, repeats=repeats)
