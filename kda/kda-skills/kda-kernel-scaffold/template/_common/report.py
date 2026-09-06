"""Verify/bench results, the M3/M4 verdict rule, and the ``report.json`` / ``REPORT.md`` writers.

Phases per workload: ``fwd`` (training forward, aux written), ``bwd``, ``infer`` (forward under
``no_grad``, nothing saved) and, for ops whose SPEC declares ``recompute.available``,
``bwd_recompute`` (training backward that recomputes the aux). Verdict rule, per required
workload and phase:

* required missing/skipped evidence -> ``incomplete``; optional skips are notes
* any numerical failure or runtime error -> ``fail``
* any failed contract row (a bad input accepted, or rejected by a CUDA error instead of a
  Python one) -> ``fail``
* the ``torch.compile(fullgraph=True)`` probe (custom-op torch only, one workload) raising or
  disagreeing with the reference -> ``fail`` ("not compile-safe": a fake that does not mirror
  the launcher, a graph break, a stride guard)
* SOL efficiency below ``SOL_THRESHOLD`` or slower than the best baseline by more than
  ``BASELINE_TOLERANCE`` -> ``tune``
* **latency floor**: when the achievable roof of a phase is under ``LATENCY_FLOOR_MS`` (a
  same-size copy takes less than ~10 us, i.e. a few MB of traffic) the phase is launch- and
  latency-bound: its time is set by kernel count and autograd overhead, not bandwidth, so the
  SOL efficiency is *reported but not gated* there and only the baseline comparison counts.
* ``bwd_recompute`` numerics always count; its performance gates only when SPEC
  ``recompute.default`` is true (``gate_recompute``)
* a kernel timed **below the datasheet roof** ``sol_ms`` by more than ``ROOF_SLACK`` cannot
  exist (a same-size copy never reaches the datasheet bandwidth; no GEMM beats the tensor-core
  peak), so the roofline is miscounted: ``reasons`` says so ("roofline miscounted?") and the
  verifier audits ``_speed_of_light.py`` before reading any ``sol_eff``. The verdict is not
  changed by it.
* otherwise -> ``pass``

SOL efficiency is ``roof_ms / kernel_ms`` where ``roof_ms`` is the *achievable* roof: a
same-size device copy or, on tensor cores, cuBLAS on the phase's GEMMs (see `bench.copy_ms`,
`bench.matmul_ms`; the datasheet peak is reached by neither a copy nor a GEMM), whichever is slower. The
datasheet number ``sol_ms`` is reported alongside. Missing peaks or a failed
``torch.compile`` baseline weaken the check rather than block it; both are recorded in
``reasons`` so the gate stays honest. The baseline comparison has a parity band: a kernel
within ``BASELINE_TOLERANCE`` of the best baseline (default 5%) is noted as "parity" and does
not gate, because the harness's own run-to-run noise on a few-us phase is of that order; on a
latency-bound phase the band is also ``LATENCY_PARITY_MS`` absolute (1 us), since 5% of a 4 us
kernel is below the profiler's jitter.
"""

from __future__ import annotations

import json

from .evidence import coverage_errors
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

SOL_THRESHOLD = 0.7
# Roof below this is latency-bound (a same-size copy under ~10 us): SOL is reported, not gated.
LATENCY_FLOOR_MS = 0.010
# Kernel within this fraction of the best baseline counts as parity, not "slower".
BASELINE_TOLERANCE = 0.05
# A gate missed by less than this (SOL 0.65-0.70, speedup 0.90-0.95x) is within a shared card's
# run-to-run spread; state this on the reason line so it is visible before the verifier's notes.
NEAR_GATE_BAND = 0.05
NEAR_GATE_NOTE = " (near gate: within the noise band of a shared card; re-run `--bench --workload <row>` twice into diagnostic paths; conflicting gates are incomplete, verify skill s4)"
# ... or, on a latency-bound phase, within this absolute gap: at a 3-4 us kernel the profiler's
# own jitter is a few tenths of a microsecond, which 5% cannot cover.
LATENCY_PARITY_MS = 0.001
# A kernel faster than the datasheet SOL by more than this fraction means the roofline
# (bytes or FLOPs) is miscounted; flagged in `reasons`, never a `pass`-changing rule.
ROOF_SLACK = 0.05
# Phases that always gate; RECOMPUTE_PHASE joins them when the SPEC default is recompute.
PHASES = ("fwd", "bwd", "infer")
RECOMPUTE_PHASE = "bwd_recompute"
ALL_PHASES = (*PHASES, RECOMPUTE_PHASE)
# Baselines have no recompute variant: the eager/compiled backward is the reference for both.
_BASELINE_PHASE = {RECOMPUTE_PHASE: "bwd"}


@dataclass
class WorkloadResult:
    name: str
    required: bool
    dtype: str
    # {output_name: CompareResult.as_dict()} per phase; None when the phase was not run.
    fwd: Optional[Dict[str, dict]] = None
    bwd: Optional[Dict[str, dict]] = None
    infer: Optional[Dict[str, dict]] = None
    bwd_recompute: Optional[Dict[str, dict]] = None
    # keys: "<baseline>_<phase>" e.g. eager_fwd, compiled_fwd, kernel_fwd, kernel_infer, kernel_bwd_recompute
    time_ms: Dict[str, float] = field(default_factory=dict)
    sol_ms: Dict[str, Optional[float]] = field(default_factory=dict)  # per phase, datasheet peaks
    roof_ms: Dict[str, Optional[float]] = field(default_factory=dict)  # per phase, achievable (copy-calibrated)
    roof_details: Dict[str, dict] = field(default_factory=dict)  # calibration provenance per phase
    error: Optional[str] = None
    # "needs X GiB, free Y GiB": required skips are incomplete; optional skips are notes.
    skipped: Optional[str] = None
    # Non-fatal problems (a baseline that failed to build or time); listed in ``reasons``, never
    # part of the verdict.
    warnings: List[str] = field(default_factory=list)
    # torch.compile(fullgraph=True) probe of the backend on this workload: outputs and gradients
    # compared with the reference; None when not run (eager backend, torch < 2.6, other workloads).
    compile: Optional[Dict[str, dict]] = None
    compile_error: Optional[str] = None
    # Contract probe rows {mutation: {"expect": "raise"|"return", "passed": bool, "detail": str}};
    # None when not probed on this workload.
    contract: Optional[Dict[str, dict]] = None

    def numerics_passed(self, phase: str) -> Optional[bool]:
        results = getattr(self, phase)
        if results is None:
            return None
        return all(r["passed"] for r in results.values())

    def contract_passed(self) -> Optional[bool]:
        if self.contract is None:
            return None
        return all(r["passed"] for r in self.contract.values())

    def sol_eff(self, phase: str) -> Optional[float]:
        """Achievable-roof efficiency; falls back to the datasheet SOL when no roof was measured."""
        details = self.roof_details.get(phase, {})
        if details.get("method") == "attention" and (not details.get("available", False) or self.roof_ms.get(phase) is None):
            return None
        roof = self.roof_ms.get(phase)
        if roof is None:
            roof = self.sol_ms.get(phase)
        actual = self.time_ms.get(f"kernel_{phase}")
        if roof is None or not actual:
            return None
        return roof / actual

    def latency_bound(self, phase: str) -> bool:
        """True when the measured achievable roof is under ``LATENCY_FLOOR_MS`` (see module doc)."""
        roof = self.roof_ms.get(phase)
        return roof is not None and roof < LATENCY_FLOOR_MS

    def below_datasheet_roof(self, phase: str) -> bool:
        """True when the kernel beat the datasheet SOL by more than ``ROOF_SLACK``: the roofline is miscounted."""
        sol = self.sol_ms.get(phase)
        actual = self.time_ms.get(f"kernel_{phase}")
        return bool(sol) and bool(actual) and actual < sol * (1.0 - ROOF_SLACK)

    def baseline_ms(self, phase: str) -> Optional[float]:
        """Fastest available reference: torch.compile if it built, else eager."""
        base_phase = _BASELINE_PHASE.get(phase, phase)
        candidates = [self.time_ms.get(f"{b}_{base_phase}") for b in ("compiled", "eager")]
        candidates = [c for c in candidates if c]
        return min(candidates) if candidates else None

    def speedup(self, phase: str) -> Optional[float]:
        base, actual = self.baseline_ms(phase), self.time_ms.get(f"kernel_{phase}")
        return base / actual if base and actual else None

    def as_dict(self) -> dict:
        d = asdict(self)
        d["derived"] = {
            phase: {"sol_eff": self.sol_eff(phase), "speedup": self.speedup(phase)}
            for phase in ALL_PHASES
            if f"kernel_{phase}" in self.time_ms
        }
        return d


def verdict(
    results: List[WorkloadResult],
    *,
    sol_threshold: float = SOL_THRESHOLD,
    gate_recompute: bool = False,
    gate_perf: bool = True,
    coverage: Optional[dict] = None,
) -> Tuple[str, List[str]]:
    """Return ``(verdict, reasons)`` over the required workloads.

    ``gate_recompute``: SPEC ``recompute.default``; when true the ``bwd_recompute`` phase gates
    performance like the others (its numerics always do).

    ``gate_perf``: false for a report timed with CUDA events (``env.bench_method == "events"``),
    which includes host launch overhead that no kernel edit can change; the SOL and baseline
    rules are then skipped and only numerics, compile and contract decide, so a diagnostic side
    report cannot say ``tune`` against a profiler-timed report of record that says ``pass``.
    """
    reasons: List[str] = []
    missing = coverage_errors([r.as_dict() for r in results], coverage)
    reasons.extend(missing)
    level = 2 if missing else 0  # pass < tune < incomplete < fail
    if not gate_perf:
        reasons.append("performance not gated: timed with CUDA events (host launch overhead included); the profiler-timed report.json is the record")
    for r in results:
        if r.skipped:
            reasons.append(f"{r.name}: skipped: {r.skipped}")
        if not r.required:
            continue
        if r.error:
            reasons.append(f"{r.name}: error: {r.error}")
            level = 3
            continue
        reasons += [f"{r.name}: warning: {w}" for w in r.warnings]
        if r.compile_error:
            reasons.append(f"{r.name}: not compile-safe: {r.compile_error}")
            level = 3
        elif r.numerics_passed("compile") is False:
            reasons.append(f"{r.name}: compiled numerics mismatch")
            level = 3
        if r.contract_passed() is False:
            failed = [f"{k} ({v['detail']})" for k, v in r.contract.items() if not v["passed"]]
            reasons.append(f"{r.name}: contract: " + "; ".join(failed))
            level = 3
        for phase in ALL_PHASES:
            ok = r.numerics_passed(phase)
            if ok is False:
                reasons.append(f"{r.name}/{phase}: numerical mismatch")
                level = 3
            if f"kernel_{phase}" not in r.time_ms or not gate_perf:
                continue
            gates = phase != RECOMPUTE_PHASE or gate_recompute
            tune_level = 1 if gates else 0
            eff = r.sol_eff(phase)
            if eff is not None and r.below_datasheet_roof(phase):
                reasons.append(
                    f"{r.name}/{phase}: kernel {r.time_ms[f'kernel_{phase}']:.3f} ms is below the datasheet roof "
                    f"{r.sol_ms[phase]:.3f} ms (SOL efficiency {eff:.2f}): roofline miscounted? audit _speed_of_light.py "
                    "(bytes once per tensor; whole GEMMs: 1 fwd, 2 bwd, 3 bwd_recompute) before reading sol_eff"
                )
            if r.roof_details.get(phase, {}).get("method") == "attention" and eff is None:
                detail = r.roof_details[phase]
                reasons.append(f"{r.name}/{phase}: attention calibration unavailable: {detail.get('reason', 'missing calibration')}")
                if gates:
                    level = max(level, 2)
            if eff is None:
                reasons.append(f"{r.name}/{phase}: SOL unavailable (peaks unknown or not benchmarked)")
            elif eff < sol_threshold:
                if r.latency_bound(phase):
                    roof_us = 1e3 * r.roof_ms[phase]
                    reasons.append(f"{r.name}/{phase}: SOL efficiency {eff:.2f} not gated: latency-bound (roof {roof_us:.1f} us < {1e3 * LATENCY_FLOOR_MS:.0f} us)")
                else:
                    near = NEAR_GATE_NOTE if eff >= sol_threshold - NEAR_GATE_BAND else ""
                    reasons.append(f"{r.name}/{phase}: SOL efficiency {eff:.2f} < {sol_threshold}" + ("" if gates else " (not gated)") + near)
                    level = max(level, tune_level)
            speedup = r.speedup(phase)
            if speedup is None:
                reasons.append(f"{r.name}/{phase}: no baseline timing")
                continue
            gap_ms = r.time_ms[f"kernel_{phase}"] - r.baseline_ms(phase)
            within_latency_band = r.latency_bound(phase) and gap_ms <= LATENCY_PARITY_MS
            if speedup < 1.0 - BASELINE_TOLERANCE and not within_latency_band:
                near = NEAR_GATE_NOTE if speedup >= 1.0 - BASELINE_TOLERANCE - NEAR_GATE_BAND else ""
                reasons.append(f"{r.name}/{phase}: slower than baseline ({speedup:.2f}x)" + ("" if gates else " (not gated)") + near)
                level = max(level, tune_level)
            elif speedup < 1.0:
                band = f"within {BASELINE_TOLERANCE:.0%}" if speedup >= 1.0 - BASELINE_TOLERANCE else f"latency-bound, gap {1e3 * gap_ms:.2f} us <= {1e3 * LATENCY_PARITY_MS:.0f} us"
                reasons.append(f"{r.name}/{phase}: parity with baseline ({speedup:.2f}x, {band}; not gated)")
    return ("pass", "tune", "incomplete", "fail")[level], reasons


def _fmt(x: Optional[float], spec: str = ".3f") -> str:
    return "n/a" if x is None else format(x, spec)


def _ok_cell(results: Optional[Dict[str, dict]]) -> str:
    if results is None:
        return "-"
    return "pass" if all(v["passed"] for v in results.values()) else "FAIL"


def render_markdown(report: Dict[str, Any]) -> str:
    env = report["env"]
    verification = report.get("verification")
    decision = verification["verdict"] if verification else report["verdict"]
    with_rc = any(f"kernel_{RECOMPUTE_PHASE}" in r["time_ms"] or r.get(RECOMPUTE_PHASE) for r in report["workloads"])
    phases = list(ALL_PHASES) if with_rc else list(PHASES)
    short = {"fwd": "fwd", "bwd": "bwd", "infer": "infer", RECOMPUTE_PHASE: "bwd_rc"}
    head = ["workload", "req", "dtype"] + [f"{short[p]} ok" for p in phases] + ["compile", "contract"]
    for p in phases:
        head += [f"kernel {short[p]} ms", f"base {short[p]} ms", f"roof {short[p]} ms", f"SOL eff {short[p]}"]
    diagnostic = report.get("gate_perf") is False  # explicit --method events side report
    lines = [
        f"# Report: `{report['op']}` ({report['backend']})" + (" - events-timed diagnostic, not the record" if diagnostic else ""),
        "",
        f"- Verdict: **{decision}**"
        + (f" (raw harness: {report['verdict']})" if verification else " (raw harness; verification unfinished)")
        + (f" (iteration {report['iteration']})" if report.get("iteration") else "")
        + (" - numerics, compile and contract only; performance not gated" if diagnostic else ""),
        f"- GPU: {env.get('gpu')} (cc {env.get('cc')}) | torch {env.get('torch')} | {report.get('kernel_backend') or 'kernel'} {env.get('kernel_backend_version') or env.get('triton')}",
        f"- Generated: {report['timestamp']}" + (f" | timing: {env['bench_method']}" if env.get("bench_method") else ""),
    ]
    if diagnostic:
        lines.append(
            "- Times below are CUDA-event wall time per call: device kernels plus host launch overhead (Python, dispatch, autograd). "
            "Subtract the profiler-timed `report.json` per phase to get the host overhead; SOL eff here is not comparable with the roofs."
        )
    lines += [
        "- SOL eff = achievable roof (same-size copy, or cuBLAS on the same GEMMs for tensor-core ops) / kernel time; base = fastest of eager and torch.compile"
        + ("; bwd_rc = backward with recompute (baseline: the eager/compiled backward)" if with_rc else "")
        + f"; a roof under {1e3 * LATENCY_FLOOR_MS:.0f} us is latency-bound (SOL eff shown, not gated); within {BASELINE_TOLERANCE:.0%} of base is parity",
        "",
        "| " + " | ".join(head) + " |",
        "|" + "---|" * len(head),
    ]

    def note(text: str) -> str:
        cells = [""] * len(head)
        cells[3] = text
        return "| " + " | ".join(cells) + " |"

    for r in report["workloads"]:
        d = r["derived"]
        t = r["time_ms"]
        roof = r.get("roof_ms", {})
        if r.get("compile_error"):
            compile_cell = "FAIL"
        elif r.get("compile") is None:
            compile_cell = "-"
        else:
            compile_cell = "pass" if all(v["passed"] for v in r["compile"].values()) else "FAIL"
        contract = r.get("contract")
        contract_cell = "-" if contract is None else ("pass" if all(v["passed"] for v in contract.values()) else "FAIL")
        cells = [r["name"], "y" if r["required"] else "n", r["dtype"]]
        cells += [_ok_cell(r.get(p)) for p in phases]
        cells += [compile_cell, contract_cell]
        for p in phases:
            base_phase = _BASELINE_PHASE.get(p, p)
            base = min([v for k, v in t.items() if k in (f"compiled_{base_phase}", f"eager_{base_phase}")], default=None)
            cells += [_fmt(t.get(f"kernel_{p}")), _fmt(base), _fmt(roof.get(p)), _fmt(d.get(p, {}).get("sol_eff"), ".2f")]
        lines.append("| " + " | ".join(cells) + " |")
        if r.get("skipped"):
            lines.append(note(f"skipped: {r['skipped']}"))
        if r["error"]:
            lines.append(note(f"error: `{r['error']}`"))
        if r.get("compile_error"):
            lines.append(note(f"not compile-safe: `{r['compile_error']}`"))
        if contract:
            failed = [f"{k}: {v['detail']}" for k, v in contract.items() if not v["passed"]]
            summary = f"contract: {len(contract) - len(failed)}/{len(contract)} rows ok"
            lines.append(note(summary + ("; " + "; ".join(failed) if failed else "")))
        for w in r.get("warnings") or []:
            lines.append(note(f"warning: `{w}`"))
    if report["reasons"]:
        lines += ["", "## Reasons", ""] + [f"- {x}" for x in report["reasons"]]
    for r in report["workloads"]:
        worst = [
            f"{phase}/{name}: max_abs={v['max_abs']:.3e} max_rel={v['max_rel']:.3e} bad={v['n_bad']}/{v['n']}"
            for phase in (*ALL_PHASES, "compile")
            if r.get(phase)
            for name, v in r[phase].items()
            if not v["passed"]
        ]
        if worst:
            lines += ["", f"## Mismatches: {r['name']}", ""] + [f"- {w}" for w in worst]
    verification = report.get("verification")
    if verification:
        lines += ["", "## Verifier decision", "",
                  f"- Verdict: **{verification['verdict']}** (raw harness: {report['verdict']})",
                  f"- Context: {verification['independence']}"]
        lines += [f"- {f['severity']}: {f['reason']} ({f['evidence']})" for f in verification['findings']]
        if verification.get("notes"):
            lines += ["", str(verification["notes"])]
        lines += ["", "Diagnosis: " + (verification.get("diagnosis") or verification['verdict']).replace("\n", " ")]
    return "\n".join(lines) + "\n"


def write_report(
    json_path: Union[str, Path],
    md_path: Union[str, Path],
    *,
    op: str,
    backend: str,
    env: Dict[str, Any],
    results: List[WorkloadResult],
    iteration: Optional[int] = None,
    sol_threshold: float = SOL_THRESHOLD,
    gate_recompute: bool = False,
    kernel_backend: Optional[str] = None,
    gate_perf: bool = True,
    coverage: Optional[dict] = None,
    source_hashes: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Write ``report.json`` and ``REPORT.md``; return the report dict.

    ``gate_perf=False`` marks a diagnostic report (``_run_dev.py --method events``, asked for
    explicitly): performance is not gated (see ``verdict``) and the Markdown says so in its
    title. A machine without CUPTI, where ``auto`` resolves to events, still gates: that is the
    only timing it has.
    """
    v, reasons = verdict(results, sol_threshold=sol_threshold, gate_recompute=gate_recompute, gate_perf=gate_perf, coverage=coverage)
    report = {
        "op": op,
        "backend": backend,
        "kernel_backend": kernel_backend,
        "env": env,
        "iteration": iteration,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "verdict": v,
        "reasons": reasons,
        "gate_recompute": gate_recompute,
        "gate_perf": gate_perf,
        "workloads": [r.as_dict() for r in results],
        "coverage": coverage,
        "source_hashes": source_hashes,
    }
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    Path(md_path).parent.mkdir(parents=True, exist_ok=True)
    Path(json_path).write_text(json.dumps(report, indent=2))
    Path(md_path).write_text(render_markdown(report))
    return report


__all__ = [
    "SOL_THRESHOLD",
    "LATENCY_FLOOR_MS",
    "BASELINE_TOLERANCE",
    "LATENCY_PARITY_MS",
    "ROOF_SLACK",
    "PHASES",
    "RECOMPUTE_PHASE",
    "ALL_PHASES",
    "WorkloadResult",
    "verdict",
    "render_markdown",
    "write_report",
]
