"""Coverage, source provenance and recoverable evidence; all operations are CPU-only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional


def source_hashes(op_dir: Path) -> Dict[str, str]:
    """Hash the contract and executable package, excluding scratch and checkpoints."""
    op_dir = Path(op_dir)
    files = [op_dir / "SPEC.md"]
    files += [p for p in op_dir.rglob("*.py")
              if not {"_scratch", "_checkpoints", "__pycache__"} & set(p.relative_to(op_dir).parts)]
    common = op_dir.parent / "_common"
    files += list(common.glob("*.py")) + [common / "VERSION"]
    return {str(p.relative_to(op_dir.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(files) if p.is_file()}


def coverage_errors(rows: List[dict], coverage: Optional[dict]) -> List[str]:
    """Find missing required evidence using expectations supplied before execution."""
    expected = (coverage or {}).get("required")
    if expected is None:
        # Legacy callers still run, but a blank required result cannot certify correctness.
        expected = {r["name"]: {"phases": ["fwd", "infer"]} for r in rows if r["required"]}
    errors = []
    by_name = {r["name"]: r for r in rows}
    if len(by_name) != len(rows):
        errors.append("duplicate workload results")
    if not rows:
        errors.append("no workload results")
    for name, need in expected.items():
        row = by_name.get(name)
        if row is None or row.get("skipped"):
            errors.append(f"{name}: required workload {'missing' if row is None else 'skipped'}")
            continue
        if not row.get("required"):
            errors.append(f"{name}: required flag disagrees with expected coverage")
        for phase in need.get("phases", []):
            if not row.get(phase):
                errors.append(f"{name}/{phase}: required numerical results missing")
        for probe in ("contract", "compile"):
            if need.get(probe) and not row.get(probe):
                errors.append(f"{name}: required {probe} probe missing")
        for key in need.get("timings", []):
            value = row.get("time_ms", {}).get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                errors.append(f"{name}/{key}: positive finite timing missing")
    return errors


def output_paths(op_dir: Path, scoped: bool, json_path: Optional[str], md_path: Optional[str]):
    """Protect the full record from scoped commands, including explicit path aliases."""
    canonical = {(op_dir / "report.json").resolve(), (op_dir / "REPORT.md").resolve()}
    if scoped and (not json_path or not md_path):
        raise ValueError("--workload requires both --json and --md diagnostic paths")
    paths = (Path(json_path) if json_path else op_dir / "report.json",
             Path(md_path) if md_path else op_dir / "REPORT.md")
    if paths[0].resolve() == paths[1].resolve():
        raise ValueError("JSON and Markdown output paths must differ")
    if scoped and any(p.resolve() in canonical for p in paths):
        raise ValueError("a scoped run cannot overwrite report.json or REPORT.md")
    if scoped and any(p.exists() for p in paths):
        raise ValueError("scoped diagnostic outputs already exist; choose new evidence paths")
    return paths


def _result(row: dict):
    from .report import WorkloadResult
    return WorkloadResult(**{k: v for k, v in row.items() if k in WorkloadResult.__dataclass_fields__})


def _phase_gates(row: dict, phase: str) -> tuple:
    """Compare SOL and baseline gates separately, with the existing runtime exemptions."""
    from .report import SOL_THRESHOLD, BASELINE_TOLERANCE, LATENCY_PARITY_MS
    result = _result(row)
    eff, speedup = result.sol_eff(phase), result.speedup(phase)
    sol_ok = None if eff is None else result.latency_bound(phase) or eff >= SOL_THRESHOLD
    baseline_ok = None
    if speedup is not None:
        gap = result.time_ms[f"kernel_{phase}"] - result.baseline_ms(phase)
        baseline_ok = speedup >= 1.0 - BASELINE_TOLERANCE or (result.latency_bound(phase) and gap <= LATENCY_PARITY_MS)
    return sol_ok, baseline_ok


def report_digest(report: dict) -> str:
    """Bind an audit to raw measurement content; finalizing again does not change this digest."""
    raw = {k: v for k, v in report.items() if k != "verification"}
    return hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest()


def _local_path(op_dir: Path, relative: str) -> Path:
    """Resolve a named evidence file inside the operation directory."""
    path = (op_dir / relative).resolve()
    if Path(relative).is_absolute() or op_dir.resolve() not in path.parents:
        raise ValueError("evidence paths must be relative files inside the operation directory")
    return path


def finalize(op_dir: Path, audit_path: Path) -> dict:
    """Finalize M3 from current full evidence and an explicit verifier audit, without CUDA."""
    from .report import ALL_PHASES, render_markdown, verdict
    op_dir = Path(op_dir)
    report_path = op_dir / "report.json"
    report = json.loads(report_path.read_text())
    audit = json.loads(Path(audit_path).read_text())
    findings = audit.get("findings", [])
    if not isinstance(findings, list) or any(
        not isinstance(f, dict) or f.get("severity") not in ("hard", "unresolved", "note")
        or not isinstance(f.get("reason"), str) or not f["reason"].strip()
        or not isinstance(f.get("evidence"), str) or not f["evidence"].strip() for f in findings
    ):
        raise ValueError("audit findings require severity (hard/unresolved/note), reason and evidence")
    findings = list(findings)

    def finding(reason: str, evidence: str, severity: str = "unresolved") -> None:
        findings.append({"severity": severity, "reason": reason, "evidence": evidence})

    audit_relative = str(Path(audit_path).resolve().relative_to(op_dir.resolve()))
    if audit.get("report_digest") != report_digest(report):
        finding("audit is not bound to the current raw report", audit_relative)
    current = source_hashes(op_dir)
    coverage = report.get("coverage")
    if not coverage or not coverage.get("required"):
        finding("legacy or missing expected coverage; update runner and rerun", "report.json")
    if report.get("source_hashes") != current:
        finding("contract, operation source or shared runtime changed since measurement", "report.json/source_hashes")
    if not coverage or not coverage.get("benchmark") or set(coverage.get("all_workloads", [])) != {r["name"] for r in report["workloads"]}:
        finding("M3 requires a complete benchmark record", "report.json/coverage")
    if report.get("backend") != report.get("kernel_backend"):
        finding("primary record must use the full kernel backend", "report.json/backend")
    if report.get("env", {}).get("bench_method") != "profiler" or report.get("gate_perf") is False:
        finding("performance evidence requires profiler timing", "report.json/env")
    raw_verdict, raw_reasons = verdict([_result(r) for r in report["workloads"]],
                                      gate_recompute=report.get("gate_recompute", False),
                                      gate_perf=report.get("gate_perf", True), coverage=coverage)
    if raw_verdict != report.get("verdict"):
        finding("stored verdict disagrees with measured evidence", "report.json/verdict")
    try:
        forward = json.loads((op_dir / "report.fwd_only.json").read_text())
    except (OSError, ValueError):
        forward = {}
    if (forward.get("source_hashes") != current or not forward.get("coverage")
            or forward.get("backend") != str(report.get("kernel_backend")) + "_fwd_only"
            or set(forward.get("coverage", {}).get("required", {})) != set((coverage or {}).get("required", {}))):
        finding("matching forward-only verification missing", "report.fwd_only.json")
    else:
        fv, _ = verdict([_result(r) for r in forward["workloads"]], gate_perf=False,
                        coverage=forward["coverage"])
        if fv != "pass":
            finding("forward-only verification did not pass", "report.fwd_only.json", "hard" if fv == "fail" else "unresolved")
    if audit.get("audit_complete") is not True:
        finding("verifier audit is unfinished", str(audit_path.name))
    independence = audit.get("independence")
    if independence not in ("independent", "same_context"):
        finding("verification context must be identified", str(audit_path.name))

    # Two scoped samples are required for each tagged near-gate workload. All supplied
    # samples remain evidence; a flipped phase is unresolved even if another phase fails.
    reruns = audit.get("reruns", {})
    if not isinstance(reruns, dict):
        raise ValueError("audit.reruns must map workload names to lists of diagnostic JSON paths")
    for row in report["workloads"]:
        name = row["name"]
        tagged = any(reason.startswith(name + "/") and "near gate:" in reason and "(not gated)" not in reason for reason in raw_reasons)
        paths = reruns.get(name, [])
        if not isinstance(paths, list) or any(not isinstance(p, str) for p in paths):
            raise ValueError("each reruns entry must be a list of paths")
        resolved_paths = [_local_path(op_dir, p) for p in paths]
        if any(p == report_path.resolve() for p in resolved_paths):
            raise ValueError("reruns must not reference the canonical report")
        if tagged and len(set(resolved_paths)) < 2:
            finding("two diagnostic repeats required for near-gate evidence", name)
        for relative in paths:
            path = _local_path(op_dir, relative)
            try:
                sample = json.loads(path.read_text())
                sr = next(r for r in sample["workloads"] if r["name"] == name)
                valid = (sample.get("source_hashes") == current
                         and sample.get("backend") == report.get("backend")
                         and sample.get("env", {}).get("bench_method") == "profiler"
                         and sample.get("coverage", {}).get("benchmark")
                         and not coverage_errors([sr], {"required": {name: (coverage or {}).get("required", {}).get(name, {})}}))
            except (OSError, ValueError, KeyError, StopIteration):
                valid = False
            if not valid:
                finding("missing, stale or incomplete diagnostic repeat", relative)
                continue
            sample_verdict, _ = verdict([_result(sr)], gate_perf=False, coverage=sample.get("coverage"))
            if sample_verdict == "fail":
                finding("diagnostic repeat failed correctness", relative, "hard")
            for phase in ALL_PHASES:
                if f"kernel_{phase}" not in row["time_ms"] or (phase == "bwd_recompute" and not report.get("gate_recompute")):
                    continue
                if _phase_gates(row, phase) != _phase_gates(sr, phase):
                    finding(f"{name}/{phase}: gate differs between record and repeat", relative)

    final = raw_verdict
    if any(f["severity"] == "hard" for f in findings):
        final = "fail"
    elif final != "fail" and any(f["severity"] == "unresolved" for f in findings):
        final = "incomplete"
    report["verification"] = {"verdict": final, "audit_complete": audit.get("audit_complete") is True,
                              "independence": independence, "findings": findings,
                              "evidence": {"audit": audit_relative, "forward_only": "report.fwd_only.json", "reruns": reruns},
                              "notes": audit.get("notes", ""), "diagnosis": audit.get("diagnosis", "")}
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    (op_dir / "REPORT.md").write_text(render_markdown(report))
    return report


def checkpoint(op_dir: Path, name: str, restore: bool = False) -> Path:
    """Snapshot/restore worker-owned files and evidence, preserving notes and run counters."""
    if not name or Path(name).name != name or name in (".", ".."):
        raise ValueError("checkpoint name must be a single directory name")
    op_dir = Path(op_dir)
    dest = op_dir / "_checkpoints" / name
    owned = ["_triton", "_tilelang", "_nvmath", "_helpers.py", "TUNED.json"]
    records = ["report.json", "REPORT.md", "report.fwd_only.json", "report.fwd_only.md", "report.verify.json", "report.verify.md", "audit.json"]

    def protected(hashes):
        return {k: v for k, v in hashes.items() if k.split("/")[0] != op_dir.name or k.split("/")[1] not in owned}

    if not restore:
        report = json.loads((op_dir / "report.json").read_text())
        hashes = source_hashes(op_dir)
        if report.get("verification", {}).get("verdict") not in ("pass", "tune") or report.get("source_hashes") != hashes:
            raise ValueError("checkpoint requires current finalized pass/tune evidence")
        references = report["verification"].get("evidence", {})
        records += [references["audit"]] if references.get("audit") else []
        for paths in references.get("reruns", {}).values():
            records += paths
        records = list(dict.fromkeys(str(_local_path(op_dir, p).relative_to(op_dir.resolve())) for p in records))
        dest.mkdir(parents=True, exist_ok=False)
        present = []
        for rel in owned + records:
            src, target = op_dir / rel, dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if src.is_dir():
                shutil.copytree(src, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            elif src.is_file():
                shutil.copy2(src, target)
            else:
                continue
            present.append(rel)
        files = {str(p.relative_to(dest)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in dest.rglob("*") if p.is_file()}
        (dest / "manifest.json").write_text(json.dumps({"source_hashes": hashes, "present": present, "files": files}, indent=2))
    else:
        manifest = json.loads((dest / "manifest.json").read_text())
        if protected(source_hashes(op_dir)) != protected(manifest["source_hashes"]):
            raise ValueError("contract or harness changed; repair/reverify instead of restoring stale evidence")
        for rel, digest in manifest["files"].items():
            path = dest / rel
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError(f"checkpoint evidence damaged: {rel}")
        for rel in list(dict.fromkeys(owned + records + manifest["present"])):
            target = _local_path(op_dir, rel)
            if target.is_dir():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()
            if rel in manifest["present"]:
                src = dest / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(src, target) if src.is_dir() else shutil.copy2(src, target)
        if source_hashes(op_dir) != manifest["source_hashes"]:
            raise ValueError("restored source does not match checkpoint; verification required")
    return dest


def main() -> None:
    """Small checkpoint CLI for orchestrators, independent of GPU/DSL imports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("op_dir", type=Path)
    parser.add_argument("name")
    parser.add_argument("--restore", action="store_true")
    args = parser.parse_args()
    try:
        print(checkpoint(args.op_dir, args.name, restore=args.restore))
    except (ValueError, OSError) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    main()
