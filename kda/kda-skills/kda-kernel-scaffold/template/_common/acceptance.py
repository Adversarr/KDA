"""Separate mathematical, integration, performance and delivery decisions."""

import math
import statistics


def measurement_errors(report):
    """Check raw samples and compatible methods before interpreting ratios."""
    errors = []
    metrics = {"profiler": "device_activity_sum_ms", "events": "stream_elapsed_ms", "wall": "region_wall_ms"}
    required = (report.get("coverage") or {}).get("required", {})
    strict = report.get("performance_policy") == "strict_kernel"
    if report.get("schema_version") == 2 and report.get("gate_perf") != strict:
        errors.append("performance policy disagrees with gate selection")
    for row in report.get("workloads", []):
        if row.get("skipped") and not row.get("required"):
            continue
        records = row.get("measurements", {})
        keys = set(required.get(row["name"], {}).get("timings", [])) | set(row.get("time_ms", {}))
        for key in keys:
            item = records.get(key, {})
            samples = item.get("samples_ms", [])
            if (item.get("status") != "complete" or not samples or item.get("unit") != "ms"
                    or item.get("metric") != metrics.get(item.get("method"))
                    or any(not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in samples)):
                errors.append(f"{row['name']}/{key}: missing or invalid raw measurements")
            elif (item.get("median_ms") != statistics.median(samples)
                  or row.get("time_ms", {}).get(key) != statistics.median(samples)):
                errors.append(f"{row['name']}/{key}: timing statistic disagrees with raw samples")
            if item and (item.get("cache_policy") not in ("cold", "warm", "context")
                         or item.get("order") != list(range(len(samples)))):
                errors.append(f"{row['name']}/{key}: missing collection policy/order")
            if strict and (item.get("method") != "profiler" or item.get("cache_policy") != "cold"):
                errors.append(f"{row['name']}/{key}: strict_kernel requires cold profiler evidence")
        execution = row.get("execution", {})
        if report.get("schema_version") == 2 and (execution.get("actual_backend") != report.get("backend")
                or execution.get("requested_backend") != execution.get("actual_backend")
                or not execution.get("callable") or not execution.get("inputs")):
            errors.append(f"{row['name']}: execution identity is missing or inconsistent")
        for phase in ("fwd", "bwd", "infer", "bwd_recompute"):
            kernel = records.get("kernel_" + phase)
            if not kernel:
                continue
            for prefix in ("eager_", "compiled_"):
                baseline = records.get(prefix + ("bwd" if phase == "bwd_recompute" else phase))
                if baseline and any(kernel.get(k) != baseline.get(k) for k in ("metric", "unit", "cache_policy", "synchronization")):
                    errors.append(f"{row['name']}/{phase}: incompatible baseline measurement")
            roof = row.get("roof_ms", {}).get(phase)
            if roof is not None and report.get("schema_version") == 2:
                detail = row.get("roof_details", {}).get(phase, {})
                try:
                    copy = detail["copy_measurement"]
                    calibration = [copy]
                    if detail.get("method") == "attention":
                        measurements = detail["measurements"]
                        calibration += [m for values in measurements.values() for m in values]
                        medians = {p: statistics.median(m["median_ms"] for m in values) for p, values in measurements.items()}
                        compute = detail["density"] * (medians["fwd"] + medians["bwd"] if phase == "bwd_recompute" else medians[phase])
                    else:
                        compute_detail = detail.get("compute") or {}
                        if compute_detail.get("method") == "sum_gemm_proxy":
                            calibration += compute_detail["measurements"]
                            compute = sum(m["median_ms"] for m in compute_detail["measurements"])
                        elif compute_detail.get("method") == "datasheet":
                            peak = report["env"]["peaks"].get("fp32_tflops" if compute_detail["unit"] == "fp32" else "bf16_tflops")
                            compute = compute_detail["flops"] / (peak * 1e12) * 1e3 if peak else None
                        else:
                            compute = None
                    for index, item in enumerate(calibration):
                        errors += measurement_errors({"performance_policy": "strict_kernel", "workloads": [
                            {"name": f"{row['name']}/{phase}/roof{index}", "measurements": {"calibration": item},
                             "time_ms": {"calibration": item["median_ms"]}}]})
                    expected = max(copy["median_ms"], compute) if compute is not None else copy["median_ms"]
                    if roof != expected or kernel.get("method") != "profiler" or kernel.get("cache_policy") != "cold":
                        errors.append(f"{row['name']}/{phase}: roof differs from compatible calibration")
                except (KeyError, TypeError, ValueError, statistics.StatisticsError):
                    errors.append(f"{row['name']}/{phase}: roof calibration evidence is incomplete")
    return errors


def operation_outcomes(report):
    """Recompute independent operation conclusions without claiming model delivery."""
    from .evidence import _result
    from .report import verdict
    import copy
    coverage = copy.deepcopy(report.get("coverage"))
    if coverage:
        for need in coverage.get("required", {}).values():
            need["timings"] = []
    correct, reasons = verdict([_result(r) for r in report["workloads"]], gate_perf=False, coverage=coverage)
    errors = measurement_errors(report)
    if any("execution identity" in error for error in errors):
        correct = "fail"
        reasons += [error for error in errors if "execution identity" in error]
    measured = bool((report.get("coverage") or {}).get("benchmark"))
    return {"correctness": {"status": correct, "reasons": reasons},
            "integration": {"status": "not_assessed", "scope": "operation"},
            "performance": {"status": "valid" if measured and not errors else "incomplete",
                            "policy": report["performance_policy"], "findings": errors,
                            "task_target": "not_assessed", "statistical_significance": "not_established"},
            "delivery": {"status": "not_assessed", "reason": "requires task integration acceptance"}}


def delivery(contract, operation_reports, integration):
    """Require current scoped evidence and task targets before authorizing delivery."""
    reasons = []
    expected = set(contract.get("operations", []))
    if set(operation_reports) != expected or not expected:
        reasons.append("operation reports do not cover the integration contract")
    for name, report in operation_reports.items():
        if (report.get("schema_version") != 2 or report.get("scope") != "operation"
                or report.get("verification", {}).get("verdict") != "pass"
                or report.get("outcomes", {}).get("correctness", {}).get("status") != "pass"
                or report.get("outcomes", {}).get("performance", {}).get("status") != "valid"
                or report.get("evidence_current") is not True):
            reasons.append(f"{name}: missing current finalized v2 correctness evidence")
        reasons += candidate_identity_errors(contract, name, report)
    if integration.get("integration", {}).get("status") != "pass":
        reasons.append("integration evidence is incomplete")
    from .experiment import summarize_ab
    performance = {}
    threshold = contract.get("target", {}).get("minimum_reduction")
    for cell in contract.get("matrix", []):
        key = cell["id"]
        saved = integration.get("performance", {}).get(key, {})
        result = summarize_ab(saved.get("samples", []), repeats=saved.get("repeats", 5))
        performance[key] = result
        if not result["measurement_valid"]:
            reasons.append(f"{key}: request A/B is incomplete")
        elif threshold is not None and result.get("deployment_reduction", -math.inf) < threshold:
            reasons.append(f"{key}: declared task target not met")
        elif threshold is None and result.get("deployment_reduction", -math.inf) <= 0:
            result["limitation"] = "positive request benefit not demonstrated"
            if not contract.get("authorization", {}).get("accept_performance_limitations"):
                reasons.append(f"{key}: observed performance limitation has no task acceptance")
    if not contract.get("authorization", {}).get("integration"):
        reasons.append("final integration authorization is missing")
    return {"schema_version": 1, "scope": "delivery", "correctness": {k: v.get("outcomes", {}).get("correctness") for k, v in operation_reports.items()},
            "integration": integration.get("integration"), "performance": performance,
            "delivery": {"status": "ready" if not reasons else "incomplete", "reasons": reasons}}


def candidate_identity_errors(contract, name, report):
    """Match verified operation sources to the candidate imported in request experiments."""
    import hashlib
    from pathlib import Path
    from .dependencies import contract_digest
    candidate = contract.get("variants", {}).get("candidate", {})
    if not candidate.get("root") or not report.get("source_hashes"):
        return [f"{name}: candidate/operation source binding missing"]
    root = Path(candidate["root"]).resolve()
    spec_path = candidate.get("operations", {}).get(name, name)
    package = Path(spec_path).parent.parent
    errors = []
    for relative, expected in report["source_hashes"].items():
        path = (root / package / relative).resolve()
        if root not in path.parents or str(path.relative_to(root)) not in candidate.get("files", []):
            errors.append(f"{name}: verified dependency not covered by request snapshot: {relative}")
            continue
        try:
            actual = contract_digest(path.read_text()) if path.name == "SPEC.md" else hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected:
                errors.append(f"{name}: request snapshot differs from verified operation: {relative}")
        except (ValueError, OSError) as exc:
            errors.append(f"{name}: unavailable candidate dependency {relative}: {exc}")
    return errors


def write_manifest(path, contract, reports, acceptance, acceptance_path):
    """Link delivery to source identity, effective reports and unresolved items."""
    import hashlib
    import json
    from pathlib import Path
    from .integration import file_hashes, contract_id
    source = contract["variants"]["candidate"]
    manifest = {"schema_version": 1, "source_root": source["root"], "revision": source.get("revision"),
        "source_hashes": file_hashes(source["root"], source["files"]),
        "execution_contract_digest": contract_id(contract),
        "reports": {name: {"digest": report.get("verification", {}).get("report_digest"),
                            "verdict": report.get("verification", {}).get("verdict"),
                            "current": report.get("evidence_current")} for name, report in reports.items()},
        "acceptance": {"path": str(Path(acceptance_path).resolve()),
                       "sha256": hashlib.sha256(Path(acceptance_path).read_bytes()).hexdigest()},
        "delivery": acceptance["delivery"], "unfinished": acceptance["delivery"]["reasons"],
        "limitations": {key: value.get("limitation") for key, value in acceptance["performance"].items() if value.get("limitation")}}
    Path(path).write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def record_experiment(ledger, *, cell, kind, decision):
    """Consume a task-wide finite experiment budget with a decision consequence."""
    if cell not in ledger["matrix"] or kind not in ("repair", "tune", "integration") or not decision.strip():
        raise ValueError("experiment requires a declared matrix cell, kind and decision consequence")
    used = sum(row["kind"] == kind for row in ledger.get("experiments", []))
    if used >= ledger["limits"][kind]:
        raise ValueError(f"task {kind} budget exhausted")
    ledger.setdefault("experiments", []).append({"cell": cell, "kind": kind, "decision": decision})
    return ledger


def load_operation(op_dir):
    """Read an operation report and independently check source and audit freshness."""
    import json
    import hashlib
    from pathlib import Path
    from .dependencies import GROUPS, operation_dependencies, invalidation
    from .evidence import report_digest
    op_dir = Path(op_dir)
    report = json.loads((op_dir / "report.json").read_text())
    current = invalidation(report.get("dependencies"), operation_dependencies(op_dir), GROUPS)["valid"]
    verification = report.get("verification", {})
    current = current and verification.get("report_digest") == report_digest(report)
    audit_path = verification.get("evidence", {}).get("audit")
    try:
        from .evidence import _local_path
        audit = _local_path(op_dir, audit_path) if audit_path else None
        current = current and audit is not None and verification.get("audit_digest") == hashlib.sha256(audit.read_bytes()).hexdigest()
    except (ValueError, OSError):
        current = False
    if report.get("schema_version") == 2:
        current = current and report.get("outcomes") == operation_outcomes(report)
    report["evidence_current"] = current
    return report


def main():
    """Write task acceptance from a contract, saved invocations and current operation reports."""
    import argparse
    import json
    from pathlib import Path
    import yaml
    from .integration import assess
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contract", type=Path)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, help="default: manifest.json beside acceptance output")
    args = parser.parse_args()
    contract = yaml.safe_load(args.contract.read_text().split("---", 2)[1])
    evidence = json.loads(args.evidence.read_text())
    integration = assess(contract, evidence["invocations"], evidence["experiments"], evidence["audit"])
    reports = {name: load_operation((args.contract.parent / name).parent) for name in contract["operations"]}
    result = delivery(contract, reports, integration)
    result["contract_digest"] = __import__(__package__ + ".dependencies", fromlist=["contract_digest"]).contract_digest(args.contract.read_text())
    result["operation_reports"] = {name: report.get("verification", {}).get("report_digest") for name, report in reports.items()}
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    write_manifest(args.manifest or args.out.with_name("manifest.json"), contract, reports, result, args.out)
    return 0 if result["delivery"]["status"] == "ready" else 3


if __name__ == "__main__":
    raise SystemExit(main())
