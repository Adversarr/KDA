"""Verify and benchmark a withheld golden using KDA's source runtime.

Use each example's ``benchmark.py --verify`` or ``benchmark.py --bench``. Scoped runs require
an explicit output path and never replace the complete record. A raw passing
verdict still requires the independent source/coverage audit before acceptance.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
from datetime import datetime, timezone
import json
import traceback
import importlib.util
import sys
from pathlib import Path


def outputs(value):
    """Normalize single-output and multiple-output kernel interfaces."""
    return value if isinstance(value, tuple) else (value,)


def differentiable(args):
    return [x for x in args if isinstance(x, torch.Tensor) and x.requires_grad]


def compare_outputs(actual, expected, computation_dtype):
    if len(actual) != len(expected):
        raise ValueError("kernel output count disagrees with eager")
    result = {}
    for i, (a, e) in enumerate(zip(actual, expected)):
        atol, rtol = verify.TOLERANCES[
            verify.lowest_precision(a.dtype, e.dtype, computation_dtype)
        ]
        result[str(i)] = verify.compare(a, e, atol=atol, rtol=rtol).as_dict()
    return result


def backward(out, args, upstream):
    return torch.autograd.grad(out, differentiable(args), upstream, retain_graph=True)


def check_baseline(fn, args, kwargs, phases, upstream, computation_dtype, reductions):
    """Certify a timing baseline against eager before accepting its timings."""
    checks = {}
    expected = outputs(cases.reference_fn(*args, **kwargs))
    actual = outputs(fn(*args, **kwargs))
    if "fwd" in phases:
        checks["fwd"] = compare_outputs(actual, expected, computation_dtype)
    if "infer" in phases:
        with torch.no_grad():
            checks["infer"] = compare_outputs(
                outputs(fn(*args, **kwargs)),
                outputs(cases.reference_fn(*args, **kwargs)),
                computation_dtype,
            )
    if "bwd" in phases:
        actual_grad = backward(actual, args, upstream)
        expected_grad = backward(expected, args, upstream)
        atol, rtol = verify.TOLERANCES[computation_dtype]
        checks["bwd"] = {
            str(i): verify.compare(a, e, atol=atol, rtol=rtol, reduced_over=n).as_dict()
            for i, (a, e, n) in enumerate(zip(actual_grad, expected_grad, reductions))
        }
    return checks


def requested_phases(row, fwd_only):
    """Select explicit workload phases, intersected with a forward-only request."""
    phases = row.get("phases", ["fwd", "infer", "bwd"])
    if (
        not phases
        or len(set(phases)) != len(phases)
        or set(phases) - {"fwd", "infer", "bwd"}
    ):
        raise ValueError("phases must be a nonempty unique subset of fwd/infer/bwd")
    return [phase for phase in phases if phase != "bwd" or not fwd_only]


def evaluate(row, timed, fwd_only, info):
    """Measure one workload with identical inputs and upstream gradients."""
    phases = requested_phases(row, fwd_only)
    torch.manual_seed(0)
    args, kwargs = row["make"]()
    computation_dtype = verify.lowest_precision(
        *(x.dtype for x in args if isinstance(x, torch.Tensor))
    )
    result = report.WorkloadResult(
        row["name"], row.get("required", True), verify.DTYPE_NAMES[computation_dtype]
    )
    expected = outputs(cases.reference_fn(*args, **kwargs))
    actual = outputs(cases.kernel_fn(*args, **kwargs))
    if "fwd" in phases:
        result.fwd = compare_outputs(actual, expected, computation_dtype)
    if "infer" in phases:
        with torch.no_grad():
            result.infer = compare_outputs(
                outputs(cases.kernel_fn(*args, **kwargs)),
                outputs(cases.reference_fn(*args, **kwargs)),
                computation_dtype,
            )
    upstream = tuple(torch.randn_like(o) for o in expected)
    if "bwd" in phases:
        expected_grad = backward(expected, args, upstream)
        actual_grad = backward(actual, args, upstream)
        reductions = row.get("grad_reductions", [1] * len(actual_grad))
        if len(reductions) != len(actual_grad):
            raise ValueError("grad_reductions must describe every differentiable input")
        atol, rtol = verify.TOLERANCES[computation_dtype]
        result.bwd = {
            str(i): verify.compare(a, e, atol=atol, rtol=rtol, reduced_over=n).as_dict()
            for i, (a, e, n) in enumerate(zip(actual_grad, expected_grad, reductions))
        }
        del actual_grad, expected_grad
    # Correctness graphs can retain large attention score planes; timing builds fresh graphs.
    del actual, expected
    # Never use numerical failures as performance evidence.
    if not timed or any(
        result.numerics_passed(p) is False for p in ("fwd", "infer", "bwd")
    ):
        return result
    functions = {"kernel": cases.kernel_fn, "eager": cases.reference_fn}
    # Each workload gets the same fresh compiler state as a scoped repeat.
    # Earlier shapes must not weaken this baseline through cache limits or generalization.
    torch._dynamo.reset()
    if hasattr(cases, "compile_fn"):
        try:
            compiled = cases.compile_fn(*args, **kwargs)
        except Exception as exc:
            result.warnings.append(f"compiled baseline setup failed: {exc}")
            compiled = None
    else:
        compiled = bench.compile_or_none(
            cases.reference_fn,
            *args,
            mode="max-autotune-no-cudagraphs" if cases.ROOF != "fp32" else None,
            **kwargs,
        )
    if compiled is not None:
        functions["compiled"] = compiled
    else:
        result.warnings.append("torch.compile baseline unavailable")
    if hasattr(cases, "baseline_fn"):
        functions["sdpa"] = cases.baseline_fn
    baseline_checks = row["_baseline_checks"] = {}
    reductions = row.get("grad_reductions", [1] * len(differentiable(args)))
    for name in list(functions):
        if name == "kernel" or name == "eager":
            continue
        try:
            checks = check_baseline(
                functions[name],
                args,
                kwargs,
                phases,
                upstream,
                computation_dtype,
                reductions,
            )
            baseline_checks[name] = checks
            passed = all(
                value["passed"]
                for phase_checks in checks.values()
                for value in phase_checks.values()
            )
            if not passed:
                result.warnings.append(f"{name} excluded: baseline numerical mismatch")
                del functions[name]
        except Exception as exc:
            baseline_checks[name] = {"error": str(exc)}
            result.warnings.append(
                f"{name} excluded: baseline verification failed: {exc}"
            )
            del functions[name]
    samples = row["_samples"] = {}
    for phase in phases:
        closures = {}
        retained = []
        for name, fn in functions.items():
            try:
                if phase == "bwd":
                    out = outputs(fn(*args, **kwargs))
                    retained.append(out)
                    closures[name] = lambda out=out: backward(out, args, upstream)
                elif phase == "infer":

                    def infer(fn=fn):
                        with torch.no_grad():
                            return fn(*args, **kwargs)

                    closures[name] = infer
                else:
                    closures[name] = lambda fn=fn: fn(*args, **kwargs)
            except Exception as exc:
                if name == "kernel":
                    result.error = f"kernel {phase} setup failed: {exc}"
                    return result
                result.warnings.append(f"{name}_{phase} excluded: setup failed: {exc}")
        # Interleave implementations so drift does not always favor the same one.
        phase_samples = samples[phase] = {name: [] for name in closures}
        for round_index in range(3):
            print("BENCH", row["name"], phase, f"round {round_index + 1}/3", flush=True)
            for name, fn in list(closures.items()):
                try:
                    phase_samples[name].append(
                        bench.bench_ms(fn, method="profiler", warmup=3, iters=10)
                    )
                except Exception as exc:
                    if name == "kernel":
                        result.error = f"kernel {phase} timing failed: {exc}"
                        return result
                    result.warnings.append(
                        f"{name}_{phase} excluded: timing failed: {exc}"
                    )
                    del closures[name]
        for name, times in phase_samples.items():
            if name in closures:
                result.time_ms[f"{name}_{phase}"] = min(times)
        nbytes, flops = cases.roof(phase, args, kwargs)
        result.sol_ms[phase] = gpu_info.sol_ms(nbytes, flops, info, roof=cases.ROOF)
        memory = bench.copy_ms(nbytes, method="profiler")
        compute = None
        details = None
        if getattr(cases, "ROOF_METHOD", "auto") == "attention":
            calibration_phase = (
                "bwd_recompute" if phase == "bwd" and kwargs.get("recompute", False)
                else phase
            )
            try:
                details = bench.attention_ms(
                    calibration_phase, **cases.attention_work(phase, args, kwargs),
                    dtype=args[0].dtype, device=args[0].device, method="profiler",
                )
            except (AttributeError, NotImplementedError, ValueError) as exc:
                details = {"method": "attention", "available": False,
                           "roof_ms": None, "reason": str(exc)}
            result.roof_details[phase] = details
            compute = details["roof_ms"]
        elif cases.ROOF != "fp32":
            shapes = (
                cases.gemm_shapes(phase, args, kwargs)
                if hasattr(cases, "gemm_shapes")
                else None
            )
            compute = bench.matmul_ms(flops, shapes=shapes, method="profiler")
        result.roof_ms[phase] = (
            None if details is not None and not details["available"]
            else max(memory, compute or 0)
        )
        row.setdefault("_roof_analysis", {})[phase] = {
            "bytes": nbytes,
            "flops": flops,
            "copy_ms": memory,
            "compute_ms": compute,
            "datasheet_ms": result.sol_ms[phase],
            "achievable_ms": result.roof_ms[phase],
            "calibration": details,
        }
        del closures, retained
        if phase == "bwd":
            del out
        del fn
    row["_samples"] = samples
    return result


def main(default_golden: Path | None = None, argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--golden", default=default_golden, required=default_golden is None, type=Path
    )
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--fwd-only", action="store_true")
    parser.add_argument("--workload")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--md", type=Path)
    parser.add_argument(
        "--adjoint",
        action="store_true",
        help="Check isolated and broadcast upstream gradients",
    )
    args = parser.parse_args(argv)
    if args.adjoint:
        if args.bench or args.fwd_only or args.md:
            parser.error(
                "--adjoint cannot be combined with --bench, --fwd-only or --md"
            )
        if args.json is None:
            parser.error("--adjoint requires --json")
        from _check_adjoint import main as check_adjoint

        options = ["--golden", str(args.golden), "--json", str(args.json)]
        if args.workload:
            options += ["--workload", args.workload]
        return check_adjoint(argv=options)
    root = args.golden.resolve()
    checkout = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(checkout / "kda/kda-skills/kda-kernel-scaffold/template"))
    sys.path.insert(0, str(root))
    global torch, cases, bench, evidence, gpu_info, report, verify
    import torch
    from _common import bench, evidence, gpu_info, report, verify

    spec = importlib.util.spec_from_file_location(
        "cases", root.parent / "benchmark_cases.py"
    )
    cases = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cases)
    backend = getattr(sys.modules.get(cases.kernel_fn.__module__), "BACKEND", "triton")
    scoped = bool(args.workload or not args.bench or args.fwd_only)
    if scoped and args.json is None:
        parser.error("scoped/verification runs require --json to preserve report.json")
    json_target = (
        args.json
        or checkout / "tmp/golden-benchmarks" / root.parent.name / "report.json"
    )
    md_target = args.md or json_target.with_suffix(".md")
    if any(
        root == path.resolve() or root in path.resolve().parents
        for path in (json_target, md_target)
    ):
        parser.error("generated evidence must be written outside golden/")
    try:
        json_path, md_path = evidence.output_paths(
            root, scoped, str(json_target), str(md_target)
        )
    except ValueError as exc:
        parser.error(str(exc))
    runtime = checkout / "kda/kda-skills/kda-kernel-scaffold/template/_common"
    reference = checkout / "kda/kda-skills/kda-kernel-implement/reference"
    validation_sources = [
        Path(__file__).resolve(),
        root.parent / "benchmark.py",
        Path(spec.origin).resolve(),
        *runtime.glob("*.py"),
        runtime / "VERSION",
        *reference.rglob("*.py"),
    ]
    golden_hashes = evidence.source_hashes(root)
    validation_hashes = {
        str(path.relative_to(checkout)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(validation_sources)
        if path.is_file()
    }
    rows = cases.cases()
    if args.workload:
        rows = [r for r in rows if r["name"] == args.workload]
        if not rows:
            parser.error("unknown workload")
    coverage = {
        "all_workloads": [r["name"] for r in rows],
        "benchmark": args.bench,
        "required": {
            r["name"]: {
                "phases": requested_phases(r, args.fwd_only),
                "timings": [f"kernel_{p}" for p in requested_phases(r, args.fwd_only)]
                if args.bench and r.get("benchmark", True)
                else [],
            }
            for r in rows
            if r.get("required", True)
        },
    }
    info = gpu_info.get_gpu_info()
    results = []

    def save():
        verdict, reasons = report.verdict(results, coverage=coverage)
        record = {
            "op": root.parent.name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "backend": backend + "_fwd_only" if args.fwd_only else backend,
            "kernel_backend": backend,
            "gate_perf": True,
            "env": {
                "bench_method": "profiler",
                "gpu": info.as_dict(),
                "torch": torch.__version__,
            },
            "coverage": coverage,
            "source_hashes": golden_hashes,
            "validation_hashes": validation_hashes,
            "verdict": verdict,
            "reasons": reasons,
            "verification": {
                "verdict": "incomplete",
                "audit_complete": False,
                "independence": "pending",
                "findings": [],
                "notes": "Raw numerics/performance only: independent source, contract and coverage audit remains required.",
            },
            "workloads": [r.as_dict() for r in results],
            "samples": {r["name"]: r.get("_samples", {}) for r in rows},
            "roof_analysis": {r["name"]: r.get("_roof_analysis", {}) for r in rows},
            "baseline_checks": {r["name"]: r.get("_baseline_checks", {}) for r in rows},
            "measurement_diagnostics": bench.measurement_diagnostics(),
        }
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(record, indent=2) + "\n")
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(report.render_markdown(record))
        return verdict, reasons

    for row in rows:
        print("RUN", row["name"], flush=True)
        try:
            result = evaluate(
                row, args.bench and row.get("benchmark", True), args.fwd_only, info
            )
        except Exception as exc:
            traceback.print_exc()
            result = report.WorkloadResult(
                row["name"], row.get("required", True), "bf16", error=str(exc)
            )
        results.append(result)
        verdict, reasons = save()
        print(row["name"], verdict, json.dumps(reasons), flush=True)
        gc.collect()
        torch.cuda.empty_cache()
    verdict, _ = save()
    raise SystemExit(0 if verdict == "pass" else 1)


if __name__ == "__main__":
    main()
