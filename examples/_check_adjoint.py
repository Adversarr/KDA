"""Supplementary adjoint checks for a withheld golden.

Example: python examples/fused_residual_rmsnorm/benchmark.py --adjoint --json tmp/adjoint-rmsnorm.json
This side check does not replace or finalize a benchmark record.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path


def _outputs(value):
    return value if isinstance(value, tuple) else (value,)


def _gradients(fn, args, kwargs, mode, index, upstream):
    """Let autograd generate the same expanded gradients produced by ordinary sums."""
    outputs = _outputs(fn(*args, **kwargs))
    inputs = [
        value
        for value in args
        if isinstance(value, torch.Tensor) and value.requires_grad
    ]
    if mode == "one_random":
        loss, grad_outputs = outputs[index], upstream[index]
    elif mode == "one_sum":
        loss, grad_outputs = outputs[index].sum(), None
    elif mode == "all_sum":
        loss, grad_outputs = sum(value.sum() for value in outputs), None
    elif mode == "all_broadcast":
        loss = outputs
        grad_outputs = tuple(
            torch.full((), i + 0.5, dtype=value.dtype, device=value.device).expand_as(
                value
            )
            for i, value in enumerate(outputs)
        )
    elif mode == "all_batch_broadcast":
        loss = outputs
        grad_outputs = tuple(
            upstream[i][:1].expand_as(value) for i, value in enumerate(outputs)
        )
    else:
        raise ValueError(f"unknown adjoint scenario: {mode}")
    gradients = torch.autograd.grad(loss, inputs, grad_outputs, allow_unused=True)
    # An unused parameter and its mathematically zero gradient are equivalent.
    return [
        torch.zeros_like(value) if grad is None else grad
        for value, grad in zip(inputs, gradients)
    ]


def main(default_golden: Path | None = None, argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--golden", default=default_golden, required=default_golden is None, type=Path
    )
    parser.add_argument("--workload")
    parser.add_argument("--json", required=True, type=Path)
    args = parser.parse_args(argv)
    global torch
    import torch

    golden = args.golden.resolve()
    checkout = Path(__file__).resolve().parents[1]
    template = checkout / "kda/kda-skills/kda-kernel-scaffold/template"
    sys.path.insert(0, str(template))
    sys.path.insert(0, str(golden))
    from _common import evidence, verify

    case_path = golden.parent / "benchmark_cases.py"
    spec = importlib.util.spec_from_file_location("adjoint_cases", case_path)
    cases = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cases)
    if any(
        golden == path.resolve() or golden in path.resolve().parents
        for path in (args.json, args.json.with_suffix(".md"))
    ):
        parser.error("generated evidence must be written outside golden/")
    try:
        json_path, md_path = evidence.output_paths(
            golden, True, str(args.json), str(args.json.with_suffix(".md"))
        )
    except ValueError as exc:
        parser.error(str(exc))
    rows = cases.cases()
    if args.workload:
        rows = [row for row in rows if row["name"] == args.workload]
    else:
        by_name = {row["name"]: row for row in rows}
        rows = [
            by_name[name]
            for name in (
                "small_tail",
                "small_rows",
                "small",
                "tail_strided",
                "tail_frame",
                "text_tail",
            )
            if name in by_name
        ][:1]
    if not rows:
        parser.error("no small case found; provide an explicit --workload")
    row = rows[0]
    torch.manual_seed(0)
    inputs, kwargs = row["make"]()
    with torch.no_grad():
        expected = _outputs(cases.reference_fn(*inputs, **kwargs))
    torch.manual_seed(1)
    upstream = tuple(torch.randn_like(value) for value in expected)
    count = len(expected)
    output_shapes = [list(value.shape) for value in expected]
    del expected
    wrt = [
        value
        for value in inputs
        if isinstance(value, torch.Tensor) and value.requires_grad
    ]
    reductions = row.get("grad_reductions", [1] * len(wrt))
    if len(reductions) != len(wrt):
        parser.error("grad_reductions must describe every differentiable input")
    dtype = verify.lowest_precision(
        *(value.dtype for value in inputs if isinstance(value, torch.Tensor))
    )
    atol, rtol = verify.TOLERANCES[dtype]
    scenarios = [
        (f"output_{index}_{mode}", mode, index)
        for index in range(count)
        for mode in ("one_random", "one_sum")
    ]
    scenarios += [
        ("all_sum", "all_sum", None),
        ("all_broadcast", "all_broadcast", None),
        ("all_batch_broadcast", "all_batch_broadcast", None),
    ]
    checks = {}
    for name, mode, index in scenarios:
        try:
            expected_grad = _gradients(
                cases.reference_fn, inputs, kwargs, mode, index, upstream
            )
            actual_grad = _gradients(
                cases.kernel_fn, inputs, kwargs, mode, index, upstream
            )
            comparisons = {
                str(i): verify.compare(
                    actual, expected, atol=atol, rtol=rtol, reduced_over=reduced
                ).as_dict()
                for i, (actual, expected, reduced) in enumerate(
                    zip(actual_grad, expected_grad, reductions)
                )
            }
            checks[name] = {
                "passed": all(value["passed"] for value in comparisons.values()),
                "gradients": comparisons,
            }
            del expected_grad, actual_grad
        except Exception as exc:
            checks[name] = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
        print(name, "PASS" if checks[name]["passed"] else "FAIL", flush=True)
    runtime = template / "_common"
    reference = checkout / "kda/kda-skills/kda-kernel-implement/reference"
    sources = [
        Path(__file__).resolve(),
        golden.parent / "benchmark.py",
        case_path.resolve(),
        *runtime.glob("*.py"),
        runtime / "VERSION",
        *reference.rglob("*.py"),
    ]
    record = {
        "op": golden.parent.name,
        "workload": row["name"],
        "kind": "multioutput_adjoint_side_check",
        "passed": all(value["passed"] for value in checks.values()),
        "checks": checks,
        "output_shapes": output_shapes,
        "grad_reductions": reductions,
        "dtype": verify.DTYPE_NAMES[dtype],
        "torch": torch.__version__,
        "source_hashes": evidence.source_hashes(golden),
        "validation_hashes": {
            str(path.relative_to(checkout)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in sorted(sources)
            if path.is_file()
        },
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(record, indent=2) + "\n")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(
        "# Multi-output adjoint side check\n\n"
        + f"Operation: `{record['op']}`; workload: `{record['workload']}`.\n\n"
        + "\n".join(
            f"- {name}: {'pass' if value['passed'] else 'FAIL'}"
            for name, value in checks.items()
        )
        + "\n\nThis is supplementary evidence, not a finalized benchmark verdict.\n"
    )
    raise SystemExit(0 if record["passed"] else 1)


if __name__ == "__main__":
    main()
