"""Real-path invocation witnesses and scoped integration acceptance."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import socket
import math
from uuid import uuid4


def contract_id(contract):
    """Bind execution obligations, retaining decision-only authorization/targets separately."""
    executable = {k: v for k, v in contract.items() if k not in ("authorization", "target")}
    return hashlib.sha256(json.dumps(executable, sort_keys=True, default=str).encode()).hexdigest()


def audit_digest(contract, invocations, experiments):
    """Bind the independent integration audit to exact execution and measurement evidence."""
    data = {"contract": contract_id(contract), "invocations": invocations, "experiments": experiments}
    return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()


def file_hashes(root, files):
    """Hash explicitly declared dependencies beneath a source snapshot."""
    root = Path(root).resolve()
    result = {}
    for name in files:
        path = (root / name).resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError(f"missing or external dependency: {name}")
        result[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def runtime_witness(root, modules, *, rank, backend, calls, cleanup):
    """Observe loaded module files inside each worker, not in its parent.

    Args:
        root: Immutable source snapshot imported by this worker.
        modules: Actual loaded module names covering the changed path.
        rank: Worker rank or zero for a single-process request.
        backend: Backend selected by instrumented calls.
        calls: Symbol-to-count mapping populated at executed replacement sites.
        cleanup: Mapping recording success and failure cleanup probe outcomes.
    """
    root = Path(root).resolve()
    files = []
    for name in modules:
        module = sys.modules.get(name)
        path = Path(getattr(module, "__file__", "")).resolve()
        if root not in path.parents:
            raise ValueError(f"{name} did not load from the declared snapshot")
        files.append(str(path.relative_to(root)))
    environment = {"python": sys.version, "executable": sys.executable, "hostname": socket.gethostname()}
    torch = sys.modules.get("torch")
    if torch is not None:
        environment.update(torch=torch.__version__, cuda=torch.version.cuda)
        if torch.cuda.is_initialized():
            environment.update(device=torch.cuda.current_device(), gpu=torch.cuda.get_device_name())
    return {"pid": os.getpid(), "rank": rank, "backend": backend, "environment": environment,
            "loaded_files": file_hashes(root, files), "calls": dict(calls),
            "cleanup": dict(cleanup)}


def validate_contract(contract):
    """Return missing integration obligations before expensive experiments."""
    required = ("operations", "entrypoint", "replacements", "command", "inputs", "configuration",
                "ranks", "branches", "variants", "lifecycle", "fallback", "quality",
                "primary_metric", "matrix", "authorization")
    errors = [f"missing contract field: {key}" for key in required if not contract.get(key)]
    if contract.get("integration_schema") != 1:
        errors.append("integration_schema must be 1")
    if contract.get("primary_metric") != "request_wall_ms":
        errors.append("this A/B assessor supports request_wall_ms; other target metrics require an explicit driver extension")
    variants = contract.get("variants", {})
    for name in ("current", "candidate", "rollback"):
        if not all(variants.get(name, {}).get(k) for k in ("root", "files", "backend")):
            errors.append(f"{name}: source root, files and expected backend required")
    matrix = contract.get("matrix", [])
    identifiers = [row.get("id") for row in matrix]
    if len(set(identifiers)) != len(identifiers) or any(not value for value in identifiers):
        errors.append("matrix ids must be unique and nonempty")
    for cell in matrix:
        if not all(cell.get(key) for key in ("data_root", "input_hashes", "config_hashes")):
            errors.append(f"{cell.get('id')}: matrix must bind data root and expected input/configuration hashes")
    return errors


def invocation(contract, variant, *, cell, witnesses, input_hashes, config_hashes, outputs,
               branch, status="success", command=None, started_at=None, finished_at=None):
    """Bind one actual invocation to its expected source and data dependencies."""
    source = contract["variants"][variant]
    return {"id": uuid4().hex, "contract_digest": contract_id(contract),
            "started_at": time.time() if started_at is None else started_at,
            "finished_at": time.time() if finished_at is None else finished_at,
            "variant": variant, "cell": cell, "branch": branch, "status": status,
            "command": command or contract["command"], "witnesses": witnesses,
            "source_hashes": file_hashes(source["root"], source["files"]),
            "input_hashes": input_hashes, "config_hashes": config_hashes, "outputs": outputs}


def assess(contract, invocations, experiments, audit):
    """Assess path coverage without inferring delivery from operation reports."""
    errors = validate_contract(contract)
    if audit.get("evidence_digest") != audit_digest(contract, invocations, experiments):
        errors.append("independent integration audit is not bound to current evidence")
    for cell in contract.get("matrix", []):
        for key in ("input_hashes", "config_hashes"):
            try:
                if file_hashes(cell["data_root"], cell[key]) != cell[key]:
                    errors.append(f"{cell['id']}: {key} source files changed")
            except (KeyError, OSError, ValueError) as exc:
                errors.append(f"{cell.get('id')}: cannot verify {key}: {exc}")
    expected = {}
    for name, source in contract.get("variants", {}).items():
        try:
            expected[name] = file_hashes(source["root"], source["files"])
        except (ValueError, OSError) as exc:
            errors.append(str(exc))
    cells = {row["id"] for row in contract.get("matrix", [])}
    ranks = set(contract.get("ranks", []))
    symbols = [row["symbol"] for row in contract.get("replacements", [])]
    coverage = set()
    identities = {}
    for row in invocations:
        variant, cell = row.get("variant"), row.get("cell")
        tag = f"{variant}/{cell}"
        if variant not in expected or cell not in cells:
            errors.append(f"{tag}: unknown variant or cell")
            continue
        planned = next(item for item in contract["matrix"] if item["id"] == cell)
        command = contract["variants"][variant].get("command", planned.get("command", contract["command"]))
        if row.get("command") != command or row.get("contract_digest") != contract_id(contract):
            errors.append(f"{tag}: command or integration contract differs")
        if any(row.get(key) != planned.get(key) for key in ("input_hashes", "config_hashes")):
            errors.append(f"{tag}: input/configuration identity differs from declared sources")
        if row.get("source_hashes") != expected[variant]:
            errors.append(f"{tag}: source changed since invocation")
        if not row.get("input_hashes") or not row.get("config_hashes") or not row.get("command"):
            errors.append(f"{tag}: missing input/configuration/command provenance")
        identity = (row.get("input_hashes"), row.get("config_hashes"))
        if cell in identities and identities[cell] != identity:
            errors.append(f"{tag}: inputs or configuration differ across variants")
        identities[cell] = identity
        if row.get("status") != "success" or row.get("outputs", {}).get("passed") is not True:
            errors.append(f"{tag}: failed invocation or unchecked output")
        witnesses = row.get("witnesses", [])
        if {w.get("rank") for w in witnesses} != ranks or len(witnesses) != len(ranks):
            errors.append(f"{tag}: missing or duplicate rank witnesses")
        if len({w.get("pid") for w in witnesses}) != len(ranks):
            errors.append(f"{tag}: rank witnesses do not identify distinct worker processes")
        for witness in witnesses:
            if not all(witness.get("environment", {}).get(key) for key in ("python", "executable", "hostname")):
                errors.append(f"{tag}: execution environment missing")
            if witness.get("backend") != contract["variants"][variant]["backend"]:
                errors.append(f"{tag}: unexpected backend on rank {witness.get('rank')}")
            loaded = witness.get("loaded_files", {})
            if not loaded or any(expected[variant].get(k) != v for k, v in loaded.items()):
                errors.append(f"{tag}: loaded code differs from snapshot")
            required_loaded = contract["variants"][variant].get("loaded_files", contract["variants"][variant]["files"])
            if not set(required_loaded) <= set(loaded):
                errors.append(f"{tag}: changed runtime path not observed")
            if variant == "candidate" and any(witness.get("calls", {}).get(symbol, 0) <= 0 for symbol in symbols):
                errors.append(f"{tag}: replacement not executed")
            if not all(witness.get("cleanup", {}).get(k) is True for k in ("success", "failure")):
                errors.append(f"{tag}: lifecycle probes incomplete")
        coverage.add((variant, cell, row.get("branch")))
    for variant in ("current", "candidate", "rollback"):
        for cell in cells:
            for branch in contract.get("branches", []):
                if (variant, cell, branch) not in coverage:
                    errors.append(f"{variant}/{cell}/{branch}: missing invocation")
    for check in ("original_contract", "integration_diff", "input_preprocessing", "quality"):
        item = audit.get(check, {})
        if item.get("passed") is not True or not item.get("evidence"):
            errors.append(f"independent audit missing: {check}")
    known = {row.get("id"): row for row in invocations if row.get("id")}
    if len(known) != len(invocations):
        errors.append("invocation ids are missing or duplicated")
    from .experiment import summarize_ab
    seen_attempts = set()
    for cell in cells:
        experiment = experiments.get(cell, {})
        samples = experiment.get("samples", [])
        if not samples:
            continue
        for sample in samples:
            linked = known.get(sample.get("outputs", {}).get("invocation_id"), {})
            if linked.get("cell") != cell or linked.get("variant") != sample.get("variant"):
                errors.append(f"{cell}: measurement is not bound to its invocation")
            identifier = linked.get("id")
            if identifier in seen_attempts:
                errors.append(f"{cell}: invocation reused for multiple attempts")
            seen_attempts.add(identifier)
            bounds = (sample.get("started_at"), linked.get("started_at"), linked.get("finished_at"), sample.get("finished_at"))
            if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in bounds) or list(bounds) != sorted(bounds):
                errors.append(f"{cell}: invocation is outside measured request boundaries")
        recalculated = summarize_ab(samples, repeats=experiment.get("repeats", 5))
        if any(experiment.get(k) != recalculated.get(k) for k in ("measurement_valid", "deployment_delta_ms", "deployment_reduction")):
            errors.append(f"{cell}: derived performance differs from raw samples")
    performance = {cell: experiments.get(cell, {}) for cell in cells}
    return {"schema_version": 1, "scope": "integration", "integration": {
        "status": "pass" if not errors else "incomplete", "findings": errors},
        "performance": performance, "audit": audit, "invocations": invocations,
        "delivery": {"status": "not_assessed"}}


def main():
    """Validate a contract and assess saved request evidence without running requests."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contract", type=Path)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    import yaml
    text = args.contract.read_text()
    contract = yaml.safe_load(text.split("---", 2)[1])
    if not args.evidence:
        errors = validate_contract(contract)
        print(json.dumps({"errors": errors}, indent=2))
        return 2 if errors else 0
    evidence = json.loads(args.evidence.read_text())
    result = assess(contract, evidence["invocations"], evidence["experiments"], evidence["audit"])
    if args.out:
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    else:
        print(json.dumps(result, indent=2))
    return 0 if result["integration"]["status"] == "pass" else 3


if __name__ == "__main__":
    raise SystemExit(main())
