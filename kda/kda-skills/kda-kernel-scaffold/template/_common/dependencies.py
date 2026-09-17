"""Bind evidence to file groups and invalidate only dependent conclusions."""

import hashlib
import json
from pathlib import Path

GROUPS = ("contract", "arithmetic", "adapter", "inputs", "measurement", "analysis", "caller")
"""File dependency groups recorded with every new report."""


def contract_digest(text):
    """Hash parsed front matter, excluding prose and formatting changes."""
    import yaml
    if not text.startswith("---\n"):
        raise ValueError("contract requires YAML front matter")
    data = yaml.safe_load(text.split("---", 2)[1])
    return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()


def snapshot(root, files):
    """Hash explicit group-to-file mappings; reject unknown groups or missing files."""
    root = Path(root).resolve()
    result = {}
    for group, names in files.items():
        if group not in GROUPS:
            raise ValueError(f"unknown dependency group: {group}")
        result[group] = {}
        for name in sorted(set(names)):
            path = (root / name).resolve()
            if root not in path.parents or not path.is_file():
                raise ValueError(f"missing or external dependency: {name}")
            result[group][name] = (contract_digest(path.read_text()) if path.name in ("SPEC.md", "INTEGRATION.md")
                                  else hashlib.sha256(path.read_bytes()).hexdigest())
    return result


def operation_dependencies(op_dir):
    """Classify generated files conservatively, including unknown executable files."""
    op_dir = Path(op_dir)
    root = op_dir.parent
    files = {group: [] for group in GROUPS}
    files["contract"] = [f"{op_dir.name}/SPEC.md"]
    files["adapter"] = ["_common/VERSION"]
    if (op_dir / "TUNED.json").is_file():
        files["arithmetic"].append(f"{op_dir.name}/TUNED.json")
    candidates = list(op_dir.rglob("*.py")) + list((root / "_common").glob("*.py"))
    for path in candidates:
        if {"_scratch", "_checkpoints", "__pycache__", "artifacts"} & set(path.relative_to(root).parts):
            continue
        name = path.name
        if name in ("_impl_fwd.py", "_impl_bwd.py", "_eager.py"):
            groups = ("arithmetic",)
        elif name in ("bench.py", "experiment.py", "gpu_info.py", "_speed_of_light.py", "roofline.py"):
            groups = ("measurement",)
        elif name in ("report.py", "evidence.py", "acceptance.py", "dependencies.py", "integration.py"):
            groups = ("analysis",)
        elif name in ("spec.py", "inputs.py"):
            groups = ("inputs",)
        elif name in ("interface.py", "backends.py", "_dispatch.py", "_register.py", "compat.py", "env.py", "plan_cache.py", "__init__.py"):
            groups = ("adapter",)
        else:
            groups = GROUPS  # Unknown harness/helpers can affect any result.
        for group in groups:
            files[group].append(str(path.relative_to(root)))
    return snapshot(root, files)


def invalidation(previous, current, depends_on):
    """Return changed groups and the minimum repeat action for one evidence item."""
    if not previous or not current or not depends_on or set(depends_on) - set(GROUPS):
        return {"valid": False, "action": "rerun", "changed": ["unknown"]}
    changed = [group for group in depends_on if previous.get(group) != current.get(group)]
    action = "reuse" if not changed else "reanalyze" if set(changed) == {"analysis"} else "rerun"
    return {"valid": not changed, "action": action, "changed": changed}
