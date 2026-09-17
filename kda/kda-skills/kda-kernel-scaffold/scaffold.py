#!/usr/bin/env python3
"""Render a new kernel package from the KDA template into a user's repo.

    python scaffold.py --op fused_residual_rmsnorm --dest /path/to/repo/kda_kernels
    python scaffold.py --op foo --dest ... --kernel-backend triton      # or nvmath (GEMM + epilogue via cuBLASLt), tilelang (experimental)
    python scaffold.py --dest ... --check-upgrade         # read-only migration inspection
    python scaffold.py --dest ... --sync-common           # refresh _common only

``template/_common`` is copied once per destination and never overwritten unless
``--sync-common`` is passed (each op's SPEC records the version it was generated against).
``template/op`` is copied with ``{{op}}``, ``{{Op}}``, ``{{common_version}}``,
``{{kernel_backend}}`` and ``{{date}}`` substituted in file contents and names. Only the
kernel directory of the chosen backend (``_triton/``, ``_nvmath/`` or ``_tilelang/``) is rendered: one
kernel backend per package. Deterministic: no LLM is involved.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "template"
OP_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "._*", ".DS_Store")
KERNEL_BACKENDS = ("triton", "nvmath", "tilelang")


def _camel(op: str) -> str:
    return "".join(part.capitalize() for part in op.split("_"))


def _template_common_version() -> str:
    return (TEMPLATE / "_common" / "VERSION").read_text().strip()


def sync_common(dest: Path, *, force: bool) -> str:
    """Install or refresh ``dest/_common``; returns the installed version."""
    src, dst = TEMPLATE / "_common", dest / "_common"
    template_version = _template_common_version()
    if not dst.exists():
        shutil.copytree(src, dst, ignore=IGNORE)
        print(f"installed _common {template_version} -> {dst}")
        return template_version
    installed = (dst / "VERSION").read_text().strip() if (dst / "VERSION").exists() else "unknown"
    if force:
        shutil.rmtree(dst)
        shutil.copytree(src, dst, ignore=IGNORE)
        print(f"updated _common {installed} -> {template_version}")
        return template_version
    if installed != template_version:
        print(
            f"note: {dst} is version {installed}, template is {template_version}; "
            "pass --sync-common to update (existing ops keep working)."
        )
    return installed


def render_op(op: str, dest: Path, *, common_version: str, force: bool, kernel_backend: str = "triton", capability: str = "training") -> Path:
    """Copy ``template/op`` to ``dest/<op>`` with placeholders substituted.

    Only ``_<kernel_backend>/`` is rendered among the kernel directories.
    """
    if kernel_backend not in KERNEL_BACKENDS:
        sys.exit(f"error: --kernel-backend must be one of {KERNEL_BACKENDS}, got {kernel_backend!r}")
    dst = dest / op
    if dst.exists():
        sys.exit(f"error: {dst} exists; merge runner/contract changes explicitly, --force cannot overwrite a package")
    subst = {
        "{{op}}": op,
        "{{Op}}": _camel(op),
        "{{common_version}}": common_version,
        "{{kernel_backend}}": kernel_backend,
        "{{capability}}": capability,
        "{{grad_inputs}}": "[]" if capability == "inference" else "null",
        "{{backward}}": {"training": "fused", "eager_grad": "eager", "inference": "none"}[capability],
        "{{date}}": _dt.date.today().isoformat(),
    }
    skipped_dirs = {f"_{b}" for b in KERNEL_BACKENDS if b != kernel_backend}

    def render(text: str) -> str:
        for key, value in subst.items():
            text = text.replace(key, value)
        return text

    for src in sorted((TEMPLATE / "op").rglob("*")):
        if "__pycache__" in src.parts or src.suffix == ".pyc" or src.name.startswith("._") or src.name == ".DS_Store":
            continue
        if skipped_dirs & set(src.relative_to(TEMPLATE / "op").parts):
            continue
        if capability != "training" and src.name == "_impl_bwd.py":
            continue
        rel = Path(render(str(src.relative_to(TEMPLATE / "op"))))
        target = dst / rel
        if src.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            content = render(src.read_text())
            if capability != "training" and src.name == "_register.py":
                content = render('''"""Capability-specific forward registration."""
from ..._common.compat import register_kernel, with_eager_backward, inference_only
from .._eager import {{op}}_eager
from ._impl_fwd import {{op}}_fwd, {{op}}_fwd_fake

N_OUTPUTS = 1
_fwd = register_kernel("kda::{{op}}_fwd", {{op}}_fwd, {{op}}_fwd_fake)
''')
                factory = (f"with_eager_backward(_fwd, {op}_eager, n_outputs=N_OUTPUTS, save_aux_arg='save_aux', recompute_arg='recompute')"
                           if capability == "eager_grad" else "inference_only(_fwd, n_outputs=N_OUTPUTS)")
                content += f"{op}_{kernel_backend} = {factory}\n"
                content += f"{op}_{kernel_backend}_fwd_only = {op}_{kernel_backend}  # compatibility alias, not independent evidence\n"
            if capability != "training" and src.name == "_impl_fwd.py":
                imports = "import torch\nfrom typing import Tuple\n"
                if kernel_backend == "nvmath":
                    imports += "from ..._common.plan_cache import PlanCache\n_PLANS = PlanCache(capacity=8)\nclear_plan_cache = _PLANS.clear\n"
                content = f'''"""Forward-only implementation; no saved auxiliaries or recompute kernel."""
{imports}

def {op}_fwd(x: torch.Tensor, save_aux: bool = False) -> Tuple[torch.Tensor]:
    """Return only user-visible outputs; adapt the signature to the observed operation."""
    raise NotImplementedError("implement {kernel_backend} forward")


def {op}_fwd_fake(x: torch.Tensor, save_aux: bool = False) -> Tuple[torch.Tensor]:
    """Describe the real forward outputs without executing arithmetic."""
    raise NotImplementedError("describe output metadata")
'''
            target.write_text(content)
    print(f"rendered op {op!r} -> {dst}")
    return dst


def upgrade_check(dest, op=None):
    """Inspect legacy packages without importing dependencies or changing any file."""
    dest = Path(dest)
    missing = []
    dirs = [dest / op] if op else [p for p in dest.iterdir() if p.is_dir() and (p / "SPEC.md").exists()] if dest.exists() else []
    if not dirs:
        missing.append("no operation packages found")
    for path in dirs:
        spec = (path / "SPEC.md").read_text() if (path / "SPEC.md").exists() else ""
        runner = (path / "_run_dev.py").read_text() if (path / "_run_dev.py").exists() else ""
        for field in ("kda_spec: 3", "capability:", "performance_policy:", "output_contracts:"):
            if field not in spec:
                missing.append(f"{path.name}: SPEC missing {field}")
        for feature in ("call_explicit", "check_workload", "clone_inputs", "performance_policy=", "dependencies="):
            if feature not in runner:
                missing.append(f"{path.name}: custom runner must merge {feature}")
        report = path / "report.json"
        data = json.loads(report.read_text()) if report.exists() else {}
        if data.get("schema_version") != 2 or not data.get("dependencies"):
            missing.append(f"{path.name}: fresh schema v2 report required; historical evidence is not acceptance")
    version = dest / "_common/VERSION"
    return {"target_spec": 3, "target_runtime": "0.7.0", "installed_runtime": version.read_text().strip() if version.exists() else None,
            "missing": missing, "action": "explicitly merge custom runner and input factory; sync-common updates runtime only"}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--op", help="snake_case op name; omit to only (re)install _common")
    p.add_argument("--dest", required=True, help="the kda_kernels package directory inside the user's repo")
    p.add_argument("--force", action="store_true", help="deprecated; cannot overwrite an existing package")
    p.add_argument("--check-upgrade", action="store_true", help="read-only package compatibility and evidence inspection")
    p.add_argument("--sync-common", action="store_true", help="overwrite dest/_common with the template version")
    p.add_argument(
        "--kernel-backend",
        default="triton",
        choices=KERNEL_BACKENDS,
        help="kernel backend of this package (SPEC kernel_backend; triton is the default, nvmath for a GEMM + epilogue that cuBLASLt expresses, tilelang is experimental); only its _<backend>/ directory is rendered",
    )
    p.add_argument("--capability", choices=("inference", "eager_grad", "training"), default="training")
    args = p.parse_args(argv)

    dest = Path(args.dest).resolve()
    if args.check_upgrade:
        result = upgrade_check(dest, args.op)
        print(json.dumps(result, indent=2))
        return 3 if result["missing"] else 0
    if args.op and (dest / args.op).exists():
        p.error("existing packages require an explicit merge; --force does not overwrite them")
    dest.mkdir(parents=True, exist_ok=True)
    init = dest / "__init__.py"
    if not init.exists():
        init.write_text('"""Kernels generated by KDA. Each subpackage is one fused op; `_common` is shared runtime."""\n')

    version = sync_common(dest, force=args.sync_common)
    if not args.op:
        return 0
    if not OP_NAME_RE.match(args.op):
        sys.exit(f"error: op name {args.op!r} must be snake_case ([a-z][a-z0-9_]*)")

    op_dir = render_op(args.op, dest, common_version=version, force=args.force, kernel_backend=args.kernel_backend, capability=args.capability)
    print(
        "\nnext steps:\n"
        f"  1. fill {op_dir / 'SPEC.md'} (front matter + prose) and {op_dir / '_eager.py'} (verbatim user code)\n"
        f"  2. write the signature and the op-specific contract checks in {op_dir / 'interface.py'},"
        f" the roofline in {op_dir / '_speed_of_light.py'}\n"
        f"  3. sanity check (verify + contract + infer), run from the repo root {dest.parent}:\n"
        f"     python -m {dest.name}.{args.op}._run_dev --verify --backend eager"
        f" --json {dest.name}/{args.op}/report.eager.json --md {dest.name}/{args.op}/report.eager.md\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
